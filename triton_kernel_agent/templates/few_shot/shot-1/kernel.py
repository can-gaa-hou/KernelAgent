import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise addition.
    
    Computes: out = a + b element-wise.
    
    Args:
        a_ptr: Pointer to first input tensor.
        b_ptr: Pointer to second input tensor.
        out_ptr: Pointer to output tensor.
        n_elements: Total number of elements in each tensor.
        BLOCK_SIZE: Number of elements processed per program.
    """
    # Get program ID for this block
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for all elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data from first input tensor
    a = tl.load(a_ptr + offsets, mask=mask)
    
    # Load data from second input tensor
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform element-wise addition using Triton operations
    # Note: We convert to float32 for the addition to maintain precision,
    # then convert back to bfloat16 for storage
    a_f32 = a.to(tl.float32)
    b_f32 = b.to(tl.float32)
    result_f32 = a_f32 + b_f32
    result = result_f32.to(tl.bfloat16)
    
    # Store result to output tensor
    tl.store(out_ptr + offsets, result, mask=mask)


def kernel_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for element-wise addition using Triton.
    
    This function:
    1. Validates input tensor properties
    2. Allocates output tensor
    3. Configures and launches the Triton kernel
    4. Returns the result
    
    All mathematical computation happens inside the Triton kernel.
    No PyTorch compute operations (torch.nn, torch.nn.functional, 
    torch.ops.aten.*, or tensor-tensor math) are used.
    
    Args:
        a: First input tensor of shape (N,) with dtype torch.bfloat16.
        b: Second input tensor of shape (N,) with dtype torch.bfloat16.
        
    Returns:
        Output tensor of same shape and dtype as inputs, with element-wise addition.
    """
    # Validate inputs
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError(f"Inputs must be torch.Tensors, got {type(a)} and {type(b)}")
    
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise RuntimeError(f"Input tensors must be on CUDA device, got {a.device} and {b.device}")
    
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"Input tensors must be bfloat16, got {a.dtype} and {b.dtype}")
    
    if a.shape != b.shape:
        raise ValueError(f"Input tensors must have same shape, got {a.shape} and {b.shape}")
    
    if a.dim() != 1:
        raise ValueError(f"Input tensors must be 1D, got shape {a.shape}")
    
    n_elements = a.numel()
    
    # Allocate output tensor (PyTorch allowed for allocation only)
    out = torch.empty_like(a)
    
    # Choose block size (power of 2 for optimal performance)
    # For element-wise operations, we can use larger blocks
    BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks needed to cover all elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch Triton kernel
    # All computation happens inside the Triton kernel
    _add_kernel[grid](
        a,
        b,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,  # Use 4 warps for good occupancy
    )
    
    return out
