import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for softmax operation.
    
    Computes: y = exp(x - max(x)) / sum(exp(x - max(x))) for each row.
    
    Implements the online softmax algorithm for numerical stability:
    1. Find maximum value in each row
    2. Compute exp(x - max) for each element
    3. Sum the exponentials for each row
    4. Normalize by the sum
    
    Args:
        x_ptr: Pointer to input tensor of shape (M, N).
        out_ptr: Pointer to output tensor of shape (M, N).
        M: Number of rows.
        N: Number of columns.
        stride_m: Stride along the row dimension.
        stride_n: Stride along the column dimension.
        BLOCK_SIZE: Number of columns processed per program.
    """
    # Get program ID for this row
    row_idx = tl.program_id(axis=0)
    
    # Check if this row is within bounds
    if row_idx >= M:
        return
    
    # Calculate pointer to the start of this row
    row_start_ptr = x_ptr + row_idx * stride_m
    
    # Initialize max value for this row (negative infinity)
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    
    # Initialize sum of exponentials for this row
    row_sum = tl.zeros((), dtype=tl.float32)
    
    # Process the row in blocks to find max and compute sum
    for col_start in range(0, N, BLOCK_SIZE):
        # Create offsets for this block
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for boundary conditions
        mask = col_offsets < N
        
        # Load block of data from this row
        x_block = tl.load(
            row_start_ptr + col_offsets * stride_n,
            mask=mask,
            other=float('-inf')
        ).to(tl.float32)
        
        # Update row max
        row_max = tl.maximum(row_max, tl.max(x_block, axis=0))
    
    # Now compute exponentials and their sum
    for col_start in range(0, N, BLOCK_SIZE):
        # Create offsets for this block
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for boundary conditions
        mask = col_offsets < N
        
        # Load block of data from this row
        x_block = tl.load(
            row_start_ptr + col_offsets * stride_n,
            mask=mask,
            other=float('-inf')
        ).to(tl.float32)
        
        # Compute x - max for numerical stability
        x_shifted = x_block - row_max
        
        # Compute exp(x - max)
        exp_block = tl.exp(x_shifted)
        
        # Accumulate sum of exponentials
        row_sum += tl.sum(exp_block, axis=0)
    
    # Finally, compute normalized softmax values
    for col_start in range(0, N, BLOCK_SIZE):
        # Create offsets for this block
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for boundary conditions
        mask = col_offsets < N
        
        # Load block of data from this row
        x_block = tl.load(
            row_start_ptr + col_offsets * stride_n,
            mask=mask,
            other=float('-inf')
        ).to(tl.float32)
        
        # Compute x - max for numerical stability
        x_shifted = x_block - row_max
        
        # Compute exp(x - max)
        exp_block = tl.exp(x_shifted)
        
        # Normalize by sum
        softmax_block = exp_block / row_sum
        
        # Convert back to bfloat16 for output
        softmax_block_bf16 = softmax_block.to(tl.bfloat16)
        
        # Calculate output pointer for this block
        out_row_start_ptr = out_ptr + row_idx * stride_m
        tl.store(
            out_row_start_ptr + col_offsets * stride_n,
            softmax_block_bf16,
            mask=mask
        )


def kernel_function(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for softmax operation using Triton.
    
    This function:
    1. Validates input tensor properties
    2. Allocates output tensor
    3. Configures and launches the Triton kernel
    4. Returns the result
    
    All mathematical computation happens inside the Triton kernel.
    No PyTorch compute operations (torch.nn, torch.nn.functional, 
    torch.ops.aten.*, or tensor-tensor math) are used.
    
    Args:
        x: Input tensor of shape (M, N) with dtype torch.bfloat16.
        
    Returns:
        Output tensor of same shape and dtype as input, with softmax applied row-wise.
    """
    # Validate input
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
    
    if x.device.type != "cuda":
        raise RuntimeError(f"Input tensor must be on CUDA device, got {x.device}")
    
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Input tensor must be bfloat16, got {x.dtype}")
    
    if x.dim() != 2:
        raise ValueError(f"Input tensor must be 2D, got shape {x.shape}")
    
    M, N = x.shape
    
    # Allocate output tensor (PyTorch allowed for allocation only)
    out = torch.empty_like(x)
    
    # Get tensor strides
    stride_m, stride_n = x.stride()
    
    # Choose block size (power of 2 for optimal performance)
    # For softmax, we process columns in blocks
    BLOCK_SIZE = 1024
    
    # Calculate grid size (one thread block per row)
    grid = (M,)
    
    # Launch Triton kernel
    # All computation happens inside the Triton kernel
    _softmax_kernel[grid](
        x,
        out,
        M,
        N,
        stride_m,
        stride_n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,  # Use more warps for better occupancy
    )
    
    return out
