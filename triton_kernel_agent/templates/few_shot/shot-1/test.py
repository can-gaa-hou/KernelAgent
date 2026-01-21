import torch
import torch.nn as nn

def test_kernel():
    """Test the Triton kernel implementation of element-wise addition.
    
    Original problem description:
    - Implements element-wise addition: result = a + b
    - Input shape: two tensors of shape (1024 * 1024,) = (1048576,)
    - Output shape: same as input
    - Uses bfloat16 precision (converted from original FP32 specification)
    """
    try:
        from kernel import kernel_function
        
        # Sanity check: kernel should be callable and self-contained
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        # Device setup
        device = "cuda"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Create test data using EXACT specifications from problem description
        N = 1024 * 1024  # 1048576
        
        # Convert from FP32 to BF16 as per requirements
        dtype = torch.bfloat16
        
        # Generate random inputs with a good range to test numerical accuracy
        # Using rand for uniform distribution in [0, 1) as specified in get_inputs()
        a = torch.rand(N, dtype=dtype, device=device)
        b = torch.rand(N, dtype=dtype, device=device)

        # Compute reference output using the PyTorch Model
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return a + b
        
        # Create reference model and compute expected output
        model = Model().to(device)
        with torch.no_grad():
            expected = model(a, b)
        
        # Call kernel_function as a normal Python function
        # The kernel should handle all Triton-specific logic internally
        result = kernel_function(a, b)
        
        # Basic validation: result should be a tensor
        if not isinstance(result, torch.Tensor):
            print(f"kernel_function did not return a torch.Tensor, got {type(result)}")
            return False
        
        # Device check (avoid comparing to literal 'cuda')
        if result.device != a.device:
            print(f"Device mismatch: input on {a.device}, result on {result.device}")
            return False
        
        # Shape check
        if result.shape != a.shape:
            print(f"Shape mismatch: input shape {a.shape}, result shape {result.shape}")
            return False
        
        # Dtype check
        if result.dtype != dtype:
            print(f"Dtype mismatch: expected {dtype}, got {result.dtype}")
            return False
        
        # Numerical verification with appropriate tolerances
        # For bfloat16: use looser tolerances due to lower precision
        rtol = 1e-2  # Relaxed for bfloat16
        atol = 2e-2  # Relaxed for bfloat16
        
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            print(f"NUMERICAL MISMATCH:")
            print(f"Input shapes: a={a.shape}, b={b.shape}")
            print(f"Input dtypes: a={a.dtype}, b={b.dtype}")
            print(f"Input ranges: a=[{torch.min(a).item():.6f}, {torch.max(a).item():.6f}], "
                  f"b=[{torch.min(b).item():.6f}, {torch.max(b).item():.6f}]")
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            
            # Sample some values for debugging
            print(f"\nSample comparison (first 10 elements):")
            print(f"a:        {a[:10].cpu().numpy()}")
            print(f"b:        {b[:10].cpu().numpy()}")
            print(f"Expected: {expected[:10].cpu().numpy()}")
            print(f"Got:      {result[:10].cpu().numpy()}")
            
            # Statistics for debugging
            diff = torch.abs(result - expected)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            
            # Calculate relative error
            rel_error = torch.abs((result - expected) / (expected + 1e-8))
            max_rel_error = torch.max(rel_error)
            mean_rel_error = torch.mean(rel_error)
            
            print(f"\nDifference statistics:")
            print(f"Max absolute difference: {max_diff.item():.6f}")
            print(f"Mean absolute difference: {mean_diff.item():.6f}")
            print(f"Max relative error: {max_rel_error.item():.6f}")
            print(f"Mean relative error: {mean_rel_error.item():.6f}")
            print(f"Used tolerances: rtol={rtol}, atol={atol}")
            
            # Check for specific patterns of errors
            print(f"\nError pattern analysis:")
            
            # Check if errors are systematic or random
            error_hist = torch.histc(diff, bins=10, min=0, max=max_diff.item())
            print(f"Error distribution (histogram): {error_hist.cpu().numpy()}")
            
            # Check if errors correlate with input magnitude
            input_magnitude = torch.abs(a) + torch.abs(b)
            correlation = torch.corrcoef(torch.stack([diff, input_magnitude]))[0, 1]
            print(f"Correlation between error and input magnitude: {correlation.item():.6f}")
            
            # Check for NaN or Inf values
            if torch.any(torch.isnan(result)):
                print(f"ERROR: Found NaN values in output")
                nan_indices = torch.nonzero(torch.isnan(result))
                print(f"NaN at indices: {nan_indices[:10].cpu().numpy()}")
            
            if torch.any(torch.isinf(result)):
                print(f"ERROR: Found Inf values in output")
                inf_indices = torch.nonzero(torch.isinf(result))
                print(f"Inf at indices: {inf_indices[:10].cpu().numpy()}")
            
            # Check if result preserves addition properties
            print(f"\nAddition property checks:")
            
            # Commutativity: a + b should equal b + a
            result_commutative = kernel_function(b, a)
            if not torch.allclose(result, result_commutative, rtol=rtol, atol=atol):
                print(f"WARNING: Addition may not be commutative")
                comm_diff = torch.max(torch.abs(result - result_commutative))
                print(f"Max difference between a+b and b+a: {comm_diff.item():.6f}")
            
            # Identity: a + 0 should equal a
            zero_tensor = torch.zeros_like(a)
            result_identity = kernel_function(a, zero_tensor)
            if not torch.allclose(result_identity, a, rtol=rtol, atol=atol):
                print(f"WARNING: Identity property may not hold")
                identity_diff = torch.max(torch.abs(result_identity - a))
                print(f"Max difference between a+0 and a: {identity_diff.item():.6f}")
            
            return False
        
        # Additional validation: check addition properties
        print("\nAdditional addition property validation:")
        
        # 1. Check for NaN or Inf values
        if torch.any(torch.isnan(result)):
            print(f"ERROR: Found NaN values in output")
            return False
        if torch.any(torch.isinf(result)):
            print(f"ERROR: Found Inf values in output")
            return False
        
        # 2. Check commutativity
        result_commutative = kernel_function(b, a)
        if not torch.allclose(result, result_commutative, rtol=rtol, atol=atol):
            print(f"WARNING: Addition may not be commutative")
        
        # 3. Check identity property
        zero_tensor = torch.zeros_like(a)
        result_identity = kernel_function(a, zero_tensor)
        if not torch.allclose(result_identity, a, rtol=rtol, atol=atol):
            print(f"WARNING: Identity property may not hold")
        
        # 4. Check range of output
        expected_min = torch.min(a + b).item()
        expected_max = torch.max(a + b).item()
        result_min = torch.min(result).item()
        result_max = torch.max(result).item()
        
        print(f"Output range: expected [{expected_min:.6f}, {expected_max:.6f}], "
              f"got [{result_min:.6f}, {result_max:.6f}]")
        
        # 5. Performance hint (not a failure, just information)
        print(f"Test completed successfully with {N:,} elements")
        
        print("Test passed!")
        return True
        
    except NameError as e:
        # Surface undefined helper issues from kernel.py clearly
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        print("Make sure all functions used in kernel.py are defined within the file.")
        return False
    except ImportError as e:
        print(f"Test failed: Could not import kernel_function: {e}")
        return False
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
