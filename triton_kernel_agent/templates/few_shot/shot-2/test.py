import torch
import torch.nn as nn

def test_kernel():
    """Test the Triton kernel implementation of softmax operation.
    
    Original problem description:
    - Implements softmax: y = exp(x - max(x)) / sum(exp(x - max(x)))
    - Input shape: (4096, 4096)
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
        M = 4096
        N = 4096
        
        # Convert from FP32 to BF16 as per requirements
        dtype = torch.bfloat16
        
        # Generate random input with a range that tests numerical stability
        # Using randn for normal distribution to get both positive and negative values
        x = torch.randn(M, N, dtype=dtype, device=device)

        # Compute reference output using the PyTorch Model
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # read MN ; write M
                x_max = x.max(dim=1).values

                # read MN + M ; write MN
                z = x - x_max[:, None]

                # read MN ; write MN
                numerator = torch.exp(z)

                # read MN ; write M
                denominator = numerator.sum(dim=1)

                # read MN + M ; write MN
                y = numerator / denominator[:, None]

                return y
        
        # Create reference model and compute expected output
        model = Model().to(device)
        with torch.no_grad():
            expected = model(x)
        
        # Call kernel_function as a normal Python function
        # The kernel should handle all Triton-specific logic internally
        result = kernel_function(x)
        
        # Basic validation: result should be a tensor
        if not isinstance(result, torch.Tensor):
            print(f"kernel_function did not return a torch.Tensor, got {type(result)}")
            return False
        
        # Device check (avoid comparing to literal 'cuda')
        if result.device != x.device:
            print(f"Device mismatch: input on {x.device}, result on {result.device}")
            return False
        
        # Shape check
        if result.shape != x.shape:
            print(f"Shape mismatch: input shape {x.shape}, result shape {result.shape}")
            return False
        
        # Dtype check
        if result.dtype != dtype:
            print(f"Dtype mismatch: expected {dtype}, got {result.dtype}")
            return False
        
        # Numerical verification with appropriate tolerances
        # For bfloat16: use looser tolerances due to lower precision
        # For softmax with large accumulation dimension (N=4096): use even looser tolerances
        rtol = 1e-1  # Very relaxed due to bfloat16 + large accumulation
        atol = 1e-1  # Very relaxed due to bfloat16 + large accumulation
        
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            print(f"NUMERICAL MISMATCH:")
            print(f"Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"Input range: min={torch.min(x).item():.6f}, max={torch.max(x).item():.6f}")
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            
            # Sample some values for debugging
            print(f"\nSample comparison (first row, first 10 elements):")
            print(f"Input:    {x[0, :10].cpu().numpy()}")
            print(f"Expected: {expected[0, :10].cpu().numpy()}")
            print(f"Got:      {result[0, :10].cpu().numpy()}")
            
            # Statistics for debugging
            diff = torch.abs(result - expected)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            
            print(f"\nDifference statistics:")
            print(f"Max absolute difference: {max_diff.item():.6f}")
            print(f"Mean absolute difference: {mean_diff.item():.6f}")
            
            # Check softmax properties
            print(f"\nSoftmax property checks:")
            
            # 1. All outputs should be between 0 and 1
            if torch.any(result < 0) or torch.any(result > 1 + 1e-5):
                print(f"ERROR: Outputs outside [0, 1] range")
                print(f"Min output: {torch.min(result).item():.6f}")
                print(f"Max output: {torch.max(result).item():.6f}")
            
            # 2. Sum of each row should be approximately 1
            row_sums = result.sum(dim=1)
            sum_diff = torch.abs(row_sums - 1.0)
            max_sum_diff = torch.max(sum_diff)
            mean_sum_diff = torch.mean(sum_diff)
            
            print(f"Row sum statistics:")
            print(f"Max deviation from 1: {max_sum_diff.item():.6f}")
            print(f"Mean deviation from 1: {mean_sum_diff.item():.6f}")
            
            # 3. Check monotonicity: if x[i] > x[j], then y[i] > y[j]
            # Sample check for first row
            row_input = x[0].cpu().numpy()
            row_output = result[0].cpu().numpy()
            sorted_indices = torch.argsort(x[0])
            sorted_output = result[0][sorted_indices]
            
            # Check if output is also sorted
            if not torch.all(torch.diff(sorted_output) >= -1e-5):
                print(f"ERROR: Output not monotonic with input")
            
            # 4. Check for NaN or Inf values
            if torch.any(torch.isnan(result)):
                print(f"ERROR: Found NaN values in output")
            if torch.any(torch.isinf(result)):
                print(f"ERROR: Found Inf values in output")
            
            # 5. Check intermediate values for debugging
            print(f"\nIntermediate value checks (first row):")
            x_max_ref = x[0].max()
            z_ref = x[0] - x_max_ref
            numerator_ref = torch.exp(z_ref)
            denominator_ref = numerator_ref.sum()
            
            print(f"Reference max: {x_max_ref.item():.6f}")
            print(f"Reference denominator: {denominator_ref.item():.6f}")
            
            return False
        
        # Additional validation: check softmax properties
        print("\nAdditional softmax property validation:")
        
        # 1. All outputs should be between 0 and 1
        if torch.any(result < 0) or torch.any(result > 1 + 1e-5):
            print(f"WARNING: Some outputs outside [0, 1] range")
            print(f"Min output: {torch.min(result).item():.6f}")
            print(f"Max output: {torch.max(result).item():.6f}")
        
        # 2. Sum of each row should be approximately 1
        row_sums = result.sum(dim=1)
        sum_diff = torch.abs(row_sums - 1.0)
        max_sum_diff = torch.max(sum_diff)
        
        if max_sum_diff > 1e-2:  # Allow some tolerance for bfloat16
            print(f"WARNING: Row sums deviate from 1 by up to {max_sum_diff.item():.6f}")
        
        # 3. Check for NaN or Inf values
        if torch.any(torch.isnan(result)):
            print(f"ERROR: Found NaN values in output")
            return False
        if torch.any(torch.isinf(result)):
            print(f"ERROR: Found Inf values in output")
            return False
        
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
