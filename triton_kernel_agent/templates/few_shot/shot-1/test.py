import torch

def test_kernel():
    """Test the fused dot-compress Triton kernel implementation."""
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

        # Use EXACT specifications from problem description
        B = 2048
        NUM_EMB = 512
        DIM = 128
        NUM_COMPRESS_EMB = 32
        
        # Use bfloat16 instead of float32 as per requirements
        dtype = torch.bfloat16
        
        # Create test data using the provided function
        x = torch.randn(
            (B, NUM_EMB, DIM),
            device=device,
            dtype=dtype,
        )
        y = torch.randn(
            (B, NUM_EMB, NUM_COMPRESS_EMB),
            device=device,
            dtype=dtype,
        )
        z = torch.randn(
            (B, DIM, NUM_COMPRESS_EMB),
            device=device,
            dtype=dtype,
        )

        # Compute reference output using the PyTorch Model
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                # Fused dot-compress operation
                # Step 1: Compute X^T @ Y (batch matrix multiplication)
                xty = torch.bmm(x.permute(0, 2, 1), y)
                
                # Step 2: Add Z and multiply with X
                out = torch.bmm(x, xty + z)
                
                return out

        model = Model().to(device=device, dtype=dtype)
        expected = model(x, y, z)

        # Call kernel_function as a normal Python function
        result = kernel_function(x, y, z)

        # Device check (avoid comparing to literal 'cuda')
        if isinstance(result, torch.Tensor):
            if result.device != x.device:
                print(f"Device mismatch: result on {result.device}, input on {x.device}")
                return False
        else:
            print(f"Result is not a torch.Tensor: {type(result)}")
            return False

        # Shape check
        if result.shape != expected.shape:
            print(f"Shape mismatch: expected {expected.shape}, got {result.shape}")
            return False

        # Dtype check
        if result.dtype != expected.dtype:
            print(f"Dtype mismatch: expected {expected.dtype}, got {result.dtype}")
            return False

        # Numerical verification with appropriate tolerances
        # For bfloat16: use looser tolerances due to lower precision
        rtol = 1e-2  # Relaxed for bfloat16
        atol = 2e-2  # Relaxed for bfloat16
        
        # Additional tolerance relaxation due to large accumulation dimension (512)
        # Matrix multiplication accumulates errors across NUM_EMB dimension
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            print(f"NUMERICAL MISMATCH:")
            print(f"Input shapes: x={x.shape}, y={y.shape}, z={z.shape}")
            print(f"All dtypes: x={x.dtype}, y={y.dtype}, z={z.dtype}")
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            
            # Sample comparison
            print(f"\nSample comparison (first 10 elements from batch 0):")
            expected_flat = expected[0].flatten()
            result_flat = result[0].flatten()
            for i in range(min(10, len(expected_flat))):
                print(f"  [{i}] Expected: {expected_flat[i]:.6f}, Got: {result_flat[i]:.6f}, "
                      f"Diff: {abs(expected_flat[i] - result_flat[i]):.6f}")
            
            # Statistical comparison
            abs_diff = torch.abs(result - expected)
            max_abs_diff = torch.max(abs_diff)
            mean_abs_diff = torch.mean(abs_diff)
            
            # Avoid division by zero for relative error
            rel_error = torch.abs((result - expected) / (expected + 1e-8))
            max_rel_error = torch.max(rel_error)
            mean_rel_error = torch.mean(rel_error)
            
            print(f"\nError statistics:")
            print(f"Max absolute difference: {max_abs_diff:.6f}")
            print(f"Mean absolute difference: {mean_abs_diff:.6f}")
            print(f"Max relative error: {max_rel_error:.6f}")
            print(f"Mean relative error: {mean_rel_error:.6f}")
            print(f"Used tolerances: rtol={rtol}, atol={atol}")
            
            return False

        print("Test passed!")
        return True

    except NameError as e:
        # Surface undefined helper issues from kernel.py clearly
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        import traceback
        traceback.print_exc()
        return False
    except ImportError as e:
        print(f"Test failed: ImportError (cannot import kernel_function): {e}")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
