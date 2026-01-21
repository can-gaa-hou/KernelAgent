import torch
import torch.nn as nn

def test_kernel():
    """Test the Triton kernel implementation of ReLU activation.
    
    Original problem description:
    - Simple model that performs ReLU activation
    - Input shape: (4096, 393216)
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
        # Convert from FP32 to BF16 as per requirements
        batch_size = 4096
        dim = 393216
        
        # Generate random input with both positive and negative values
        # Using randn for normal distribution to test both sides of ReLU
        x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device=device)

        # Compute reference output using the PyTorch Model
        class Model(nn.Module):
            """
            Simple model that performs a ReLU activation.
            """
            def __init__(self):
                super(Model, self).__init__()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Applies ReLU activation to the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of any shape.

                Returns:
                    torch.Tensor: Output tensor with ReLU applied, same shape as input.
                """
                return torch.relu(x)
        
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
        if result.dtype != torch.bfloat16:
            print(f"Dtype mismatch: expected bfloat16, got {result.dtype}")
            return False
        
        # Numerical verification with appropriate tolerances for bfloat16
        # bfloat16 has lower precision than float32, so we use looser tolerances
        rtol = 1e-2  # Relaxed for bfloat16
        atol = 2e-2  # Relaxed for bfloat16
        
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            print(f"NUMERICAL MISMATCH:")
            print(f"Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            
            # Sample some values for debugging
            print(f"\nSample comparison (first 10 elements):")
            print(f"Expected: {expected.flatten()[:10].cpu().numpy()}")
            print(f"Got:      {result.flatten()[:10].cpu().numpy()}")
            
            # Statistics for debugging
            diff = torch.abs(result - expected)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            
            print(f"\nDifference statistics:")
            print(f"Max absolute difference: {max_diff.item()}")
            print(f"Mean absolute difference: {mean_diff.item()}")
            
            # Check for common ReLU implementation errors
            print(f"\nReLU-specific checks:")
            print(f"Input min: {torch.min(x).item()}, max: {torch.max(x).item()}")
            print(f"Result min: {torch.min(result).item()}, max: {torch.max(result).item()}")
            print(f"Expected min: {torch.min(expected).item()}, max: {torch.max(expected).item()}")
            
            # Check if negative values were properly zeroed
            negative_mask = x < 0
            if torch.any(negative_mask):
                negative_result = result[negative_mask]
                if torch.any(negative_result != 0):
                    print(f"ERROR: Found {torch.sum(negative_result != 0)} non-zero outputs for negative inputs")
                    print(f"Non-zero values: {negative_result[negative_result != 0][:5]}")
            
            # Check if positive values were preserved
            positive_mask = x > 0
            if torch.any(positive_mask):
                positive_diff = torch.abs(result[positive_mask] - x[positive_mask])
                max_positive_diff = torch.max(positive_diff)
                if max_positive_diff > atol:
                    print(f"ERROR: Positive values not preserved accurately")
                    print(f"Max difference for positive inputs: {max_positive_diff.item()}")
            
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
