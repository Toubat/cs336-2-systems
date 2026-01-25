"""
Modal script to test the Triton weighted_sum kernel against PyTorch reference.
"""

import modal

app = modal.App("test-weighted-sum")
image = modal.Image.debian_slim().apt_install("wget", "gnupg").uv_sync().add_local_python_source("cs336_systems")


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
)
def test_weighted_sum():
    import torch

    from cs336_systems.kernels.weighted_sum import f_weighted_sum

    def pytorch_weighted_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Reference PyTorch implementation: sum over last dimension of x * weight."""
        return (x * weight).sum(dim=-1)

    def check_close(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ) -> tuple[bool, float]:
        """Check if two tensors are close, return (passed, max_error)."""
        max_error = (actual - expected).abs().max().item()
        passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
        return passed, max_error

    print("=" * 80)
    print("WEIGHTED SUM TRITON KERNEL TEST")
    print("=" * 80)

    # Test configurations: (batch_dims, d)
    test_configs = [
        # 2D inputs
        ((16,), 64),
        ((128,), 256),
        ((1000,), 512),
        # 3D inputs
        ((4, 16), 64),
        ((8, 32), 128),
        ((16, 64), 256),
        # 4D inputs
        ((2, 4, 8), 64),
        ((4, 8, 16), 128),
        # Edge cases
        ((1,), 16),  # Single row
        ((7,), 33),  # Non-power-of-2 dimensions
        ((13, 17), 41),  # Prime dimensions
        ((100,), 1024),  # Larger embedding
    ]

    all_passed = True
    results = []

    for batch_dims, d in test_configs:
        shape = batch_dims + (d,)
        print(f"\nTesting shape: {shape}")
        print("-" * 40)

        # Create inputs with gradients
        x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
        weight = torch.randn(d, device="cuda", dtype=torch.float32, requires_grad=True)

        # Clone for PyTorch reference
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)

        # Forward pass
        triton_out = f_weighted_sum(x, weight)
        pytorch_out = pytorch_weighted_sum(x_ref, weight_ref)

        fwd_passed, fwd_error = check_close("Forward", triton_out, pytorch_out)
        print(f"  Forward:  {'PASS' if fwd_passed else 'FAIL'} (max error: {fwd_error:.2e})")

        # Backward pass
        grad_output = torch.randn_like(triton_out)
        triton_out.backward(grad_output)
        pytorch_out.backward(grad_output)

        grad_x_passed, grad_x_error = check_close("grad_x", x.grad, x_ref.grad)
        grad_w_passed, grad_w_error = check_close("grad_weight", weight.grad, weight_ref.grad)

        print(f"  grad_x:   {'PASS' if grad_x_passed else 'FAIL'} (max error: {grad_x_error:.2e})")
        print(f"  grad_w:   {'PASS' if grad_w_passed else 'FAIL'} (max error: {grad_w_error:.2e})")

        test_passed = fwd_passed and grad_x_passed and grad_w_passed
        all_passed = all_passed and test_passed

        results.append(
            {
                "shape": shape,
                "forward_passed": fwd_passed,
                "forward_error": fwd_error,
                "grad_x_passed": grad_x_passed,
                "grad_x_error": grad_x_error,
                "grad_w_passed": grad_w_passed,
                "grad_w_error": grad_w_error,
                "all_passed": test_passed,
            }
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Shape':<25} {'Forward':<15} {'grad_x':<15} {'grad_w':<15} {'Status':<10}")
    print("-" * 80)

    for r in results:
        shape_str = str(r["shape"])
        fwd_str = f"{r['forward_error']:.2e}"
        gx_str = f"{r['grad_x_error']:.2e}"
        gw_str = f"{r['grad_w_error']:.2e}"
        status = "PASS" if r["all_passed"] else "FAIL"
        print(f"{shape_str:<25} {fwd_str:<15} {gx_str:<15} {gw_str:<15} {status:<10}")

    print("-" * 80)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)

    return {"all_passed": all_passed, "results": results}


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
)
def benchmark_weighted_sum():
    """Benchmark Triton vs PyTorch weighted_sum."""
    import time

    import torch

    from cs336_systems.kernels.weighted_sum import f_weighted_sum

    def pytorch_weighted_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (x * weight).sum(dim=-1)

    print("=" * 80)
    print("WEIGHTED SUM BENCHMARK (Triton vs PyTorch)")
    print("=" * 80)

    configs = [
        ((1024,), 512),
        ((4096,), 1024),
        ((8, 1024), 512),
        ((16, 2048), 1024),
        ((32, 4096), 2048),
    ]

    num_warmup = 10
    num_iters = 100

    results = []

    for batch_dims, d in configs:
        shape = batch_dims + (d,)
        print(f"\nShape: {shape}")

        x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
        weight = torch.randn(d, device="cuda", dtype=torch.float32, requires_grad=True)

        # Warmup Triton
        for _ in range(num_warmup):
            out = f_weighted_sum(x, weight)
            out.sum().backward()
            x.grad = None
            weight.grad = None

        torch.cuda.synchronize()

        # Benchmark Triton forward
        start = time.perf_counter()
        for _ in range(num_iters):
            out = f_weighted_sum(x, weight)
            torch.cuda.synchronize()
        triton_fwd_time = (time.perf_counter() - start) / num_iters * 1000

        # Benchmark Triton backward
        grad_out = torch.randn_like(out)
        start = time.perf_counter()
        for _ in range(num_iters):
            x.grad = None
            weight.grad = None
            out = f_weighted_sum(x, weight)
            out.backward(grad_out)
            torch.cuda.synchronize()
        triton_total_time = (time.perf_counter() - start) / num_iters * 1000
        triton_bwd_time = triton_total_time - triton_fwd_time

        # Warmup PyTorch
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        for _ in range(num_warmup):
            out = pytorch_weighted_sum(x_ref, weight_ref)
            out.sum().backward()
            x_ref.grad = None
            weight_ref.grad = None

        torch.cuda.synchronize()

        # Benchmark PyTorch forward
        start = time.perf_counter()
        for _ in range(num_iters):
            out = pytorch_weighted_sum(x_ref, weight_ref)
            torch.cuda.synchronize()
        pytorch_fwd_time = (time.perf_counter() - start) / num_iters * 1000

        # Benchmark PyTorch backward
        start = time.perf_counter()
        for _ in range(num_iters):
            x_ref.grad = None
            weight_ref.grad = None
            out = pytorch_weighted_sum(x_ref, weight_ref)
            out.backward(grad_out)
            torch.cuda.synchronize()
        pytorch_total_time = (time.perf_counter() - start) / num_iters * 1000
        pytorch_bwd_time = pytorch_total_time - pytorch_fwd_time

        print(f"  Triton:  fwd={triton_fwd_time:.3f}ms, bwd={triton_bwd_time:.3f}ms")
        print(f"  PyTorch: fwd={pytorch_fwd_time:.3f}ms, bwd={pytorch_bwd_time:.3f}ms")
        print(
            f"  Speedup: fwd={pytorch_fwd_time / triton_fwd_time:.2f}x, bwd={pytorch_bwd_time / triton_bwd_time:.2f}x"
        )

        results.append(
            {
                "shape": shape,
                "triton_fwd_ms": triton_fwd_time,
                "triton_bwd_ms": triton_bwd_time,
                "pytorch_fwd_ms": pytorch_fwd_time,
                "pytorch_bwd_ms": pytorch_bwd_time,
            }
        )

    return results


@app.local_entrypoint()
def main(benchmark: bool = False):
    # Run correctness tests
    result = test_weighted_sum.remote()
    print(f"\nTest result: {'PASSED' if result['all_passed'] else 'FAILED'}")

    # Optionally run benchmarks
    if benchmark:
        print("\n" + "=" * 80)
        print("Running benchmarks...")
        benchmark_weighted_sum.remote()
