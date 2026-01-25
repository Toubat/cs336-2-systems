"""
Benchmark PyTorch attention implementation at different scales.
- Batch size fixed to 8
- No multihead attention (single head)
- Iterates through d_model x seq_len configurations
- Times 100 forward and backward passes
- Supports torch.compile for JIT compilation
"""

import json
import time
from typing import Callable

import chz
import numpy as np
import torch

from cs336_systems.nn.utils import scaled_dot_product_attention

# Configuration space
D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]
BATCH_SIZE = 8
NUM_STEPS = 100
WARMUP_STEPS = 5


def benchmark_attention_config(
    d_model: int,
    seq_len: int,
    batch_size: int,
    num_steps: int,
    warmup_steps: int,
    use_compile: bool = False,
) -> dict:
    """Benchmark a single attention configuration."""
    device = "cuda:0"
    compile_str = "compiled" if use_compile else "eager"

    print(f"\n{'=' * 60}")
    print(f"Benchmarking attention ({compile_str}): d_model={d_model}, seq_len={seq_len}")
    print(f"Batch size: {batch_size}, Steps: {num_steps}")
    print(f"{'=' * 60}")

    try:
        # Create causal mask
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()

        # Get attention function (compiled or not)
        attn_fn: Callable = scaled_dot_product_attention
        if use_compile:
            print("Compiling attention function with torch.compile...")
            attn_fn = torch.compile(scaled_dot_product_attention)

        # Warmup (extra important for compiled version)
        print(f"Running {warmup_steps} warm-up steps...")
        for _ in range(warmup_steps):
            # Create fresh inputs for warmup
            q_warm = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            k_warm = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            v_warm = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

            out = attn_fn(q_warm, k_warm, v_warm, mask)
            out.sum().backward()
            torch.cuda.synchronize()

        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Benchmark forward pass (100 iterations)
        print(f"Benchmarking forward pass ({num_steps} steps)...")
        forward_times: list[float] = []

        for _ in range(num_steps):
            # Create fresh inputs to avoid caching effects
            q_fwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=False)
            k_fwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=False)
            v_fwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=False)

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = attn_fn(q_fwd, k_fwd, v_fwd, mask)
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - start) * 1000)  # ms

        # Measure memory before backward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Do one forward pass to set up for memory measurement
        q_mem = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        k_mem = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        v_mem = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        _ = attn_fn(q_mem, k_mem, v_mem, mask)
        torch.cuda.synchronize()

        memory_before_backward_mb = torch.cuda.memory_allocated() / (1024**2)
        print(f"Memory before backward: {memory_before_backward_mb:.2f} MB")

        # Benchmark backward pass (100 iterations)
        print(f"Benchmarking backward pass ({num_steps} steps)...")
        backward_times: list[float] = []

        for _ in range(num_steps):
            # Create fresh inputs with gradients
            q_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            k_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            v_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

            # Forward pass
            out = attn_fn(q_bwd, k_bwd, v_bwd, mask)
            torch.cuda.synchronize()

            # Time backward pass only
            start = time.perf_counter()
            out.sum().backward()
            torch.cuda.synchronize()
            backward_times.append((time.perf_counter() - start) * 1000)  # ms

        forward_times_arr = np.array(forward_times)
        backward_times_arr = np.array(backward_times)

        results = {
            "d_model": d_model,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "use_compile": use_compile,
            "mode": compile_str,
            "forward_mean_ms": float(forward_times_arr.mean()),
            "forward_std_ms": float(forward_times_arr.std()),
            "backward_mean_ms": float(backward_times_arr.mean()),
            "backward_std_ms": float(backward_times_arr.std()),
            "memory_before_backward_mb": memory_before_backward_mb,
            "status": "OK",
        }

        print(
            f"\nResults: forward={results['forward_mean_ms']:.2f}±{results['forward_std_ms']:.2f}ms, "
            f"backward={results['backward_mean_ms']:.2f}±{results['backward_std_ms']:.2f}ms"
        )

        return results

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM Error: {e}")
        torch.cuda.empty_cache()
        return {
            "d_model": d_model,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "use_compile": use_compile,
            "mode": compile_str,
            "status": "OOM",
            "error": str(e),
        }


@chz.chz(typecheck=True)
class AttentionBenchmarkConfig:
    d_model: int = 64
    seq_len: int = 1024
    batch_size: int = BATCH_SIZE
    num_steps: int = NUM_STEPS
    warmup_steps: int = WARMUP_STEPS
    use_compile: bool = False  # Use torch.compile for JIT compilation


def main(config: AttentionBenchmarkConfig):
    """Run a single attention benchmark configuration."""
    result = benchmark_attention_config(
        d_model=config.d_model,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_steps=config.num_steps,
        warmup_steps=config.warmup_steps,
        use_compile=config.use_compile,
    )

    # Output encoded result for Modal parsing
    print("\nATTENTION_RESULT_START")
    print(json.dumps(result))
    print("ATTENTION_RESULT_END")

    return result


if __name__ == "__main__":
    chz.nested_entrypoint(main)
