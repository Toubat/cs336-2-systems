import time
from contextlib import nullcontext

import chz
import numpy as np
import torch

from cs336_systems.config import ModelConfig
from cs336_systems.transformer_lm import TransformerLM
from cs336_systems.utils import get_random_batch


def benchmark_model(
    size: str,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    warmup_steps: int,
    num_steps: int,
    batch_size: int,
    use_amp: bool = False,
):
    device = "cuda:0"
    config = ModelConfig()
    try:
        nvtx = torch.cuda.nvtx
    except Exception:
        nvtx = None

    precision_str = "BF16" if use_amp else "FP32"
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {size} model: d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"Precision: {precision_str}")
    print(f"{'=' * 60}")

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=config.theta,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e9:.2f}B)")

    x = get_random_batch(
        vocab_size=config.vocab_size,
        batch_size=batch_size,
        context_length=config.context_length,
        device=device,
    )

    # Warmup
    print(f"Running {warmup_steps} warm-up steps...")
    if nvtx is not None:
        nvtx.range_push("warmup")
    for _ in range(warmup_steps):
        with amp_context:
            logits = model(x)
            logits.mean().backward()
        model.zero_grad()
    if nvtx is not None:
        nvtx.range_pop()
    torch.cuda.synchronize()

    # Benchmark forward pass
    print(f"Benchmarking forward pass ({num_steps} steps)...")
    forward_times: list[float] = []
    if nvtx is not None:
        nvtx.range_push("forward")
    for _ in range(num_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with amp_context:
            _ = model(x)
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)  # ms
    if nvtx is not None:
        nvtx.range_pop()

    # Benchmark backward pass (forward + backward)
    print(f"Benchmarking forward+backward pass ({num_steps} steps)...")
    backward_times: list[float] = []
    if nvtx is not None:
        nvtx.range_push("forward+backward")
    for _ in range(num_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with amp_context:
            logits = model(x)
            logits.mean().backward()
        torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)  # ms
        model.zero_grad()
    if nvtx is not None:
        nvtx.range_pop()

    forward_times_arr = np.array(forward_times)
    backward_times_arr = np.array(backward_times)

    results = {
        "size": size,
        "num_params": num_params,
        "batch_size": batch_size,
        "use_amp": use_amp,
        "precision": precision_str,
        "forward_mean": float(forward_times_arr.mean()),
        "forward_std": float(forward_times_arr.std()),
        "backward_mean": float(backward_times_arr.mean()),
        "backward_std": float(backward_times_arr.std()),
    }

    return results


@chz.chz(typecheck=True)
class BenchmarkConfig:
    size: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    warmup_steps: int = 10
    num_steps: int = 20
    batch_size: int = 4
    use_amp: bool = False  # Enable BF16 mixed precision


def main(config: BenchmarkConfig):
    results = benchmark_model(
        size=config.size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        warmup_steps=config.warmup_steps,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        use_amp=config.use_amp,
    )
    print(results)


if __name__ == "__main__":
    chz.nested_entrypoint(main)


"""
================================================================================
SUMMARY (baseline with no warmup steps)
================================================================================
Size       Params          Forward (ms)         Backward (ms)       
--------------------------------------------------------------------------------
small      0.13B         332.11 ± 912.81      138.79 ± 189.63
medium     0.42B         393.73 ± 948.10      274.08 ± 189.79
large      0.97B         480.69 ± 981.22      473.54 ± 175.91
xl         2.00B         590.70 ± 957.01      858.04 ± 156.91
2.7B       3.41B         638.48 ± 891.29      1153.99 ± 142.38

================================================================================
SUMMARY (baseline with 1 warmup step)
================================================================================
Size       Params          Forward (ms)         Backward (ms)       
--------------------------------------------------------------------------------
small      0.13B         28.59 ± 7.65      72.88 ± 2.69
medium     0.42B         67.84 ± 7.60      195.97 ± 5.54
large      0.97B         144.81 ± 12.43      420.34 ± 10.32
xl         2.00B         311.57 ± 34.22      815.77 ± 5.45
2.7B       3.41B         342.14 ± 3.07      1118.18 ± 23.78 

================================================================================
SUMMARY (baseline with 5 warmup steps)
================================================================================
Size       Params          Forward (ms)         Backward (ms)
--------------------------------------------------------------------------------
small      0.13B          22.93 ± 1.78         74.21 ± 2.54
medium     0.42B          74.14 ± 7.76         192.62 ± 4.30
large      0.97B          158.85 ± 9.93        419.77 ± 7.16
xl         2.00B          282.74 ± 3.86        823.70 ± 24.34
2.7B       3.41B          343.50 ± 3.02        1104.09 ± 1.65
"""
