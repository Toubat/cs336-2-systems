import modal

from cs336_systems.transformer_lm import TransformerLM

app = modal.App("benchmark-e2e")
image = modal.Image.debian_slim().uv_sync().add_local_python_source("cs336_systems")

# Model configurations from Table 1
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

with image.imports():
    import time

    import numpy as np
    import torch

    from cs336_systems.config import ModelConfig
    from cs336_systems.utils import get_random_batch


@app.function(
    image=image,
    gpu="H100",
    cpu=32,
    timeout=10000,
)
def benchmark_model(
    size: str,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    warmup_steps: int,
    num_steps: int,
    batch_size: int,
):
    device = "cuda:0"
    config = ModelConfig()

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {size} model: d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}, num_heads={num_heads}")
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
    for _ in range(warmup_steps):
        logits = model(x)
        logits.mean().backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Benchmark forward pass
    print(f"Benchmarking forward pass ({num_steps} steps)...")
    forward_times: list[float] = []
    for _ in range(num_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)  # ms

    # Benchmark backward pass (forward + backward)
    print(f"Benchmarking forward+backward pass ({num_steps} steps)...")
    backward_times: list[float] = []
    for _ in range(num_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(x)
        logits.mean().backward()
        torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)  # ms
        model.zero_grad()

    forward_times_arr = np.array(forward_times)
    backward_times_arr = np.array(backward_times)

    results = {
        "size": size,
        "num_params": num_params,
        "batch_size": batch_size,
        "forward_mean": forward_times_arr.mean(),
        "forward_std": forward_times_arr.std(),
        "backward_mean": backward_times_arr.mean(),
        "backward_std": backward_times_arr.std(),
    }

    print(f"\nResults for {size}:")
    print(f"  Forward:  {results['forward_mean']:.2f} ± {results['forward_std']:.2f} ms")
    print(f"  Backward: {results['backward_mean']:.2f} ± {results['backward_std']:.2f} ms")

    return results


@app.local_entrypoint()
async def main(warmup_steps: int = 1, num_steps: int = 10, batch_size: int = 4):
    import asyncio

    print(f"Running benchmarks with warmup_steps={warmup_steps}, num_steps={num_steps}, batch_size={batch_size}")

    tasks = [
        benchmark_model.remote.aio(
            size=size,
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            warmup_steps=warmup_steps,
            num_steps=num_steps,
            batch_size=batch_size,
        )
        for size, config in MODEL_CONFIGS.items()
    ]
    all_results = await asyncio.gather(*tasks)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Size':<10} {'Params':<15} {'Forward (ms)':<20} {'Backward (ms)':<20}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['size']:<10} {r['num_params'] / 1e9:.2f}B{'':<8} "
            f"{r['forward_mean']:.2f} ± {r['forward_std']:.2f}{'':<5} "
            f"{r['backward_mean']:.2f} ± {r['backward_std']:.2f}"
        )


"""
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
"""
