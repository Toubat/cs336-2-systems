"""
Benchmark script using torch.profiler instead of nsys.
This works on Modal and shows CUDA kernel information.
"""

import chz
import torch
from torch.profiler import ProfilerActivity, profile

from cs336_systems.config import ModelConfig
from cs336_systems.profile import encode_profile, extract_profiler_stats
from cs336_systems.transformer_lm import TransformerLM
from cs336_systems.utils import get_random_batch


def run_profiled(model, x, num_steps: int, include_backward: bool = False):
    """Run model with profiling enabled."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(num_steps):
            if include_backward:
                logits = model(x)
                logits.mean().backward()
                model.zero_grad()
            else:
                with torch.no_grad():
                    _ = model(x)
            torch.cuda.synchronize()
    return prof


def benchmark_model(
    size: str,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    warmup_steps: int,
    num_steps: int,
    batch_size: int,
    context_length: int = 128,
):
    device = "cuda:0"
    config = ModelConfig()

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {size} model: d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"Context length: {context_length}, Batch size: {batch_size}")
    print(f"{'=' * 60}")

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=config.theta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e9:.2f}B)")

    x = get_random_batch(
        vocab_size=config.vocab_size,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # Warmup
    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        logits = model(x)
        logits.mean().backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Profile forward only
    print("\nProfiling forward pass...")
    prof_forward = run_profiled(model, x, num_steps, include_backward=False)

    print("\n" + "=" * 80)
    print("FORWARD PASS - CUDA Kernel Summary")
    print("=" * 80)
    print(prof_forward.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    # Profile forward + backward
    print("\nProfiling forward+backward pass...")
    prof_backward = run_profiled(model, x, num_steps, include_backward=True)

    print("\n" + "=" * 80)
    print("FORWARD+BACKWARD PASS - CUDA Kernel Summary")
    print("=" * 80)
    print(prof_backward.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    # Export Chrome traces
    prof_forward.export_chrome_trace(f"/tmp/trace_forward_{size}.json")
    prof_backward.export_chrome_trace(f"/tmp/trace_backward_{size}.json")
    print(f"\nChrome traces exported to /tmp/trace_forward_{size}.json and /tmp/trace_backward_{size}.json")

    # Build results using ProfileStats
    forward_stats = extract_profiler_stats(prof_forward)
    backward_stats = extract_profiler_stats(prof_backward)

    results = {
        "size": size,
        "num_params": num_params,
        "batch_size": batch_size,
        "context_length": context_length,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_steps": num_steps,
        "forward": forward_stats.to_dict(),
        "backward": backward_stats.to_dict(),
    }

    # Output compressed data for efficient Modal transfer
    encoded = encode_profile(results)
    print(f"\nCompressed size: {len(encoded)} bytes (vs ~{len(str(results))} uncompressed)")
    print("\nENCODED_RESULTS_START")
    print(encoded)
    print("ENCODED_RESULTS_END")

    return results


@chz.chz(typecheck=True)
class BenchmarkConfig:
    size: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    warmup_steps: int = 5
    num_steps: int = 10
    batch_size: int = 4
    context_length: int = 128


def main(config: BenchmarkConfig):
    benchmark_model(
        size=config.size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        warmup_steps=config.warmup_steps,
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        context_length=config.context_length,
    )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
