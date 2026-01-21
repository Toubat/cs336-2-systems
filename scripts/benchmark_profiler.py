"""
Benchmark script using torch.profiler instead of nsys.
This works on Modal and shows CUDA kernel information.
"""

from contextlib import nullcontext

import chz
import torch
from torch.profiler import ProfilerActivity, profile

from cs336_systems.config import ModelConfig
from cs336_systems.optim import AdamW
from cs336_systems.profile import encode_profile, extract_profiler_stats
from cs336_systems.transformer_lm import TransformerLM
from cs336_systems.utils import get_random_batch


def run_profiled(model, x, num_steps: int, mode: str = "forward", optimizer=None, use_amp: bool = False):
    """Run model with profiling enabled.

    Args:
        model: The model to profile
        x: Input tensor
        num_steps: Number of steps to run
        mode: One of "forward", "backward", or "train"
        optimizer: Required if mode is "train"
        use_amp: Whether to use automatic mixed precision (BF16)
    """
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(num_steps):
            with amp_context:
                if mode == "forward":
                    with torch.no_grad():
                        _ = model(x)
                elif mode == "backward":
                    logits = model(x)
                    logits.mean().backward()
                    model.zero_grad()
                elif mode == "train":
                    assert optimizer is not None, "optimizer required for train mode"
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    logits.mean().backward()
                    optimizer.step()
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
    use_amp: bool = False,
):
    device = "cuda:0"
    config = ModelConfig()
    precision_str = "BF16 (mixed precision)" if use_amp else "FP32 (full precision)"

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {size} model: d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}, num_heads={num_heads}")
    print(f"Context length: {context_length}, Batch size: {batch_size}, Precision: {precision_str}")
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

    optimizer = AdamW(model.parameters())

    x = get_random_batch(
        vocab_size=config.vocab_size,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # Warmup (full training step)
    print(f"Running {warmup_steps} warm-up steps...")
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    for _ in range(warmup_steps):
        with amp_context:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            logits.mean().backward()
            optimizer.step()
    torch.cuda.synchronize()

    # Profile forward only
    print(f"\nProfiling forward pass ({precision_str})...")
    prof_forward = run_profiled(model, x, num_steps, mode="forward", use_amp=use_amp)

    print("\n" + "=" * 80)
    print(f"FORWARD PASS ({precision_str}) - CUDA Kernel Summary")
    print("=" * 80)
    print(prof_forward.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    # Profile forward + backward
    print(f"\nProfiling forward+backward pass ({precision_str})...")
    prof_backward = run_profiled(model, x, num_steps, mode="backward", use_amp=use_amp)

    print("\n" + "=" * 80)
    print(f"FORWARD+BACKWARD PASS ({precision_str}) - CUDA Kernel Summary")
    print("=" * 80)
    print(prof_backward.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    # Profile full training step (forward + backward + optimizer)
    print(f"\nProfiling full training step ({precision_str})...")
    prof_train = run_profiled(model, x, num_steps, mode="train", optimizer=optimizer, use_amp=use_amp)

    print("\n" + "=" * 80)
    print(f"FULL TRAINING STEP ({precision_str}) - CUDA Kernel Summary")
    print("=" * 80)
    print(prof_train.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    # Export Chrome traces
    amp_suffix = "_amp" if use_amp else ""
    prof_forward.export_chrome_trace(f"/tmp/trace_forward_{size}{amp_suffix}.json")
    prof_backward.export_chrome_trace(f"/tmp/trace_backward_{size}{amp_suffix}.json")
    prof_train.export_chrome_trace(f"/tmp/trace_train_{size}{amp_suffix}.json")
    print(f"\nChrome traces exported to /tmp/trace_*_{size}{amp_suffix}.json")

    # Build results using ProfileStats
    forward_stats = extract_profiler_stats(prof_forward)
    backward_stats = extract_profiler_stats(prof_backward)
    train_stats = extract_profiler_stats(prof_train)

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
        "use_amp": use_amp,
        "precision": "BF16" if use_amp else "FP32",
        "forward": forward_stats.to_dict(),
        "backward": backward_stats.to_dict(),
        "train": train_stats.to_dict(),
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
    use_amp: bool = False  # Enable BF16 mixed precision


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
        use_amp=config.use_amp,
    )


if __name__ == "__main__":
    chz.nested_entrypoint(main)
