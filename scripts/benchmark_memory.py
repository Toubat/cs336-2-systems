"""
Memory profiling script for the 2.7B model.
Runs a single configuration (context_length, mode, precision).
Outputs pickle file for pytorch.org/memory_viz and JSON summary.
"""

import json
import os
from contextlib import nullcontext

import chz
import torch

from cs336_systems.config import ModelConfig
from cs336_systems.optim import AdamW
from cs336_systems.transformer_lm import TransformerLM
from cs336_systems.utils import get_random_batch

# 2.7B model configuration (fixed)
MODEL_CONFIG = {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
}


def run_memory_profile(
    context_length: int,
    batch_size: int,
    mode: str,
    use_amp: bool,
    output_dir: str,
) -> dict:
    """Run memory profiling for a single configuration."""
    device = "cuda:0"
    config = ModelConfig()
    precision_str = "bf16" if use_amp else "fp32"

    print(f"\n{'=' * 60}")
    print("Memory Profiling: 2.7B model")
    print(f"Context length: {context_length}, Batch size: {batch_size}")
    print(f"Mode: {mode}, Precision: {precision_str.upper()}")
    print(f"{'=' * 60}")

    # Clear any existing memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=context_length,
        num_layers=MODEL_CONFIG["num_layers"],
        d_model=MODEL_CONFIG["d_model"],
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
        theta=config.theta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e9:.2f}B)")

    # Create optimizer (needed for train mode)
    optimizer = AdamW(model.parameters()) if mode == "train" else None

    # Create input batch
    x = get_random_batch(
        vocab_size=config.vocab_size,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # AMP context
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # Warmup (important for accurate memory profiling)
    print("Running warmup...")
    for _ in range(2):
        with amp_context:
            if mode == "forward":
                with torch.no_grad():
                    _ = model(x)
            else:  # train
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                logits.mean().backward()
                optimizer.step()
        torch.cuda.synchronize()

    # Clear memory stats after warmup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Start recording memory history with full stack traces
    print("Recording memory snapshot with stack traces...")
    torch.cuda.memory._record_memory_history(max_entries=1000000, stacks="all")

    # Run the profiled step
    with amp_context:
        if mode == "forward":
            with torch.no_grad():
                _ = model(x)
        else:  # train
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            logits.mean().backward()
            optimizer.step()
    torch.cuda.synchronize()

    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    allocated_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    reserved_memory = torch.cuda.memory_reserved() / (1024**2)  # MB

    print("\nMemory Stats:")
    print(f"  Peak allocated: {peak_memory:.2f} MB")
    print(f"  Current allocated: {allocated_memory:.2f} MB")
    print(f"  Reserved: {reserved_memory:.2f} MB")

    # Save memory snapshot (pickle file for pytorch.org/memory_viz)
    os.makedirs(output_dir, exist_ok=True)
    pickle_filename = f"memory_{mode}_{context_length}_{precision_str}.pickle"
    pickle_path = os.path.join(output_dir, pickle_filename)
    torch.cuda.memory._dump_snapshot(pickle_path)
    print(f"\nMemory snapshot saved: {pickle_path}")

    # Stop recording
    torch.cuda.memory._record_memory_history(enabled=None)

    # Build summary result
    result = {
        "context_length": context_length,
        "batch_size": batch_size,
        "mode": mode,
        "precision": precision_str.upper(),
        "use_amp": use_amp,
        "num_params": num_params,
        "peak_memory_mb": peak_memory,
        "allocated_memory_mb": allocated_memory,
        "reserved_memory_mb": reserved_memory,
        "pickle_file": pickle_filename,
    }

    # Save JSON summary for this run
    json_filename = f"memory_{mode}_{context_length}_{precision_str}.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON summary saved: {json_path}")

    return result


@chz.chz(typecheck=True)
class MemoryBenchmarkConfig:
    context_length: int = 128
    batch_size: int = 4
    mode: str = "forward"  # "forward" or "train"
    use_amp: bool = False
    output_dir: str = "./memory_snapshots"


def main(config: MemoryBenchmarkConfig):
    try:
        result = run_memory_profile(
            context_length=config.context_length,
            batch_size=config.batch_size,
            mode=config.mode,
            use_amp=config.use_amp,
            output_dir=config.output_dir,
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Context: {result['context_length']}, Mode: {result['mode']}, Precision: {result['precision']}")
        print(f"Peak Memory: {result['peak_memory_mb']:.2f} MB")
        print(f"Pickle file: {result['pickle_file']}")

        # Output encoded results for Modal parsing
        print("\nMEMORY_RESULT_START")
        print(json.dumps(result))
        print("MEMORY_RESULT_END")

        return result

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM Error: {e}")
        error_result = {
            "context_length": config.context_length,
            "batch_size": config.batch_size,
            "mode": config.mode,
            "precision": "BF16" if config.use_amp else "FP32",
            "use_amp": config.use_amp,
            "error": "OOM",
        }
        print("\nMEMORY_RESULT_START")
        print(json.dumps(error_result))
        print("MEMORY_RESULT_END")
        return error_result


if __name__ == "__main__":
    chz.nested_entrypoint(main)
