"""
Modal benchmark using torch.profiler instead of nsys.
This works on Modal and shows CUDA kernel information!
"""

import asyncio
import json
import os
from datetime import datetime

import modal
from cs336_systems.profile import decode_profile

app = modal.App("benchmark-profiler")
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .uv_sync()
    .add_local_python_source("cs336_systems")
    .add_local_dir("scripts", remote_path="/scripts")
)

# Model configurations from Table 1
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

OUTPUT_DIR = "./profiler_results"


def parse_encoded_from_stdout(stdout: str) -> dict | None:
    """Extract and decode compressed results from stdout."""
    start_marker = "ENCODED_RESULTS_START"
    end_marker = "ENCODED_RESULTS_END"

    if start_marker not in stdout or end_marker not in stdout:
        return None

    start_idx = stdout.index(start_marker) + len(start_marker)
    end_idx = stdout.index(end_marker)
    encoded = stdout[start_idx:end_idx].strip()

    try:
        result = decode_profile(encoded)
        return result if isinstance(result, dict) else result[0] if result else None
    except Exception as e:
        print(f"Failed to decode: {e}")
        return None


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
    context_length: int,
):
    import subprocess

    cmd = [
        "python",
        "/scripts/benchmark_profiler.py",
        f"size={size}",
        f"d_model={d_model}",
        f"d_ff={d_ff}",
        f"num_layers={num_layers}",
        f"num_heads={num_heads}",
        f"warmup_steps={warmup_steps}",
        f"num_steps={num_steps}",
        f"batch_size={batch_size}",
        f"context_length={context_length}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return {
        "size": size,
        "context_length": context_length,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.local_entrypoint()
async def main(
    warmup_steps: int = 5,
    num_steps: int = 10,
    batch_size: int = 4,
    context_length: int = 128,
):
    print(f"Running profiler benchmark: context_length={context_length}, batch_size={batch_size}")

    results = await asyncio.gather(
        *[
            benchmark_model.remote.aio(
                size=size,
                d_model=config["d_model"],
                d_ff=config["d_ff"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                warmup_steps=warmup_steps,
                num_steps=num_steps,
                batch_size=batch_size,
                context_length=context_length,
            )
            for size, config in MODEL_CONFIGS.items()
        ]
    )

    # Parse and decode compressed results, save combined to local disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = [parsed for result in results if (parsed := parse_encoded_from_stdout(result["stdout"]))]

    if all_results:
        filename = f"{OUTPUT_DIR}/profile_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved: {filename}")

    print(f"\nCompleted {len(all_results)} benchmark(s)")
