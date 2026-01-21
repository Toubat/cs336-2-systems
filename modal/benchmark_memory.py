"""
Modal memory profiling for the 2.7B model with different context lengths.
Launches parallel tasks and uses ephemeral volume for file transfer.
"""

import json
import os
from datetime import datetime

import modal

app = modal.App("benchmark-memory")
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .uv_sync()
    .add_local_python_source("cs336_systems")
    .add_local_dir("scripts", remote_path="/scripts")
)

# Default configurations
CONTEXT_LENGTHS = [128, 256, 512]
MODES = ["forward", "train"]
OUTPUT_DIR = "./memory_snapshots"
REMOTE_OUTPUT_DIR = "/tmp/memory_snapshots"


def parse_result_from_stdout(stdout: str) -> dict | None:
    """Extract memory result from stdout."""
    start_marker = "MEMORY_RESULT_START"
    end_marker = "MEMORY_RESULT_END"

    if start_marker not in stdout or end_marker not in stdout:
        return None

    start_idx = stdout.index(start_marker) + len(start_marker)
    end_idx = stdout.index(end_marker)
    json_str = stdout[start_idx:end_idx].strip()

    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse results: {e}")
        return None


@app.function(
    image=image,
    gpu="H100",
    cpu=32,
    timeout=10000,
)
def run_memory_profile(
    context_length: int,
    batch_size: int,
    mode: str,
    use_amp: bool,
) -> dict:
    """Run memory profiling for a single configuration on Modal."""
    import subprocess

    cmd = [
        "python",
        "/scripts/benchmark_memory.py",
        f"context_length={context_length}",
        f"batch_size={batch_size}",
        f"mode={mode}",
        f"use_amp={use_amp}",
        f"output_dir={REMOTE_OUTPUT_DIR}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Read the pickle file to return
    precision_str = "bf16" if use_amp else "fp32"
    pickle_filename = f"memory_{mode}_{context_length}_{precision_str}.pickle"
    pickle_path = f"{REMOTE_OUTPUT_DIR}/{pickle_filename}"

    pickle_data = None
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            pickle_data = f.read()
        print(f"Read pickle file: {pickle_path} ({len(pickle_data)} bytes)")

    # Parse result from stdout
    parsed_result = parse_result_from_stdout(result.stdout)

    return {
        "context_length": context_length,
        "mode": mode,
        "use_amp": use_amp,
        "precision": precision_str.upper(),
        "returncode": result.returncode,
        "parsed_result": parsed_result,
        "pickle_data": pickle_data,
        "pickle_filename": pickle_filename,
    }


@app.local_entrypoint()
def main(
    batch_size: int = 4,
    use_amp: bool = False,
    compare_precision: bool = False,  # Run both FP32 and BF16
    context_length: int | None = None,  # Single context length (None = all)
    mode: str | None = None,  # Single mode (None = all)
):
    """Run memory profiling on Modal H100."""

    # Determine what to run
    context_lengths = [context_length] if context_length else CONTEXT_LENGTHS
    modes = [mode] if mode else MODES
    amp_modes = [False, True] if compare_precision else [use_amp]

    precision_str = "FP32 vs BF16" if compare_precision else ("BF16" if use_amp else "FP32")
    print(f"Running memory profiling ({precision_str})")
    print(f"Context lengths: {context_lengths}")
    print(f"Modes: {modes}")
    print(f"Batch size: {batch_size}")
    print()

    # Launch all tasks in parallel using starmap
    configs = [(ctx_len, batch_size, m, amp) for ctx_len in context_lengths for m in modes for amp in amp_modes]

    print(f"Launching {len(configs)} parallel tasks...")
    results = list(run_memory_profile.starmap(configs))

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save pickle files locally
    all_summaries = []
    for r in results:
        # Save pickle file
        if r.get("pickle_data"):
            pickle_path = os.path.join(OUTPUT_DIR, r["pickle_filename"])
            with open(pickle_path, "wb") as f:
                f.write(r["pickle_data"])
            print(f"Saved: {pickle_path}")

        # Collect summary
        if r.get("parsed_result"):
            all_summaries.append(r["parsed_result"])

    # Save combined JSON summary
    if all_summaries:
        suffix = "_compare" if compare_precision else ("_bf16" if use_amp else "_fp32")
        json_filename = f"memory_summary_{timestamp}{suffix}.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        with open(json_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSaved summary: {json_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("MEMORY PROFILING SUMMARY")
    print("=" * 100)
    header = f"{'Context':<10} {'Mode':<10} {'Precision':<10} {'Peak (MB)':<15} {'Allocated (MB)':<15} {'Status':<10}"
    print(header)
    print("-" * 100)

    for r in sorted(
        all_summaries,
        key=lambda x: (
            x.get("context_length", 0),
            x.get("mode", ""),
            x.get("precision", ""),
        ),
    ):
        if "error" in r:
            print(
                f"{r.get('context_length', '--'):<10} "
                f"{r.get('mode', '--'):<10} "
                f"{r.get('precision', '--'):<10} "
                f"{'--':<15} "
                f"{'--':<15} "
                f"{r.get('error', 'ERROR'):<10}"
            )
        else:
            print(
                f"{r.get('context_length', '--'):<10} "
                f"{r.get('mode', '--'):<10} "
                f"{r.get('precision', '--'):<10} "
                f"{r.get('peak_memory_mb', 0):<15.2f} "
                f"{r.get('allocated_memory_mb', 0):<15.2f} "
                f"{'OK':<10}"
            )

    print(f"\nCompleted {len(results)} benchmark(s)")
    print(f"Pickle files saved to: {OUTPUT_DIR}/")
    print("Upload pickle files to https://pytorch.org/memory_viz for visualization")
