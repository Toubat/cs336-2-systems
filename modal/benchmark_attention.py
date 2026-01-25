"""
Modal benchmark for PyTorch attention at different scales.
Launches parallel tasks for each (d_model, seq_len) configuration.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import modal

app = modal.App("benchmark-attention")
image = (
    modal.Image.debian_slim()
    .apt_install("wget", "gnupg")
    .uv_sync()
    .add_local_python_source("cs336_systems")
    .add_local_dir("scripts", remote_path="/scripts")
)

# Configuration space
D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]
BATCH_SIZE = 8
NUM_STEPS = 100
WARMUP_STEPS = 5

OUTPUT_DIR = "./attention_benchmarks"


def create_heatmaps(all_results: list[dict], output_dir: str, timestamp: str, mode: str = "eager"):
    """Create heatmap visualizations for forward time, backward time, and memory."""
    # Build 2D matrices
    d_models = D_MODEL_VALUES
    seq_lens = SEQ_LEN_VALUES

    # Initialize matrices with NaN for missing/OOM values
    forward_matrix = np.full((len(d_models), len(seq_lens)), np.nan)
    backward_matrix = np.full((len(d_models), len(seq_lens)), np.nan)
    memory_matrix = np.full((len(d_models), len(seq_lens)), np.nan)

    # Fill matrices from results
    for r in all_results:
        if r.get("status") != "OK":
            continue
        d_idx = d_models.index(r["d_model"])
        s_idx = seq_lens.index(r["seq_len"])
        forward_matrix[d_idx, s_idx] = r.get("forward_mean_ms", np.nan)
        backward_matrix[d_idx, s_idx] = r.get("backward_mean_ms", np.nan)
        memory_matrix[d_idx, s_idx] = r.get("memory_before_backward_mb", np.nan)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Heatmap settings
    seq_len_labels = [str(s) for s in seq_lens]
    d_model_labels = [str(d) for d in d_models]

    # Forward time heatmap
    im0 = axes[0].imshow(forward_matrix, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(len(seq_lens)))
    axes[0].set_xticklabels(seq_len_labels, rotation=45, ha="right")
    axes[0].set_yticks(range(len(d_models)))
    axes[0].set_yticklabels(d_model_labels)
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("d_model")
    axes[0].set_title("Forward Time (ms)")
    plt.colorbar(im0, ax=axes[0])

    # Add text annotations
    for i in range(len(d_models)):
        for j in range(len(seq_lens)):
            val = forward_matrix[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
            else:
                axes[0].text(j, i, "OOM", ha="center", va="center", fontsize=8, color="gray")

    # Backward time heatmap
    im1 = axes[1].imshow(backward_matrix, cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(len(seq_lens)))
    axes[1].set_xticklabels(seq_len_labels, rotation=45, ha="right")
    axes[1].set_yticks(range(len(d_models)))
    axes[1].set_yticklabels(d_model_labels)
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("d_model")
    axes[1].set_title("Backward Time (ms)")
    plt.colorbar(im1, ax=axes[1])

    # Add text annotations
    for i in range(len(d_models)):
        for j in range(len(seq_lens)):
            val = backward_matrix[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
            else:
                axes[1].text(j, i, "OOM", ha="center", va="center", fontsize=8, color="gray")

    # Memory heatmap
    im2 = axes[2].imshow(memory_matrix, cmap="YlGnBu", aspect="auto")
    axes[2].set_xticks(range(len(seq_lens)))
    axes[2].set_xticklabels(seq_len_labels, rotation=45, ha="right")
    axes[2].set_yticks(range(len(d_models)))
    axes[2].set_yticklabels(d_model_labels)
    axes[2].set_xlabel("Sequence Length")
    axes[2].set_ylabel("d_model")
    axes[2].set_title("Memory Before Backward (MB)")
    plt.colorbar(im2, ax=axes[2])

    # Add text annotations
    for i in range(len(d_models)):
        for j in range(len(seq_lens)):
            val = memory_matrix[i, j]
            if not np.isnan(val):
                axes[2].text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
            else:
                axes[2].text(j, i, "OOM", ha="center", va="center", fontsize=8, color="gray")

    plt.suptitle(f"PyTorch Attention Benchmark - {mode.upper()} (batch_size={BATCH_SIZE})", fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, f"attention_heatmap_{mode}_{timestamp}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved heatmap: {fig_path}")

    plt.close()


def create_comparison_heatmaps(
    eager_results: list[dict], compiled_results: list[dict], output_dir: str, timestamp: str
):
    """Create heatmap showing speedup from torch.compile (compiled / eager)."""
    d_models = D_MODEL_VALUES
    seq_lens = SEQ_LEN_VALUES

    # Build lookup dicts
    def build_lookup(results: list[dict]) -> dict:
        lookup = {}
        for r in results:
            if r.get("status") == "OK":
                lookup[(r["d_model"], r["seq_len"])] = r
        return lookup

    eager_lookup = build_lookup(eager_results)
    compiled_lookup = build_lookup(compiled_results)

    # Compute speedup matrices
    forward_speedup = np.full((len(d_models), len(seq_lens)), np.nan)
    backward_speedup = np.full((len(d_models), len(seq_lens)), np.nan)

    for i, d in enumerate(d_models):
        for j, s in enumerate(seq_lens):
            key = (d, s)
            if key in eager_lookup and key in compiled_lookup:
                eager_fwd = eager_lookup[key].get("forward_mean_ms", 0)
                compiled_fwd = compiled_lookup[key].get("forward_mean_ms", 0)
                if compiled_fwd > 0:
                    forward_speedup[i, j] = eager_fwd / compiled_fwd

                eager_bwd = eager_lookup[key].get("backward_mean_ms", 0)
                compiled_bwd = compiled_lookup[key].get("backward_mean_ms", 0)
                if compiled_bwd > 0:
                    backward_speedup[i, j] = eager_bwd / compiled_bwd

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seq_len_labels = [str(s) for s in seq_lens]
    d_model_labels = [str(d) for d in d_models]

    # Forward speedup heatmap
    im0 = axes[0].imshow(forward_speedup, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.0)
    axes[0].set_xticks(range(len(seq_lens)))
    axes[0].set_xticklabels(seq_len_labels, rotation=45, ha="right")
    axes[0].set_yticks(range(len(d_models)))
    axes[0].set_yticklabels(d_model_labels)
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("d_model")
    axes[0].set_title("Forward Speedup (eager/compiled)")
    plt.colorbar(im0, ax=axes[0])

    for i in range(len(d_models)):
        for j in range(len(seq_lens)):
            val = forward_speedup[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=8)

    # Backward speedup heatmap
    im1 = axes[1].imshow(backward_speedup, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.0)
    axes[1].set_xticks(range(len(seq_lens)))
    axes[1].set_xticklabels(seq_len_labels, rotation=45, ha="right")
    axes[1].set_yticks(range(len(d_models)))
    axes[1].set_yticklabels(d_model_labels)
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("d_model")
    axes[1].set_title("Backward Speedup (eager/compiled)")
    plt.colorbar(im1, ax=axes[1])

    for i in range(len(d_models)):
        for j in range(len(seq_lens)):
            val = backward_speedup[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=8)

    plt.suptitle("torch.compile Speedup (>1 means compiled is faster)", fontsize=14)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"attention_compile_speedup_{timestamp}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved speedup heatmap: {fig_path}")

    plt.close()


def parse_result_from_stdout(stdout: str) -> dict | None:
    """Extract single attention result from stdout."""
    start_marker = "ATTENTION_RESULT_START"
    end_marker = "ATTENTION_RESULT_END"

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
def run_attention_benchmark(
    d_model: int,
    seq_len: int,
    batch_size: int,
    num_steps: int,
    warmup_steps: int,
    use_compile: bool = False,
) -> dict:
    """Run attention benchmark for a single configuration on Modal."""
    import subprocess

    cmd = [
        "python",
        "/scripts/benchmark_attention.py",
        f"d_model={d_model}",
        f"seq_len={seq_len}",
        f"batch_size={batch_size}",
        f"num_steps={num_steps}",
        f"warmup_steps={warmup_steps}",
        f"use_compile={use_compile}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse result from stdout
    parsed_result = parse_result_from_stdout(result.stdout)

    return {
        "d_model": d_model,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "use_compile": use_compile,
        "returncode": result.returncode,
        "parsed_result": parsed_result,
    }


@app.local_entrypoint()
def main(
    batch_size: int = BATCH_SIZE,
    num_steps: int = NUM_STEPS,
    warmup_steps: int = WARMUP_STEPS,
    compare_compile: bool = True,  # Run both eager and compiled versions
):
    """Run attention benchmarks on Modal H100."""

    print("Running PyTorch Attention Benchmark")
    print(f"d_model values: {D_MODEL_VALUES}")
    print(f"seq_len values: {SEQ_LEN_VALUES}")
    print(f"Batch size: {batch_size}, Steps: {num_steps}, Warmup: {warmup_steps}")
    print(f"Compare compile: {compare_compile}")
    print()

    # Determine compile modes to run
    compile_modes = [False, True] if compare_compile else [False]

    # Create all configurations
    configs = [
        (d_model, seq_len, batch_size, num_steps, warmup_steps, use_compile)
        for d_model in D_MODEL_VALUES
        for seq_len in SEQ_LEN_VALUES
        for use_compile in compile_modes
    ]

    print(f"Launching {len(configs)} parallel tasks...")
    results = list(run_attention_benchmark.starmap(configs))

    # Collect all parsed results
    all_results: list[dict] = []
    for r in results:
        if r.get("parsed_result"):
            all_results.append(r["parsed_result"])

    # Separate eager and compiled results
    eager_results = [r for r in all_results if not r.get("use_compile", False)]
    compiled_results = [r for r in all_results if r.get("use_compile", False)]

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_filename = f"attention_benchmark_{timestamp}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {json_path}")

    # Print summary tables
    for mode_name, mode_results in [("EAGER", eager_results), ("COMPILED", compiled_results)]:
        if not mode_results:
            continue

        print("\n" + "=" * 120)
        print(f"ATTENTION BENCHMARK SUMMARY - {mode_name}")
        print("=" * 120)
        print(f"{'d_model':<10} {'seq_len':<10} {'Forward (ms)':<22} {'Backward (ms)':<22} {'Mem (MB)':<15} {'Status':<10}")
        print("-" * 120)

        sorted_results = sorted(mode_results, key=lambda x: (x.get("d_model", 0), x.get("seq_len", 0)))

        for r in sorted_results:
            d_model = r.get("d_model", "--")
            seq_len = r.get("seq_len", "--")
            status = r.get("status", "UNKNOWN")

            if status == "OOM":
                print(f"{d_model:<10} {seq_len:<10} {'--':<22} {'--':<22} {'--':<15} {'OOM':<10}")
            elif status == "OK":
                forward = f"{r.get('forward_mean_ms', 0):.2f} ± {r.get('forward_std_ms', 0):.2f}"
                backward = f"{r.get('backward_mean_ms', 0):.2f} ± {r.get('backward_std_ms', 0):.2f}"
                mem = f"{r.get('memory_before_backward_mb', 0):.2f}"
                print(f"{d_model:<10} {seq_len:<10} {forward:<22} {backward:<22} {mem:<15} {'OK':<10}")
            else:
                print(f"{d_model:<10} {seq_len:<10} {'--':<22} {'--':<22} {'--':<15} {status:<10}")

    print(f"\nCompleted {len(all_results)} benchmark(s)")

    # Print OOM analysis if any
    oom_configs = [r for r in all_results if r.get("status") == "OOM"]
    if oom_configs:
        print("\n" + "=" * 60)
        print("OUT OF MEMORY CONFIGURATIONS")
        print("=" * 60)
        for r in oom_configs:
            mode = "compiled" if r.get("use_compile") else "eager"
            print(f"  d_model={r['d_model']}, seq_len={r['seq_len']} ({mode})")

    # Create heatmap visualizations
    if eager_results:
        create_heatmaps(eager_results, OUTPUT_DIR, timestamp, mode="eager")
    if compiled_results:
        create_heatmaps(compiled_results, OUTPUT_DIR, timestamp, mode="compiled")

    # Create comparison heatmap if both modes were run
    if eager_results and compiled_results:
        create_comparison_heatmaps(eager_results, compiled_results, OUTPUT_DIR, timestamp)
