import modal

app = modal.App("benchmark-e2e")
image = (
    modal.Image.debian_slim()
    .apt_install("wget", "gnupg")
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

with image.imports():
    import ast


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
    use_amp: bool = False,
) -> dict:
    import subprocess

    result = subprocess.run(
        [
            "python",
            "/scripts/benchmark.py",
            f"size={size}",
            f"d_model={d_model}",
            f"d_ff={d_ff}",
            f"num_layers={num_layers}",
            f"num_heads={num_heads}",
            f"warmup_steps={warmup_steps}",
            f"num_steps={num_steps}",
            f"batch_size={batch_size}",
            f"use_amp={use_amp}",
        ],
        capture_output=True,
        text=True,
    )

    # Print subprocess output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse the results dict from the last line
    output = result.stdout.strip()
    last_line = output.split("\n")[-1]
    parsed = ast.literal_eval(last_line)

    return parsed


@app.local_entrypoint()
async def main(
    warmup_steps: int = 5,
    num_steps: int = 10,
    batch_size: int = 4,
    use_amp: bool = False,
    compare_precision: bool = False,  # Run both FP32 and BF16 for comparison
):
    import asyncio

    if compare_precision:
        print(
            f"Running benchmarks (FP32 vs BF16) with warmup_steps={warmup_steps}, num_steps={num_steps}, batch_size={batch_size}"
        )
        amp_modes = [False, True]
    else:
        precision_str = "BF16" if use_amp else "FP32"
        print(
            f"Running benchmarks ({precision_str}) with warmup_steps={warmup_steps}, num_steps={num_steps}, batch_size={batch_size}"
        )
        amp_modes = [use_amp]
    print()

    all_results = await asyncio.gather(
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
                use_amp=amp,
            )
            for size, config in MODEL_CONFIGS.items()
            for amp in amp_modes
        ]
    )

    # Print summary table
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"{'Size':<10} {'Precision':<10} {'Params':<12} {'Forward (ms)':<22} {'Backward (ms)':<22}")
    print("-" * 110)

    # Sort by model size order, then by precision
    size_order = ["small", "medium", "large", "xl", "2.7B"]
    sorted_results = sorted(
        all_results,
        key=lambda x: (size_order.index(x["size"]) if x["size"] in size_order else 999, x.get("use_amp", False)),
    )

    for r in sorted_results:
        precision = r.get("precision", "BF16" if r.get("use_amp") else "FP32")
        params = f"{r.get('num_params', 0) / 1e9:.2f}B"
        forward = f"{r.get('forward_mean', 0):.2f} ± {r.get('forward_std', 0):.2f}"
        backward = f"{r.get('backward_mean', 0):.2f} ± {r.get('backward_std', 0):.2f}"
        print(f"{r['size']:<10} {precision:<10} {params:<12} {forward:<22} {backward:<22}")


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
