"""Phase 5.2 — Scaling analysis script.

Reads scaling sweep results and generates:
  - scaling_runtime.png   (log-log: runtime vs N)
  - scaling_circuit.png   (log-log: circuit depth vs N)
  - scaling_summary.csv   (table: N, n, qubits, depth, runtime)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.serialization import load_experiment_result

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300,
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]


def plot_scaling(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generate log-log scaling plots for runtime and circuit depth vs N."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    N_vals = summary_df["N"].values
    runtime_vals = summary_df["runtime_sec_mean"].values
    depth_vals = summary_df["circuit_depth"].values

    # --- Plot 1: Runtime vs N ---
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(N_vals, runtime_vals, "o-", color="#2b5c8f", linewidth=2.2,
              markersize=8, label="Measured Runtime")

    log_N = np.log2(N_vals.astype(float))
    ref_N = np.array(N_vals, dtype=float)

    # Fit polynomial reference line (N^2 expected for statevector)
    if len(N_vals) >= 2:
        log_T = np.log2(runtime_vals.astype(float))
        coeffs = np.polyfit(log_N, log_T, 1)
        fitted_exp = coeffs[0]
        ref_T = 2 ** np.polyval(coeffs, log_N)
        ax.loglog(ref_N, ref_T, "--", color="#d95f02", linewidth=1.5,
                  label=f"Poly fit slope ≈ {fitted_exp:.2f}")

    ax.set_xlabel("Modulus $N$")
    ax.set_ylabel("Runtime per Configuration (s)")
    ax.set_title("Simulation Runtime Scaling vs. Modulus $N$ (log-log)")
    ax.set_xticks(list(N_vals))
    ax.set_xticklabels([str(int(n)) for n in N_vals])
    ax.legend(frameon=True)

    p1 = output_dir / "scaling_runtime.png"
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # --- Plot 2: Circuit Depth vs N ---
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.loglog(N_vals, depth_vals, "s-", color="#7570b3", linewidth=2.2,
              markersize=8, label="Circuit Depth")

    if len(N_vals) >= 2:
        log_D = np.log2(depth_vals.astype(float))
        coeffs_d = np.polyfit(log_N, log_D, 1)
        slope_d = coeffs_d[0]
        ref_D = 2 ** np.polyval(coeffs_d, log_N)
        ax.loglog(ref_N, ref_D, "--", color="#d95f02", linewidth=1.5,
                  label=f"Poly fit slope ≈ {slope_d:.2f}")

    ax.set_xlabel("Modulus $N$")
    ax.set_ylabel("QFT Circuit Depth (gates)")
    ax.set_title("Circuit Depth Scaling vs. Modulus $N$ (log-log)")
    ax.set_xticks(list(N_vals))
    ax.set_xticklabels([str(int(n)) for n in N_vals])
    ax.legend(frameon=True)

    p2 = output_dir / "scaling_circuit_depth.png"
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 5.2 scaling results.")
    parser.add_argument("--input", type=str,
                        default="results/raw/dcp_scaling/dcp_scaling_sweep.parquet")
    parser.add_argument("--output-dir", type=str,
                        default="results/figures/dcp_scaling")
    parser.add_argument("--agg-dir", type=str,
                        default="results/aggregated")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    df = load_experiment_result(input_path)
    print(f"Loaded {len(df):,} trials.")

    # Build scaling summary: one row per N (aggregate over shots)
    summary = df.groupby(["N", "n"]).agg(
        circuit_depth=("circuit_depth", "first") if "circuit_depth" in df.columns else ("N", "count"),
        num_qubits=("num_qubits", "first") if "num_qubits" in df.columns else ("n", "first"),
        runtime_sec_mean=("runtime_sec", "mean") if "runtime_sec" in df.columns else ("N", "count"),
    ).reset_index()

    # Handle case where trial-level df doesn't have per-row runtime
    # We need the summary rows - read from the metadata JSON
    import json
    meta_path = input_path.parent / (input_path.stem + "_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        summary_rows = meta.get("summary", [])
        if summary_rows:
            summary = pd.DataFrame(summary_rows)
            summary = summary.rename(columns={"runtime_sec": "runtime_sec_mean"})

    print("\nScaling Summary:")
    print(summary.to_string(index=False))

    # Save scaling table
    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)
    csv_path = agg_dir / "scaling_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved scaling table to: {csv_path}")

    # Generate plots
    out_dir = Path(args.output_dir)
    paths = plot_scaling(summary, out_dir)
    for p in paths:
        print(f"  [scaling plot]  {p}")


if __name__ == "__main__":
    main()
