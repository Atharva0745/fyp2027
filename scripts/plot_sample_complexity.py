"""Plotting script for Phase 4 sample complexity experiments."""

import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.serialization import load_experiment_result

# Configure cohesive publication visual style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]


def plot_recovery_vs_samples(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    # Calculate success probabilities
    if "mirror_correct" in df.columns:
        metric_col = "mirror_correct"
    else:
        metric_col = "correct"
        
    grouped = df.groupby(["N", "k", "m"]).agg(
        p_success=(metric_col, "mean")
    ).reset_index()

    moduli = sorted(list(grouped["N"].unique()))

    for N in moduli:
        sub_N = grouped[grouped["N"] == N]
        k_values = sorted(list(sub_N["k"].unique()))
        
        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
        
        for idx, k in enumerate(k_values):
            sub_k = sub_N[sub_N["k"] == k].sort_values("m")
            color = PALETTE[idx % len(PALETTE)]
            ax.plot(
                sub_k["m"], 
                sub_k["p_success"], 
                "o-", 
                label=f"k = {k}", 
                color=color, 
                linewidth=2.2, 
                markersize=7
            )
            
        # Random baseline
        ax.axhline(
            1.0 / N,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="Random Guess"
        )

        ax.set_xlabel("Number of Independent Samples ($m$)")
        ax.set_ylabel(f"Recovery Probability (using {metric_col})")
        ax.set_title(f"Sample Complexity of DCP Secret Recovery ($N={N}$)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xscale("log", base=2)
        
        # Ensure x-ticks match the m values tested
        m_vals = sorted(list(sub_N["m"].unique()))
        ax.set_xticks(m_vals)
        ax.set_xticklabels([str(int(m)) for m in m_vals])

        ax.legend(title="Retained Bits ($k$)", frameon=True)

        save_path = output_dir / f"recovery_vs_samples_N{N}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)
        
    return saved_paths

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sample complexity results.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/raw/dcp_sample_complexity/dcp_sample_complexity_sweep.parquet",
        help="Path to raw trial-level parquet results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures/dcp_sample_complexity",
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input dataset not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {input_path}")
    df = load_experiment_result(input_path)
    
    print(f"Loaded {len(df):,} trials.")
    out_dir = Path(args.output_dir)
    paths = plot_recovery_vs_samples(df, out_dir)
    
    print("Generated sample complexity plots:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
