"""Phase 5.1 — Plotting script for noise robustness experiments.

Generates:
  - recovery_vs_noise_N{N}_k{k}.png  (P_success vs epsilon per (N,k))
  - noise_heatmap_N{N}.png            (heatmap of P_success across (k, epsilon))
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

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300,
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]


def plot_recovery_vs_noise(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot P_success vs epsilon for each (N, k) combination."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    metric = "mirror_correct" if "mirror_correct" in df.columns else "correct"
    grouped = df.groupby(["N", "k", "epsilon"]).agg(
        p_success=(metric, "mean")
    ).reset_index()

    for N in sorted(grouped["N"].unique()):
        sub_N = grouped[grouped["N"] == N]
        k_values = sorted(sub_N["k"].unique())
        n_bits = max(1, int(N - 1).bit_length() if hasattr(int(N - 1), 'bit_length') else 1)

        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

        for idx, k in enumerate(k_values):
            sub_k = sub_N[sub_N["k"] == k].sort_values("epsilon")
            color = PALETTE[idx % len(PALETTE)]
            ax.plot(
                sub_k["epsilon"], sub_k["p_success"],
                "o-", label=f"k = {k}",
                color=color, linewidth=2.2, markersize=7,
            )

        # Random baseline
        ax.axhline(1.0 / N, color="gray", linestyle=":", alpha=0.7, label="Random Guess")

        ax.set_xlabel("Bit-Flip Noise Level ($\\varepsilon$)")
        ax.set_ylabel(f"Recovery Probability")
        ax.set_title(f"Noise Robustness of DCP Recovery ($N={N}$)\n"
                     f"(P includes $s$ or $N-s$ as correct)")
        ax.set_xlim(-0.01, 0.22)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(title="Retained Bits ($k$)", frameon=True)

        save_path = output_dir / f"recovery_vs_noise_N{N}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(save_path)

    return saved


def plot_noise_heatmaps(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Heatmap of P_success(k, epsilon) for each N."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    metric = "mirror_correct" if "mirror_correct" in df.columns else "correct"
    grouped = df.groupby(["N", "k", "epsilon"]).agg(
        p_success=(metric, "mean")
    ).reset_index()

    for N in sorted(grouped["N"].unique()):
        sub_N = grouped[grouped["N"] == N]
        pivot = sub_N.pivot(index="k", columns="epsilon", values="p_success")

        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        sns.heatmap(
            pivot,
            annot=True, fmt=".2f",
            cmap="RdYlGn",
            vmin=0.0, vmax=1.0,
            linewidths=0.5,
            cbar_kws={"label": "P(recovery)"},
            ax=ax,
        )
        ax.set_title(f"P(Recovery) Heatmap — ($N={N}$, $k$ vs $\\varepsilon$)")
        ax.set_xlabel("Noise Level ($\\varepsilon$)")
        ax.set_ylabel("Retained Bits ($k$)")

        save_path = output_dir / f"noise_heatmap_N{N}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(save_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 5.1 noise robustness results.")
    parser.add_argument("--input", type=str,
                        default="results/raw/dcp_noise/dcp_noise_sweep.parquet")
    parser.add_argument("--output-dir", type=str,
                        default="results/figures/dcp_noise")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    df = load_experiment_result(input_path)
    print(f"Loaded {len(df):,} trials.")

    out_dir = Path(args.output_dir)

    paths1 = plot_recovery_vs_noise(df, out_dir)
    for p in paths1:
        print(f"  [noise curves]  {p}")

    paths2 = plot_noise_heatmaps(df, out_dir)
    for p in paths2:
        print(f"  [heatmap]       {p}")


if __name__ == "__main__":
    main()
