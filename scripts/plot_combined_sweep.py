"""Phase 5.3 — Plotting script for the combined parameter sweep.

Generates:
  - combined_heatmap_k_eps_N{N}_m{m}.png   P_success(k, epsilon) for each (N, m)
  - combined_heatmap_k_m_N{N}_eps{eps}.png  P_success(k, m)       for each (N, epsilon)
  - combined_dashboard_N{N}.png              4-panel summary per modulus
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


def _get_metric(df: pd.DataFrame) -> str:
    """Return 'mirror_correct' if available, else 'correct'."""
    return "mirror_correct" if "mirror_correct" in df.columns else "correct"


def plot_heatmap_k_epsilon(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Heatmap of P_success(k, epsilon) for each (N, m) combination."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = _get_metric(df)

    grouped = df.groupby(["N", "m", "k", "epsilon"]).agg(
        p_success=(metric, "mean")
    ).reset_index()

    saved: list[Path] = []
    for N in sorted(grouped["N"].unique()):
        sub_N = grouped[grouped["N"] == N]
        for m in sorted(sub_N["m"].unique()):
            sub = sub_N[sub_N["m"] == m]
            pivot = sub.pivot(index="k", columns="epsilon", values="p_success")

            # Sort index and columns
            pivot = pivot.sort_index(ascending=False)  # higher k at top

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
            ax.set_title(
                f"P(Recovery) — $k$ vs $\\varepsilon$\n"
                f"($N={N}$, $m={m}$, metric: {metric})"
            )
            ax.set_xlabel("Noise Level ($\\varepsilon$)")
            ax.set_ylabel("Retained Bits ($k$)")

            fname = output_dir / f"combined_heatmap_k_eps_N{N}_m{m}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved.append(fname)
            print(f"  [k-ε heatmap]  {fname}")

    return saved


def plot_heatmap_k_m(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Heatmap of P_success(k, m) for each (N, epsilon) combination."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = _get_metric(df)

    grouped = df.groupby(["N", "epsilon", "k", "m"]).agg(
        p_success=(metric, "mean")
    ).reset_index()

    saved: list[Path] = []
    for N in sorted(grouped["N"].unique()):
        sub_N = grouped[grouped["N"] == N]
        for eps in sorted(sub_N["epsilon"].unique()):
            sub = sub_N[sub_N["epsilon"] == eps]
            pivot = sub.pivot(index="k", columns="m", values="p_success")
            pivot = pivot.sort_index(ascending=False)  # higher k at top

            fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
            sns.heatmap(
                pivot,
                annot=True, fmt=".2f",
                cmap="Blues",
                vmin=0.0, vmax=1.0,
                linewidths=0.5,
                cbar_kws={"label": "P(recovery)"},
                ax=ax,
            )
            ax.set_title(
                f"P(Recovery) — $k$ vs $m$\n"
                f"($N={N}$, $\\varepsilon={eps}$, metric: {metric})"
            )
            ax.set_xlabel("Number of Samples ($m$)")
            ax.set_ylabel("Retained Bits ($k$)")

            eps_tag = str(eps).replace(".", "p")
            fname = output_dir / f"combined_heatmap_k_m_N{N}_eps{eps_tag}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved.append(fname)
            print(f"  [k-m heatmap]  {fname}")

    return saved


def plot_combined_dashboard(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """4-panel per-N dashboard: two heatmaps + two line plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = _get_metric(df)

    grouped = df.groupby(["N", "k", "m", "epsilon"]).agg(
        p_success=(metric, "mean")
    ).reset_index()

    saved: list[Path] = []
    for N in sorted(grouped["N"].unique()):
        sub_N = grouped[grouped["N"] == N]
        n_bits = max(1, int(N - 1).bit_length())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        # --- Panel A: P_success vs k (fixed eps=0, varying m) ---
        ax_a = axes[0, 0]
        sub_eps0 = sub_N[sub_N["epsilon"] == 0.0]
        for idx, m in enumerate(sorted(sub_eps0["m"].unique())):
            sub_m = sub_eps0[sub_eps0["m"] == m].sort_values("k")
            ax_a.plot(
                sub_m["k"], sub_m["p_success"],
                "o-", label=f"m={m}", color=PALETTE[idx % len(PALETTE)],
                linewidth=2.2, markersize=7,
            )
        ax_a.axhline(1.0 / N, color="gray", linestyle=":", alpha=0.7, label="Random")
        ax_a.set_title(f"A: P(Recovery) vs $k$ ($\\varepsilon=0$)", fontweight="bold")
        ax_a.set_xlabel("Retained Bits ($k$)")
        ax_a.set_ylabel("P(Recovery)")
        ax_a.set_ylim(-0.02, 1.02)
        ax_a.legend(title="Samples ($m$)")

        # --- Panel B: P_success vs epsilon (fixed m=1, varying k) ---
        ax_b = axes[0, 1]
        sub_m1 = sub_N[sub_N["m"] == sorted(sub_N["m"].unique())[0]]
        for idx, k in enumerate(sorted(sub_m1["k"].unique())):
            sub_k = sub_m1[sub_m1["k"] == k].sort_values("epsilon")
            ax_b.plot(
                sub_k["epsilon"], sub_k["p_success"],
                "s-", label=f"k={k}", color=PALETTE[idx % len(PALETTE)],
                linewidth=2.2, markersize=7,
            )
        ax_b.axhline(1.0 / N, color="gray", linestyle=":", alpha=0.7, label="Random")
        ax_b.set_title(f"B: P(Recovery) vs $\\varepsilon$ ($m=1$)", fontweight="bold")
        ax_b.set_xlabel("Noise Level ($\\varepsilon$)")
        ax_b.set_ylabel("P(Recovery)")
        ax_b.set_ylim(-0.02, 1.02)
        ax_b.legend(title="Retained Bits ($k$)")

        # --- Panel C: Heatmap P_success(k, eps) for m=1 ---
        ax_c = axes[1, 0]
        m_val_c = sorted(sub_N["m"].unique())[0]
        sub_c = sub_N[sub_N["m"] == m_val_c]
        pivot_c = sub_c.pivot(index="k", columns="epsilon", values="p_success")
        pivot_c = pivot_c.sort_index(ascending=False)
        sns.heatmap(
            pivot_c,
            annot=True, fmt=".2f",
            cmap="RdYlGn",
            vmin=0.0, vmax=1.0,
            linewidths=0.5,
            cbar_kws={"label": "P(recovery)"},
            ax=ax_c,
        )
        ax_c.set_title(f"C: Heatmap P(recovery) — $k$ vs $\\varepsilon$ ($m={m_val_c}$)", fontweight="bold")
        ax_c.set_xlabel("Noise Level ($\\varepsilon$)")
        ax_c.set_ylabel("Retained Bits ($k$)")

        # --- Panel D: Heatmap P_success(k, m) for eps=0 ---
        ax_d = axes[1, 1]
        sub_d = sub_N[sub_N["epsilon"] == 0.0]
        pivot_d = sub_d.pivot(index="k", columns="m", values="p_success")
        pivot_d = pivot_d.sort_index(ascending=False)
        sns.heatmap(
            pivot_d,
            annot=True, fmt=".2f",
            cmap="Blues",
            vmin=0.0, vmax=1.0,
            linewidths=0.5,
            cbar_kws={"label": "P(recovery)"},
            ax=ax_d,
        )
        ax_d.set_title(f"D: Heatmap P(recovery) — $k$ vs $m$ ($\\varepsilon=0$)", fontweight="bold")
        ax_d.set_xlabel("Number of Samples ($m$)")
        ax_d.set_ylabel("Retained Bits ($k$)")

        fig.suptitle(
            f"DCP Combined Parameter Sweep — $N={N}$ ($n={n_bits}$ bits)\n"
            f"metric: {metric}",
            fontsize=14, fontweight="bold",
        )

        fname = output_dir / f"combined_dashboard_N{N}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
        print(f"  [dashboard]    {fname}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Phase 5.3 combined parameter sweep results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/raw/dcp_combined/dcp_combined_sweep.parquet",
        help="Path to combined sweep Parquet results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures/dcp_heatmaps",
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    df = load_experiment_result(input_path)
    print(f"Loaded {len(df):,} trials from {input_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"N values: {sorted(df['N'].unique())}")
    if "k" in df.columns:
        print(f"k values: {sorted(df['k'].unique())}")
    if "m" in df.columns:
        print(f"m values: {sorted(df['m'].unique())}")
    if "epsilon" in df.columns:
        print(f"epsilon values: {sorted(df['epsilon'].unique())}")

    out_dir = Path(args.output_dir)

    print("\nGenerating k-epsilon heatmaps...")
    plot_heatmap_k_epsilon(df, out_dir)

    print("\nGenerating k-m heatmaps...")
    plot_heatmap_k_m(df, out_dir)

    print("\nGenerating combined dashboards...")
    plot_combined_dashboard(df, out_dir)

    print("\nDone. All Phase 5.3 figures saved.")


if __name__ == "__main__":
    main()
