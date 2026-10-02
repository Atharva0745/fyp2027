"""Phase 6.3 — Plotting script for DCP vs. EDCP comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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


def plot_dcp_vs_edcp(df_dcp: pd.DataFrame, df_edcp: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot comparison of P_success for DCP vs EDCP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    metric_dcp = "mirror_correct" if "mirror_correct" in df_dcp.columns else "correct"
    metric_edcp = "mirror_correct" if "mirror_correct" in df_edcp.columns else "correct"

    g_dcp = df_dcp.groupby(["N", "k"]).agg(p_success=(metric_dcp, "mean")).reset_index()
    g_edcp = df_edcp.groupby(["N", "k"]).agg(p_success=(metric_edcp, "mean")).reset_index()

    common_moduli = sorted(list(set(g_dcp["N"].unique()) & set(g_edcp["N"].unique())))

    for N in common_moduli:
        sub_dcp = g_dcp[g_dcp["N"] == N].sort_values("k")
        sub_edcp = g_edcp[g_edcp["N"] == N].sort_values("k")

        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

        ax.plot(
            sub_dcp["k"], sub_dcp["p_success"],
            "o-", label="DCP (2-term)", color="#2b5c8f", linewidth=2.2, markersize=7
        )
        
        ax.plot(
            sub_edcp["k"], sub_edcp["p_success"],
            "s-", label="EDCP (LWE-like 4-term)", color="#d95f02", linewidth=2.2, markersize=7
        )

        ax.axhline(1.0 / N, color="gray", linestyle=":", alpha=0.7, label="Random Guess")

        ax.set_xlabel("Retained Fourier Bits ($k$)")
        ax.set_ylabel("Recovery Probability")
        ax.set_title(f"DCP vs EDCP Secret Recovery Comparison ($N={N}$)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=True)

        p = output_dir / f"dcp_vs_edcp_N{N}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print(f"  [comparison plot] {p}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 6.3 DCP vs EDCP comparison.")
    parser.add_argument("--dcp-input", type=str, default="results/raw/dcp_truncation_core/dcp_truncation_sweep.parquet")
    parser.add_argument("--edcp-input", type=str, default="results/raw/edcp_comparison/edcp_sweep.parquet")
    parser.add_argument("--output-dir", type=str, default="results/figures/edcp_comparison")
    args = parser.parse_args()

    dcp_path = Path(args.dcp_input)
    edcp_path = Path(args.edcp_input)

    if not dcp_path.exists():
        print(f"Error: DCP input not found at {dcp_path}", file=sys.stderr)
        sys.exit(1)
    if not edcp_path.exists():
        print(f"Error: EDCP input not found at {edcp_path}", file=sys.stderr)
        sys.exit(1)

    df_dcp = load_experiment_result(dcp_path)
    df_edcp = load_experiment_result(edcp_path)

    out_dir = Path(args.output_dir)
    print("\nGenerating DCP vs EDCP comparison plots...")
    plot_dcp_vs_edcp(df_dcp, df_edcp, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
