"""CLI script to analyze experiment results, aggregate statistics, and generate publication plots."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engines.stats_engine import StatisticsEngine
from src.utils.serialization import load_experiment_result
from src.visualization.plots import (
    plot_bit_recovery_heatmap,
    plot_information_loss,
    plot_mi_vs_truncation,
    plot_posterior,
    plot_recovery_vs_truncation,
    plot_summary_dashboard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze results and generate publication figures.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/raw/dcp_truncation_core/dcp_truncation_sweep.parquet",
        help="Path to raw trial-level parquet results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures/dcp_core",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--agg-dir",
        type=str,
        default="results/aggregated",
        help="Directory to save aggregated summary tables.",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default="dcp_core_summary",
        help="Base name for aggregated summary datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input dataset not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"============================================================")
    print(f"Loading raw trial results from: {input_path}")
    print(f"Figures Destination:            {args.output_dir}")
    print(f"Aggregated Data Destination:    {args.agg_dir}")
    print(f"============================================================")

    # 1. Load dataset
    df = load_experiment_result(input_path)
    print(f"Loaded {len(df):,} trial records across {df['N'].nunique()} moduli.")

    # 2. Run Statistics Engine
    stats_engine = StatisticsEngine(confidence_level=0.95)
    agg_df = stats_engine.aggregate_sweep_dataframe(df)
    summary_table = stats_engine.generate_summary_table(agg_df)

    print("\n" + "=" * 80)
    print("AGGREGATED EXPERIMENTAL & INFORMATION-THEORETIC SUMMARY")
    print("=" * 80)
    print(summary_table.to_string(index=False))
    print("=" * 80)

    # 3. Persist Aggregated Summary Dataset
    agg_pq, agg_csv, agg_json = stats_engine.save_aggregated_summary(
        aggregated_df=agg_df,
        output_dir=args.agg_dir,
        file_prefix=args.file_prefix,
    )
    print(f"\nPersisted Aggregated Datasets:")
    print(f"  Parquet: {agg_pq}")
    print(f"  CSV:     {agg_csv}")
    print(f"  JSON:    {agg_json}")

    # 4. Generate Publication Figures
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating Publication Figures in: {out_dir}")

    # Plot 1: Recovery vs. Truncation
    p1 = plot_recovery_vs_truncation(df, output_dir=out_dir)
    print(f"  [1/5] Generated: {p1.name}")

    # Plot 2: Exact Mutual Information
    p2 = plot_mi_vs_truncation(output_dir=out_dir, moduli=sorted(list(df["N"].unique())))
    print(f"  [2/5] Generated: {p2.name}")

    # Plot 3: Information Loss Ratio
    p3 = plot_information_loss(output_dir=out_dir, moduli=sorted(list(df["N"].unique())))
    print(f"  [3/5] Generated: {p3.name}")

    # Plot 4: Bit Recovery Heatmaps
    heatmaps = plot_bit_recovery_heatmap(df, output_dir=out_dir)
    for hm in heatmaps:
        print(f"  [4/5] Generated Heatmap: {hm.name}")

    # Plot 5: Multi-panel Dashboard
    p5 = plot_summary_dashboard(df, output_dir=out_dir)
    print(f"  [5/5] Generated Dashboard: {p5.name}")

    # Example Posterior Visualization
    sample_posterior = {s: (1.0 / 16 if s % 2 == 1 else 0.0) for s in range(16)}
    sample_posterior[11] = 0.45
    p_post = plot_posterior(sample_posterior, s_true=11, output_path=out_dir / "sample_posterior_N16.png")
    print(f"  [Bonus] Generated Sample Posterior: {p_post.name}")

    print("\nPhase 3 Core Analysis & Plotting completed successfully.")


if __name__ == "__main__":
    main()
