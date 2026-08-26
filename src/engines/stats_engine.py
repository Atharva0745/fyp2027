"""Statistics Engine for aggregating trial data and computing information-theoretic metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.info.mutual_information import compute_mutual_information, information_loss_ratio
from src.utils.math_utils import wilson_score_interval
from src.utils.serialization import save_experiment_result


@dataclass
class AggregatedMetrics:
    """Dataclass holding statistical and information metrics for an experiment configuration."""

    N: int
    n: int
    k: int
    s: int
    m: int
    epsilon: float
    shots: int
    recovery_prob: float
    ci_lower: float
    ci_upper: float
    mean_bit_accuracy: float
    bit_recovery_probs: list[float]
    bit_advantages: list[float]
    mi_truncated: float
    mi_full: float
    info_loss_ratio: float
    mean_confidence: float


class StatisticsEngine:
    """Engine for statistical analysis and information-theoretic metric calculation."""

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level

    def compute_group_metrics(self, group_df: pd.DataFrame) -> dict[str, Any]:
        """Compute statistical and MI metrics for a homogeneous group of trials.

        Args:
            group_df: Subset DataFrame sharing the same (N, n, k, s, m, epsilon, truncation_mode).

        Returns:
            Dictionary of aggregated metrics.
        """
        first = group_df.iloc[0]
        N = int(first["N"])
        n = int(first.get("n", max(1, (N - 1).bit_length())))
        k = int(first.get("k", n))
        s = int(first.get("s_true", first.get("s", 0)))
        m = int(first.get("m", 1))
        epsilon = float(first.get("epsilon", 0.0))
        trunc_mode = str(first.get("truncation_mode", "msb"))
        shots = len(group_df)

        successes = int(group_df["correct"].sum())
        rec_prob = float(successes / shots)
        ci_lower, ci_upper = wilson_score_interval(successes, shots, confidence=self.confidence_level)

        # Mirror recovery: s_hat == s OR s_hat == (N-s) % N
        if "mirror_correct" in group_df.columns:
            mirror_successes = int(group_df["mirror_correct"].sum())
        else:
            mirror_successes = successes
        mirror_prob = float(mirror_successes / shots)
        mirror_ci_lower, mirror_ci_upper = wilson_score_interval(mirror_successes, shots, confidence=self.confidence_level)

        mean_bit_acc = float(group_df["mean_bit_accuracy"].mean()) if "mean_bit_accuracy" in group_df else rec_prob
        mean_conf = float(group_df["confidence"].mean()) if "confidence" in group_df else 0.0

        # Per-bit success and advantages
        bit_probs: list[float] = []
        bit_advs: list[float] = []
        for bit_i in range(n):
            col = f"bit_correct_{bit_i}"
            if col in group_df.columns:
                p_bit = float(group_df[col].mean())
                bit_probs.append(p_bit)
                bit_advs.append(p_bit - 0.5)

        # Theoretical Mutual Information
        mi_full = compute_mutual_information(N=N, k=n, n=n, mode=trunc_mode, include_flag=True)
        mi_trunc = compute_mutual_information(N=N, k=k, n=n, mode=trunc_mode, include_flag=True)
        info_loss = information_loss_ratio(mi_full, mi_trunc)

        return {
            "N": N,
            "n": n,
            "k": k,
            "s": s,
            "m": m,
            "epsilon": epsilon,
            "truncation_mode": trunc_mode,
            "shots": shots,
            "recovery_prob": rec_prob,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mirror_recovery_prob": mirror_prob,
            "mirror_ci_lower": mirror_ci_lower,
            "mirror_ci_upper": mirror_ci_upper,
            "mean_bit_accuracy": mean_bit_acc,
            "bit_recovery_probs": bit_probs,
            "bit_advantages": bit_advs,
            "mi_truncated": mi_trunc,
            "mi_full": mi_full,
            "info_loss_ratio": info_loss,
            "mean_confidence": mean_conf,
        }

    def aggregate_sweep_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a trial-level DataFrame into a summary DataFrame grouped by parameters.

        Args:
            raw_df: DataFrame of individual simulation trials.

        Returns:
            Aggregated DataFrame with one row per (N, k, s, m, epsilon) configuration.
        """
        group_cols = ["N", "k"]
        for col in ["n", "s_true", "m", "epsilon", "truncation_mode", "strategy"]:
            if col in raw_df.columns:
                group_cols.append(col)

        rows: list[dict[str, Any]] = []
        for _, group in raw_df.groupby(group_cols, sort=False):
            metrics = self.compute_group_metrics(group)
            rows.append(metrics)

        agg_df = pd.DataFrame(rows)
        # Sort naturally by N then k
        agg_df = agg_df.sort_values(by=["N", "k"]).reset_index(drop=True)
        return agg_df

    def generate_summary_table(self, aggregated_df: pd.DataFrame) -> pd.DataFrame:
        """Format a clean, human-readable summary table for presentation and reporting."""
        display_cols = [
            "N",
            "n",
            "k",
            "s",
            "shots",
            "recovery_prob",
            "ci_lower",
            "ci_upper",
            "mirror_recovery_prob",
            "mirror_ci_lower",
            "mirror_ci_upper",
            "mi_truncated",
            "info_loss_ratio",
            "mean_bit_accuracy",
        ]
        cols_present = [c for c in display_cols if c in aggregated_df.columns]
        return aggregated_df[cols_present].copy()

    def save_aggregated_summary(
        self,
        aggregated_df: pd.DataFrame,
        output_dir: str | Path,
        file_prefix: str = "dcp_core_summary",
    ) -> tuple[Path, Path, Path]:
        """Save aggregated dataset in Parquet, CSV, and JSON formats.

        Args:
            aggregated_df: Aggregated summary DataFrame.
            output_dir: Target directory.
            file_prefix: Base file name.

        Returns:
            Tuple of (parquet_path, csv_path, json_path).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        meta = {
            "num_configurations": len(aggregated_df),
            "columns": list(aggregated_df.columns),
            "moduli": sorted(list(aggregated_df["N"].unique())),
        }

        pq_path, json_path = save_experiment_result(
            result_df=aggregated_df,
            metadata=meta,
            output_dir=out_path,
            file_prefix=file_prefix,
        )

        csv_path = out_path / f"{file_prefix}.csv"
        aggregated_df.to_csv(csv_path, index=False)

        return pq_path, csv_path, json_path
