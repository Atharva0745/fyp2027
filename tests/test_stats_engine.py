"""Unit tests for StatisticsEngine aggregation and metrics."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.engines.stats_engine import StatisticsEngine
from src.utils.serialization import load_experiment_result, load_metadata


def test_statistics_engine_group_metrics():
    engine = StatisticsEngine(confidence_level=0.95)

    # Create synthetic group of 100 trials
    df_group = pd.DataFrame({
        "trial": list(range(100)),
        "N": [4] * 100,
        "n": [2] * 100,
        "k": [2] * 100,
        "s_true": [3] * 100,
        "s_hat": [3] * 80 + [1] * 20,
        "correct": [True] * 80 + [False] * 20,
        "confidence": [0.9] * 100,
        "mean_bit_accuracy": [0.85] * 100,
        "bit_correct_0": [True] * 85 + [False] * 15,
        "bit_correct_1": [True] * 85 + [False] * 15,
        "m": [1] * 100,
        "epsilon": [0.0] * 100,
        "truncation_mode": ["msb"] * 100,
    })

    metrics = engine.compute_group_metrics(df_group)

    assert metrics["N"] == 4
    assert metrics["k"] == 2
    assert metrics["shots"] == 100
    assert np.isclose(metrics["recovery_prob"], 0.80)
    assert 0.70 <= metrics["ci_lower"] <= 0.80
    assert 0.80 <= metrics["ci_upper"] <= 0.90
    assert np.isclose(metrics["mean_bit_accuracy"], 0.85)
    assert len(metrics["bit_recovery_probs"]) == 2
    assert metrics["mi_full"] > 0
    assert metrics["mi_truncated"] > 0


def test_statistics_engine_sweep_aggregation_and_save():
    engine = StatisticsEngine()

    df_sweep = pd.DataFrame({
        "trial": list(range(40)),
        "N": [4] * 20 + [8] * 20,
        "n": [2] * 20 + [3] * 20,
        "k": [1] * 10 + [2] * 10 + [1] * 10 + [3] * 10,
        "s_true": [1] * 20 + [5] * 20,
        "s_hat": [1] * 40,
        "correct": [True] * 40,
        "confidence": [0.95] * 40,
        "mean_bit_accuracy": [1.0] * 40,
        "m": [1] * 40,
        "epsilon": [0.0] * 40,
        "truncation_mode": ["msb"] * 40,
    })

    agg_df = engine.aggregate_sweep_dataframe(df_sweep)
    assert len(agg_df) == 4  # 4 combinations of (N, k)

    summary_table = engine.generate_summary_table(agg_df)
    assert "recovery_prob" in summary_table.columns
    assert "mi_truncated" in summary_table.columns
    assert "info_loss_ratio" in summary_table.columns

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq, csv, jf = engine.save_aggregated_summary(agg_df, output_dir=tmp_dir, file_prefix="test_agg")
        assert Path(pq).exists()
        assert Path(csv).exists()
        assert Path(jf).exists()

        loaded_df = load_experiment_result(pq)
        assert len(loaded_df) == 4
