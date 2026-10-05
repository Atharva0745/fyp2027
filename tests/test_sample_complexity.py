"""Unit and integration tests for Phase 4: Sample Complexity Experiments."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.orchestrator import Orchestrator
from src.recovery.bayesian import bayesian_recovery
from src.visualization.plots import plot_recovery_vs_samples


def test_multisample_bayesian_posterior_concentration():
    """Verify that adding more observations (m) increases posterior concentration on true secret."""
    N, s, n, k = 8, 5, 3, 2
    orch = Orchestrator()
    cfg_m1 = ExperimentConfig(N=N, s=s, k=k, m=1, shots=50, seed=42, recovery_strategy="bayesian")
    cfg_m8 = ExperimentConfig(N=N, s=s, k=k, m=8, shots=50, seed=42, recovery_strategy="bayesian")

    res_m1 = orch.run(cfg_m1)
    res_m8 = orch.run(cfg_m8)

    # Recovery probability with m=8 should be >= recovery probability with m=1
    assert res_m8.statistics is not None
    assert res_m1.statistics is not None
    assert res_m8.statistics.recovery_prob >= res_m1.statistics.recovery_prob, (
        f"Expected P_success(m=8) [{res_m8.statistics.recovery_prob:.3f}] >= P_success(m=1) [{res_m1.statistics.recovery_prob:.3f}]"
    )


def test_sample_complexity_sweep_execution():
    """Test running a mini sample complexity grid sweep via Orchestrator."""
    base_cfg = ExperimentConfig(N=4, s=3, k=1, m=1, shots=20, seed=123, recovery_strategy="bayesian")
    orch = Orchestrator()

    sweep_df = orch.run_sweep(base_config=base_cfg, param_grid={"m": [1, 2, 4]})
    assert len(sweep_df) == 60  # 3 parameter settings x 20 shots
    assert sorted(list(sweep_df["m"].unique())) == [1, 2, 4]




def test_plot_recovery_vs_samples(tmp_path: Path):
    """Verify plot_recovery_vs_samples generates a plot without errors."""
    data = []
    for N in [4, 8]:
        n = max(1, (N - 1).bit_length())
        for k in range(1, n + 1):
            for m in [1, 2, 4]:
                p_succ = min(1.0, (k / n) * (m / 4.0))
                data.append({
                    "N": N,
                    "k": k,
                    "m": m,
                    "p_success": p_succ,
                    "correct": True if p_succ > 0.5 else False,
                    "mean_bit_accuracy": p_succ,
                })
    df = pd.DataFrame(data)

    save_path = plot_recovery_vs_samples(df, output_dir=tmp_path)
    assert save_path.exists()
    assert save_path.name == "recovery_vs_samples.png"
