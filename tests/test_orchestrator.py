"""Integration tests for the Orchestrator and serialization subsystem."""

from pathlib import Path
import tempfile
import pytest

from src.config import ExperimentConfig
from src.orchestrator import Orchestrator
from src.utils.serialization import load_experiment_result, load_metadata


def test_orchestrator_single_run():
    orchestrator = Orchestrator()
    config = ExperimentConfig(
        N=4,
        s=3,
        k=2,
        m=1,
        epsilon=0.0,
        shots=50,
        seed=42,
        recovery_strategy="brute_force",
    )

    result = orchestrator.run(config)

    assert result.statistics is not None
    assert 0.0 <= result.statistics.recovery_prob <= 1.0
    assert result.statistics.num_samples == 50
    assert len(result.statistics.bit_recovery_probs) == 2
    assert result.statistics.circuit_depth > 0
    assert result.statistics.num_qubits == 3

    # Check raw dataframe columns
    df = result.statistics.raw_data
    assert len(df) == 50
    expected_cols = {"trial", "N", "s_true", "s_hat", "correct", "confidence", "k", "n", "m"}
    assert expected_cols.issubset(set(df.columns))


def test_orchestrator_sweep_and_serialization():
    orchestrator = Orchestrator()
    base_config = ExperimentConfig(
        N=4,
        s=1,
        shots=20,
        seed=123,
    )

    param_grid = {
        "k": [1, 2],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        combined_df = orchestrator.run_sweep(
            base_config=base_config,
            param_grid=param_grid,
            output_dir=tmp_dir,
            file_prefix="test_sweep",
        )

        assert len(combined_df) == 40  # 2 configs * 20 shots

        parquet_path = Path(tmp_dir) / "test_sweep.parquet"
        json_path = Path(tmp_dir) / "test_sweep_metadata.json"

        assert parquet_path.exists()
        assert json_path.exists()

        # Load back
        loaded_df = load_experiment_result(parquet_path)
        assert len(loaded_df) == 40
        assert list(loaded_df.columns) == list(combined_df.columns)

        loaded_meta = load_metadata(json_path)
        assert loaded_meta["num_configurations"] == 2
        assert "summary" in loaded_meta
