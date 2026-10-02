"""Unit tests for configuration system."""

import pytest
from src.config import ExperimentConfig, validate_config, load_config


def test_default_experiment_config():
    cfg = ExperimentConfig(N=16, s=5)
    assert cfg.N == 16
    assert cfg.s == 5
    assert cfg.n == 4
    assert cfg.m == 1
    assert cfg.k is None
    assert cfg.epsilon == 0.0
    assert cfg.shots == 1000
    assert cfg.seed == 42
    assert cfg.problem_type == "dcp"
    validate_config(cfg)


def test_load_yaml_config(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
parameters:
  N: 8
  s: 3
  m: 2
  k: 2
  epsilon: 0.05
  shots: 500
  seed: 123
  problem_type: dcp
  recovery_strategy: brute_force
  truncation_mode: msb
  backend: statevector
"""
    )
    cfg = load_config(config_file)
    assert cfg.N == 8
    assert cfg.s == 3
    assert cfg.n == 3
    assert cfg.m == 2
    assert cfg.k == 2
    assert cfg.epsilon == 0.05
    assert cfg.shots == 500
    assert cfg.seed == 123


def test_invalid_modulus():
    cfg = ExperimentConfig(N=1, s=0)
    with pytest.raises(ValueError, match="Modulus N must be >= 2"):
        validate_config(cfg)


def test_invalid_secret():
    cfg = ExperimentConfig(N=8, s=8)
    with pytest.raises(ValueError, match="Secret s must be in"):
        validate_config(cfg)

    cfg2 = ExperimentConfig(N=8, s=-1)
    with pytest.raises(ValueError, match="Secret s must be in"):
        validate_config(cfg2)


def test_invalid_k():
    cfg = ExperimentConfig(N=8, s=3, k=4)  # n=3 for N=8
    with pytest.raises(ValueError, match="k must be in"):
        validate_config(cfg)


def test_invalid_epsilon():
    cfg = ExperimentConfig(N=8, s=3, epsilon=1.5)
    with pytest.raises(ValueError, match="Noise epsilon must be in"):
        validate_config(cfg)


def test_load_base_yaml():
    cfg = load_config("configs/dcp_base.yaml")
    assert cfg.N == 16
    assert cfg.s == 5
    assert cfg.n == 4
