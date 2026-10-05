"""Configuration system for DCP/EDCP experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ExperimentConfig:
    """Configuration dataclass for DCP and EDCP experiments."""

    N: int
    s: int
    n: int = 0
    m: int = 1
    k: int | None = None
    epsilon: float = 0.0
    shots: int = 1000
    seed: int = 42
    problem_type: str = "dcp"
    recovery_strategy: str = "brute_force"
    truncation_mode: str = "msb"
    backend: str = "statevector"
    edcp_chi: dict[int, complex] | None = None
    mod_halving_iterations: int = 0

    def __post_init__(self) -> None:
        if self.n <= 0:
            self.n = max(1, (self.N - 1).bit_length())


def validate_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration parameters.

    Raises:
        ValueError: If any parameter value is invalid.
    """
    if config.N < 2:
        raise ValueError(f"Modulus N must be >= 2, got {config.N}")
    if not (0 <= config.s < config.N):
        raise ValueError(f"Secret s must be in [0, {config.N}), got {config.s}")
    if config.k is not None and not (0 <= config.k <= config.n):
        raise ValueError(f"k must be in [0, {config.n}], got {config.k}")
    if config.m < 1:
        raise ValueError(f"Sample count m must be >= 1, got {config.m}")
    if not (0.0 <= config.epsilon <= 1.0):
        raise ValueError(f"Noise epsilon must be in [0, 1], got {config.epsilon}")
    if config.shots < 1:
        raise ValueError(f"Shots must be >= 1, got {config.shots}")
    if config.problem_type not in ("dcp", "edcp"):
        raise ValueError(
            f"problem_type must be 'dcp' or 'edcp', got {config.problem_type}"
        )
    if config.truncation_mode not in ("msb", "lsb", "custom"):
        raise ValueError(
            f"truncation_mode must be 'msb', 'lsb', or 'custom', got {config.truncation_mode}"
        )
    if config.backend not in ("statevector", "shots"):
        raise ValueError(
            f"backend must be 'statevector' or 'shots', got {config.backend}"
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an ExperimentConfig from a YAML file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config YAML must contain a dictionary")

    # If the YAML structure has a 'parameters' sub-dict, extract it
    params = data.get("parameters", data)
    if not isinstance(params, dict):
        raise ValueError("Configuration parameters must be a dictionary")

    # Filter recognized parameters for ExperimentConfig
    valid_keys = {
        "N",
        "s",
        "n",
        "m",
        "k",
        "epsilon",
        "shots",
        "seed",
        "problem_type",
        "recovery_strategy",
        "truncation_mode",
        "backend",
        "edcp_chi",
        "mod_halving_iterations",
    }
    filtered_params = {k: v for k, v in params.items() if k in valid_keys}

    if "N" not in filtered_params or "s" not in filtered_params:
        raise ValueError("Configuration must specify 'N' and 's'")

    config = ExperimentConfig(**filtered_params)
    validate_config(config)
    return config
