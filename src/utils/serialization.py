"""Results serialization and persistence layer supporting Parquet and JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy types, complex numbers, and Path objects for JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "__dict__"):
        return _sanitize_for_json(obj.__dict__)
    return obj


def save_experiment_result(
    result_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str | Path,
    file_prefix: str,
) -> tuple[Path, Path]:
    """Save experiment trial data to Parquet and metadata to JSON.

    Args:
        result_df: Pandas DataFrame containing trial-by-trial experiment data.
        metadata: Dictionary containing configuration, aggregate statistics, and metadata.
        output_dir: Destination directory.
        file_prefix: Base name for output files (without extension).

    Returns:
        Tuple of (parquet_path, json_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_file = out_path / f"{file_prefix}.parquet"
    json_file = out_path / f"{file_prefix}_metadata.json"

    # Save DataFrame to Parquet
    try:
        result_df.to_parquet(parquet_file, index=False, engine="pyarrow")
    except Exception:
        # Fallback to default engine
        result_df.to_parquet(parquet_file, index=False)

    # Save metadata to JSON
    clean_metadata = _sanitize_for_json(metadata)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(clean_metadata, f, indent=2)

    return parquet_file, json_file


def load_experiment_result(parquet_path: str | Path) -> pd.DataFrame:
    """Load experiment results DataFrame from a Parquet file.

    Args:
        parquet_path: Path to the .parquet file.

    Returns:
        Loaded pandas DataFrame.
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pd.read_parquet(p)


def load_metadata(json_path: str | Path) -> dict[str, Any]:
    """Load metadata dictionary from a JSON file.

    Args:
        json_path: Path to the .json metadata file.

    Returns:
        Dictionary of loaded metadata.
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
