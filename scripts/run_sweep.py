"""CLI script to execute parameter sweeps (e.g. Fourier truncation sweep)."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import pandas as pd
import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.orchestrator import Orchestrator
from src.utils.serialization import save_experiment_result

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DCP parameter sweep.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dcp_truncation_sweep.yaml",
        help="Path to YAML sweep configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default=None,
        help="Override output file prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    exp_info = data.get("experiment", {})
    base_params = data.get("base_parameters", {})
    sweep_grid = data.get("sweep_grid", [])
    output_info = data.get("output", {})

    output_dir = args.output_dir or output_info.get("dir", "results/raw/dcp_truncation_core")
    file_prefix = args.file_prefix or output_info.get("file_prefix", "dcp_truncation_sweep")

    print(f"============================================================")
    print(f"Executing Sweep: {exp_info.get('name', 'DCP Sweep')}")
    print(f"Description:     {exp_info.get('description', '')}")
    print(f"Output Target:   {output_dir}/{file_prefix}.parquet")
    print(f"============================================================")

    orchestrator = Orchestrator()
    all_trials_dfs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    # Flatten sweep jobs
    jobs = []
    for item in sweep_grid:
        N = item["N"]
        s = item["s"]
        k_values = item["k_values"]
        for k in k_values:
            jobs.append((N, s, k))

    start_time = time.perf_counter()

    for N, s, k in tqdm(jobs, desc="Sweep Progress"):
        cfg = ExperimentConfig(
            N=N,
            s=s,
            k=k,
            m=base_params.get("m", 1),
            epsilon=base_params.get("epsilon", 0.0),
            shots=base_params.get("shots", 1000),
            seed=base_params.get("seed", 42),
            problem_type=base_params.get("problem_type", "dcp"),
            recovery_strategy=base_params.get("recovery_strategy", "brute_force"),
            truncation_mode=base_params.get("truncation_mode", "msb"),
            backend=base_params.get("backend", "statevector"),
        )

        res = orchestrator.run(cfg)
        stats = res.statistics
        assert stats is not None

        all_trials_dfs.append(stats.raw_data)
        summary_rows.append({
            "N": N,
            "n": cfg.n,
            "k": k,
            "s": s,
            "shots": cfg.shots,
            "recovery_prob": stats.recovery_prob,
            "mirror_recovery_prob": stats.mirror_recovery_prob,
            "ci_lower": stats.recovery_prob_ci[0],
            "ci_upper": stats.recovery_prob_ci[1],
            "runtime_sec": stats.runtime_seconds,
        })

    total_time = time.perf_counter() - start_time
    combined_trials_df = pd.concat(all_trials_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"Total Sweep Execution Time: {total_time:.2f} s")

    # Persist results
    metadata = {
        "experiment": exp_info,
        "base_parameters": base_params,
        "sweep_grid": sweep_grid,
        "total_jobs": len(jobs),
        "total_runtime_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "summary": summary_rows,
    }

    pq_path, json_path = save_experiment_result(
        result_df=combined_trials_df,
        metadata=metadata,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    print(f"\nPersisted Raw Trials to: {pq_path}")
    print(f"Persisted Metadata to:   {json_path}")


if __name__ == "__main__":
    main()
