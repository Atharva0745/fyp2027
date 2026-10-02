"""Phase 6 Master Orchestrator — EDCP and Modulus Halving.

Runs:
  6.1/6.3 EDCP Comparison Sweep
  6.2 Modulus Halving Toy Run
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.orchestrator import Orchestrator
from scripts.run_phase5 import _run_sweep_from_yaml, _banner
from src.utils.serialization import save_experiment_result
from src.engines.mod_halving_engine import ModulusHalvingEngine


def run_edcp_comparison(orchestrator: Orchestrator) -> None:
    _banner("Phase 6.3 — EDCP vs DCP Comparison Sweep")

    config_path = ROOT / "configs" / "edcp_comparison_sweep.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    output_info = data.get("output", {})
    output_dir = ROOT / output_info.get("dir", "results/raw/edcp_comparison")
    file_prefix = output_info.get("file_prefix", "edcp_sweep")

    print(f"  Config:      {config_path}")
    print(f"  Output:      {output_dir}/{file_prefix}.parquet")

    t0 = time.perf_counter()
    combined_df, summary_rows, jobs = _run_sweep_from_yaml(config_path, orchestrator)
    elapsed = time.perf_counter() - t0

    print(f"\n  Completed {len(jobs)} jobs in {elapsed:.1f}s")

    # Persist results
    metadata = {
        "experiment": data.get("experiment", {}),
        "base_parameters": data.get("base_parameters", {}),
        "total_jobs": len(jobs),
        "total_runtime_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "summary": summary_rows,
    }
    pq, js = save_experiment_result(combined_df, metadata, output_dir, file_prefix)
    print(f"\n  Saved Parquet:  {pq}")
    print(f"  Saved Metadata: {js}")

    # Call plotting script logic
    dcp_input = ROOT / "results" / "raw" / "dcp_truncation_core" / "dcp_truncation_sweep.parquet"
    if not dcp_input.exists():
        print(f"\n  [!] Warning: DCP baseline data not found at {dcp_input}")
        print("      Run Phase 2 (or 'python scripts/run_sweep.py') first to generate DCP comparison data.")
    else:
        import subprocess
        print("\n  Generating comparison plots...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_edcp_comparison.py")])

    _banner("Phase 6.3 COMPLETE")


def run_modulus_halving() -> None:
    _banner("Phase 6.2 — Toy Modulus Halving Pipeline")
    
    engine = ModulusHalvingEngine()
    N_start = 64
    s_start = 42  # 101010 in binary (even!)
    iterations = 3
    
    print(f"  Starting parameters: N={N_start}, s={s_start}, iterations={iterations}")
    print(f"  Using LWE-like error distribution (4-term)\n")
    
    results = engine.run_pipeline(N_start, s_start, chi_name="lwe-like", iterations=iterations)
    
    for res in results:
        it = res["iteration"]
        print(f"  --- Iteration {it} ---")
        print(f"    N: {res['N_prev']} -> {res['N_next']}")
        print(f"    s: {res['s_prev']} -> {res['s_next']} (LSB guessed as {res['s_lsb']})")
        print(f"    Observed Full Label: {res['y_observed']}")
        print()
        
    _banner("Phase 6.2 COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Master Orchestrator")
    parser.add_argument("--task", type=str, choices=["edcp", "halving", "all"], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 6 — EDCP & Modulus Halving")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Task:    {args.task}")
    print("=" * 70)

    orc = Orchestrator()
    t_start = time.perf_counter()

    if args.task in ("edcp", "all"):
        run_edcp_comparison(orc)

    if args.task in ("halving", "all"):
        run_modulus_halving()

    total = time.perf_counter() - t_start
    _banner(f"PHASE 6 DONE — total wall time {total:.1f}s")


if __name__ == "__main__":
    main()
