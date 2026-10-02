"""Phase 5 Master Orchestrator — Noise Robustness, Modulus Scaling, and Combined Sweep.

Runs all three Phase 5 sub-tasks in sequence:

  5.1  Noise Robustness Sweep
       Config:   configs/dcp_noise_sweep.yaml
       Output:   results/raw/dcp_noise/
       Figures:  results/figures/dcp_noise/

  5.2  Modulus Scaling (N = 4, 8, 16, 32, 64)
       Config:   configs/dcp_scaling_sweep.yaml
       Output:   results/raw/dcp_scaling/
       Figures:  results/figures/dcp_scaling/
       Table:    results/aggregated/scaling_summary.csv

  5.3  Combined Parameter Sweep (N, k, m, epsilon)
       Config:   configs/dcp_combined_sweep.yaml
       Output:   results/raw/dcp_combined/
       Figures:  results/figures/dcp_heatmaps/

Usage
-----
  python scripts/run_phase5.py               # run all three tasks
  python scripts/run_phase5.py --task noise  # run only 5.1
  python scripts/run_phase5.py --task scaling
  python scripts/run_phase5.py --task combined
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.orchestrator import Orchestrator
from src.utils.serialization import save_experiment_result, load_experiment_result

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _run_sweep_from_yaml(config_path: Path, orchestrator: Orchestrator) -> tuple[pd.DataFrame, list[dict], list]:
    """Parse a sweep YAML and execute all jobs, returning (combined_df, summary_rows, jobs)."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base_params = data.get("base_parameters", {})
    sweep_grid = data.get("sweep_grid", [])

    jobs = []
    for item in sweep_grid:
        N = item["N"]
        s = item["s"]
        k_values = item.get("k_values", [None])
        m_values = item.get("m_values", [base_params.get("m", 1)])
        epsilon_values = item.get("epsilon_values", [base_params.get("epsilon", 0.0)])
        for k in k_values:
            for m in m_values:
                for eps in epsilon_values:
                    jobs.append((N, s, k, m, eps))

    all_dfs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for N, s, k, m, epsilon in tqdm(jobs, desc="  Progress"):
        cfg = ExperimentConfig(
            N=N,
            s=s,
            k=k,
            m=m,
            epsilon=epsilon,
            shots=base_params.get("shots", 500),
            seed=base_params.get("seed", 42),
            problem_type=base_params.get("problem_type", "dcp"),
            recovery_strategy=base_params.get("recovery_strategy", "bayesian"),
            truncation_mode=base_params.get("truncation_mode", "msb"),
            backend=base_params.get("backend", "statevector"),
        )
        res = orchestrator.run(cfg)
        stats = res.statistics
        assert stats is not None

        all_dfs.append(stats.raw_data)
        summary_rows.append({
            "N": N,
            "n": cfg.n,
            "k": k if k is not None else cfg.n,
            "m": m,
            "epsilon": epsilon,
            "s": s,
            "shots": cfg.shots,
            "recovery_prob": stats.recovery_prob,
            "mirror_recovery_prob": stats.mirror_recovery_prob,
            "ci_lower": stats.recovery_prob_ci[0],
            "ci_upper": stats.recovery_prob_ci[1],
            "circuit_depth": stats.circuit_depth,
            "num_qubits": stats.num_qubits,
            "runtime_sec": stats.runtime_seconds,
        })

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, summary_rows, jobs


# ──────────────────────────────────────────────────────────────────────────────
# Task 5.1 — Noise Robustness
# ──────────────────────────────────────────────────────────────────────────────

def run_noise_sweep(orchestrator: Orchestrator) -> None:
    _banner("Phase 5.1 — Noise Robustness Sweep")

    config_path = ROOT / "configs" / "dcp_noise_sweep.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    exp_info = data.get("experiment", {})
    output_info = data.get("output", {})
    output_dir = ROOT / output_info.get("dir", "results/raw/dcp_noise")
    file_prefix = output_info.get("file_prefix", "dcp_noise_sweep")

    print(f"  Config:      {config_path}")
    print(f"  Output:      {output_dir}/{file_prefix}.parquet")

    t0 = time.perf_counter()
    combined_df, summary_rows, jobs = _run_sweep_from_yaml(config_path, orchestrator)
    elapsed = time.perf_counter() - t0

    print(f"\n  Completed {len(jobs)} jobs in {elapsed:.1f}s")
    print(f"\n  Noise Sweep Summary:")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df[["N", "k", "epsilon", "recovery_prob", "mirror_recovery_prob"]].to_string(index=False))

    # Persist results
    with open(config_path) as f:
        data = yaml.safe_load(f)

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

    # Generate noise plots
    print("\n  Generating noise robustness plots...")
    fig_dir = ROOT / "results" / "figures" / "dcp_noise"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metric = "mirror_correct" if "mirror_correct" in combined_df.columns else "correct"
    grouped = combined_df.groupby(["N", "k", "epsilon"]).agg(
        p_success=(metric, "mean")
    ).reset_index()

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})

        PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]

        # Plot 1: P_success vs epsilon for each (N, k)
        for N in sorted(grouped["N"].unique()):
            sub_N = grouped[grouped["N"] == N]
            fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
            for idx, k in enumerate(sorted(sub_N["k"].unique())):
                sub_k = sub_N[sub_N["k"] == k].sort_values("epsilon")
                ax.plot(
                    sub_k["epsilon"], sub_k["p_success"],
                    "o-", label=f"k={k}",
                    color=PALETTE[idx % len(PALETTE)], linewidth=2.2, markersize=7,
                )
            ax.axhline(1.0 / N, color="gray", linestyle=":", alpha=0.7, label="Random Guess")
            ax.set_xlabel("Bit-Flip Noise Level ($\\varepsilon$)")
            ax.set_ylabel("Recovery Probability")
            ax.set_title(f"Noise Robustness of DCP Recovery ($N={N}$)\n(mirror: s or N-s)")
            ax.set_xlim(-0.01, 0.22)
            ax.set_ylim(-0.02, 1.02)
            ax.legend(title="Retained Bits ($k$)", frameon=True)
            p = fig_dir / f"recovery_vs_noise_N{N}.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [noise curve]   {p}")

        # Plot 2: Heatmap P_success(k, epsilon) per N
        for N in sorted(grouped["N"].unique()):
            sub_N = grouped[grouped["N"] == N]
            pivot = sub_N.pivot(index="k", columns="epsilon", values="p_success")
            pivot = pivot.sort_index(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            sns.heatmap(
                pivot,
                annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0.0, vmax=1.0,
                linewidths=0.5,
                cbar_kws={"label": "P(recovery)"},
                ax=ax,
            )
            ax.set_title(f"P(Recovery) Heatmap — $k$ vs $\\varepsilon$ ($N={N}$)")
            ax.set_xlabel("Noise Level ($\\varepsilon$)")
            ax.set_ylabel("Retained Bits ($k$)")
            p = fig_dir / f"noise_heatmap_N{N}.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [noise heatmap] {p}")

    except Exception as exc:
        print(f"  Warning: plotting failed — {exc}")

    # Analysis summary
    print("\n  Analysis:")
    for N in sorted(grouped["N"].unique()):
        sub_N = grouped[grouped["N"] == N]
        for k in sorted(sub_N["k"].unique()):
            sub_k = sub_N[sub_N["k"] == k].sort_values("epsilon")
            p0 = sub_k[sub_k["epsilon"] == 0.0]["p_success"].values
            p_max_eps = sub_k.iloc[-1]["p_success"]
            if len(p0) > 0:
                degradation = float(p0[0]) - float(p_max_eps)
                print(f"    N={N}, k={k}: P(eps=0)={float(p0[0]):.3f}  "
                      f"P(eps_max)={float(p_max_eps):.3f}  "
                      f"Δ={degradation:.3f}")

    _banner("Phase 5.1 COMPLETE")


# ──────────────────────────────────────────────────────────────────────────────
# Task 5.2 — Modulus Scaling
# ──────────────────────────────────────────────────────────────────────────────

def run_scaling_sweep(orchestrator: Orchestrator) -> None:
    _banner("Phase 5.2 — Modulus Scaling (N = 4, 8, 16, 32, 64)")

    config_path = ROOT / "configs" / "dcp_scaling_sweep.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    exp_info = data.get("experiment", {})
    output_info = data.get("output", {})
    output_dir = ROOT / output_info.get("dir", "results/raw/dcp_scaling")
    file_prefix = output_info.get("file_prefix", "dcp_scaling_sweep")

    print(f"  Config:      {config_path}")
    print(f"  Output:      {output_dir}/{file_prefix}.parquet")

    t0 = time.perf_counter()
    combined_df, summary_rows, jobs = _run_sweep_from_yaml(config_path, orchestrator)
    elapsed = time.perf_counter() - t0

    print(f"\n  Completed {len(jobs)} configurations in {elapsed:.1f}s")

    # Build scaling summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\n  Scaling Summary Table:")
    print(summary_df[["N", "n", "k", "circuit_depth", "num_qubits", "runtime_sec"]].to_string(index=False))

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

    # Save scaling CSV summary
    agg_dir = ROOT / "results" / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    csv_path = agg_dir / "scaling_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved Scaling Table: {csv_path}")

    # Generate scaling plots
    print("\n  Generating scaling plots...")
    fig_dir = ROOT / "results" / "figures" / "dcp_scaling"
    fig_dir.mkdir(parents=True, exist_ok=True)

    slope_rt = 0.0
    slope_d = 0.0
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})

        N_vals = summary_df["N"].values.astype(float)
        runtime_vals = summary_df["runtime_sec"].values.astype(float)
        depth_vals = summary_df["circuit_depth"].values.astype(float)
        qubit_vals = summary_df["num_qubits"].values.astype(float)

        # Log-log fit helper
        def loglog_fit(x, y):
            log_x = np.log2(x)
            log_y = np.log2(y + 1e-12)
            coeffs = np.polyfit(log_x, log_y, 1)
            return coeffs[0], coeffs

        # Plot 1: Runtime vs N (log-log)
        slope_rt, coeffs_rt = loglog_fit(N_vals, runtime_vals)
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.loglog(N_vals, runtime_vals, "o-", color="#2b5c8f", linewidth=2.2, markersize=8, label="Measured Runtime")
        ref_rt = 2 ** np.polyval(coeffs_rt, np.log2(N_vals))
        ax.loglog(N_vals, ref_rt, "--", color="#d95f02", linewidth=1.5, label=f"Fit slope ≈ {slope_rt:.2f}")
        ax.set_xlabel("Modulus $N$")
        ax.set_ylabel("Runtime per Configuration (s)")
        ax.set_title("Simulation Runtime Scaling vs. $N$ (log-log)")
        ax.set_xticks(list(N_vals))
        ax.set_xticklabels([str(int(n)) for n in N_vals])
        ax.legend(frameon=True)
        p1 = fig_dir / "scaling_runtime.png"
        fig.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [scaling]  {p1}")

        # Plot 2: Circuit Depth vs N (log-log)
        slope_d, coeffs_d = loglog_fit(N_vals, depth_vals)
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.loglog(N_vals, depth_vals, "s-", color="#7570b3", linewidth=2.2, markersize=8, label="Circuit Depth")
        ref_d = 2 ** np.polyval(coeffs_d, np.log2(N_vals))
        ax.loglog(N_vals, ref_d, "--", color="#d95f02", linewidth=1.5, label=f"Fit slope ≈ {slope_d:.2f}")
        ax.set_xlabel("Modulus $N$")
        ax.set_ylabel("QFT Circuit Depth (gates)")
        ax.set_title("Circuit Depth Scaling vs. $N$ (log-log)")
        ax.set_xticks(list(N_vals))
        ax.set_xticklabels([str(int(n)) for n in N_vals])
        ax.legend(frameon=True)
        p2 = fig_dir / "scaling_circuit_depth.png"
        fig.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [scaling]  {p2}")

        # Plot 3: Combined 2-panel scaling dashboard
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        ax_l, ax_r = axes

        ax_l.loglog(N_vals, runtime_vals, "o-", color="#2b5c8f", linewidth=2.2, markersize=8, label="Runtime")
        ax_l.loglog(N_vals, ref_rt, "--", color="#d95f02", linewidth=1.5, label=f"slope≈{slope_rt:.2f}")
        ax_l.set_xlabel("Modulus $N$")
        ax_l.set_ylabel("Runtime (s)")
        ax_l.set_title("A: Runtime Scaling (log-log)", fontweight="bold")
        ax_l.set_xticks(list(N_vals))
        ax_l.set_xticklabels([str(int(n)) for n in N_vals])
        ax_l.legend(frameon=True)

        ax_r.loglog(N_vals, depth_vals, "s-", color="#7570b3", linewidth=2.2, markersize=8, label="Depth")
        ax_r.loglog(N_vals, ref_d, "--", color="#d95f02", linewidth=1.5, label=f"slope≈{slope_d:.2f}")
        ax_r.set_xlabel("Modulus $N$")
        ax_r.set_ylabel("Circuit Depth (gates)")
        ax_r.set_title("B: Circuit Depth Scaling (log-log)", fontweight="bold")
        ax_r.set_xticks(list(N_vals))
        ax_r.set_xticklabels([str(int(n)) for n in N_vals])
        ax_r.legend(frameon=True)

        fig.suptitle("DCP Simulation Scaling Analysis", fontsize=14, fontweight="bold")
        p3 = fig_dir / "scaling_dashboard.png"
        fig.savefig(p3, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [scaling]  {p3}")

    except Exception as exc:
        print(f"  Warning: scaling plots failed — {exc}")

    # Scaling analysis
    print("\n  Scaling Analysis:")
    print(f"    Runtime slope (log-log fit): {slope_rt:.3f}")
    print(f"    Circuit depth slope:         {slope_d:.3f}")
    print(f"    (Expected O(n^2) ≈ O(log^2 N): slope ~ 2 in log-log plot)")
    print(f"    N=64 completed: {'YES' if 64 in summary_df['N'].values else 'NO'}")

    _banner("Phase 5.2 COMPLETE")


# ──────────────────────────────────────────────────────────────────────────────
# Task 5.3 — Combined Parameter Sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_combined_sweep(orchestrator: Orchestrator) -> None:
    _banner("Phase 5.3 — Combined Parameter Sweep (N, k, m, epsilon)")

    config_path = ROOT / "configs" / "dcp_combined_sweep.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    exp_info = data.get("experiment", {})
    output_info = data.get("output", {})
    output_dir = ROOT / output_info.get("dir", "results/raw/dcp_combined")
    file_prefix = output_info.get("file_prefix", "dcp_combined_sweep")

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

    # Generate combined heatmaps and dashboards
    print("\n  Generating combined heatmaps and dashboards...")
    fig_dir = ROOT / "results" / "figures" / "dcp_heatmaps"

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})
        PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]
        fig_dir.mkdir(parents=True, exist_ok=True)

        metric = "mirror_correct" if "mirror_correct" in combined_df.columns else "correct"

        # Aggregate
        grouped = combined_df.groupby(["N", "k", "m", "epsilon"]).agg(
            p_success=(metric, "mean")
        ).reset_index()

        # Heatmaps: P_success(k, epsilon) for each (N, m)
        print("\n  k-ε heatmaps:")
        gke = combined_df.groupby(["N", "m", "k", "epsilon"]).agg(
            p_success=(metric, "mean")
        ).reset_index()
        for N in sorted(gke["N"].unique()):
            sub_N = gke[gke["N"] == N]
            for m in sorted(sub_N["m"].unique()):
                sub = sub_N[sub_N["m"] == m]
                pivot = sub.pivot(index="k", columns="epsilon", values="p_success")
                pivot = pivot.sort_index(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                            vmin=0.0, vmax=1.0, linewidths=0.5,
                            cbar_kws={"label": "P(recovery)"}, ax=ax)
                ax.set_title(f"P(Recovery) — $k$ vs $\\varepsilon$ ($N={N}$, $m={m}$)")
                ax.set_xlabel("Noise Level ($\\varepsilon$)")
                ax.set_ylabel("Retained Bits ($k$)")
                fname = fig_dir / f"combined_heatmap_k_eps_N{N}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  [k-ε heatmap]  {fname}")

        # Heatmaps: P_success(k, m) for each (N, epsilon)
        print("\n  k-m heatmaps:")
        gkm = combined_df.groupby(["N", "epsilon", "k", "m"]).agg(
            p_success=(metric, "mean")
        ).reset_index()
        for N in sorted(gkm["N"].unique()):
            sub_N = gkm[gkm["N"] == N]
            for eps in sorted(sub_N["epsilon"].unique()):
                sub = sub_N[sub_N["epsilon"] == eps]
                pivot = sub.pivot(index="k", columns="m", values="p_success")
                pivot = pivot.sort_index(ascending=False)
                fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues",
                            vmin=0.0, vmax=1.0, linewidths=0.5,
                            cbar_kws={"label": "P(recovery)"}, ax=ax)
                ax.set_title(f"P(Recovery) — $k$ vs $m$ ($N={N}$, $\\varepsilon={eps}$)")
                ax.set_xlabel("Number of Samples ($m$)")
                ax.set_ylabel("Retained Bits ($k$)")
                eps_tag = str(eps).replace(".", "p")
                fname = fig_dir / f"combined_heatmap_k_m_N{N}_eps{eps_tag}.png"
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  [k-m heatmap]  {fname}")

        # Combined dashboard per N
        print("\n  Combined dashboards:")
        for N in sorted(grouped["N"].unique()):
            sub_N = grouped[grouped["N"] == N]
            n_bits = max(1, int(N - 1).bit_length())
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

            # Panel A: P(rec) vs k for eps=0, varying m
            ax_a = axes[0, 0]
            sub_0 = sub_N[sub_N["epsilon"] == 0.0]
            for idx, m in enumerate(sorted(sub_0["m"].unique())):
                sub_m = sub_0[sub_0["m"] == m].sort_values("k")
                ax_a.plot(sub_m["k"], sub_m["p_success"], "o-", label=f"m={m}",
                          color=PALETTE[idx % len(PALETTE)], linewidth=2.2, markersize=7)
            ax_a.axhline(1.0 / N, color="gray", linestyle=":", alpha=0.7, label="Random")
            ax_a.set_title("A: P(Recovery) vs $k$ ($\\varepsilon=0$)", fontweight="bold")
            ax_a.set_xlabel("Retained Bits ($k$)")
            ax_a.set_ylabel("P(Recovery)")
            ax_a.set_ylim(-0.02, 1.02)
            ax_a.legend(title="Samples ($m$)")

            # Panel B: P(rec) vs epsilon for m=1, varying k
            ax_b = axes[0, 1]
            m_val = sorted(sub_N["m"].unique())[0]
            sub_m1 = sub_N[sub_N["m"] == m_val]
            for idx, k in enumerate(sorted(sub_m1["k"].unique())):
                sub_k = sub_m1[sub_m1["k"] == k].sort_values("epsilon")
                ax_b.plot(sub_k["epsilon"], sub_k["p_success"], "s-", label=f"k={k}",
                          color=PALETTE[idx % len(PALETTE)], linewidth=2.2, markersize=7)
            ax_b.axhline(1.0 / N, color="gray", linestyle=":", alpha=0.7, label="Random")
            ax_b.set_title(f"B: P(Recovery) vs $\\varepsilon$ ($m={m_val}$)", fontweight="bold")
            ax_b.set_xlabel("Noise Level ($\\varepsilon$)")
            ax_b.set_ylabel("P(Recovery)")
            ax_b.set_ylim(-0.02, 1.02)
            ax_b.legend(title="Retained Bits ($k$)")

            # Panel C: Heatmap k-epsilon for m=1
            ax_c = axes[1, 0]
            sub_c = sub_N[sub_N["m"] == m_val]
            try:
                pivot_c = sub_c.pivot(index="k", columns="epsilon", values="p_success").sort_index(ascending=False)
                sns.heatmap(pivot_c, annot=True, fmt=".2f", cmap="RdYlGn",
                            vmin=0.0, vmax=1.0, linewidths=0.5,
                            cbar_kws={"label": "P(recovery)"}, ax=ax_c)
                ax_c.set_title(f"C: Heatmap $k$ vs $\\varepsilon$ ($m={m_val}$)", fontweight="bold")
                ax_c.set_xlabel("Noise Level ($\\varepsilon$)")
                ax_c.set_ylabel("Retained Bits ($k$)")
            except Exception:
                ax_c.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax_c.transAxes)

            # Panel D: Heatmap k-m for eps=0
            ax_d = axes[1, 1]
            sub_d = sub_N[sub_N["epsilon"] == 0.0]
            try:
                pivot_d = sub_d.pivot(index="k", columns="m", values="p_success").sort_index(ascending=False)
                sns.heatmap(pivot_d, annot=True, fmt=".2f", cmap="Blues",
                            vmin=0.0, vmax=1.0, linewidths=0.5,
                            cbar_kws={"label": "P(recovery)"}, ax=ax_d)
                ax_d.set_title("D: Heatmap $k$ vs $m$ ($\\varepsilon=0$)", fontweight="bold")
                ax_d.set_xlabel("Number of Samples ($m$)")
                ax_d.set_ylabel("Retained Bits ($k$)")
            except Exception:
                ax_d.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax_d.transAxes)

            fig.suptitle(
                f"DCP Combined Parameter Sweep — $N={N}$ ($n={n_bits}$ bits)",
                fontsize=14, fontweight="bold",
            )
            fname = fig_dir / f"combined_dashboard_N{N}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [dashboard]    {fname}")

    except Exception as exc:
        print(f"  Warning: combined plotting failed — {exc}")
        import traceback
        traceback.print_exc()

    _banner("Phase 5.3 COMPLETE")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 Master Orchestrator (Noise, Scaling, Combined Sweep)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["noise", "scaling", "combined", "all"],
        default="all",
        help="Which Phase 5 sub-task to run (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.task

    print("=" * 70)
    print("  PHASE 5 — Noise & Scaling Experiments")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Task:    {task}")
    print("=" * 70)

    orchestrator = Orchestrator()
    t_start = time.perf_counter()

    if task in ("noise", "all"):
        run_noise_sweep(orchestrator)

    if task in ("scaling", "all"):
        run_scaling_sweep(orchestrator)

    if task in ("combined", "all"):
        run_combined_sweep(orchestrator)

    total = time.perf_counter() - t_start
    _banner(f"PHASE 5 DONE — total wall time {total:.1f}s")
    print(f"  Results: {ROOT / 'results'}")
    print()


if __name__ == "__main__":
    main()
