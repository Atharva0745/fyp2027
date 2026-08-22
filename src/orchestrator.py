"""Experiment Orchestrator executing DCP/EDCP simulation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd

from src.config import ExperimentConfig, validate_config
from src.engines.dcp_engine import DCPEngine, DCPState
from src.engines.info_engine import InformationEngine, InformationResult
from src.engines.qft_engine import QFTEngine, QFTResult
from src.engines.recovery_engine import RecoveryEngine, RecoveryResult
from src.utils.math_utils import wilson_score_interval
from src.utils.serialization import save_experiment_result


@dataclass
class StatisticsResult:
    """Dataclass holding statistical metrics aggregated across trials."""

    recovery_prob: float
    mirror_recovery_prob: float
    recovery_prob_ci: tuple[float, float]
    mirror_recovery_prob_ci: tuple[float, float]
    bit_recovery_probs: list[float]
    bit_advantages: list[float]
    runtime_seconds: float
    circuit_depth: int
    num_qubits: int
    num_samples: int
    raw_data: pd.DataFrame


@dataclass
class ExperimentResult:
    """Dataclass holding single experiment run results and telemetry."""

    config: ExperimentConfig
    dcp_state: DCPState | None
    qft_result: QFTResult | None
    info_result: InformationResult | None
    recovery_result: RecoveryResult | None
    statistics: StatisticsResult | None
    timestamp: str


class Orchestrator:
    """Top-level experiment runner orchestrating the entire quantum pipeline."""

    def __init__(self) -> None:
        self.dcp_engine = DCPEngine()
        self.qft_engine = QFTEngine()

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run an end-to-end experiment according to the configuration.

        Args:
            config: ExperimentConfig instance.

        Returns:
            ExperimentResult containing trial data and summary statistics.
        """
        validate_config(config)

        start_time = time.perf_counter()
        rng = np.random.default_rng(config.seed)
        info_engine = InformationEngine(rng=rng)
        recovery_engine = RecoveryEngine(rng=rng)

        N = config.N
        s = config.s
        n = config.n
        k = config.k if config.k is not None else n
        m = config.m
        shots = config.shots

        trials: list[dict[str, Any]] = []

        last_dcp_state: DCPState | None = None
        last_qft_res: QFTResult | None = None
        last_info_res: InformationResult | None = None
        last_rec_res: RecoveryResult | None = None

        circuit_depth = 0
        num_qubits = n + 1

        for trial_idx in range(shots):
            # For m samples, generate independent offsets x_i
            offsets = [int(rng.integers(0, N)) for _ in range(m)]
            obs_list: list[InformationResult] = []

            for x_i in offsets:
                dcp_state = self.dcp_engine.create_state(N=N, s=s, x=x_i)
                qft_res = self.qft_engine.transform(dcp_state)
                info_res = info_engine.process(
                    qft_result=qft_res,
                    k=config.k,
                    noise_level=config.epsilon,
                    truncation_mode=config.truncation_mode,
                    rng=rng,
                )
                obs_list.append(info_res)

                if trial_idx == 0:
                    last_dcp_state = dcp_state
                    last_qft_res = qft_res
                    last_info_res = info_res
                    circuit_depth = dcp_state.circuit.depth()

            rec_res = recovery_engine.recover(
                observations=obs_list,
                N=N,
                s_true=s,
                k=config.k,
                n=n,
                strategy=config.recovery_strategy,
                truncation_mode=config.truncation_mode,
                rng=rng,
            )

            if trial_idx == 0:
                last_rec_res = rec_res

            # Record trial metrics
            trial_record: dict[str, Any] = {
                "trial": trial_idx,
                "N": N,
                "n": n,
                "k": k,
                "s_true": s,
                "s_hat": rec_res.s_hat,
                "correct": bool(rec_res.correct),
                "mirror_correct": bool(rec_res.mirror_correct),
                "confidence": float(rec_res.confidence),
                "m": m,
                "epsilon": config.epsilon,
                "truncation_mode": config.truncation_mode,
                "strategy": config.recovery_strategy,
                "mean_bit_accuracy": float(np.mean(rec_res.bit_correct)),
            }

            for bit_i, is_bit_corr in enumerate(rec_res.bit_correct):
                trial_record[f"bit_correct_{bit_i}"] = bool(is_bit_corr)

            trials.append(trial_record)

        total_runtime = time.perf_counter() - start_time
        raw_df = pd.DataFrame(trials)

        # Compute aggregate statistical metrics
        successes = int(raw_df["correct"].sum())
        mirror_successes = int(raw_df["mirror_correct"].sum())
        rec_prob = float(successes / shots)
        mirror_prob = float(mirror_successes / shots)
        ci_lower, ci_upper = wilson_score_interval(successes, shots, confidence=0.95)
        mirror_ci_lower, mirror_ci_upper = wilson_score_interval(mirror_successes, shots, confidence=0.95)

        bit_probs: list[float] = []
        bit_advs: list[float] = []
        for bit_i in range(n):
            col = f"bit_correct_{bit_i}"
            if col in raw_df.columns:
                p_bit = float(raw_df[col].mean())
                bit_probs.append(p_bit)
                bit_advs.append(p_bit - 0.5)

        stats_res = StatisticsResult(
            recovery_prob=rec_prob,
            mirror_recovery_prob=mirror_prob,
            recovery_prob_ci=(ci_lower, ci_upper),
            mirror_recovery_prob_ci=(mirror_ci_lower, mirror_ci_upper),
            bit_recovery_probs=bit_probs,
            bit_advantages=bit_advs,
            runtime_seconds=total_runtime,
            circuit_depth=circuit_depth,
            num_qubits=num_qubits,
            num_samples=shots,
            raw_data=raw_df,
        )

        return ExperimentResult(
            config=config,
            dcp_state=last_dcp_state,
            qft_result=last_qft_res,
            info_result=last_info_res,
            recovery_result=last_rec_res,
            statistics=stats_res,
            timestamp=datetime.now().isoformat(),
        )

    def run_sweep(
        self,
        base_config: ExperimentConfig,
        param_grid: dict[str, list[Any]],
        output_dir: str | Path | None = None,
        file_prefix: str = "sweep_results",
    ) -> pd.DataFrame:
        """Run parameter sweep over the Cartesian product of param_grid.

        Args:
            base_config: Template ExperimentConfig.
            param_grid: Dictionary mapping parameter names to lists of values to test.
            output_dir: Optional directory to persist raw results and metadata.
            file_prefix: Base filename for saved artifacts.

        Returns:
            Concatenated DataFrame of all trial-level results across sweep.
        """
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        combinations = list(itertools.product(*value_lists))

        all_dfs: list[pd.DataFrame] = []
        summary_records: list[dict[str, Any]] = []

        for combo in combinations:
            cfg_dict = base_config.__dict__.copy()
            for k, v in zip(keys, combo):
                cfg_dict[k] = v

            # If N changed, recompute n
            cfg_dict["n"] = max(1, (cfg_dict["N"] - 1).bit_length())
            # Ensure s < N
            if cfg_dict["s"] >= cfg_dict["N"]:
                cfg_dict["s"] = cfg_dict["s"] % cfg_dict["N"]

            current_cfg = ExperimentConfig(**cfg_dict)
            exp_res = self.run(current_cfg)

            assert exp_res.statistics is not None
            all_dfs.append(exp_res.statistics.raw_data)

            summary_records.append({
                "N": current_cfg.N,
                "n": current_cfg.n,
                "k": current_cfg.k,
                "s": current_cfg.s,
                "m": current_cfg.m,
                "epsilon": current_cfg.epsilon,
                "recovery_prob": exp_res.statistics.recovery_prob,
                "ci_lower": exp_res.statistics.recovery_prob_ci[0],
                "ci_upper": exp_res.statistics.recovery_prob_ci[1],
                "runtime_seconds": exp_res.statistics.runtime_seconds,
            })

        combined_df = pd.concat(all_dfs, ignore_index=True)

        if output_dir is not None:
            metadata = {
                "base_config": base_config.__dict__,
                "param_grid": param_grid,
                "timestamp": datetime.now().isoformat(),
                "num_configurations": len(combinations),
                "summary": summary_records,
            }
            save_experiment_result(
                result_df=combined_df,
                metadata=metadata,
                output_dir=output_dir,
                file_prefix=file_prefix,
            )

        return combined_df
