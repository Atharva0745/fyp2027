"""Experiment Orchestrator executing DCP/EDCP simulation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path
import time
from typing import Any, Generator
import numpy as np
import pandas as pd

from src.config import ExperimentConfig, validate_config
from src.engines.dcp_engine import DCPEngine, DCPState
from src.engines.info_engine import InformationEngine, InformationResult
from src.engines.qft_engine import QFTEngine, QFTResult
from src.engines.recovery_engine import RecoveryEngine, RecoveryResult
from src.recovery.bayesian import bayesian_recovery
from src.utils.math_utils import wilson_score_interval
from src.utils.serialization import save_experiment_result
from src.verification import VerificationReport, build_verification_report


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.engines.edcp_engine import EDCPState


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
    dcp_state: DCPState | EDCPState | None
    qft_result: QFTResult | None
    info_result: InformationResult | None
    recovery_result: RecoveryResult | None
    statistics: StatisticsResult | None
    verification: VerificationReport | None = None
    timestamp: str = ""


@dataclass(frozen=True)
class TraceEvent:
    """One observable stage in a traced experiment run."""

    stage: str
    trial: int
    sample: int | None
    payload: dict[str, Any]


class Orchestrator:
    """Top-level experiment runner orchestrating the entire quantum pipeline."""

    def __init__(self) -> None:
        self.dcp_engine = DCPEngine()
        self.qft_engine = QFTEngine()

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run an experiment without collecting intermediate trace events."""
        events = self._run_events(config, offset_override=None, trace_enabled=False)
        while True:
            try:
                next(events)
            except StopIteration as completed:
                return completed.value

    def run_traced(
        self,
        config: ExperimentConfig,
        offset_override: int | None = None,
    ) -> Generator[TraceEvent, None, ExperimentResult]:
        """Run an experiment while yielding its state, QFT, sample, and inference steps.

        An optional offset override fixes the first sample's random offset without
        changing the random stream used to sample measurement outcomes.
        """
        return self._run_events(
            config,
            offset_override=offset_override,
            trace_enabled=True,
        )

    def _run_events(
        self,
        config: ExperimentConfig,
        offset_override: int | None,
        trace_enabled: bool,
    ) -> Generator[TraceEvent, None, ExperimentResult]:
        """Run an end-to-end experiment according to the configuration.

        Args:
            config: ExperimentConfig instance.

        Returns:
            ExperimentResult containing trial data and summary statistics.
        """
        validate_config(config)
        if offset_override is not None and not (0 <= offset_override < config.N):
            raise ValueError(
                f"offset_override must be in [0, {config.N}), got {offset_override}"
            )

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

        from src.engines.edcp_engine import EDCPEngine, EDCPState, get_standard_chi

        last_dcp_state: DCPState | EDCPState | None = None
        last_qft_res: QFTResult | None = None
        last_info_res: InformationResult | None = None
        last_rec_res: RecoveryResult | None = None

        circuit_depth = 0
        num_qubits = n + 1

        is_edcp = config.problem_type == "edcp"
        edcp_engine = EDCPEngine() if is_edcp else None
        chi = (config.edcp_chi if config.edcp_chi else get_standard_chi("2-term")) if is_edcp else None

        for trial_idx in range(shots):
            # For m samples, generate independent offsets x_i
            offsets = [int(rng.integers(0, N)) for _ in range(m)]
            if trial_idx == 0 and offset_override is not None:
                offsets[0] = offset_override
            obs_list: list[InformationResult] = []

            for sample_idx, x_i in enumerate(offsets):
                if is_edcp and edcp_engine is not None and chi is not None:
                    state = edcp_engine.create_state(N=N, s=s, x=x_i, chi=chi)
                else:
                    state = self.dcp_engine.create_state(N=N, s=s, x=x_i)

                if trace_enabled and trial_idx == 0 and sample_idx == 0:
                    state_data = state.statevector.data
                    data_dimension = 1 << n
                    amplitudes = [
                        {
                            "basis": (
                                f"|{basis_index // data_dimension}>"
                                f"|{basis_index % data_dimension}>"
                            ),
                            "magnitude": float(abs(amplitude)),
                            "probability": float(abs(amplitude) ** 2),
                        }
                        for basis_index, amplitude in enumerate(state_data)
                        if abs(amplitude) > 1e-10
                    ]
                    yield TraceEvent(
                        stage="state",
                        trial=trial_idx,
                        sample=sample_idx,
                        payload={
                            "N": N,
                            "s": s,
                            "x": x_i,
                            "target": (x_i + s) % N,
                            "amplitudes": amplitudes,
                            "circuit_depth": state.circuit.depth(),
                            "circuit_text": str(state.circuit.draw(output="text")),
                        },
                    )

                qft_res = self.qft_engine.transform(state)
                if trace_enabled and trial_idx == 0 and sample_idx == 0:
                    yield TraceEvent(
                        stage="qft",
                        trial=trial_idx,
                        sample=sample_idx,
                        payload={
                            "N": N,
                            "distribution": qft_res.fourier_distribution.copy(),
                            "phases": qft_res.phases.copy(),
                        },
                    )

                info_res = info_engine.process(
                    qft_result=qft_res,
                    k=config.k,
                    noise_level=config.epsilon,
                    truncation_mode=config.truncation_mode,
                    rng=rng,
                )
                obs_list.append(info_res)

                if trace_enabled:
                    yield TraceEvent(
                        stage="sample",
                        trial=trial_idx,
                        sample=sample_idx,
                        payload={
                            "x": x_i,
                            "Y_full": info_res.Y_full,
                            "Y_noisy": info_res.Y_noisy,
                            "Y_truncated": info_res.Y_truncated,
                            "b": info_res.b,
                            "b_sampled": info_res.b_sampled,
                            "bit_flips": info_res.bit_flips.copy(),
                            "flag_bit_flipped": info_res.flag_bit_flipped,
                            "k": info_res.k,
                            "n": info_res.n,
                            "truncation_mode": info_res.truncation_mode,
                        },
                    )
                    _, posterior, confidence = bayesian_recovery(
                        observations=[
                            (observation.Y_truncated, observation.b)
                            for observation in obs_list
                        ],
                        k=config.k,
                        n=n,
                        N=N,
                        mode=config.truncation_mode,
                        rng=np.random.default_rng(0),
                    )
                    yield TraceEvent(
                        stage="posterior",
                        trial=trial_idx,
                        sample=sample_idx,
                        payload={
                            "posterior": posterior,
                            "confidence": float(confidence),
                            "samples_seen": sample_idx + 1,
                        },
                    )

                if trial_idx == 0:
                    last_dcp_state = state
                    last_qft_res = qft_res
                    last_info_res = info_res
                    circuit_depth = state.circuit.depth()

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
                "correct": rec_res.correct,
                "mirror_correct": rec_res.mirror_correct,
                "confidence": float(rec_res.confidence),
                "m": m,
                "epsilon": config.epsilon,
                "truncation_mode": config.truncation_mode,
                "strategy": config.recovery_strategy,
                "mean_bit_accuracy": float(np.mean(rec_res.bit_correct)),
            }

            for bit_i, is_bit_corr in enumerate(rec_res.bit_correct):
                trial_record[f"bit_correct_{bit_i}"] = is_bit_corr

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

        label_values = []
        if last_info_res is not None:
            label_values = [
                int(last_info_res.Y_truncated),
            ]
        if trials:
            label_values = [
                int(trial["s_hat"]) if "s_hat" in trial else 0 for trial in trials
            ]

        verification_report = build_verification_report(
            qft_distribution=list(last_qft_res.fourier_distribution.values()) if last_qft_res is not None else None,
            labels=label_values,
            amplitudes_h0=[complex(v) for v in (last_qft_res.phases.values() if last_qft_res is not None else [])],
            amplitudes_h1=[complex(v) for v in (last_qft_res.phases.values() if last_qft_res is not None else [])],
            branch_h0=[float(x) for x in (last_qft_res.fourier_distribution.values() if last_qft_res is not None else [])],
            branch_h1=[float(x) for x in (last_qft_res.fourier_distribution.values() if last_qft_res is not None else [])],
            prob_a=[float(rec_prob)],
            prob_b=[float(1.0 - rec_prob)] if rec_prob < 1.0 else [0.0, 1.0],
        )

        result = ExperimentResult(
            config=config,
            dcp_state=last_dcp_state,
            qft_result=last_qft_res,
            info_result=last_info_res,
            recovery_result=last_rec_res,
            statistics=stats_res,
            verification=verification_report,
            timestamp=datetime.now().isoformat(),
        )
        if trace_enabled:
            assert last_rec_res is not None
            yield TraceEvent(
                stage="verdict",
                trial=0,
                sample=None,
                payload={
                    "s_hat": last_rec_res.s_hat,
                    "s_true": last_rec_res.s_true,
                    "correct": last_rec_res.correct,
                    "mirror_correct": last_rec_res.mirror_correct,
                    "confidence": last_rec_res.confidence,
                    "bit_correct": last_rec_res.bit_correct,
                    "posterior": last_rec_res.posterior,
                    "recovery_prob": rec_prob,
                    "mirror_recovery_prob": mirror_prob,
                },
            )
        return result

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
