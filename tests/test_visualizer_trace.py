"""Tests for the visualizer's event trace and run equivalence."""

import pytest

from src.config import ExperimentConfig
from src.orchestrator import Orchestrator, TraceEvent


@pytest.mark.parametrize(
    "strategy",
    ["bayesian", "brute_force", "ml", "bitwise"],
)
def test_traced_run_preserves_single_trial_recovery(strategy: str) -> None:
    config = ExperimentConfig(
        N=8,
        s=3,
        k=2,
        m=3,
        epsilon=0.05,
        shots=1,
        seed=1234,
        recovery_strategy=strategy,
    )

    normal_result = Orchestrator().run(config)
    events = list(Orchestrator().run_traced(config, offset_override=5))

    assert events
    assert all(isinstance(event, TraceEvent) for event in events)
    assert events[0].stage == "state"
    assert events[0].payload["x"] == 5
    assert events[0].payload["target"] == (5 + config.s) % config.N
    assert [event.stage for event in events].count("sample") == config.m
    assert [event.stage for event in events].count("posterior") == config.m
    sample = next(event for event in events if event.stage == "sample")
    assert sample.payload["Y_noisy"] is not None
    assert sample.payload["b_sampled"] in (0, 1)
    assert sample.payload["flag_bit_flipped"] in (True, False)

    normal_recovery = normal_result.recovery_result
    verdict = next(event for event in events if event.stage == "verdict")
    assert normal_recovery is not None
    assert verdict.payload["s_hat"] == normal_recovery.s_hat
    assert verdict.payload["correct"] == normal_recovery.correct
    assert verdict.payload["mirror_correct"] == normal_recovery.mirror_correct
    assert verdict.payload["confidence"] == pytest.approx(normal_recovery.confidence)


def test_bayesian_trace_posterior_matches_recovery_engine() -> None:
    config = ExperimentConfig(
        N=16,
        s=5,
        k=3,
        m=4,
        epsilon=0.0,
        shots=1,
        seed=42,
        recovery_strategy="bayesian",
    )
    result = Orchestrator().run(config)
    events = list(Orchestrator().run_traced(config, offset_override=11))

    recovery = result.recovery_result
    posterior_event = [
        event for event in events if event.stage == "posterior"
    ][-1]
    verdict = next(event for event in events if event.stage == "verdict")

    assert recovery is not None
    assert posterior_event.payload["posterior"] == pytest.approx(recovery.posterior)
    assert verdict.payload["posterior"] == pytest.approx(recovery.posterior)


def test_traced_run_rejects_out_of_range_offset() -> None:
    config = ExperimentConfig(N=8, s=2, shots=1)
    events = Orchestrator().run_traced(config, offset_override=8)

    with pytest.raises(ValueError, match="offset_override"):
        next(events)
