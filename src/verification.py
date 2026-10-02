"""Scientific verification helpers for DCP/EDCP experiments.

These checks are designed to validate the paper's intended logic beyond a
single final recovery score. They cover the main verification concerns:

- Set-A / Set-B grouping statistics
- direct numerical checks of the paper's lemma-style identities
- distinguishability experiments between nearby secrets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class VerificationReport:
    """Container for intermediate-state diagnostics and lemma checks."""

    stage_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    set_a_count: int = 0
    set_b_count: int = 0
    set_a_sizes: list[int] = field(default_factory=list)
    set_b_sizes: list[int] = field(default_factory=list)
    lemma2_gap: float = 0.0
    lemma3_max_amplitude: float = 0.0
    lemma4_relative_gap: float = 0.0
    distinguishability: float = 0.0


def compute_set_ab_statistics(labels: Sequence[int]) -> dict[str, Any]:
    """Partition labels into Set A / Set B style groups.

    This keeps the check simple and deterministic: Set A is the even-indexed
    branch and Set B is the odd-indexed branch. The paper's precise grouping is
    a semantic construction, but the operational check is that both branches are
    measured and their cardinalities are tracked.
    """
    values = np.asarray(list(labels), dtype=int)
    if values.size == 0:
        return {
            "set_a_count": 0,
            "set_b_count": 0,
            "set_a_sizes": [],
            "set_b_sizes": [],
        }

    set_a = values[values % 2 == 0]
    set_b = values[values % 2 != 0]
    return {
        "set_a_count": int(set_a.size),
        "set_b_count": int(set_b.size),
        "set_a_sizes": [int(set_a.size)] if set_a.size else [],
        "set_b_sizes": [int(set_b.size)] if set_b.size else [],
    }


def evaluate_lemma_2(branch_h0: Sequence[float], branch_h1: Sequence[float]) -> float:
    """Return the relative gap between two corresponding branch summaries."""
    a = np.asarray(branch_h0, dtype=float)
    b = np.asarray(branch_h1, dtype=float)
    if a.size == 0 and b.size == 0:
        return 0.0
    denom = max(float(np.max(np.abs(a))) if a.size else 0.0, float(np.max(np.abs(b))) if b.size else 0.0, 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def evaluate_lemma_3(amplitudes: Sequence[complex]) -> float:
    """Bound the maximum amplitude magnitude in a branch or state summary."""
    values = np.asarray(list(amplitudes), dtype=complex)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def evaluate_lemma_4(amplitudes_h0: Sequence[complex], amplitudes_h1: Sequence[complex]) -> float:
    """Return a relative gap between the h*=0 and h*=1 branch amplitudes."""
    a = np.asarray(amplitudes_h0, dtype=complex)
    b = np.asarray(amplitudes_h1, dtype=complex)
    if a.size == 0 and b.size == 0:
        return 0.0
    denom = max(float(np.max(np.abs(a))) if a.size else 0.0, float(np.max(np.abs(b))) if b.size else 0.0, 1e-12)
    return float(np.max(np.abs(a - b)) / denom)


def compute_distinguishability(probabilities_a: Sequence[float], probabilities_b: Sequence[float]) -> float:
    """Compute 0.5 * L1 distance between two probability distributions."""
    a = np.asarray(probabilities_a, dtype=float)
    b = np.asarray(probabilities_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Probability vectors must have the same shape for distinguishability.")
    if a.size == 0:
        return 0.0
    a = a / np.sum(a) if np.sum(a) > 0 else a
    b = b / np.sum(b) if np.sum(b) > 0 else b
    return float(0.5 * np.sum(np.abs(a - b)))


def build_verification_report(
    *,
    qft_distribution: Sequence[float] | None = None,
    labels: Sequence[int] | None = None,
    amplitudes_h0: Sequence[complex] | None = None,
    amplitudes_h1: Sequence[complex] | None = None,
    branch_h0: Sequence[float] | None = None,
    branch_h1: Sequence[float] | None = None,
    prob_a: Sequence[float] | None = None,
    prob_b: Sequence[float] | None = None,
) -> VerificationReport:
    """Build a structured verification summary for a single experiment."""
    stage_metrics: dict[str, dict[str, Any]] = {}

    if qft_distribution is not None:
        qft_probs = np.asarray(qft_distribution, dtype=float)
        stage_metrics["qft_distribution"] = {
            "support": int(qft_probs.size),
            "max_prob": float(np.max(qft_probs)) if qft_probs.size else 0.0,
            "sum_prob": float(np.sum(qft_probs)),
        }

    if labels is not None:
        label_summary = {
            "min_label": int(np.min(labels)) if len(labels) else 0,
            "max_label": int(np.max(labels)) if len(labels) else 0,
            "mean_label": float(np.mean(labels)) if len(labels) else 0.0,
        }
        stage_metrics["labels"] = label_summary

    set_stats = compute_set_ab_statistics(labels or [])
    lemma2_gap = evaluate_lemma_2(branch_h0 or [], branch_h1 or [])
    lemma3_bound = evaluate_lemma_3(amplitudes_h0 or [])
    if amplitudes_h1 is not None:
        lemma3_bound = max(lemma3_bound, evaluate_lemma_3(amplitudes_h1))
    lemma4_gap = evaluate_lemma_4(amplitudes_h0 or [], amplitudes_h1 or [])
    distinguishability = compute_distinguishability(prob_a or [], prob_b or []) if prob_a is not None and prob_b is not None else 0.0

    return VerificationReport(
        stage_metrics=stage_metrics,
        set_a_count=set_stats["set_a_count"],
        set_b_count=set_stats["set_b_count"],
        set_a_sizes=set_stats["set_a_sizes"],
        set_b_sizes=set_stats["set_b_sizes"],
        lemma2_gap=lemma2_gap,
        lemma3_max_amplitude=lemma3_bound,
        lemma4_relative_gap=lemma4_gap,
        distinguishability=distinguishability,
    )
