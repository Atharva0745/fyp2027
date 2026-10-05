"""Scientific verification helpers for DCP/EDCP experiments.

These checks are designed to validate the paper's intended logic beyond a
single final recovery score. They cover the main verification concerns:

- Set-A / Set-B grouping statistics
- direct numerical checks of the paper's lemma-style identities
- distinguishability experiments between nearby secrets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Hashable, Sequence

import numpy as np


@dataclass
class VerificationReport:
    """Container for intermediate-state diagnostics and lemma checks."""

    stage_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    set_a_count: int | None = None
    set_b_count: int | None = None
    set_a_sizes: list[int] = field(default_factory=list)
    set_b_sizes: list[int] = field(default_factory=list)
    lemma2_gap: float | None = None
    lemma3_max_amplitude: float | None = None
    lemma4_relative_gap: float | None = None
    distinguishability: float | None = None


@dataclass(frozen=True)
class IndependenceTestResult:
    """Empirical dependence metrics for paired ``(h_star, B)`` observations."""

    outcome_values: tuple[Hashable, ...]
    probabilities_b_given_h0: tuple[float, ...]
    probabilities_b_given_h1: tuple[float, ...]
    count_h0: int
    count_h1: int
    probability_h1: float
    total_variation: float
    kl_divergence_h0_to_h1: float
    mutual_information_bits: float


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


def _normalize_probability_distribution(probabilities: Sequence[float]) -> np.ndarray:
    distribution = np.asarray(probabilities, dtype=float)
    if distribution.ndim != 1 or distribution.size == 0:
        raise ValueError("Probability distribution must be a non-empty vector")
    if not np.all(np.isfinite(distribution)) or np.any(distribution < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    total = float(np.sum(distribution))
    if total <= 0:
        raise ValueError("Probability distribution must have positive mass")
    return distribution / total


def compute_kl_divergence(
    probabilities_p: Sequence[float],
    probabilities_q: Sequence[float],
) -> float:
    """Compute ``D_KL(P || Q)`` in bits from explicit probability vectors."""
    p = _normalize_probability_distribution(probabilities_p)
    q = _normalize_probability_distribution(probabilities_q)
    if p.shape != q.shape:
        raise ValueError("Probability vectors must have the same shape for KL divergence")

    positive_p = p > 0
    if np.any(q[positive_p] == 0):
        return float("inf")
    return float(np.sum(p[positive_p] * np.log2(p[positive_p] / q[positive_p])))


def compute_mutual_information_from_conditionals(
    probabilities_b_given_h0: Sequence[float],
    probabilities_b_given_h1: Sequence[float],
    probability_h1: float = 0.5,
) -> float:
    """Compute ``I(H; B)`` in bits from ``P(B|H=0)`` and ``P(B|H=1)``."""
    if not np.isfinite(probability_h1) or not 0.0 <= probability_h1 <= 1.0:
        raise ValueError("probability_h1 must be between 0 and 1")

    p_b_h0 = _normalize_probability_distribution(probabilities_b_given_h0)
    p_b_h1 = _normalize_probability_distribution(probabilities_b_given_h1)
    if p_b_h0.shape != p_b_h1.shape:
        raise ValueError("Conditional probability vectors must have the same shape")

    p_h1 = float(probability_h1)
    marginal_b = (1.0 - p_h1) * p_b_h0 + p_h1 * p_b_h1
    mutual_information = 0.0
    for p_h, conditional in ((1.0 - p_h1, p_b_h0), (p_h1, p_b_h1)):
        if p_h == 0.0:
            continue
        positive = conditional > 0
        mutual_information += p_h * float(
            np.sum(conditional[positive] * np.log2(conditional[positive] / marginal_b[positive]))
        )
    return max(0.0, mutual_information)


def analyze_conditional_independence(
    h_star_values: Sequence[int],
    b_outcomes: Sequence[Hashable],
) -> IndependenceTestResult:
    """Estimate ``P(B|h*=0/1)`` and compare the conditional distributions.

    Inputs must be paired outcomes from repeated trials. This function does not
    generate the attack's ``h*`` or B outcomes; it analyzes supplied data.
    """
    if len(h_star_values) != len(b_outcomes):
        raise ValueError("h_star_values and b_outcomes must have equal lengths")
    if not h_star_values:
        raise ValueError("at least one paired observation is required")

    paired_values: list[tuple[int, Hashable]] = []
    for h_star, b_outcome in zip(h_star_values, b_outcomes):
        if not isinstance(h_star, Integral) or h_star not in (0, 1):
            raise ValueError("h_star_values must contain only 0 or 1")
        try:
            hash(b_outcome)
        except TypeError as error:
            raise ValueError("each B outcome must be a hashable category") from error
        paired_values.append((int(h_star), b_outcome))

    count_h0 = sum(h_star == 0 for h_star, _ in paired_values)
    count_h1 = len(paired_values) - count_h0
    if count_h0 == 0 or count_h1 == 0:
        raise ValueError("observations must include both h*=0 and h*=1")

    outcome_values = tuple(
        sorted({outcome for _, outcome in paired_values}, key=repr)
    )
    outcome_index = {outcome: index for index, outcome in enumerate(outcome_values)}
    counts_h0 = np.zeros(len(outcome_values), dtype=float)
    counts_h1 = np.zeros(len(outcome_values), dtype=float)
    for h_star, outcome in paired_values:
        target = counts_h1 if h_star else counts_h0
        target[outcome_index[outcome]] += 1.0

    probabilities_h0 = counts_h0 / count_h0
    probabilities_h1 = counts_h1 / count_h1
    probability_h1 = count_h1 / len(paired_values)
    conditional_h0 = tuple(float(value) for value in probabilities_h0)
    conditional_h1 = tuple(float(value) for value in probabilities_h1)

    return IndependenceTestResult(
        outcome_values=outcome_values,
        probabilities_b_given_h0=conditional_h0,
        probabilities_b_given_h1=conditional_h1,
        count_h0=count_h0,
        count_h1=count_h1,
        probability_h1=float(probability_h1),
        total_variation=compute_distinguishability(conditional_h0, conditional_h1),
        kl_divergence_h0_to_h1=compute_kl_divergence(conditional_h0, conditional_h1),
        mutual_information_bits=compute_mutual_information_from_conditionals(
            conditional_h0,
            conditional_h1,
            probability_h1=probability_h1,
        ),
    )


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

    set_stats = compute_set_ab_statistics(labels) if labels is not None else None
    lemma2_gap = (
        evaluate_lemma_2(branch_h0, branch_h1)
        if branch_h0 is not None and branch_h1 is not None
        else None
    )
    amplitude_sets = [values for values in (amplitudes_h0, amplitudes_h1) if values is not None]
    lemma3_bound = (
        max(evaluate_lemma_3(values) for values in amplitude_sets)
        if amplitude_sets
        else None
    )
    lemma4_gap = (
        evaluate_lemma_4(amplitudes_h0, amplitudes_h1)
        if amplitudes_h0 is not None and amplitudes_h1 is not None
        else None
    )
    distinguishability = (
        compute_distinguishability(prob_a, prob_b)
        if prob_a is not None and prob_b is not None
        else None
    )

    return VerificationReport(
        stage_metrics=stage_metrics,
        set_a_count=set_stats["set_a_count"] if set_stats is not None else None,
        set_b_count=set_stats["set_b_count"] if set_stats is not None else None,
        set_a_sizes=set_stats["set_a_sizes"] if set_stats is not None else [],
        set_b_sizes=set_stats["set_b_sizes"] if set_stats is not None else [],
        lemma2_gap=lemma2_gap,
        lemma3_max_amplitude=lemma3_bound,
        lemma4_relative_gap=lemma4_gap,
        distinguishability=distinguishability,
    )
