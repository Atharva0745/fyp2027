"""Mathematical and bit-manipulation utility functions."""

from __future__ import annotations

from typing import Sequence
import numpy as np


def int_to_bits(x: int, n: int) -> list[int]:
    """Convert an integer x into a list of n bits (LSB at index 0).

    Args:
        x: Integer to convert.
        n: Number of bits.

    Returns:
        List of 0s and 1s of length n, where result[0] is the LSB.
    """
    return [(x >> i) & 1 for i in range(n)]


def bits_to_int(bits: Sequence[int | bool]) -> int:
    """Convert a sequence of bits (LSB at index 0) back to an integer.

    Args:
        bits: Sequence of 0/1 or True/False.

    Returns:
        Integer value.
    """
    val = 0
    for i, b in enumerate(bits):
        if b:
            val |= 1 << i
    return val


def hamming_distance(a: int, b: int) -> int:
    """Compute the Hamming distance (number of differing bits) between two integers."""
    return (a ^ b).bit_count()


def bit_accuracy(s_hat: int, s_true: int, n: int) -> list[bool]:
    """Compute per-bit correctness between estimated secret and ground truth.

    Args:
        s_hat: Estimated secret integer.
        s_true: Ground truth secret integer.
        n: Total bit width.

    Returns:
        List of booleans where index i is True if the i-th bit matches.
    """
    return [((s_hat >> i) & 1) == ((s_true >> i) & 1) for i in range(n)]


def binary_entropy(p: float) -> float:
    """Compute binary Shannon entropy H_2(p) in bits.

    Args:
        p: Probability in [0, 1].

    Returns:
        Entropy in bits (0.0 <= H_2(p) <= 1.0).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successful trials.
        total: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound)
    """
    if total <= 0:
        return 0.0, 1.0

    from scipy.stats import norm

    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = successes / total

    denominator = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denominator
    margin = (
        z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)
    ) / denominator

    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return lower, upper
