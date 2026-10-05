"""Bit-wise secret recovery by marginalizing posterior distributions."""

from __future__ import annotations

from typing import Mapping


def bitwise_recovery(posterior: Mapping[int, float], n: int) -> list[bool]:
    """Infer individual bit values s_i by marginalizing the posterior distribution.

    For each bit position i in 0 .. n-1:
        P(s_i = 1 | data) = sum_{s: (s >> i) & 1 == 1} P(s | data)

    The MAP estimate for bit i is True (1) if P(s_i = 1) > 0.5, else False (0).

    Args:
        posterior: Dict mapping candidate secret integer s to its posterior probability.
        n: Number of bits in secret representation (n = ceil(log2(N))).

    Returns:
        List of booleans [s_0, s_1, ..., s_{n-1}] where s_0 is the LSB.
    """
    bit_estimates: list[bool] = []
    for i in range(n):
        # Marginal probability that the i-th bit is 1
        p_bit_1 = sum(prob for s, prob in posterior.items() if (s >> i) & 1)
        bit_estimates.append(p_bit_1 > 0.5)

    return bit_estimates


def compute_bit_probabilities(posterior: Mapping[int, float], n: int) -> list[float]:
    """Compute marginal probability P(s_i = 1) for each bit position.

    Args:
        posterior: Dict mapping candidate secret integer s to its posterior probability.
        n: Number of bits.

    Returns:
        List of floats [P(s_0=1), ..., P(s_{n-1}=1)].
    """
    probs: list[float] = []
    for i in range(n):
        p_bit_1 = sum(prob for s, prob in posterior.items() if (s >> i) & 1)
        probs.append(float(p_bit_1))
    return probs
