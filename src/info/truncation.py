"""Fourier label truncation, noise injection, and distribution sampling."""

from __future__ import annotations

from typing import Sequence
import numpy as np


def truncate_label(
    y: int,
    n: int,
    k: int | None,
    mode: str = "msb",
) -> int:
    """Truncate an n-bit Fourier label to k bits.

    Args:
        y: Full Fourier label integer (0 <= y < 2^n).
        n: Total number of bits in the register.
        k: Number of bits to retain (None means full, k <= n).
        mode: Truncation mode ("msb" keeps top k bits, "lsb" keeps bottom k bits).

    Returns:
        Truncated Fourier label integer.

    Raises:
        ValueError: If an unknown truncation mode is provided.
    """
    if k is None or k >= n:
        return y
    if k <= 0:
        return 0

    if mode == "msb":
        # Keep top k bits: right shift by (n - k)
        return y >> (n - k)
    elif mode == "lsb":
        # Keep bottom k bits: mask with (2^k - 1)
        return y & ((1 << k) - 1)
    else:
        raise ValueError(f"Unknown truncation mode: '{mode}'. Expected 'msb' or 'lsb'.")


def inject_noise(
    y: int,
    n: int,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> tuple[int, list[int]]:
    """Flip each bit of an n-bit integer independently with probability epsilon.

    Args:
        y: Original integer label.
        n: Total bit width.
        epsilon: Bit-flip error probability (0.0 <= epsilon <= 1.0).
        rng: Optional NumPy random Generator.

    Returns:
        Tuple of (noisy_y, flipped_bit_indices).
    """
    if rng is None:
        rng = np.random.default_rng()

    if epsilon <= 0.0:
        return y, []

    flipped: list[int] = []
    y_noisy = y

    for i in range(n):
        if rng.random() < epsilon:
            y_noisy ^= (1 << i)
            flipped.append(i)

    return y_noisy, flipped


def sample_fourier_label(
    distribution: dict[int, float],
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a Fourier label y according to probability distribution P(y).

    Args:
        distribution: Dict mapping Fourier label integer y to probability P(y).
        rng: Optional NumPy random Generator.

    Returns:
        Sampled Fourier label integer y.
    """
    if rng is None:
        rng = np.random.default_rng()

    labels = list(distribution.keys())
    probs = np.array([distribution[y] for y in labels], dtype=np.float64)

    total_prob = probs.sum()
    if total_prob > 0:
        probs /= total_prob
    else:
        probs = np.ones(len(labels)) / len(labels)

    idx = rng.choice(len(labels), p=probs)
    return int(labels[idx])
