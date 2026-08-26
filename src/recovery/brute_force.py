"""Brute-force secret recovery strategy."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from src.info.truncation import truncate_label


def compute_likelihood(
    y_k: int,
    s_candidate: int,
    k: int | None,
    n: int,
    N: int,
    b: int | None = None,
    mode: str = "msb",
) -> float:
    """Compute the likelihood P(y_k, [b] | s_candidate).

    Likelihood model:
        P(y_k, b | s) = sum_{y_full in trunc_preimage(y_k)} P(y_full, b | s)
        where P(y_full, b | s) = (1 / (2N)) * (1 + (-1)^b * cos(2π * s * y_full / N))

    If b is None (marginalized over flag outcomes):
        P(y_k | s) = sum_{y_full in trunc_preimage(y_k)} (1 / N) * cos^2(π * s * y_full / N)
    """
    total = 0.0
    for y_full in range(N):
        y_trunc = truncate_label(y_full, n, k, mode)
        if y_trunc == y_k:
            phase = 2.0 * np.pi * s_candidate * y_full / N
            if b is not None:
                prob = (1.0 / (2.0 * N)) * (1.0 + ((-1) ** b) * np.cos(phase))
            else:
                prob = (1.0 / N) * (1.0 + np.cos(phase)) / 2.0
            total += prob
    return max(total, 1e-15)


def brute_force_recovery(
    observations: Sequence[int | tuple[int, int]],
    k: int | None,
    n: int,
    N: int,
    mode: str = "msb",
    rng: np.random.Generator | None = None,
) -> tuple[int, dict[int, float], float]:
    """Exhaustively evaluate all secrets s in Z_N and return the MAP estimate.

    Args:
        observations: Sequence of truncated labels y_k, or (y_k, b) tuples.
        k: Number of retained bits.
        n: Total bits (n = ceil(log2(N))).
        N: Modulus.
        mode: Truncation mode ("msb" or "lsb").
        rng: Optional random generator for breaking ties.

    Returns:
        (s_hat, posterior_dict, confidence)
    """
    if rng is None:
        rng = np.random.default_rng()

    log_posterior = {s: 0.0 for s in range(N)}

    for obs in observations:
        if isinstance(obs, tuple):
            y_k, b = obs
        else:
            y_k, b = obs, None

        for s in range(N):
            lik = compute_likelihood(y_k, s, k, n, N, b=b, mode=mode)
            log_posterior[s] += np.log(lik)

    # Log-sum-exp normalization
    max_log = max(log_posterior.values())
    exp_vals = {s: np.exp(lp - max_log) for s, lp in log_posterior.items()}
    total = sum(exp_vals.values())
    posterior = {s: v / total for s, v in exp_vals.items()}

    max_prob = max(posterior.values())
    candidates = [s for s, p in posterior.items() if np.isclose(p, max_prob, atol=1e-9)]
    # Random tie-breaking: when s and N-s are exactly tied (always true for single DCP sample),
    # pick uniformly at random. mirror_correct in the orchestrator tracks whether we found
    # s OR its indistinguishable mirror N-s.
    s_hat = int(rng.choice(candidates))
    confidence = posterior[s_hat]

    return s_hat, posterior, confidence
