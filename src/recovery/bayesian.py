"""Bayesian secret recovery with sequential log-posterior updating."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from src.recovery.brute_force import compute_likelihood


def bayesian_recovery(
    observations: Sequence[int | tuple[int, int]],
    k: int | None,
    n: int,
    N: int,
    prior: dict[int, float] | None = None,
    mode: str = "msb",
    rng: np.random.Generator | None = None,
) -> tuple[int, dict[int, float], float]:
    """Bayesian secret recovery using sequential log-space updates.

    Args:
        observations: Sequence of observations (y_k or (y_k, b)).
        k: Retained bits.
        n: Total register bits.
        N: Modulus.
        prior: Optional prior distribution over Z_N. Default is uniform (1/N).
        mode: Truncation mode ("msb" or "lsb").
        rng: Optional random generator for tie-breaking.

    Returns:
        (s_hat, posterior_dict, confidence)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize log prior
    if prior is None:
        log_posterior = {s: -np.log(N) for s in range(N)}
    else:
        log_posterior = {s: np.log(max(prior.get(s, 1e-15), 1e-15)) for s in range(N)}

    for obs in observations:
        if isinstance(obs, tuple):
            y_k, b = obs
        else:
            y_k, b = obs, None

        for s in range(N):
            lik = compute_likelihood(y_k, s, k, n, N, b=b, mode=mode)
            log_posterior[s] += np.log(lik)

    max_log = max(log_posterior.values())
    exp_vals = {s: np.exp(lp - max_log) for s, lp in log_posterior.items()}
    total = sum(exp_vals.values())
    posterior = {s: v / total for s, v in exp_vals.items()}

    max_prob = max(posterior.values())
    candidates = [s for s, p in posterior.items() if np.isclose(p, max_prob, atol=1e-12)]
    s_hat = int(rng.choice(candidates))
    confidence = posterior[s_hat]

    return s_hat, posterior, confidence
