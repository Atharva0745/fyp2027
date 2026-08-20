"""Maximum-likelihood secret recovery."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from src.recovery.brute_force import brute_force_recovery


def maximum_likelihood_recovery(
    observations: Sequence[int | tuple[int, int]],
    k: int | None,
    n: int,
    N: int,
    mode: str = "msb",
    rng: np.random.Generator | None = None,
) -> tuple[int, dict[int, float], float]:
    """Maximum-likelihood estimation across candidate secrets."""
    return brute_force_recovery(
        observations=observations,
        k=k,
        n=n,
        N=N,
        mode=mode,
        rng=rng,
    )
