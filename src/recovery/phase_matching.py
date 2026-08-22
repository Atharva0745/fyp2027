"""Phase-matching secret recovery strategy using Fourier phase factors."""

from __future__ import annotations

from typing import Mapping
import numpy as np


def phase_matching_recovery(
    phases: Mapping[int, complex],
    N: int,
    rng: np.random.Generator | None = None,
) -> tuple[int, dict[int, float], float]:
    """Recover secret s by matching extracted Fourier phases exp(2π i s y / N).

    Args:
        phases: Dict mapping Fourier label y to complex relative phase.
        N: Modulus.
        rng: Optional random generator for tie-breaking.

    Returns:
        (s_hat, posterior_dict, confidence)
    """
    if rng is None:
        rng = np.random.default_rng()

    scores = {s: 0.0 for s in range(N)}

    for y, phase in phases.items():
        if y == 0 or abs(phase) < 1e-10:
            continue
        # Extracted phase angle
        extracted_angle = np.angle(phase)

        for s in range(N):
            expected_angle = 2.0 * np.pi * s * y / N
            # Cosine distance between angles (in [-1, 1], 1 is exact match)
            diff = extracted_angle - expected_angle
            scores[s] += float(np.cos(diff))

    # Convert scores to normalized softmax probabilities
    max_score = max(scores.values()) if scores else 0.0
    exp_scores = {s: np.exp(score - max_score) for s, score in scores.items()}
    total = sum(exp_scores.values())
    posterior = {s: float(v / total) for s, v in exp_scores.items()}

    max_prob = max(posterior.values())
    candidates = [s for s, p in posterior.items() if np.isclose(p, max_prob, atol=1e-8)]
    s_hat = int(rng.choice(candidates))
    confidence = posterior[s_hat]

    return s_hat, posterior, confidence
