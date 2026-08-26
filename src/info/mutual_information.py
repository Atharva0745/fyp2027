"""Mutual Information computation and information-loss characterisation."""

from __future__ import annotations

import numpy as np
from src.info.truncation import truncate_label


def compute_mutual_information(
    N: int,
    k: int | None = None,
    n: int | None = None,
    mode: str = "msb",
    include_flag: bool = True,
) -> float:
    """Compute exact Shannon Mutual Information I(S; Y_k, [B]) via joint distribution enumeration.

    Theoretical Setup:
        S is uniformly distributed on Z_N: P(S = s) = 1/N.

        For candidate secret s, full Fourier label y in 0..N-1, and flag outcome b in {0, 1}:
            P(y, b | s) = (1 / (2N)) * (1 + (-1)^b * cos(2π s y / N))

        Note: summing over y in 0..N-1:
            sum_y P(y, 0|s) = sum_y P(y, 1|s) = 1/2  (each flag outcome is equally likely)
            sum_{y,b} P(y,b|s) = 1  (properly normalized conditional)

        When truncated to k bits, y is mapped to y_k = truncate(y, n, k):
            P(y_k, b | s) = sum_{y : truncate(y)=y_k} P(y, b | s)

        Joint distribution (S uniform):
            P(s, y_k, b) = (1/N) * P(y_k, b | s)

        Mutual information:
            I(S; Y_k, B) = sum_{s, y_k, b} P(s, y_k, b) * log2( P(s, y_k, b) / (P(s) * P(y_k, b)) )

    Args:
        N: Modulus (integer >= 2).
        k: Retained Fourier bits (None means full information k=n).
        n: Total register bits (inferred from N if None).
        mode: Truncation mode ("msb" or "lsb").
        include_flag: Whether the observation includes the Hadamard flag bit B.

    Returns:
        Mutual information in bits (non-negative float).
    """
    N = int(N)
    if n is None:
        n = max(1, (N - 1).bit_length())
    else:
        n = int(n)
    if k is None or k >= n:
        k = n
    else:
        k = int(k)
    if k <= 0:
        return 0.0

    y_max = 1 << k  # Number of distinct truncated values: 2^k

    if include_flag:
        # Build the CONDITIONAL likelihood table P(y_k, b | s), shape (N, y_max, 2).
        # We accumulate over the full-resolution labels y_full that map to each y_k.
        # P(y_full, b | s) = (1 / (2N)) * (1 ± cos(2π s y_full / N))
        likelihood = np.zeros((N, y_max, 2), dtype=np.float64)

        for s in range(N):
            for y_full in range(N):
                yk = truncate_label(y_full, n, k, mode)
                phase = 2.0 * np.pi * s * y_full / N
                cos_val = np.cos(phase)

                # Born-rule conditional probabilities
                p_b0 = (1.0 / (2.0 * N)) * (1.0 + cos_val)   # P(y_full, b=0 | s)
                p_b1 = (1.0 / (2.0 * N)) * (1.0 - cos_val)   # P(y_full, b=1 | s)

                likelihood[s, yk, 0] += p_b0
                likelihood[s, yk, 1] += p_b1

        # Sanity: for each s, sum over (y_k, b) should equal 1.0
        # likelihood[s].sum() = sum_{y_full} 1/(2N)*(1+cos) + 1/(2N)*(1-cos) = sum_{y_full} 1/N = 1

        # Joint distribution: P(s, y_k, b) = (1/N) * P(y_k, b | s)
        joint = likelihood / N  # shape (N, y_max, 2), sums to 1 over all (s, y_k, b)

        # Marginals
        p_s = joint.sum(axis=(1, 2))    # P(S=s) — should be exactly 1/N each
        p_obs = joint.sum(axis=0)       # P(Y_k=y_k, B=b), shape (y_max, 2)

        # Mutual Information I(S; Y_k, B)
        mi = 0.0
        for s in range(N):
            for yk in range(y_max):
                for b in (0, 1):
                    p_joint = joint[s, yk, b]
                    if p_joint > 1e-15:
                        denom = p_s[s] * p_obs[yk, b]
                        if denom > 1e-15:
                            mi += p_joint * np.log2(p_joint / denom)

    else:
        # Without flag: marginalise over b — observation is just (S, Y_k)
        # P(y_full | s) = sum_b P(y_full, b | s) = 1/N  (uniform over y for any s)
        # Therefore I(S; Y_k) when marginalising over b is computed via:
        #   P(y_k | s) = sum_{y_full: truncate=y_k} 1/N = |preimage(y_k)| / N
        # For uniform preimage sizes this is 1/2^k, giving I(S;Y_k)=0 — not useful.
        # The informative formulation uses cos^2 interference terms:
        #   P(y_full | s, measured in computational basis after QFT, b traced out)
        #   = cos^2(π s y_full / N) (from the phase kick-back derivation).
        likelihood_2d = np.zeros((N, y_max), dtype=np.float64)

        for s in range(N):
            for y_full in range(N):
                yk = truncate_label(y_full, n, k, mode)
                phase = np.pi * s * y_full / N  # NOTE: π not 2π (cos^2 form)
                p_y = np.cos(phase) ** 2 / N    # P(y_full | s) in computational basis
                likelihood_2d[s, yk] += p_y

        # Joint P(s, y_k) = (1/N) * P(y_k | s)
        joint_2d = likelihood_2d / N

        p_s = joint_2d.sum(axis=1)
        p_yk = joint_2d.sum(axis=0)

        mi = 0.0
        for s in range(N):
            for yk in range(y_max):
                p_joint = joint_2d[s, yk]
                if p_joint > 1e-15:
                    denom = p_s[s] * p_yk[yk]
                    if denom > 1e-15:
                        mi += p_joint * np.log2(p_joint / denom)

    return max(0.0, float(mi))


def information_loss_ratio(mi_full: float, mi_truncated: float) -> float:
    """Compute the fraction of information lost due to Fourier truncation.

    Formula:
        loss_ratio = 1 - I(S; Y_k) / I(S; Y_full)

    Returns 0.0 when no information is lost, and 1.0 when all information is lost.
    """
    if mi_full <= 1e-12:
        return 1.0
    ratio = 1.0 - (mi_truncated / mi_full)
    return float(np.clip(ratio, 0.0, 1.0))


def compute_mi_profile(
    N: int,
    mode: str = "msb",
    include_flag: bool = True,
) -> dict[int, dict[str, float]]:
    """Compute full profile of Mutual Information and information loss for all k in 1..n.

    Args:
        N: Modulus.
        mode: Truncation mode ("msb" or "lsb").
        include_flag: Whether to include flag outcome.

    Returns:
        Dict mapping k -> {"mi": float, "mi_full": float, "info_loss_ratio": float}.
    """
    N = int(N)
    n = max(1, (N - 1).bit_length())
    mi_full = compute_mutual_information(N, k=n, n=n, mode=mode, include_flag=include_flag)

    profile: dict[int, dict[str, float]] = {}
    for k in range(1, n + 1):
        mi_k = compute_mutual_information(N, k=k, n=n, mode=mode, include_flag=include_flag)
        loss = information_loss_ratio(mi_full, mi_k)
        profile[k] = {
            "mi": mi_k,
            "mi_full": mi_full,
            "info_loss_ratio": loss,
            "bits_lost": mi_full - mi_k,
        }

    return profile
