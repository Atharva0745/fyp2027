"""Secret Recovery Engine coordinating various recovery strategies."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence, TypedDict
import numpy as np

from src.engines.info_engine import InformationResult
from src.recovery.bayesian import bayesian_recovery
from src.recovery.bitwise import bitwise_recovery, compute_bit_probabilities
from src.recovery.brute_force import brute_force_recovery
from src.recovery.maximum_likelihood import maximum_likelihood_recovery
from src.recovery.phase_matching import phase_matching_recovery
from src.utils.math_utils import bit_accuracy, bits_to_int


class SecretBitRecoveryResult(TypedDict):
    bit_index: int
    bit_hat: int
    probability_one: float
    confidence: float
    correct: bool | None
    samples_used: int


class DCPSecretRecoveryResult(TypedDict):
    d_hat: int
    correct: bool | None
    mirror_correct: bool | None
    confidence: float
    iterations: int
    posterior: dict[int, float]


def _validated_dcp_samples(
    samples: Sequence[tuple[int, int]],
    N: int,
) -> list[tuple[int, int]]:
    if N < 2:
        raise ValueError("N must be at least 2")
    if not samples:
        raise ValueError("at least one (y, b) sample is required")

    validated: list[tuple[int, int]] = []
    for sample in samples:
        if len(sample) != 2:
            raise ValueError("each sample must be a (y, b) pair")
        y_value, flag_bit = sample
        if not isinstance(y_value, Integral) or not 0 <= y_value < N:
            raise ValueError(f"Fourier label y must be in [0, {N})")
        if not isinstance(flag_bit, Integral) or flag_bit not in (0, 1):
            raise ValueError("sample bit b must be 0 or 1")
        validated.append((int(y_value), int(flag_bit)))
    return validated


def recover_secret_bit(
    samples: Sequence[tuple[int, int]],
    N: int,
    bit_index: int,
    true_secret: int | None = None,
    rng: np.random.Generator | None = None,
) -> SecretBitRecoveryResult:
    """Infer one secret bit by marginalizing the existing DCP posterior.

    This is Bayesian DCP inference and is not Simon's paper-specific
    recursive bit-recovery procedure.
    """
    validated_samples = _validated_dcp_samples(samples, N)
    n = max(1, (N - 1).bit_length())
    if not isinstance(bit_index, Integral) or not 0 <= bit_index < n:
        raise ValueError(f"bit_index must be in [0, {n})")
    if true_secret is not None and not 0 <= true_secret < N:
        raise ValueError(f"true_secret must be in [0, {N})")

    _, posterior, _ = bayesian_recovery(
        observations=validated_samples,
        k=n,
        n=n,
        N=N,
        rng=rng,
    )
    probability_one = compute_bit_probabilities(posterior, n)[int(bit_index)]
    bit_hat = int(probability_one > 0.5)
    true_bit = ((true_secret >> int(bit_index)) & 1) if true_secret is not None else None
    return SecretBitRecoveryResult(
        bit_index=int(bit_index),
        bit_hat=bit_hat,
        probability_one=probability_one,
        confidence=max(probability_one, 1.0 - probability_one),
        correct=(bit_hat == true_bit) if true_bit is not None else None,
        samples_used=len(validated_samples),
    )


def recover_secret(
    samples: Sequence[tuple[int, int]],
    N: int,
    true_secret: int | None = None,
    rng: np.random.Generator | None = None,
) -> DCPSecretRecoveryResult:
    """Return the MAP DCP secret estimate and posterior diagnostics.

    ``iterations`` is the number of sequential sample updates. DCP's
    sign/mirror ambiguity is surfaced explicitly; this is not Simon-specific
    recursive recovery.
    """
    validated_samples = _validated_dcp_samples(samples, N)
    if true_secret is not None and not 0 <= true_secret < N:
        raise ValueError(f"true_secret must be in [0, {N})")
    n = max(1, (N - 1).bit_length())
    d_hat, posterior, confidence = bayesian_recovery(
        observations=validated_samples,
        k=n,
        n=n,
        N=N,
        rng=rng,
    )
    mirror_secret = (N - true_secret) % N if true_secret is not None else None
    return DCPSecretRecoveryResult(
        d_hat=d_hat,
        correct=(d_hat == true_secret) if true_secret is not None else None,
        mirror_correct=(
            d_hat == true_secret or d_hat == mirror_secret
            if true_secret is not None
            else None
        ),
        confidence=float(confidence),
        iterations=len(validated_samples),
        posterior=posterior,
    )


@dataclass
class RecoveryResult:
    """Dataclass holding secret recovery results and diagnostic information."""

    s_hat: int
    s_true: int
    correct: bool
    mirror_correct: bool
    posterior: dict[int, float]
    bit_correct: list[bool]
    confidence: float
    strategy: str
    num_samples_used: int


class RecoveryEngine:
    """Engine for inferring hidden secret s from Fourier observations."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def recover(
        self,
        observations: Sequence[InformationResult | int | tuple[int, int]] | Mapping[int, complex],
        N: int,
        s_true: int,
        k: int | None = None,
        n: int | None = None,
        strategy: str = "brute_force",
        prior: dict[int, float] | None = None,
        truncation_mode: str = "msb",
        rng: np.random.Generator | None = None,
    ) -> RecoveryResult:
        """Infer hidden secret s given observations.

        Args:
            observations: Sequence of InformationResult instances, integer truncated labels,
                          (label, flag) tuples, or dict of complex Fourier phases.
            N: Modulus.
            s_true: Ground truth secret.
            k: Number of retained bits (inferred from observations if InformationResult).
            n: Total register bits (inferred from N if None).
            strategy: Recovery strategy ("brute_force", "ml", "maximum_likelihood",
                      "bayesian", "bitwise", "phase_match").
            prior: Optional prior distribution for Bayesian strategy.
            truncation_mode: Truncation mode ("msb" or "lsb").
            rng: Optional random generator for tie-breaking.

        Returns:
            RecoveryResult dataclass.
        """
        active_rng = rng if rng is not None else self.rng
        if n is None:
            n = max(1, (N - 1).bit_length())

        strat_lower = strategy.lower().strip()

        if strat_lower in ("phase_match", "phase_matching"):
            if isinstance(observations, dict):
                phases_dict = observations
            else:
                phases_dict = {}
            s_hat, posterior, confidence = phase_matching_recovery(
                phases=phases_dict,
                N=N,
                rng=active_rng,
            )
            num_samples = len(phases_dict)
        else:
            # Extract raw observations and determine k if given InformationResult objects
            raw_obs: list[int | tuple[int, int]] = []
            eff_k = k
            eff_mode = truncation_mode

            if isinstance(observations, (list, tuple)):
                for obs in observations:
                    if isinstance(obs, InformationResult):
                        raw_obs.append((obs.Y_truncated, obs.b))
                        eff_k = obs.k if eff_k is None else eff_k
                        eff_mode = obs.truncation_mode
                    else:
                        raw_obs.append(obs)

            num_samples = len(raw_obs)

            if strat_lower in ("brute_force", "bf"):
                s_hat, posterior, confidence = brute_force_recovery(
                    observations=raw_obs,
                    k=eff_k,
                    n=n,
                    N=N,
                    mode=eff_mode,
                    rng=active_rng,
                )
            elif strat_lower in ("ml", "maximum_likelihood"):
                s_hat, posterior, confidence = maximum_likelihood_recovery(
                    observations=raw_obs,
                    k=eff_k,
                    n=n,
                    N=N,
                    mode=eff_mode,
                    rng=active_rng,
                )
            elif strat_lower in ("bayesian", "bayes"):
                s_hat, posterior, confidence = bayesian_recovery(
                    observations=raw_obs,
                    k=eff_k,
                    n=n,
                    N=N,
                    prior=prior,
                    mode=eff_mode,
                    rng=active_rng,
                )
            elif strat_lower in ("bitwise", "bit_wise"):
                # First compute full posterior using brute force
                _, posterior, _ = brute_force_recovery(
                    observations=raw_obs,
                    k=eff_k,
                    n=n,
                    N=N,
                    mode=eff_mode,
                    rng=active_rng,
                )
                bit_est = bitwise_recovery(posterior, n=n)
                s_hat = bits_to_int(bit_est) % N
                confidence = posterior.get(s_hat, 0.0)
            else:
                raise ValueError(
                    f"Unknown recovery strategy: '{strategy}'. Supported: "
                    "'brute_force', 'ml', 'bayesian', 'bitwise', 'phase_match'."
                )

        correct = (s_hat == s_true)
        # DCP sign-ambiguity: s and N-s produce identical measurement distributions.
        # A recovery is "mirror correct" if it finds s or the indistinguishable mirror N-s.
        mirror_s = (N - s_true) % N
        mirror_correct = bool(s_hat == s_true or s_hat == mirror_s)
        bit_corr = bit_accuracy(s_hat, s_true, n)

        return RecoveryResult(
            s_hat=s_hat,
            s_true=s_true,
            correct=correct,
            mirror_correct=mirror_correct,
            posterior=posterior,
            bit_correct=bit_corr,
            confidence=float(confidence),
            strategy=strategy,
            num_samples_used=num_samples,
        )
