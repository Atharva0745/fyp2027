"""Information Engine for Fourier label sampling, truncation, and noise injection."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from src.engines.qft_engine import QFTResult
from src.info.truncation import inject_noise, sample_fourier_label, truncate_label


@dataclass
class InformationResult:
    """Dataclass holding Fourier observation information after truncation and noise."""

    Y_full: int
    Y_truncated: int
    k: int | None
    n: int
    noise_applied: bool
    bit_flips: list[int]
    truncation_mode: str


class InformationEngine:
    """Engine for systematically manipulating Fourier information."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def process(
        self,
        qft_result: QFTResult,
        k: int | None = None,
        noise_level: float = 0.0,
        truncation_mode: str = "msb",
        rng: np.random.Generator | None = None,
    ) -> InformationResult:
        """Sample a Fourier label from QFTResult, optionally inject noise, and truncate.

        Args:
            qft_result: Output from QFTEngine containing Fourier distribution.
            k: Number of bits to retain (None means full information).
            noise_level: Bit-flip error probability epsilon (0.0 to 1.0).
            truncation_mode: Truncation mode ("msb" or "lsb").
            rng: Optional NumPy random Generator (overrides instance default).

    Returns:
        InformationResult containing original, noisy, and truncated labels.
    """
        active_rng = rng if rng is not None else self.rng
        n = max(1, (qft_result.N - 1).bit_length())

        # Step 1: Sample Fourier label y from P(y)
        y_full = sample_fourier_label(qft_result.fourier_distribution, rng=active_rng)

        # Step 2: Inject noise if epsilon > 0
        if noise_level > 0.0:
            y_noisy, flipped = inject_noise(y_full, n, noise_level, rng=active_rng)
            noise_applied = len(flipped) > 0
        else:
            y_noisy = y_full
            flipped = []
            noise_applied = False

        # Step 3: Truncate Fourier label
        y_truncated = truncate_label(y_noisy, n, k, mode=truncation_mode)

        return InformationResult(
            Y_full=y_full,
            Y_truncated=y_truncated,
            k=k if k is not None else n,
            n=n,
            noise_applied=noise_applied,
            bit_flips=flipped,
            truncation_mode=truncation_mode,
        )
