"""Information Engine for Fourier label sampling, truncation, and noise injection."""

from __future__ import annotations

from dataclasses import dataclass
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
    b: int = 0  # Flag qubit measurement outcome in Hadamard/X basis


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
        """Sample a Fourier label and flag outcome, optionally inject noise, and truncate.

        Args:
            qft_result: Output from QFTEngine containing Fourier statevector/distribution.
            k: Number of bits to retain (None means full information).
            noise_level: Bit-flip error probability epsilon (0.0 to 1.0).
            truncation_mode: Truncation mode ("msb" or "lsb").
            rng: Optional NumPy random Generator (overrides instance default).

        Returns:
            InformationResult containing original, noisy, and truncated labels, and flag outcome.
        """
        active_rng = rng if rng is not None else self.rng
        N = qft_result.N
        n = max(1, (N - 1).bit_length())

        # Step 1: Sample joint measurement (y, b) in Hadamard basis on the flag qubit
        if qft_result.statevector is not None:
            data = qft_result.statevector.data
            dim_data = 1 << n
            joint_probs = np.zeros(2 * N, dtype=np.float64)

            for y in range(N):
                amp_0y = data[0 * dim_data + y]
                amp_1y = data[1 * dim_data + y]
                # Hadamard basis on flag: |+> = (0 + 1)/sqrt(2), |-> = (0 - 1)/sqrt(2)
                amp_plus = (amp_0y + amp_1y) / np.sqrt(2.0)
                amp_minus = (amp_0y - amp_1y) / np.sqrt(2.0)

                joint_probs[0 * N + y] = float(abs(amp_plus) ** 2)
                joint_probs[1 * N + y] = float(abs(amp_minus) ** 2)

            total_prob = joint_probs.sum()
            if total_prob > 0:
                joint_probs /= total_prob
            else:
                joint_probs = np.ones(2 * N) / (2 * N)

            idx = active_rng.choice(2 * N, p=joint_probs)
            b_sampled = int(idx // N)
            y_full = int(idx % N)
        else:
            y_full = sample_fourier_label(qft_result.fourier_distribution, rng=active_rng)
            b_sampled = 0

        # Step 2: Inject noise if epsilon > 0
        flipped: list[int] = []
        if noise_level > 0.0:
            y_noisy, flipped = inject_noise(y_full, n, noise_level, rng=active_rng)
            b_noisy = 1 - b_sampled if active_rng.random() < noise_level else b_sampled
            noise_applied = len(flipped) > 0 or (b_noisy != b_sampled)
        else:
            y_noisy = y_full
            b_noisy = b_sampled
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
            b=b_noisy,
        )
