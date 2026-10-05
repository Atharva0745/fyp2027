"""Simon attack bookkeeping for explicitly supplied group measurements.

These helpers implement the arithmetic and classical data handling described
in the project roadmap. They do not synthesize the paper's joint quantum state
or produce Hadamard measurement outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from numbers import Integral
from typing import Mapping, Sequence

from src.engines.subset_sum_engine import GroupPartition
from src.utils.math_utils import wilson_score_interval


@dataclass(frozen=True)
class BMeasurementResult:
    """Recorded measured s_j values for groups assigned to B."""

    group_ids: tuple[int, ...]
    measured_s_values: tuple[int, ...]
    distribution: dict[int, float]


@dataclass(frozen=True)
class SimonSecretBitEstimate:
    """Majority estimate from supplied final Hadamard measurement outcomes."""

    d_n_hat: int
    empirical_confidence: float
    confidence_interval: tuple[float, float]
    observations: int
    correct: bool | None


@dataclass(frozen=True)
class SimonSecretEstimate:
    """Full estimate assembled from per-recursion Hadamard outcomes."""

    d_hat: int
    correct: bool | None
    confidence: float
    iterations: int
    bit_estimates: tuple[SimonSecretBitEstimate, ...]


class SimonAttackEngine:
    """Run paper-defined classical bookkeeping on supplied measurements."""

    @staticmethod
    def record_b_s_measurements(
        partition: GroupPartition,
        measured_s_values: Mapping[int, int],
        n: int,
    ) -> BMeasurementResult:
        """Validate and record the measured s_j values for every B group."""
        if not isinstance(n, Integral) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        expected_group_ids = tuple(group.group_id for group in partition.groups_b)
        if set(measured_s_values) != set(expected_group_ids):
            raise ValueError("measured_s_values must contain every B group exactly once")

        value_limit = 1 << ceil(log2(int(n)))
        values: list[int] = []
        for group_id in expected_group_ids:
            value = measured_s_values[group_id]
            if not isinstance(value, Integral) or not 0 <= value < value_limit:
                raise ValueError(f"measured s_j values must be in [0, {value_limit})")
            values.append(int(value))

        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        total = len(values)
        distribution = (
            {value: count / total for value, count in sorted(counts.items())}
            if total
            else {}
        )
        return BMeasurementResult(
            group_ids=expected_group_ids,
            measured_s_values=tuple(values),
            distribution=distribution,
        )

    @staticmethod
    def construct_s_star(s_values: Sequence[int], n: int) -> int:
        """Compute ``sum(s_j) mod n`` as specified in the supplied rule."""
        if not isinstance(n, Integral) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if any(not isinstance(value, Integral) or value < 0 for value in s_values):
            raise ValueError("s_values must contain non-negative integers")
        return sum(int(value) for value in s_values) % int(n)

    @staticmethod
    def erase_s_a(s_star: int, other_s_values: Sequence[int], n: int) -> int:
        """Recover the erased group value from ``s*`` and the other values."""
        if not isinstance(n, Integral) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if not isinstance(s_star, Integral) or not 0 <= s_star < n:
            raise ValueError("s_star must be in [0, n)")
        if any(not isinstance(value, Integral) or value < 0 for value in other_s_values):
            raise ValueError("other_s_values must contain non-negative integers")
        return (int(s_star) - sum(int(value) for value in other_s_values)) % int(n)

    @staticmethod
    def derive_h_star(s_star: int, n: int, l_s_star: int) -> int | None:
        """Return the MSB of s* only when the supplied condition bit is zero."""
        if not isinstance(n, Integral) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if not isinstance(s_star, Integral) or not 0 <= s_star < n:
            raise ValueError("s_star must be in [0, n)")
        if l_s_star not in (0, 1):
            raise ValueError("l_s_star must be 0 or 1")
        if l_s_star != 0:
            return None
        msb_index = ceil(log2(int(n))) - 1
        return (int(s_star) >> msb_index) & 1

    @staticmethod
    def transfer_h_bit(h: int, h_star: int) -> int:
        """Compute the classical bookkeeping relation ``h' = h XOR h*``."""
        if h not in (0, 1) or h_star not in (0, 1):
            raise ValueError("h and h_star must each be 0 or 1")
        return int(h) ^ int(h_star)

    @staticmethod
    def recover_secret_msb(
        hadamard_measurements: Sequence[int],
        true_bit: int | None = None,
    ) -> SimonSecretBitEstimate:
        """Estimate d_n by majority vote over supplied H(h*) measurement bits.

        Measurement outcomes must come from the complete quantum-state
        simulation or experiment; this method only aggregates them.
        """
        if not hadamard_measurements:
            raise ValueError("at least one Hadamard measurement is required")
        if any(bit not in (0, 1) for bit in hadamard_measurements):
            raise ValueError("Hadamard measurements must contain only 0 or 1")
        if true_bit is not None and true_bit not in (0, 1):
            raise ValueError("true_bit must be 0, 1, or None")

        ones = sum(int(bit) for bit in hadamard_measurements)
        zeros = len(hadamard_measurements) - ones
        estimate = int(ones > zeros)
        supporting_count = ones if estimate else zeros
        lower, upper = wilson_score_interval(supporting_count, len(hadamard_measurements))
        return SimonSecretBitEstimate(
            d_n_hat=estimate,
            empirical_confidence=supporting_count / len(hadamard_measurements),
            confidence_interval=(lower, upper),
            observations=len(hadamard_measurements),
            correct=(estimate == true_bit) if true_bit is not None else None,
        )

    @staticmethod
    def recover_secret(
        hadamard_measurements_by_bit: Mapping[int, Sequence[int]],
        N: int,
        true_secret: int | None = None,
    ) -> SimonSecretEstimate:
        """Assemble per-recursion measurement streams into a secret estimate.

        This consumes already generated outcomes for each bit-recovery round;
        it does not generate Simon's quantum circuit, erased samples, or the
        paper's recursive measurement state.
        """
        if not isinstance(N, Integral) or N < 2:
            raise ValueError("N must be an integer of at least 2")
        n = max(1, (int(N) - 1).bit_length())
        if set(hadamard_measurements_by_bit) != set(range(n)):
            raise ValueError("measurement streams must be supplied for every secret bit")
        if true_secret is not None and not 0 <= true_secret < N:
            raise ValueError("true_secret must be in [0, N)")

        bit_estimates = tuple(
            SimonAttackEngine.recover_secret_msb(
                hadamard_measurements_by_bit[bit_index],
                true_bit=((true_secret >> bit_index) & 1) if true_secret is not None else None,
            )
            for bit_index in range(n)
        )
        bit_string_value = sum(
            estimate.d_n_hat << bit_index
            for bit_index, estimate in enumerate(bit_estimates)
        )
        d_hat = bit_string_value % int(N)

        # Bonferroni-adjust each Wilson interval to bound all bit estimates jointly.
        interval_confidence = 1.0 - 0.05 / n
        simultaneous_lower_bounds: list[float] = []
        for estimate in bit_estimates:
            supporting_count = round(estimate.empirical_confidence * estimate.observations)
            lower, _ = wilson_score_interval(
                supporting_count,
                estimate.observations,
                confidence=interval_confidence,
            )
            simultaneous_lower_bounds.append(lower)
        confidence = max(0.0, 1.0 - sum(1.0 - lower for lower in simultaneous_lower_bounds))

        return SimonSecretEstimate(
            d_hat=d_hat,
            correct=(d_hat == true_secret) if true_secret is not None else None,
            confidence=confidence,
            iterations=sum(estimate.observations for estimate in bit_estimates),
            bit_estimates=bit_estimates,
        )
