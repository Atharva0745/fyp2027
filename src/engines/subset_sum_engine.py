"""Classical basis-value primitives for Simon-style subset sums."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, log2
from numbers import Integral
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SubsetSumAnalysis:
    """Classical basis-value details for one subset-sum calculation."""

    y_values: tuple[int, ...]
    b_register: tuple[int, ...]
    z: int
    lower_bits: int
    remaining_high_bits: int


@dataclass(frozen=True)
class GroupAnalysis:
    """Inspectable arithmetic and measurement record for one sample group.

    ``s_j`` stores the top ``ceil(log2(n))`` bits of the fixed-width r_j
    register. ``low_bits_result`` is the optional Phase 3 low-bit projection.
    ``measurement_result`` stores the per-bit Hadamard outcomes used for the
    Phase 5 A/B partition.
    """

    group_id: int
    sample_indices: tuple[int, ...]
    y_values: tuple[int, ...]
    b_register: tuple[int, ...]
    r_j: int
    measurement_result: tuple[int, ...] | None = None
    low_bits_result: int | None = None
    remaining_high_bits: int | None = None
    s_j: int | None = None
    belongs_to_A: bool | None = None
    belongs_to_B: bool | None = None


@dataclass(frozen=True)
class GroupPartition:
    """Auditable result of the paper's first-all-zero-groups partition."""

    groups: tuple[GroupAnalysis, ...]
    groups_a: tuple[GroupAnalysis, ...]
    groups_b: tuple[GroupAnalysis, ...]
    partition_rule: str
    random_seed: int | None
    target_a_size: int
    measurement_distribution_a: dict[tuple[int, ...], float]
    measurement_distribution_b: dict[tuple[int, ...], float]
    s_distribution_a: dict[int, float]
    s_distribution_b: dict[int, float]

    @property
    def count_a(self) -> int:
        return len(self.groups_a)

    @property
    def count_b(self) -> int:
        return len(self.groups_b)


class SubsetSumEngine:
    """Compute subset sums and extract low bits from basis values.

    These helpers do not simulate a superposition or quantum measurement.
    """

    def compute_subset_sum(
        self,
        y_values: Sequence[int],
        b_register: Sequence[int],
    ) -> int:
        """Return the integer sum ``sum(b_i * y_i)`` for binary ``b_i``."""
        if len(y_values) != len(b_register):
            raise ValueError("y_values and b_register must have the same length")

        total = 0
        for y_value, coefficient in zip(y_values, b_register):
            if not isinstance(y_value, Integral) or y_value < 0:
                raise ValueError("y_values must contain non-negative integers")
            if coefficient not in (0, 1):
                raise ValueError("b_register must contain only 0 or 1")
            total += int(y_value) * int(coefficient)

        return total

    @staticmethod
    def recommended_group_size(n: int, c: float) -> int:
        """Return ``ceil(c * log2(n))`` samples per group."""
        if not isinstance(n, Integral) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if c <= 0:
            raise ValueError("c must be positive")
        return max(1, ceil(c * log2(int(n))))

    @staticmethod
    def target_a_size(n: int) -> int:
        """Return ``ceil(n / log2(n))`` to realize the paper's A-size target."""
        if not isinstance(n, Integral) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        return ceil(int(n) / log2(int(n)))

    def measure_low_bits(self, z: int, bits_to_keep: int) -> int:
        """Return the low-bit outcome for a computational-basis value ``z``."""
        if not isinstance(z, Integral) or z < 0:
            raise ValueError("z must be a non-negative integer")
        if not isinstance(bits_to_keep, Integral) or bits_to_keep < 0:
            raise ValueError("bits_to_keep must be a non-negative integer")
        if bits_to_keep == 0:
            return 0
        return int(z) & ((1 << int(bits_to_keep)) - 1)

    def analyze_subset_sum(
        self,
        y_values: Sequence[int],
        b_register: Sequence[int],
        bits_to_keep: int,
    ) -> SubsetSumAnalysis:
        """Record inputs, subset sum, low-bit outcome, and remaining high bits."""
        z = self.compute_subset_sum(y_values, b_register)
        lower_bits = self.measure_low_bits(z, bits_to_keep)
        return SubsetSumAnalysis(
            y_values=tuple(int(value) for value in y_values),
            b_register=tuple(int(value) for value in b_register),
            z=z,
            lower_bits=lower_bits,
            remaining_high_bits=z >> bits_to_keep,
        )

    def create_groups(
        self,
        samples: Sequence[tuple[int, int]],
        group_size: int,
        n: int,
        bits_to_keep: int | None = None,
    ) -> list[GroupAnalysis]:
        """Group ordered ``(y, b)`` samples and compute ``r_j`` and ``s_j``.

        ``r_j`` is held in a fixed-width register of ``n + ceil(log2(group_size))``
        bits. ``s_j`` is its most-significant ``ceil(log2(n))`` bits. If
        ``bits_to_keep`` is supplied, the low-bit projection is also recorded;
        it is distinct from the Hadamard outcomes added during partitioning.
        """
        if not isinstance(group_size, Integral) or group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if not isinstance(n, Integral) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if bits_to_keep is not None and (
            not isinstance(bits_to_keep, Integral) or bits_to_keep < 0
        ):
            raise ValueError("bits_to_keep must be a non-negative integer")

        groups: list[GroupAnalysis] = []
        r_register_width = int(n) + ceil(log2(int(group_size)))
        s_width = ceil(log2(int(n)))
        s_shift = r_register_width - s_width
        for group_id, start in enumerate(range(0, len(samples), int(group_size))):
            end = min(start + int(group_size), len(samples))
            group_samples = samples[start:end]
            y_values: list[int] = []
            b_register: list[int] = []

            for sample in group_samples:
                if len(sample) != 2:
                    raise ValueError("each sample must be a (y, b) pair")
                y_value, coefficient = sample
                y_values.append(y_value)
                b_register.append(coefficient)

            analysis = (
                self.analyze_subset_sum(y_values, b_register, bits_to_keep)
                if bits_to_keep is not None
                else None
            )
            r_j = (
                analysis.z
                if analysis is not None
                else self.compute_subset_sum(y_values, b_register)
            )
            if any(y_value >= (1 << int(n)) for y_value in y_values):
                raise ValueError(f"Fourier labels must fit in the specified n={n} bits")
            groups.append(
                GroupAnalysis(
                    group_id=group_id,
                    sample_indices=tuple(range(start, end)),
                    y_values=tuple(y_values),
                    b_register=tuple(b_register),
                    r_j=r_j,
                    low_bits_result=analysis.lower_bits if analysis else None,
                    remaining_high_bits=(
                        analysis.remaining_high_bits if analysis else None
                    ),
                    s_j=(r_j >> s_shift) & ((1 << s_width) - 1),
                )
            )

        return groups

    def partition_groups(
        self,
        groups: Sequence[GroupAnalysis],
        hadamard_outcomes: Mapping[int, Sequence[int]],
        target_a_size: int | None = None,
        n: int | None = None,
        partition_rule: str = "first target_a_size groups with all-zero Hadamard outcomes enter A",
        random_seed: int | None = None,
    ) -> GroupPartition:
        """Select the first target number of all-zero Hadamard groups into A.

        Hadamard outcomes are explicit inputs because this classical helper
        cannot produce outcomes from the algorithm's joint quantum state.
        """
        if not partition_rule.strip():
            raise ValueError("partition_rule must describe the partition rule")
        if target_a_size is None:
            if n is None:
                raise ValueError("provide target_a_size or n to derive the paper target")
            target_a_size = self.target_a_size(n)
        if not isinstance(target_a_size, Integral) or target_a_size < 0:
            raise ValueError("target_a_size must be a non-negative integer")
        if random_seed is not None and (
            not isinstance(random_seed, Integral) or random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer or None")

        group_ids = [group.group_id for group in groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("group_id values must be unique")
        if set(hadamard_outcomes) != set(group_ids):
            raise ValueError("Hadamard outcomes must be supplied for every group exactly once")

        normalized_outcomes: dict[int, tuple[int, ...]] = {}
        zero_group_ids: list[int] = []
        for group in groups:
            outcome = tuple(hadamard_outcomes[group.group_id])
            if len(outcome) != len(group.b_register):
                raise ValueError(
                    f"Hadamard outcome for group {group.group_id} must contain "
                    f"{len(group.b_register)} bits"
                )
            if any(not isinstance(bit, Integral) or bit not in (0, 1) for bit in outcome):
                raise ValueError("Hadamard outcomes must contain only 0 or 1")
            normalized_outcomes[group.group_id] = tuple(int(bit) for bit in outcome)
            if all(bit == 0 for bit in outcome):
                zero_group_ids.append(group.group_id)

        if len(zero_group_ids) < target_a_size:
            raise ValueError(
                f"Only {len(zero_group_ids)} all-zero groups; "
                f"need {target_a_size} for A (reject and restart)"
            )
        selected_a = set(zero_group_ids[: int(target_a_size)])
        partitioned_groups = tuple(
            replace(
                group,
                measurement_result=normalized_outcomes[group.group_id],
                belongs_to_A=group.group_id in selected_a,
                belongs_to_B=group.group_id not in selected_a,
            )
            for group in groups
        )
        groups_a = tuple(group for group in partitioned_groups if group.belongs_to_A)
        groups_b = tuple(group for group in partitioned_groups if group.belongs_to_B)

        return GroupPartition(
            groups=partitioned_groups,
            groups_a=groups_a,
            groups_b=groups_b,
            partition_rule=partition_rule.strip(),
            random_seed=int(random_seed) if random_seed is not None else None,
            target_a_size=int(target_a_size),
            measurement_distribution_a=self._measurement_distribution(groups_a),
            measurement_distribution_b=self._measurement_distribution(groups_b),
            s_distribution_a=self._s_distribution(groups_a),
            s_distribution_b=self._s_distribution(groups_b),
        )

    @staticmethod
    def _measurement_distribution(
        groups: Sequence[GroupAnalysis],
    ) -> dict[tuple[int, ...], float]:
        outcomes = [
            group.measurement_result
            for group in groups
            if group.measurement_result is not None
        ]
        if not outcomes:
            return {}
        counts: dict[tuple[int, ...], int] = {}
        for outcome in outcomes:
            counts[outcome] = counts.get(outcome, 0) + 1
        total = len(outcomes)
        return {outcome: count / total for outcome, count in sorted(counts.items())}

    @staticmethod
    def _s_distribution(groups: Sequence[GroupAnalysis]) -> dict[int, float]:
        outcomes = [group.s_j for group in groups if group.s_j is not None]
        if not outcomes:
            return {}
        counts: dict[int, int] = {}
        for outcome in outcomes:
            counts[outcome] = counts.get(outcome, 0) + 1
        total = len(outcomes)
        return {outcome: count / total for outcome, count in sorted(counts.items())}