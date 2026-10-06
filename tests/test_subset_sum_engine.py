import pytest

from src.engines.subset_sum_engine import GroupAnalysis, SubsetSumAnalysis, SubsetSumEngine


def test_compute_subset_sum_selects_requested_values() -> None:
    engine = SubsetSumEngine()

    assert engine.compute_subset_sum([3, 5, 9], [1, 0, 1]) == 12


def test_compute_subset_sum_allows_empty_registers() -> None:
    assert SubsetSumEngine().compute_subset_sum([], []) == 0


def test_analyze_subset_sum_tracks_inputs_sum_and_bit_partition() -> None:
    analysis = SubsetSumEngine().analyze_subset_sum(
        [3, 5, 9],
        [1, 0, 1],
        bits_to_keep=3,
    )

    assert analysis == SubsetSumAnalysis(
        y_values=(3, 5, 9),
        b_register=(1, 0, 1),
        z=12,
        lower_bits=4,
        remaining_high_bits=1,
    )


def test_create_groups_preserves_indices_and_computes_group_sums() -> None:
    groups = SubsetSumEngine().create_groups(
        [(3, 1), (5, 0), (9, 1)],
        group_size=2,
        n=4,
    )

    assert [(group.group_id, group.sample_indices, group.r_j) for group in groups] == [
        (0, (0, 1), 3),
        (1, (2,), 9),
    ]
    assert groups[0].y_values == (3, 5)
    assert groups[0].b_register == (1, 0)
    assert groups[0].measurement_result is None
    assert groups[0].s_j == 0
    assert groups[0].belongs_to_A is None
    assert groups[0].belongs_to_B is None


def test_create_groups_records_low_measurement_and_remaining_high_bits() -> None:
    groups = SubsetSumEngine().create_groups(
        [(3, 1), (5, 0), (9, 1)],
        group_size=2,
        n=4,
        bits_to_keep=2,
    )

    assert isinstance(groups[0], GroupAnalysis)
    assert (groups[0].r_j, groups[0].low_bits_result, groups[0].remaining_high_bits) == (3, 3, 0)
    assert (groups[1].r_j, groups[1].low_bits_result, groups[1].remaining_high_bits) == (9, 1, 2)
    assert (groups[0].s_j, groups[1].s_j) == (0, 1)


def test_s_j_uses_fixed_width_most_significant_bits():
    group = SubsetSumEngine().create_groups(
        [(250, 1), (180, 1)],
        group_size=2,
        n=8,
    )[0]

    assert group.r_j == 430
    assert group.s_j == 0b110


def test_partition_groups_records_assignment_rule_and_distributions() -> None:
    engine = SubsetSumEngine()
    groups = engine.create_groups(
        [(1, 1), (3, 1), (2, 1), (1, 1)],
        group_size=1,
        n=8,
        bits_to_keep=2,
    )

    partition = engine.partition_groups(
        groups,
        hadamard_outcomes={0: (0,), 1: (1,), 2: (0,), 3: (0,)},
        target_a_size=2,
        random_seed=17,
    )

    assert partition.count_a == 2
    assert partition.count_b == 2
    assert [group.group_id for group in partition.groups_a] == [0, 2]
    assert [group.group_id for group in partition.groups_b] == [1, 3]
    assert partition.target_a_size == 2
    assert partition.partition_rule.startswith("first target_a_size groups")
    assert partition.random_seed == 17
    assert partition.measurement_distribution_a == {(0,): 1.0}
    assert partition.measurement_distribution_b == {(0,): 0.5, (1,): 0.5}
    assert partition.s_distribution_a == {0: 1.0}
    assert partition.s_distribution_b == {0: 1.0}
    assert all(group.belongs_to_A != group.belongs_to_B for group in partition.groups)


def test_partition_groups_rejects_incomplete_outcomes_and_insufficient_zero_groups() -> None:
    engine = SubsetSumEngine()
    groups = engine.create_groups([(1, 1), (2, 1)], group_size=1, n=4)

    with pytest.raises(ValueError, match="every group exactly once"):
        engine.partition_groups(groups, {0: (0,)}, target_a_size=1)
    with pytest.raises(ValueError, match="reject and restart"):
        engine.partition_groups(
            groups,
            {0: (0,), 1: (1,)},
            target_a_size=2,
        )


def test_paper_a_target_is_derived_from_n_and_selects_first_qualifiers():
    engine = SubsetSumEngine()
    groups = engine.create_groups(
        [(1, 1), (2, 1), (3, 1), (4, 1)],
        group_size=1,
        n=8,
    )

    partition = engine.partition_groups(
        groups,
        hadamard_outcomes={0: (0,), 1: (0,), 2: (0,), 3: (1,)},
        n=8,
    )

    assert engine.target_a_size(8) == 3
    assert partition.target_a_size == 3
    assert [group.group_id for group in partition.groups_a] == [0, 1, 2]
    assert [group.group_id for group in partition.groups_b] == [3]


def test_create_groups_accepts_empty_samples() -> None:
    assert SubsetSumEngine().create_groups([], group_size=2, n=4) == []


@pytest.mark.parametrize("group_size", [0, -1])
def test_create_groups_rejects_non_positive_group_size(group_size) -> None:
    with pytest.raises(ValueError, match="group_size"):
        SubsetSumEngine().create_groups([(1, 1)], group_size, n=4)


def test_create_groups_rejects_malformed_samples() -> None:
    with pytest.raises(ValueError, match=r"\(y, b\) pair"):
        SubsetSumEngine().create_groups([(1,)], group_size=1, n=4)


@pytest.mark.parametrize(
    ("y_values", "b_register"),
    [
        ([1, 2], [1]),
        ([1, -2], [1, 0]),
        ([1, 2], [1, 2]),
    ],
)
def test_compute_subset_sum_rejects_invalid_inputs(y_values, b_register) -> None:
    with pytest.raises(ValueError):
        SubsetSumEngine().compute_subset_sum(y_values, b_register)


@pytest.mark.parametrize(
    ("z", "bits_to_keep", "expected"),
    [(0b110101, 3, 0b101), (0b110101, 6, 0b110101), (0b110101, 0, 0)],
)
def test_measure_low_bits(z, bits_to_keep, expected) -> None:
    assert SubsetSumEngine().measure_low_bits(z, bits_to_keep) == expected


@pytest.mark.parametrize(("z", "bits_to_keep"), [(-1, 1), (1, -1)])
def test_measure_low_bits_rejects_negative_inputs(z, bits_to_keep) -> None:
    with pytest.raises(ValueError):
        SubsetSumEngine().measure_low_bits(z, bits_to_keep)