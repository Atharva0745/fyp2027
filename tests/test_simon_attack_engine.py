import pytest

from src.engines.simon_attack_engine import SimonAttackEngine
from src.engines.subset_sum_engine import SubsetSumEngine
from src.verification import analyze_conditional_independence


def _test_partition():
    engine = SubsetSumEngine()
    groups = engine.create_groups(
        [(1, 1), (3, 1), (2, 1), (1, 1)],
        group_size=1,
        n=8,
    )
    return engine.partition_groups(
        groups,
        hadamard_outcomes={0: (0,), 1: (1,), 2: (0,), 3: (0,)},
        target_a_size=2,
    )


def test_records_measured_s_values_for_groups_in_b():
    result = SimonAttackEngine.record_b_s_measurements(
        _test_partition(),
        measured_s_values={1: 3, 3: 5},
        n=8,
    )

    assert result.group_ids == (1, 3)
    assert result.measured_s_values == (3, 5)
    assert result.distribution == {3: 0.5, 5: 0.5}


def test_records_b_measurements_rejects_missing_group():
    with pytest.raises(ValueError, match="every B group exactly once"):
        SimonAttackEngine.record_b_s_measurements(
            _test_partition(),
            measured_s_values={1: 3},
            n=8,
        )


def test_constructs_s_star_and_recovers_erased_group_value():
    s_values = [3, 4, 2]
    s_star = SimonAttackEngine.construct_s_star(s_values, n=8)
    erased = SimonAttackEngine.erase_s_a(s_star, [4, 2], n=8)

    assert s_star == 1
    assert erased == 3


def test_h_star_is_only_defined_when_condition_bit_is_zero():
    assert SimonAttackEngine.derive_h_star(s_star=4, n=8, l_s_star=0) == 1
    assert SimonAttackEngine.derive_h_star(s_star=4, n=8, l_s_star=1) is None


@pytest.mark.parametrize(
    ("h", "h_star", "expected"),
    [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_transfers_bit_with_xor(h, h_star, expected):
    assert SimonAttackEngine.transfer_h_bit(h, h_star) == expected


def test_recovers_secret_bit_by_majority_and_reports_uncertainty():
    estimate = SimonAttackEngine.recover_secret_msb(
        [1, 1, 1, 0, 1],
        true_bit=1,
    )

    assert estimate.d_n_hat == 1
    assert estimate.empirical_confidence == 0.8
    assert estimate.observations == 5
    assert estimate.correct is True
    assert estimate.confidence_interval[0] < 0.8 < estimate.confidence_interval[1]


def test_recovers_zero_bit_on_tie_and_requires_measurements():
    estimate = SimonAttackEngine.recover_secret_msb([1, 0])

    assert estimate.d_n_hat == 0
    assert estimate.correct is None
    with pytest.raises(ValueError, match="at least one"):
        SimonAttackEngine.recover_secret_msb([])


def test_recovers_full_secret_from_per_recursion_measurement_streams():
    result = SimonAttackEngine.recover_secret(
        hadamard_measurements_by_bit={
            0: [1, 1, 1, 0, 1],
            1: [0, 0, 0, 1, 0],
            2: [1, 1, 1, 1, 0],
        },
        N=8,
        true_secret=5,
    )

    assert result.d_hat == 5
    assert result.correct is True
    assert result.iterations == 15
    assert len(result.bit_estimates) == 3
    assert 0.0 <= result.confidence <= 1.0


def test_full_secret_recovery_requires_each_bit_stream():
    with pytest.raises(ValueError, match="every secret bit"):
        SimonAttackEngine.recover_secret({0: [1, 1]}, N=8)


def test_full_secret_recovery_returns_residue_for_non_power_of_two_modulus():
    result = SimonAttackEngine.recover_secret(
        hadamard_measurements_by_bit={
            0: [1, 1, 1],
            1: [1, 1, 1],
            2: [1, 1, 1],
        },
        N=6,
    )

    assert result.d_hat == 1


def test_supplied_measurements_flow_through_phases_5_to_7():
    subset_engine = SubsetSumEngine()
    groups = subset_engine.create_groups(
        [(32, 1), (96, 1), (160, 1), (224, 1)],
        group_size=1,
        n=8,
    )
    partition = subset_engine.partition_groups(
        groups,
        hadamard_outcomes={0: (0,), 1: (0,), 2: (0,), 3: (1,)},
        n=8,
    )
    b_measurements = SimonAttackEngine.record_b_s_measurements(
        partition,
        measured_s_values={group.group_id: group.s_j for group in partition.groups_b},
        n=8,
    )
    all_s_values = [group.s_j for group in partition.groups]
    s_star = SimonAttackEngine.construct_s_star(all_s_values, n=8)
    erased_a_value = SimonAttackEngine.erase_s_a(
        s_star,
        all_s_values[1:],
        n=8,
    )
    h_star = SimonAttackEngine.derive_h_star(s_star, n=8, l_s_star=0)
    h_prime = SimonAttackEngine.transfer_h_bit(h=1, h_star=h_star)
    bit_estimate = SimonAttackEngine.recover_secret_msb([1, 1, 1, 0])
    independence = analyze_conditional_independence(
        h_star_values=[0, 0, 1, 1],
        b_outcomes=[(7,), (7,), (7,), (6,)],
    )

    assert [group.group_id for group in partition.groups_a] == [0, 1, 2]
    assert [group.group_id for group in partition.groups_b] == [3]
    assert b_measurements.measured_s_values == (7,)
    assert all_s_values == [1, 3, 5, 7]
    assert s_star == 0
    assert erased_a_value == 1
    assert h_star == 0
    assert h_prime == 1
    assert bit_estimate.d_n_hat == 1
    assert independence.total_variation > 0.0