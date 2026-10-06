import numpy as np
import pytest

from src.verification import (
    analyze_conditional_independence,
    build_verification_report,
    compute_distinguishability,
    compute_kl_divergence,
    compute_mutual_information_from_conditionals,
    compute_set_ab_statistics,
    evaluate_lemma_2,
    evaluate_lemma_3,
    evaluate_lemma_4,
)


def test_compute_set_ab_statistics_tracks_groups() -> None:
    stats = compute_set_ab_statistics([0, 2, 4, 1, 3, 5])
    assert stats["set_a_count"] == 3
    assert stats["set_b_count"] == 3
    assert stats["set_a_sizes"] == [3]
    assert stats["set_b_sizes"] == [3]


def test_lemma_2_gap_is_zero_for_matching_branches() -> None:
    gap = evaluate_lemma_2([0.4, 0.6], [0.4, 0.6])
    assert gap == 0.0


def test_lemma_3_reports_max_amplitude_bound() -> None:
    bound = evaluate_lemma_3([0.2 + 0.1j, 0.8 + 0.0j, 0.1 + 0.2j])
    assert np.isclose(bound, 0.8)


def test_lemma_4_relative_gap_is_small_for_close_branches() -> None:
    gap = evaluate_lemma_4([1.0, 2.0], [1.02, 2.04])
    assert gap < 0.05


def test_distinguishability_detects_secret_difference() -> None:
    d = compute_distinguishability([0.8, 0.2], [0.2, 0.8])
    assert np.isclose(d, 0.6)


def test_verification_report_marks_uncomputed_diagnostics_unavailable() -> None:
    report = build_verification_report(qft_distribution=[0.5, 0.5])

    assert report.stage_metrics["qft_distribution"]["sum_prob"] == 1.0
    assert report.set_a_count is None
    assert report.set_b_count is None
    assert report.lemma2_gap is None
    assert report.lemma3_max_amplitude is None
    assert report.lemma4_relative_gap is None
    assert report.distinguishability is None


def test_verification_report_computes_tv_from_explicit_distributions() -> None:
    report = build_verification_report(
        prob_a=[0.8, 0.2],
        prob_b=[0.2, 0.8],
    )

    assert np.isclose(report.distinguishability, 0.6)


def test_kl_divergence_is_measured_in_bits() -> None:
    assert np.isclose(compute_kl_divergence([1.0, 0.0], [0.5, 0.5]), 1.0)


def test_kl_divergence_is_infinite_for_missing_support() -> None:
    assert compute_kl_divergence([1.0, 0.0], [0.0, 1.0]) == float("inf")


def test_conditional_mutual_information_is_zero_for_equal_distributions() -> None:
    information = compute_mutual_information_from_conditionals(
        [0.8, 0.2],
        [0.8, 0.2],
    )

    assert np.isclose(information, 0.0)


def test_conditional_mutual_information_detects_perfect_dependence() -> None:
    information = compute_mutual_information_from_conditionals(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert np.isclose(information, 1.0)


def test_conditional_mutual_information_validates_prior() -> None:
    with pytest.raises(ValueError, match="probability_h1"):
        compute_mutual_information_from_conditionals([1.0], [1.0], probability_h1=1.1)


def test_independence_analysis_detects_identical_empirical_conditionals() -> None:
    result = analyze_conditional_independence(
        h_star_values=[0, 0, 1, 1],
        b_outcomes=[2, 5, 5, 2],
    )

    assert result.outcome_values == (2, 5)
    assert result.count_h0 == result.count_h1 == 2
    assert result.probabilities_b_given_h0 == (0.5, 0.5)
    assert result.probabilities_b_given_h1 == (0.5, 0.5)
    assert result.total_variation == 0.0
    assert result.kl_divergence_h0_to_h1 == 0.0
    assert np.isclose(result.mutual_information_bits, 0.0)


def test_independence_analysis_detects_perfect_empirical_dependence() -> None:
    result = analyze_conditional_independence(
        h_star_values=[0, 0, 1, 1],
        b_outcomes=[2, 2, 5, 5],
    )

    assert result.total_variation == 1.0
    assert result.kl_divergence_h0_to_h1 == float("inf")
    assert np.isclose(result.mutual_information_bits, 1.0)


def test_independence_analysis_accepts_full_b_measurement_vectors() -> None:
    result = analyze_conditional_independence(
        h_star_values=[0, 0, 1, 1],
        b_outcomes=[(1, 2), (1, 2), (3, 4), (3, 4)],
    )

    assert result.outcome_values == ((1, 2), (3, 4))
    assert result.total_variation == 1.0
    assert np.isclose(result.mutual_information_bits, 1.0)


def test_independence_analysis_requires_both_h_star_values() -> None:
    with pytest.raises(ValueError, match=r"both h\*=0 and h\*=1"):
        analyze_conditional_independence([0, 0], [1, 2])
