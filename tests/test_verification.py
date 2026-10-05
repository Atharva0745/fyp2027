import numpy as np

from src.verification import (
    compute_distinguishability,
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
