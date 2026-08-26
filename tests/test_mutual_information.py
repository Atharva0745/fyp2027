"""Unit tests for Mutual Information computation and information-loss metrics."""

import numpy as np
import pytest

from src.info.mutual_information import (
    compute_mi_profile,
    compute_mutual_information,
    information_loss_ratio,
)


def test_mutual_information_n2():
    """For N=2, compute exact MI."""
    # Joint (Y, B) measurement
    mi_flag = compute_mutual_information(N=2, k=1, include_flag=True)
    assert 0.0 < mi_flag <= 1.0

    # Interference distribution without flag
    mi_no_flag = compute_mutual_information(N=2, k=1, include_flag=False)
    assert 0.0 < mi_no_flag <= 1.0


def test_mutual_information_monotonicity():
    """MI must be monotonically non-decreasing in retained bits k."""
    for N in [4, 8, 16, 32]:
        n = max(1, (N - 1).bit_length())
        mis = [compute_mutual_information(N=N, k=k, n=n, mode="msb") for k in range(1, n + 1)]
        # Check non-decreasing
        for i in range(len(mis) - 1):
            assert mis[i] <= mis[i + 1] + 1e-12, f"Monotonicity violation at N={N}, k={i+1}"


def test_information_loss_ratio_properties():
    """Verify bounds and behavior of information loss ratio."""
    # Full info -> ratio is 0.0
    assert information_loss_ratio(2.0, 2.0) == 0.0

    # Zero info -> ratio is 1.0
    assert information_loss_ratio(2.0, 0.0) == 1.0

    # Partial info
    ratio = information_loss_ratio(2.0, 1.5)
    assert np.isclose(ratio, 0.25)

    # Edge case: zero full MI
    assert information_loss_ratio(0.0, 0.0) == 1.0


def test_compute_mi_profile():
    """Verify compute_mi_profile returns valid profile dict for all k in 1..n."""
    N = 16
    prof = compute_mi_profile(N)
    assert len(prof) == 4  # n=4 for N=16
    assert list(prof.keys()) == [1, 2, 3, 4]

    for k, metrics in prof.items():
        assert "mi" in metrics
        assert "mi_full" in metrics
        assert "info_loss_ratio" in metrics
        assert 0.0 <= metrics["info_loss_ratio"] <= 1.0
        assert metrics["mi"] <= metrics["mi_full"] + 1e-12

    # At k=4, info loss ratio should be 0.0
    assert np.isclose(prof[4]["info_loss_ratio"], 0.0, atol=1e-10)
