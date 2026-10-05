"""Unit tests for QFT Engine and Phase Extraction."""

import numpy as np
import pytest
from src.engines.dcp_engine import DCPEngine
from src.engines.qft_engine import QFTEngine, extract_fourier_info, verify_phases


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_qft_fourier_distribution_uniformity(N):
    """Test that Fourier distribution for DCP state is uniform (1/N for each y)."""
    dcp_engine = DCPEngine()
    qft_engine = QFTEngine()

    for s in [1, N // 2, N - 1]:
        for x in [0, 1, N - 1]:
            state = dcp_engine.create_state(N=N, s=s, x=x)
            qft_res = qft_engine.transform(state)

            assert qft_res.N == N
            assert len(qft_res.fourier_distribution) == N

            for y, prob in qft_res.fourier_distribution.items():
                assert np.isclose(prob, 1.0 / N, atol=1e-8), (
                    f"Expected probability 1/{N}={1.0/N}, got {prob} at y={y}"
                )


@pytest.mark.parametrize("N", [4, 8, 16, 32, 64])
def test_qft_phase_extraction_correctness(N):
    """Test that extracted phases match exp(2π i s y / N) within 1e-8 tolerance."""
    dcp_engine = DCPEngine()
    qft_engine = QFTEngine()

    test_s_values = [0, 1, 3, N // 2, N - 1]
    for s in test_s_values:
        x = (s * 3 + 1) % N
        state = dcp_engine.create_state(N=N, s=s, x=x)
        qft_res = qft_engine.transform(state)

        assert qft_engine.verify_phases(qft_res.phases, s=s, N=N, tol=1e-8)


def test_qft_phase_verification_failure():
    """Test that verify_phases raises AssertionError when given incorrect secret."""
    dcp_engine = DCPEngine()
    qft_engine = QFTEngine()

    state = dcp_engine.create_state(N=16, s=5, x=11)
    qft_res = qft_engine.transform(state)

    # Verifying with wrong secret s=4 should fail
    with pytest.raises(AssertionError, match="Phase mismatch"):
        qft_engine.verify_phases(qft_res.phases, s=4, N=16, tol=1e-8)
