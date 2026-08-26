"""Unit tests for Secret Recovery Engine and strategies."""

import numpy as np
import pytest

from src.engines.dcp_engine import DCPEngine
from src.engines.info_engine import InformationEngine
from src.engines.qft_engine import QFTEngine
from src.engines.recovery_engine import RecoveryEngine, RecoveryResult
from src.recovery.bayesian import bayesian_recovery
from src.recovery.bitwise import bitwise_recovery, compute_bit_probabilities
from src.recovery.brute_force import brute_force_recovery, compute_likelihood
from src.recovery.phase_matching import phase_matching_recovery
from src.utils.math_utils import bit_accuracy, bits_to_int, int_to_bits, wilson_score_interval


def test_bit_conversions():
    # 13 in binary: 1101 (LSB at index 0 -> [1, 0, 1, 1])
    bits = int_to_bits(13, 4)
    assert bits == [1, 0, 1, 1]
    assert bits_to_int(bits) == 13

    # Bit accuracy
    assert bit_accuracy(13, 13, 4) == [True, True, True, True]
    assert bit_accuracy(13, 12, 4) == [False, True, True, True]  # 13 (1101) vs 12 (1100)


def test_wilson_interval():
    low, high = wilson_score_interval(50, 100, confidence=0.95)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert np.isclose((low + high) / 2, 0.5, atol=0.05)


def test_bitwise_recovery_simple():
    # If posterior is peaked at s=3 (binary 011 in 3 bits)
    posterior = {0: 0.05, 1: 0.05, 2: 0.1, 3: 0.8}
    bit_est = bitwise_recovery(posterior, n=3)
    # 3 in binary is [1, 1, 0]
    assert bit_est == [True, True, False]
    probs = compute_bit_probabilities(posterior, n=3)
    assert probs[0] > 0.5
    assert probs[1] > 0.5
    assert probs[2] < 0.5


def test_phase_matching_recovery():
    """Phase matching on full QFT phases should recover the exact secret."""
    dcp = DCPEngine()
    qft = QFTEngine()
    rec = RecoveryEngine()

    for N in [4, 8, 16]:
        for s_true in [1, 3, N - 1]:
            state = dcp.create_state(N=N, s=s_true, x=0)
            qft_res = qft.transform(state)
            res = rec.recover(qft_res.phases, N=N, s_true=s_true, strategy="phase_match")
            assert res.correct
            assert res.s_hat == s_true
            assert res.confidence > 0.5


def test_recovery_truncated_vs_full_multisample():
    """Full information (k=n) should achieve significantly higher recovery than heavy truncation (k=1)."""
    dcp = DCPEngine()
    qft = QFTEngine()
    info = InformationEngine(rng=np.random.default_rng(42))
    rec = RecoveryEngine(rng=np.random.default_rng(42))

    N = 8
    s_true = 3
    m = 8
    shots = 50

    full_correct = 0
    trunc_correct = 0

    for _ in range(shots):
        obs_full = []
        obs_trunc = []
        for _ in range(m):
            x = int(np.random.randint(0, N))
            state = dcp.create_state(N=N, s=s_true, x=x)
            qft_res = qft.transform(state)

            info_f = info.process(qft_res, k=3)
            obs_full.append(info_f)

            info_t = info.process(qft_res, k=1)
            obs_trunc.append(info_t)

        res_full = rec.recover(obs_full, N=N, s_true=s_true, strategy="bayesian")
        if res_full.correct:
            full_correct += 1

        res_trunc = rec.recover(obs_trunc, N=N, s_true=s_true, strategy="bayesian")
        if res_trunc.correct:
            trunc_correct += 1

    assert full_correct > trunc_correct
    # Truncated recovery on N=8 should be low (~ random baseline)
    assert (trunc_correct / shots) <= 0.25
