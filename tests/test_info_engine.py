"""Unit tests for Information Engine and Truncation."""

import numpy as np
import pytest
from src.engines.dcp_engine import DCPEngine
from src.engines.info_engine import InformationEngine
from src.engines.qft_engine import QFTEngine
from src.info.truncation import inject_noise, sample_fourier_label, truncate_label


def test_truncate_label_msb():
    # y = 182 = 0b10110110 (n=8)
    y = 182
    n = 8
    assert truncate_label(y, n=n, k=8, mode="msb") == 182   # 10110110
    assert truncate_label(y, n=n, k=6, mode="msb") == 45    # 00101101 (182 >> 2)
    assert truncate_label(y, n=n, k=4, mode="msb") == 11    # 00001011 (182 >> 4)
    assert truncate_label(y, n=n, k=2, mode="msb") == 2     # 00000010 (182 >> 6)
    assert truncate_label(y, n=n, k=0, mode="msb") == 0
    assert truncate_label(y, n=n, k=None, mode="msb") == 182


def test_truncate_label_lsb():
    # y = 182 = 0b10110110 (n=8)
    y = 182
    n = 8
    assert truncate_label(y, n=n, k=8, mode="lsb") == 182
    assert truncate_label(y, n=n, k=4, mode="lsb") == 6     # 0110 = 6
    assert truncate_label(y, n=n, k=2, mode="lsb") == 2     # 10 = 2


def test_truncate_invalid_mode():
    with pytest.raises(ValueError, match="Unknown truncation mode"):
        truncate_label(10, n=4, k=2, mode="invalid")


def test_inject_noise_rate():
    rng = np.random.default_rng(42)
    n = 10
    epsilon = 0.1
    shots = 5000
    total_flips = 0

    for _ in range(shots):
        _, flips = inject_noise(0, n=n, epsilon=epsilon, rng=rng)
        total_flips += len(flips)

    observed_rate = total_flips / (shots * n)
    assert np.isclose(observed_rate, epsilon, atol=0.01)


def test_info_engine_process():
    dcp = DCPEngine()
    qft = QFTEngine()
    info = InformationEngine(rng=np.random.default_rng(123))

    state = dcp.create_state(N=16, s=5, x=11)
    qft_res = qft.transform(state)

    res = info.process(qft_res, k=2, noise_level=0.0, truncation_mode="msb")
    assert 0 <= res.Y_full < 16
    assert res.n == 4
    assert res.k == 2
    assert res.Y_truncated == (res.Y_full >> 2)
    assert not res.noise_applied
    assert res.bit_flips == []
