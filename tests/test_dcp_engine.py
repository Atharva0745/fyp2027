"""Unit tests for DCP State Construction and DCP Engine."""

import pytest
from src.engines.dcp_engine import DCPEngine, verify_dcp_state


def test_dcp_engine_n4_exhaustive():
    """Verify statevector for all (s, x) combinations with N=4."""
    engine = DCPEngine()
    N = 4
    for s in range(N):
        for x in range(N):
            state = engine.create_state(N=N, s=s, x=x)
            assert state.N == N
            assert state.s == s
            assert state.x == x
            assert state.n_qubits == 3  # 2 data qubits + 1 flag qubit
            assert engine.verify_state(state)


def test_dcp_engine_n8_spot_checks():
    """Spot-check multiple (s, x) pairs with N=8."""
    engine = DCPEngine()
    N = 8
    test_cases = [
        (0, 0),
        (1, 0),
        (3, 5),
        (5, 7),
        (7, 1),
        (4, 4),
    ]
    for s, x in test_cases:
        state = engine.create_state(N=N, s=s, x=x)
        assert state.n_qubits == 4
        assert engine.verify_state(state)


@pytest.mark.parametrize("N,s,x", [
    (16, 5, 11),
    (16, 0, 7),
    (16, 15, 1),
    (32, 7, 19),
    (32, 13, 31),
    (64, 23, 42),
])
def test_dcp_engine_larger_moduli(N, s, x):
    """Verify DCP state for larger moduli (N=16, 32, 64)."""
    engine = DCPEngine()
    state = engine.create_state(N=N, s=s, x=x)
    assert engine.verify_state(state)
