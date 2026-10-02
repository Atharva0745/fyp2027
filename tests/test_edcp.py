"""Unit tests for EDCP State Construction and Verification."""

import pytest
import numpy as np

from src.engines.edcp_engine import EDCPEngine, get_standard_chi
from src.engines.mod_halving_engine import ModulusHalvingEngine


def test_edcp_dcp_equivalence():
    """Verify that a 2-term EDCP state is equivalent to a DCP state."""
    N = 16
    s = 5
    x = 3
    chi = get_standard_chi("2-term")
    
    engine = EDCPEngine()
    state = engine.create_state(N, s, x, chi)
    
    # 2-term chi means m=1, so total qubits should be n+1
    n = max(1, (N - 1).bit_length())
    assert state.n_qubits == n + 1
    
    # Verify correctness against theoretical EDCP state
    assert engine.verify_state(state)


def test_edcp_lwe_like():
    """Verify that an LWE-like 4-term EDCP state is generated correctly."""
    N = 32
    s = 19
    x = 10
    chi = get_standard_chi("lwe-like")
    
    engine = EDCPEngine()
    state = engine.create_state(N, s, x, chi)
    
    # 4-term chi means m=2, so total qubits should be n+2
    n = max(1, (N - 1).bit_length())
    assert state.n_qubits == n + 2
    
    assert engine.verify_state(state)


def test_modulus_halving_valid_iteration():
    """Test a single valid iteration of modulus halving."""
    engine = ModulusHalvingEngine(rng=np.random.default_rng(42))
    
    N = 64
    s = 42 # Even secret
    chi = get_standard_chi("2-term")
    
    res = engine.run_iteration(N, s, chi)
    
    assert res["N_prev"] == 64
    assert res["N_next"] == 32
    assert res["s_prev"] == 42
    assert res["s_next"] == 21
    assert res["s_lsb"] == 0
    assert "y_observed" in res


def test_modulus_halving_invalid_odd_N():
    """Test that modulus halving correctly raises an error for odd N."""
    engine = ModulusHalvingEngine()
    
    with pytest.raises(ValueError, match="must be even for halving"):
        engine.run_iteration(31, 15, get_standard_chi("2-term"))


def test_modulus_halving_pipeline():
    """Test the multi-iteration modulus halving pipeline."""
    engine = ModulusHalvingEngine(rng=np.random.default_rng(42))
    
    results = engine.run_pipeline(N_start=64, s_start=42, chi_name="4-term", iterations=3)
    
    assert len(results) == 3
    
    assert results[0]["N_prev"] == 64
    assert results[0]["N_next"] == 32
    assert results[0]["s_prev"] == 42
    assert results[0]["s_next"] == 21
    
    assert results[1]["N_prev"] == 32
    assert results[1]["N_next"] == 16
    assert results[1]["s_prev"] == 21
    assert results[1]["s_next"] == 10
    
    assert results[2]["N_prev"] == 16
    assert results[2]["N_next"] == 8
    assert results[2]["s_prev"] == 10
    assert results[2]["s_next"] == 5
