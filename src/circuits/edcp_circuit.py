"""EDCP quantum circuit construction."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from src.circuits.modular_add import apply_modular_addition


def build_edcp_circuit(N: int, s: int, x: int, chi: dict[int, complex]) -> QuantumCircuit:
    """Construct the EDCP quantum circuit.

    The circuit prepares the EDCP state:
        |ψ_{x,s}> = \\sum_e c_e |e>|x + e * s mod N>

    Register layout:
        - Qubits 0 .. n-1: Data register (|x>)
        - Qubits n .. n+m-1: Flag register (|e>)

    Args:
        N: Modulus (integer >= 2).
        s: Hidden secret (0 <= s < N).
        x: Random offset (0 <= x < N).
        chi: Dictionary mapping error e to complex amplitude c_e.
             The values must have a squared norm sum of 1.0.

    Returns:
        QuantumCircuit of n + m qubits.
    """
    n = max(1, (N - 1).bit_length())
    max_e = max(chi.keys()) if chi else 0
    m = max(1, max_e.bit_length())

    qc = QuantumCircuit(n + m, name=f"EDCP(N={N},s={s},x={x})")
    data_qubits = list(range(n))
    flag_qubits = list(range(n, n + m))

    # Stage 1 — Initialise data register to |x>
    for i in range(n):
        if (x >> i) & 1:
            qc.x(data_qubits[i])

    # Stage 2 — Create superposition on flag register according to chi
    # We construct a statevector for the flag register and initialize it
    flag_dim = 1 << m
    flag_state = np.zeros(flag_dim, dtype=complex)
    for e, amp in chi.items():
        flag_state[e] = amp

    # Normalize just in case, though chi should be normalized
    norm = np.linalg.norm(flag_state)
    if norm > 0:
        flag_state /= norm

    qc.initialize(flag_state.tolist(), flag_qubits)

    # Stage 3 — Conditional modular additions for each bit of the flag register
    # If the j-th bit of the flag register is 1, we add (2^j * s) mod N to the data register
    for j, flag_q in enumerate(flag_qubits):
        val_to_add = (s * (1 << j)) % N
        if val_to_add != 0:
            apply_modular_addition(qc, data_qubits, val_to_add, N, control_qubit=flag_q)

    return qc
