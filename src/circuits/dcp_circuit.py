"""DCP quantum circuit construction."""

from __future__ import annotations

from qiskit import QuantumCircuit
from src.circuits.modular_add import apply_modular_addition


def build_dcp_circuit(N: int, s: int, x: int) -> QuantumCircuit:
    """Construct the DCP quantum circuit.

    The circuit prepares the DCP state:
        |ψ_{x,s}> = (|0>|x> + |1>|x + s mod N>) / sqrt(2)

    Register layout:
        - Qubits 0 .. n-1: Data register (|x>)
        - Qubit n: Flag register (|0> or |1>)

    Args:
        N: Modulus (integer >= 2).
        s: Hidden secret (0 <= s < N).
        x: Random offset (0 <= x < N).

    Returns:
        QuantumCircuit of n + 1 qubits.
    """
    n = max(1, (N - 1).bit_length())
    qc = QuantumCircuit(n + 1, name=f"DCP(N={N},s={s},x={x})")
    data_qubits = list(range(n))
    flag_qubit = n

    # Stage 1 — Initialise data register to |x>
    for i in range(n):
        if (x >> i) & 1:
            qc.x(data_qubits[i])

    # Stage 2 — Create superposition on flag qubit
    qc.h(flag_qubit)

    # Stage 3 — Conditional modular addition of s mod N
    apply_modular_addition(qc, data_qubits, s, N, control_qubit=flag_qubit)

    return qc
