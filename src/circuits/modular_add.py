"""Modular addition quantum circuits."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate


def apply_modular_addition(
    circuit: QuantumCircuit,
    data_qubits: Sequence[int],
    a: int,
    N: int,
    control_qubit: int | None = None,
) -> None:
    """In-place modular addition: |x> -> |(x + a) mod N>.

    Uses QFT-based phase arithmetic (Draper adder).
    When control_qubit is provided, the addition is conditionally applied
    only when the control qubit is in state |1>.

    Args:
        circuit: QuantumCircuit to append gates to.
        data_qubits: Sequence of qubit indices holding the data integer x.
        a: Integer constant to add (0 <= a < N).
        N: Modulus (typically a power of 2, N = 2^n).
        control_qubit: Optional control qubit index.
    """
    n = len(data_qubits)
    a_eff = a % N
    if a_eff == 0:
        # Adding 0 (or multiple of N) is a no-op
        return

    # Step 1: Transform data register into Fourier basis
    circuit.append(QFTGate(num_qubits=n), data_qubits)

    # Step 2: Apply phase rotations for adding a_eff
    for j in range(n):
        angle = 2.0 * np.pi * a_eff / (2 ** (n - j))
        target_q = data_qubits[j]
        if control_qubit is not None:
            circuit.cp(angle, control_qubit, target_q)
        else:
            circuit.p(angle, target_q)

    # Step 3: Transform back to computational basis
    circuit.append(QFTGate(num_qubits=n).inverse(), data_qubits)


def build_modular_adder_circuit(
    n: int,
    a: int,
    N: int,
    controlled: bool = False,
) -> QuantumCircuit:
    """Build a standalone modular adder circuit.

    Args:
        n: Number of data qubits (n = ceil(log2(N))).
        a: Integer constant to add.
        N: Modulus.
        controlled: Whether to prepend a control qubit.

    Returns:
        QuantumCircuit implementing the modular adder.
    """
    if controlled:
        qc = QuantumCircuit(n + 1, name=f"CADD({a} mod {N})")
        control_qubit = 0
        data_qubits = list(range(1, n + 1))
        apply_modular_addition(qc, data_qubits, a, N, control_qubit=control_qubit)
    else:
        qc = QuantumCircuit(n, name=f"ADD({a} mod {N})")
        data_qubits = list(range(n))
        apply_modular_addition(qc, data_qubits, a, N, control_qubit=None)
    return qc
