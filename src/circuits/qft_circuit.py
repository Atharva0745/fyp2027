"""Quantum Fourier Transform circuits."""

from __future__ import annotations

from typing import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate


def apply_qft_to_register(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    inverse: bool = False,
) -> None:
    """Apply the Quantum Fourier Transform to a register of qubits.

    Args:
        circuit: QuantumCircuit to append the QFT gate to.
        qubits: List or sequence of qubit indices to transform.
        inverse: If True, apply inverse QFT.
    """
    n = len(qubits)
    qft = QFTGate(num_qubits=n)
    if inverse:
        circuit.append(qft.inverse(), qubits)
    else:
        circuit.append(qft, qubits)


def build_qft_circuit(n_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """Build a standalone QFT circuit on n qubits.

    Args:
        n_qubits: Number of qubits.
        inverse: If True, returns inverse QFT circuit.

    Returns:
        QuantumCircuit implementing QFT (or inverse QFT).
    """
    name = f"IQFT({n_qubits})" if inverse else f"QFT({n_qubits})"
    qc = QuantumCircuit(n_qubits, name=name)
    apply_qft_to_register(qc, list(range(n_qubits)), inverse=inverse)
    return qc
