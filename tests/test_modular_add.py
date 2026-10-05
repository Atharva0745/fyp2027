"""Unit tests for modular addition circuit."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.circuits.modular_add import apply_modular_addition, build_modular_adder_circuit


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_uncontrolled_modular_addition(N):
    n = max(1, (N - 1).bit_length())
    test_a_values = [0, 1, 2, N // 2, N - 1]
    test_x_values = [0, 1, N // 2, N - 1]

    for a in test_a_values:
        for x in test_x_values:
            qc = QuantumCircuit(n)
            for i in range(n):
                if (x >> i) & 1:
                    qc.x(i)

            apply_modular_addition(qc, list(range(n)), a, N)
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities()
            res = np.argmax(probs)
            expected = (x + a) % N
            assert res == expected, f"Failed for N={N}, a={a}, x={x}: got {res}, expected {expected}"
            assert np.isclose(probs[res], 1.0, atol=1e-10)


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_controlled_modular_addition(N):
    n = max(1, (N - 1).bit_length())
    control_qubit = n
    data_qubits = list(range(n))

    for a in [1, 3, N - 1]:
        for x in [0, 1, N - 2]:
            # Control = 0: should not add
            qc0 = QuantumCircuit(n + 1)
            for i in range(n):
                if (x >> i) & 1:
                    qc0.x(data_qubits[i])
            apply_modular_addition(qc0, data_qubits, a, N, control_qubit=control_qubit)
            sv0 = Statevector.from_instruction(qc0)
            res0 = np.argmax(sv0.probabilities())
            assert res0 == x, f"Control 0 failed for N={N}, a={a}, x={x}"

            # Control = 1: should add a
            qc1 = QuantumCircuit(n + 1)
            qc1.x(control_qubit)
            for i in range(n):
                if (x >> i) & 1:
                    qc1.x(data_qubits[i])
            apply_modular_addition(qc1, data_qubits, a, N, control_qubit=control_qubit)
            sv1 = Statevector.from_instruction(qc1)
            res1 = np.argmax(sv1.probabilities())
            expected1 = (1 << n) + ((x + a) % N)
            assert res1 == expected1, f"Control 1 failed for N={N}, a={a}, x={x}"


def test_build_modular_adder_circuit():
    qc_uncontrolled = build_modular_adder_circuit(n=3, a=2, N=8, controlled=False)
    assert qc_uncontrolled.num_qubits == 3

    qc_controlled = build_modular_adder_circuit(n=3, a=2, N=8, controlled=True)
    assert qc_controlled.num_qubits == 4
