"""DCP Engine for preparing and verifying DCP quantum states."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.circuits.dcp_circuit import build_dcp_circuit


@dataclass
class DCPState:
    """Dataclass encapsulating a prepared DCP state."""

    circuit: QuantumCircuit
    statevector: Statevector
    N: int
    s: int
    x: int
    n_qubits: int


def verify_dcp_state(
    state: Statevector | DCPState,
    x: int,
    s: int,
    N: int,
    n: int | None = None,
    tol: float = 1e-8,
) -> bool:
    """Verify that a statevector matches the theoretical DCP state.

    The expected state is:
        |ψ_{x,s}> = (|0>|x> + |1>|(x + s) mod N>) / sqrt(2)

    Indexed as |flag, data> where flag is the most significant qubit
    and data is the n-qubit register.

    Args:
        state: Statevector instance or DCPState object.
        x: Random offset.
        s: Hidden secret.
        N: Modulus.
        n: Data register width in qubits (auto-computed if None).
        tol: Numerical tolerance for amplitude verification.

    Returns:
        True if the statevector matches theoretical expectation.

    Raises:
        AssertionError: If any verification condition fails.
    """
    sv = state.statevector if isinstance(state, DCPState) else state
    if n is None:
        n = max(1, (N - 1).bit_length())

    total_dim = 1 << (n + 1)
    data = sv.data

    idx_0 = 0 * (1 << n) + (x % N)
    idx_1 = 1 * (1 << n) + ((x + s) % N)

    actual_probs = np.abs(data) ** 2

    # Check that all other basis states have zero amplitude
    for i in range(total_dim):
        if i not in (idx_0, idx_1):
            assert actual_probs[i] < tol, (
                f"Unexpected non-zero amplitude at basis index {i}: "
                f"probability {actual_probs[i]}"
            )

    # Check amplitude magnitudes
    if idx_0 == idx_1:
        # Special edge case: s = 0 and flag is in superposition |+>
        assert np.isclose(actual_probs[idx_0], 1.0, atol=tol), (
            f"Expected probability 1.0 at index {idx_0}, got {actual_probs[idx_0]}"
        )
    else:
        assert np.isclose(actual_probs[idx_0], 0.5, atol=tol), (
            f"Expected probability 0.5 at index {idx_0} (|0>|{x}>), got {actual_probs[idx_0]}"
        )
        assert np.isclose(actual_probs[idx_1], 0.5, atol=tol), (
            f"Expected probability 0.5 at index {idx_1} (|1>|{(x+s)%N}>), got {actual_probs[idx_1]}"
        )

        # Check relative phase is real and positive (0 phase difference)
        phase_ratio = data[idx_1] / data[idx_0]
        assert np.isclose(phase_ratio, 1.0, atol=tol), (
            f"Expected relative phase 1.0 between |1,x+s> and |0,x>, got {phase_ratio}"
        )

    return True


class DCPEngine:
    """Engine for creating and managing DCP quantum states."""

    def __init__(self, backend: str = "statevector") -> None:
        self.backend = backend

    def create_state(self, N: int, s: int, x: int) -> DCPState:
        """Create a DCPState for given parameters (N, s, x)."""
        n = max(1, (N - 1).bit_length())
        circuit = build_dcp_circuit(N, s, x)
        statevector = Statevector.from_instruction(circuit)

        return DCPState(
            circuit=circuit,
            statevector=statevector,
            N=N,
            s=s,
            x=x,
            n_qubits=n + 1,
        )

    def verify_state(self, state: DCPState, tol: float = 1e-8) -> bool:
        """Verify the correctness of a DCPState."""
        return verify_dcp_state(
            state=state,
            x=state.x,
            s=state.s,
            N=state.N,
            n=state.n_qubits - 1,
            tol=tol,
        )
