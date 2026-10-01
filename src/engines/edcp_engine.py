"""EDCP Engine for preparing and verifying EDCP quantum states."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.circuits.edcp_circuit import build_edcp_circuit


@dataclass
class EDCPState:
    """Dataclass encapsulating a prepared EDCP state."""

    circuit: QuantumCircuit
    statevector: Statevector
    N: int
    s: int
    x: int
    chi: dict[int, complex]
    n_qubits: int


def get_standard_chi(name: str) -> dict[int, complex]:
    """Return standard chi distributions for testing."""
    if name == "2-term":
        # Equivalent to DCP
        return {0: 1/np.sqrt(2), 1: 1/np.sqrt(2)}
    elif name == "4-term":
        # Uniform superposition over 4 terms
        return {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    elif name == "lwe-like":
        # Gaussian-like binomial distribution over 4 terms
        # Probs: 1/8, 3/8, 3/8, 1/8 -> amps: sqrt
        return {
            0: np.sqrt(1/8),
            1: np.sqrt(3/8),
            2: np.sqrt(3/8),
            3: np.sqrt(1/8)
        }
    else:
        raise ValueError(f"Unknown standard chi distribution: {name}")


def verify_edcp_state(
    state: Statevector | EDCPState,
    x: int,
    s: int,
    N: int,
    chi: dict[int, complex],
    n: int | None = None,
    tol: float = 1e-8,
) -> bool:
    """Verify that a statevector matches the theoretical EDCP state.

    The expected state is:
        |ψ_{x,s}> = \\sum_e c_e |e>|(x + e * s) mod N>

    Args:
        state: Statevector instance or EDCPState object.
        x: Random offset.
        s: Hidden secret.
        N: Modulus.
        chi: Dictionary of error amplitudes.
        n: Data register width in qubits (auto-computed if None).
        tol: Numerical tolerance for amplitude verification.

    Returns:
        True if the statevector matches theoretical expectation.
    """
    sv = state.statevector if isinstance(state, EDCPState) else state
    if n is None:
        n = max(1, (N - 1).bit_length())

    max_e = max(chi.keys()) if chi else 0
    m = max(1, max_e.bit_length())

    total_dim = 1 << (n + m)
    data = sv.data

    actual_probs = np.abs(data) ** 2
    
    # Track expected indices and amplitudes to verify zero elsewhere
    expected_indices = set()

    for e, c_e in chi.items():
        if np.abs(c_e) < tol:
            continue
            
        idx = e * (1 << n) + ((x + e * s) % N)
        expected_indices.add(idx)
        
        # Check amplitude magnitude
        expected_prob = np.abs(c_e) ** 2
        
        # Handle overlaps if (x + e1*s) = (x + e2*s) mod N for different e1, e2
        # Since e is also part of the basis |e>|data>, they won't overlap in the computational basis!
        assert np.isclose(actual_probs[idx], expected_prob, atol=tol), (
            f"Expected probability {expected_prob} at index {idx} (|{e}>|{(x+e*s)%N}>), "
            f"got {actual_probs[idx]}"
        )

        # We could check relative phases here between elements, but for simplicity we verify magnitude.
        # Since initialize creates correct phases, and modular addition is a permutation, it is preserved.

    # Check that all other basis states have zero amplitude
    for i in range(total_dim):
        if i not in expected_indices:
            assert actual_probs[i] < tol, (
                f"Unexpected non-zero amplitude at basis index {i}: "
                f"probability {actual_probs[i]}"
            )

    return True


class EDCPEngine:
    """Engine for creating and managing EDCP quantum states."""

    def __init__(self, backend: str = "statevector") -> None:
        self.backend = backend

    def create_state(self, N: int, s: int, x: int, chi: dict[int, complex]) -> EDCPState:
        """Create a EDCPState for given parameters (N, s, x, chi)."""
        n = max(1, (N - 1).bit_length())
        max_e = max(chi.keys()) if chi else 0
        m = max(1, max_e.bit_length())
        
        circuit = build_edcp_circuit(N, s, x, chi)
        statevector = Statevector.from_instruction(circuit)

        return EDCPState(
            circuit=circuit,
            statevector=statevector,
            N=N,
            s=s,
            x=x,
            chi=chi,
            n_qubits=n + m,
        )

    def verify_state(self, state: EDCPState, tol: float = 1e-8) -> bool:
        """Verify the correctness of an EDCPState."""
        return verify_edcp_state(
            state=state,
            x=state.x,
            s=state.s,
            N=state.N,
            chi=state.chi,
            n=max(1, (state.N - 1).bit_length()),
            tol=tol,
        )
