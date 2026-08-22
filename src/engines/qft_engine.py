"""QFT Engine for applying QFT and extracting Fourier information."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.circuits.qft_circuit import apply_qft_to_register
from src.engines.dcp_engine import DCPState


@dataclass
class QFTResult:
    """Dataclass holding post-QFT quantum state and extracted Fourier information."""

    statevector: Statevector
    circuit: QuantumCircuit
    fourier_distribution: dict[int, float]  # y -> P(y)
    phases: dict[int, complex]              # y -> relative phase factor
    N: int


def extract_fourier_info(
    statevector: Statevector,
    n: int,
    N: int,
    circuit: QuantumCircuit | None = None,
) -> QFTResult:
    """Extract the Fourier label distribution and secret-dependent phases
    from the post-QFT statevector.

    The statevector has 2^(n+1) amplitudes indexed as |flag, data>.
    For each y in 0..N-1:
        P(y) = |<0,y|psi>|^2 + |<1,y|psi>|^2
        phase(y) = <1,y|psi> / <0,y|psi>  (when both are non-zero)

    Args:
        statevector: Post-QFT Statevector.
        n: Number of data qubits (n = ceil(log2(N))).
        N: Modulus.
        circuit: Optional QuantumCircuit associated with the statevector.

    Returns:
        QFTResult object containing distributions and phases.
    """
    distribution: dict[int, float] = {}
    phases: dict[int, complex] = {}
    data = statevector.data

    dim_data = 1 << n
    for y in range(N):
        amp_0y = data[0 * dim_data + y]
        amp_1y = data[1 * dim_data + y]

        prob_y = float(abs(amp_0y) ** 2 + abs(amp_1y) ** 2)
        distribution[y] = prob_y

        if prob_y > 1e-12 and abs(amp_0y) > 1e-12:
            phase = amp_1y / amp_0y
            phases[y] = complex(phase)

    if circuit is None:
        circuit = QuantumCircuit(n + 1, name=f"QFT_state(N={N})")

    return QFTResult(
        statevector=statevector,
        circuit=circuit,
        fourier_distribution=distribution,
        phases=phases,
        N=N,
    )


def verify_phases(
    phases: dict[int, complex],
    s: int,
    N: int,
    tol: float = 1e-8,
) -> bool:
    """Verify that extracted Fourier phases match theoretical predictions.

    For DCP states, the relative phase for Fourier label y should equal:
        exp(2π i s y / N)

    Args:
        phases: Dictionary mapping Fourier label y to complex phase.
        s: Hidden secret.
        N: Modulus.
        tol: Tolerance for complex difference.

    Returns:
        True if all extracted phases match theory.

    Raises:
        AssertionError: If any phase deviates beyond tol.
    """
    for y, phase in phases.items():
        expected = np.exp(2j * np.pi * s * y / N)
        diff = abs(phase - expected)
        assert diff < tol, (
            f"Phase mismatch for Fourier label y={y}: "
            f"extracted={phase}, theoretical={expected}, error={diff}"
        )
    return True


class QFTEngine:
    """Engine for applying Quantum Fourier Transform and extracting phase information."""

    def __init__(self, backend: str = "statevector") -> None:
        self.backend = backend

    def transform(self, dcp_state: DCPState) -> QFTResult:
        """Apply QFT to the data register of a DCP state and extract information.

        Args:
            dcp_state: Prepared DCPState.

        Returns:
            QFTResult with transformed statevector, distribution, and phases.
        """
        N = dcp_state.N
        n = dcp_state.n_qubits - 1
        data_qubits = list(range(n))

        # Build full circuit: DCP state preparation + QFT on data register
        full_circuit = dcp_state.circuit.copy()
        apply_qft_to_register(full_circuit, data_qubits, inverse=False)

        # Simulate statevector
        post_qft_statevector = Statevector.from_instruction(full_circuit)

        # Extract Fourier information
        return extract_fourier_info(
            statevector=post_qft_statevector,
            n=n,
            N=N,
            circuit=full_circuit,
        )

    def verify_phases(
        self,
        phases: dict[int, complex],
        s: int,
        N: int,
        tol: float = 1e-8,
    ) -> bool:
        """Verify extracted phases match theoretical expectation exp(2π i s y / N)."""
        return verify_phases(phases=phases, s=s, N=N, tol=tol)
