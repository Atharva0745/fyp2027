import argparse
from pathlib import Path
import sys

# Ensure src is importable when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.engines.dcp_engine import DCPEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify DCP statevector correctness against theoretical model."
    )
    parser.add_argument("--N", type=int, default=16, help="Modulus N (default: 16)")
    parser.add_argument("--s", type=int, default=5, help="Hidden secret s (default: 5)")
    parser.add_argument("--x", type=int, default=11, help="Random offset x (default: 11)")
    args = parser.parse_args()

    N = args.N
    s = args.s
    x = args.x

    print("=" * 60)
    print("DCP STATE VERIFICATION")
    print("=" * 60)
    print(f"Parameters: Modulus N={N}, Secret s={s}, Offset x={x}")

    engine = DCPEngine()
    try:
        state = engine.create_state(N=N, s=s, x=x)
        engine.verify_state(state)
    except Exception as e:
        print(f"\n[FAIL] Verification failed with error: {e}")
        sys.exit(1)

    n = state.n_qubits - 1
    idx_0 = 0 * (1 << n) + (x % N)
    idx_1 = 1 * (1 << n) + ((x + s) % N)

    print("\nState details:")
    print(f"  Total qubits: {state.n_qubits} (1 flag + {n} data qubits)")
    print(f"  Circuit depth: {state.circuit.depth()}")
    print(f"  Non-zero amplitudes:")
    print(f"    |0>|{x}>          (basis index {idx_0:2d}): {state.statevector.data[idx_0]:.5f} (prob = {abs(state.statevector.data[idx_0])**2:.4f})")
    print(f"    |1>|{(x+s)%N}>    (basis index {idx_1:2d}): {state.statevector.data[idx_1]:.5f} (prob = {abs(state.statevector.data[idx_1])**2:.4f})")
    print("\n[SUCCESS] DCP statevector matches theoretical expectation |psi> = (|0>|x> + |1>|x+s>) / sqrt(2)!")
    print("=" * 60)


if __name__ == "__main__":
    main()
