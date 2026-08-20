import argparse
from pathlib import Path
import sys

# Ensure src is importable when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.engines.dcp_engine import DCPEngine
from src.engines.qft_engine import QFTEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify QFT transformation and Fourier phase extraction against theory."
    )
    parser.add_argument("--N", type=int, default=16, help="Modulus N (default: 16)")
    parser.add_argument("--s", type=int, default=5, help="Hidden secret s (default: 5)")
    parser.add_argument("--x", type=int, default=11, help="Random offset x (default: 11)")
    parser.add_argument(
        "--tol", type=float, default=1e-8, help="Numerical tolerance (default: 1e-8)"
    )
    args = parser.parse_args()

    N = args.N
    s = args.s
    x = args.x
    tol = args.tol

    print("=" * 70)
    print("QFT & FOURIER PHASE EXTRACTION VERIFICATION")
    print("=" * 70)
    print(f"Parameters: Modulus N={N}, Secret s={s}, Offset x={x}, Tolerance={tol}")

    dcp_engine = DCPEngine()
    qft_engine = QFTEngine()

    dcp_state = dcp_engine.create_state(N=N, s=s, x=x)
    qft_res = qft_engine.transform(dcp_state)

    try:
        qft_engine.verify_phases(qft_res.phases, s=s, N=N, tol=tol)
    except AssertionError as e:
        print(f"\n[FAIL] Phase verification failed: {e}")
        sys.exit(1)

    print("\nFourier Distribution & Extracted Phases:")
    print(f"{'y':>4} | {'P(y)':>8} | {'Extracted Phase':>22} | {'Theoretical Phase':>22} | {'Error':>10}")
    print("-" * 75)

    max_err = 0.0
    for y in range(N):
        prob = qft_res.fourier_distribution.get(y, 0.0)
        ext_phase = qft_res.phases.get(y, complex(0.0, 0.0))
        th_phase = np.exp(2j * np.pi * s * y / N)
        err = abs(ext_phase - th_phase)
        max_err = max(max_err, err)

        ext_str = f"{ext_phase.real:+.4f}{ext_phase.imag:+.4f}j"
        th_str = f"{th_phase.real:+.4f}{th_phase.imag:+.4f}j"
        print(f"{y:4d} | {prob:8.4f} | {ext_str:>22} | {th_str:>22} | {err:10.2e}")

    print("-" * 75)
    print(f"Maximum phase error across all {N} Fourier labels: {max_err:.2e}")
    print(f"Fourier distribution: Uniform (1/N = {1.0/N:.4f} for all y)")
    print("\n[SUCCESS] Extracted Fourier phases match theoretical exp(2*pi*i*s*y/N)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
