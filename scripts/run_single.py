"""CLI script to execute a single DCP/EDCP experiment run."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, load_config
from src.orchestrator import Orchestrator
from src.utils.serialization import save_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single DCP experiment.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--N", type=int, default=None, help="Modulus N (e.g. 16).")
    parser.add_argument("--s", type=int, default=None, help="Hidden secret s (0 <= s < N).")
    parser.add_argument("--k", type=int, default=None, help="Retained Fourier bits (None for full).")
    parser.add_argument("--m", type=int, default=1, help="Number of samples per trial.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Bit-flip noise probability.")
    parser.add_argument("--shots", type=int, default=1000, help="Number of simulation trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--strategy", type=str, default="brute_force", help="Recovery strategy.")
    parser.add_argument("--truncation-mode", type=str, default="msb", help="Truncation mode ('msb' or 'lsb').")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output.")
    parser.add_argument("--file-prefix", type=str, default="dcp_single_run", help="Prefix for saved file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        config = load_config(args.config)
        # Apply any explicit CLI overrides
        if args.N is not None:
            config.N = args.N
        if args.s is not None:
            config.s = args.s
        if args.k is not None:
            config.k = args.k
        if args.shots is not None:
            config.shots = args.shots
        if args.strategy:
            config.recovery_strategy = args.strategy
    else:
        if args.N is None or args.s is None:
            print("Error: When --config is omitted, both --N and --s must be provided.", file=sys.stderr)
            sys.exit(1)
        config = ExperimentConfig(
            N=args.N,
            s=args.s,
            k=args.k,
            m=args.m,
            epsilon=args.epsilon,
            shots=args.shots,
            seed=args.seed,
            recovery_strategy=args.strategy,
            truncation_mode=args.truncation_mode,
        )

    print(f"============================================================")
    print(f"Running Experiment: N={config.N}, s={config.s}, k={config.k}, m={config.m}, shots={config.shots}")
    print(f"Recovery Strategy: {config.recovery_strategy}, Truncation: {config.truncation_mode}, Noise: {config.epsilon}")
    print(f"============================================================")

    orchestrator = Orchestrator()
    result = orchestrator.run(config)
    stats = result.statistics

    assert stats is not None
    print(f"\n--- Results Summary ---")
    print(f"Recovery Probability: {stats.recovery_prob:.4f}  (95% CI: [{stats.recovery_prob_ci[0]:.4f}, {stats.recovery_prob_ci[1]:.4f}])")
    print(f"Circuit Depth:        {stats.circuit_depth} gates")
    print(f"Total Qubits:         {stats.num_qubits}")
    print(f"Simulation Time:      {stats.runtime_seconds:.2f} s")
    print(f"\nPer-Bit Probabilities & Advantages:")
    for bit_i, (p_bit, adv) in enumerate(zip(stats.bit_recovery_probs, stats.bit_advantages)):
        print(f"  Bit {bit_i}: P(correct) = {p_bit:.4f}, Advantage = {adv:+.4f}")

    if args.output_dir:
        meta = {
            "config": config.__dict__,
            "statistics": {
                "recovery_prob": stats.recovery_prob,
                "recovery_prob_ci": stats.recovery_prob_ci,
                "bit_recovery_probs": stats.bit_recovery_probs,
                "bit_advantages": stats.bit_advantages,
                "runtime_seconds": stats.runtime_seconds,
            },
            "timestamp": result.timestamp,
        }
        pq, jf = save_experiment_result(stats.raw_data, meta, args.output_dir, args.file_prefix)
        print(f"\nSaved results to:\n  {pq}\n  {jf}")


if __name__ == "__main__":
    main()
