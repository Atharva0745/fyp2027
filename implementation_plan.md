# Implementation Plan — Phase 2: Core Pipeline (Information Engine, Recovery Engine, and Truncation Experiments)

This plan covers all tasks in **Phase 2** as specified in [phase.md](file:///c:/Users/visha/OneDrive/문서/fyp2027/phase.md), [architecture.md](file:///c:/Users/visha/OneDrive/문서/fyp2027/architecture.md), and [implementation.md](file:///c:/Users/visha/OneDrive/문서/fyp2027/implementation.md). It creates the `requirements.txt` file, builds the information manipulation engine, secret recovery strategies, experiment orchestrator, serialization/persistence layer, and runs the core truncation sweep.

## User Review Required

> [!IMPORTANT]
> - Phase 2 addresses the central research question: how does Fourier information truncation ($k$ bits) affect hidden-secret recovery across moduli $N \in \{4, 8, 16, 32\}$?
> - Parquet and JSON serialization will be used to store trial-level experiment data and metadata in `results/raw/`.

---

## Proposed Changes

### 1. Requirements File

#### [NEW] [requirements.txt](file:///c:/Users/visha/OneDrive/문서/fyp2027/requirements.txt)
- Pin core dependencies: `qiskit`, `qiskit-aer`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`, `pyarrow`, `pytest`, `tqdm`.

---

### 2. Information Engine (Task 2.1)

#### [NEW] [src/info/truncation.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/info/truncation.py)
- `truncate_label(y: int, n: int, k: int | None, mode: str = "msb") -> int`:
  - `mode="msb"`: retain $k$ most significant bits ($y \gg (n - k)$).
  - `mode="lsb"`: retain $k$ least significant bits ($y \ \& \ ((1 \ll k) - 1)$).
- `inject_noise(y: int, n: int, epsilon: float, rng: np.random.Generator) -> tuple[int, list[int]]`:
  - Flips bit $i$ independently with probability $\epsilon$.
- `sample_fourier_label(distribution: dict[int, float], rng: np.random.Generator) -> int`:
  - Samples $y$ from the marginal Fourier distribution $P(y)$.

#### [NEW] [src/engines/info_engine.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/engines/info_engine.py)
- `InformationResult` dataclass (`Y_full`, `Y_truncated`, `k`, `n`, `noise_applied`, `bit_flips`, `truncation_mode`).
- `InformationEngine` class:
  - `process(qft_result: QFTResult, k: int | None, noise_level: float = 0.0, truncation_mode: str = "msb", rng: np.random.Generator | None = None) -> InformationResult`.

#### [NEW] [tests/test_info_engine.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/tests/test_info_engine.py)
- Tests for MSB/LSB truncation, noise injection, and Fourier sampling distributions.

---

### 3. Secret Recovery Engine (Task 2.2)

#### [NEW] [src/recovery/brute_force.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/recovery/brute_force.py)
- `brute_force_recovery(observations: list[InformationResult | tuple[int, int]], k: int | None, n: int, N: int, mode: str = "msb") -> tuple[int, dict[int, float], float]`:
  - Likelihood model: evaluates likelihood $P(y_k \mid s) = \sum_{y_{\text{full}} \to y_k} P(y_{\text{full}} \mid s)$ over all candidate $s \in \mathbb{Z}_N$.

#### [NEW] [src/recovery/maximum_likelihood.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/recovery/maximum_likelihood.py)
- `maximum_likelihood_recovery`: Maximum-likelihood estimation across candidate secrets.

#### [NEW] [src/recovery/bayesian.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/recovery/bayesian.py)
- `bayesian_recovery`: Sequential log-space posterior updating with log-sum-exp normalization for $m \ge 1$ samples.

#### [NEW] [src/recovery/bitwise.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/recovery/bitwise.py)
- `bitwise_recovery(posterior: dict[int, float], n: int) -> list[bool]`:
  - Marginalizes posterior over individual bit positions $s_i$.

#### [NEW] [src/engines/recovery_engine.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/engines/recovery_engine.py)
- `RecoveryResult` dataclass (`s_hat`, `s_true`, `correct`, `posterior`, `bit_correct`, `confidence`, `strategy`, `num_samples_used`).
- `RecoveryEngine` class supporting `"brute_force"`, `"ml"`, `"bayesian"`, `"bitwise"`.

#### [NEW] [tests/test_recovery_engine.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/tests/test_recovery_engine.py)
- Verification tests for full info ($k=n$) vs truncated ($k=1$), multiple samples, and bit correctness.

---

### 4. Utilities & Orchestration (Task 2.3)

#### [NEW] [src/utils/math_utils.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/utils/math_utils.py)
- Bit conversion, binary entropy, and number theoretic helpers.

#### [NEW] [src/utils/serialization.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/utils/serialization.py)
- `save_experiment_result(result_df, metadata, output_dir, file_prefix)`: Parquet/JSON writer.
- `load_experiment_result(parquet_path)`: Parquet reader.

#### [NEW] [src/orchestrator.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/src/orchestrator.py)
- `Orchestrator` class:
  - `run(config: ExperimentConfig) -> ExperimentResult`: End-to-end execution of $M$ shots for a given config.
  - `run_sweep(base_config: ExperimentConfig, param_grid: dict) -> pd.DataFrame`.

#### [NEW] [tests/test_orchestrator.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/tests/test_orchestrator.py)
- Integration test running the full pipeline end-to-end and checking result structure and degradation with truncation.

---

### 5. Truncation Experiment Execution (Task 2.4)

#### [NEW] [configs/dcp_truncation_sweep.yaml](file:///c:/Users/visha/OneDrive/문서/fyp2027/configs/dcp_truncation_sweep.yaml)
- Sweep config over $N \in \{4, 8, 16, 32\}$ and $k \in [1, \dots, n]$.

#### [NEW] [scripts/run_single.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/scripts/run_single.py)
- CLI script to execute a single experiment configuration from YAML or CLI flags.

#### [NEW] [scripts/run_sweep.py](file:///c:/Users/visha/OneDrive/문서/fyp2027/scripts/run_sweep.py)
- CLI script to run the truncation sweep and save parquet results in `results/raw/dcp_truncation_core/`.

---

### 6. Documentation Updates
- Update [context.md](file:///c:/Users/visha/OneDrive/문서/fyp2027/context.md) with Phase 2 implementation details and experimental findings.
- Update [walkthrough.md](file:///C:/Users/visha/.gemini/antigravity-ide/brain/606c882b-1622-4d26-b1c0-a1fd09e4708e/walkthrough.md).

---

## Verification Plan

### Automated Tests
- Run all test suites:
  ```bash
  python -m pytest tests/ -v
  ```
- All Phase 1 and Phase 2 tests (config, modular arithmetic, dcp, qft, info engine, recovery engine, orchestrator) must pass with 0 errors.

### Experimental Sweep Verification
- Execute `python scripts/run_sweep.py --config configs/dcp_truncation_sweep.yaml` and verify:
  1. $P_{\text{success}} \to 1$ as $k \to n$.
  2. $P_{\text{success}} \to 1/N$ as $k \to 1$ (random baseline).
  3. Results parquet file exists in `results/raw/dcp_truncation_core/` and can be loaded with Pandas.
