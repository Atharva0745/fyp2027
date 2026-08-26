# Architecture — DCP/EDCP Quantum Information Analysis Framework

> **Version:** 1.0  
> **Date:** 2026-08-17  
> **Status:** Design Specification

---

## 1. Architectural Overview

The framework is a **modular, pipeline-based quantum simulation system** built on Qiskit. It prepares DCP and EDCP quantum states, applies the Quantum Fourier Transform, systematically manipulates the resulting Fourier information, and measures the impact on hidden-secret recovery. Every component communicates through well-defined Python data classes, and the entire pipeline is driven by a declarative experiment configuration layer.

### 1.1 High-Level Architecture

```text
+====================================================================+
|                    EXPERIMENT CONFIGURATION                        |
|   (parameters: N, s, m, k, epsilon, shots, problem_type, ...)     |
+====================================================================+
                               |
                               v
+====================================================================+
|                      ORCHESTRATOR                                  |
|  - Parses experiment config                                         |
|  - Dispatches to the correct pipeline (DCP or EDCP)                |
|  - Collects results from all downstream components                  |
|  - Persists results to disk                                         |
+====================================================================+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
+===========================+   +==============================+
|      DCP PIPELINE         |   |      EDCP PIPELINE            |
|                           |   |                              |
|  DCP Engine               |   |  EDCP Engine                 |
|    v                      |   |    v                         |
|  QFT Engine               |   |  QFT Engine                  |
|    v                      |   |    v                         |
|  Information Engine       |   |  Modulus-Halving Engine      |
|    v                      |   |    v                         |
|  Secret Recovery Engine   |   |  Information Engine          |
|    v                      |   |    v                         |
|  Statistics Engine        |   |  Secret Recovery Engine      |
|                           |   |    v                         |
|                           |   |  Statistics Engine           |
+===========================+   +==============================+
              |                                 |
              +----------------+----------------+
                               |
                               v
+====================================================================+
|                     RESULTS STORE                                  |
|  - Per-experiment results (Parquet / JSON)                         |
|  - Aggregated datasets                                             |
|  - Plots and figures (PDF / PNG)                                   |
+====================================================================+
                               |
                               v
+====================================================================+
|                   ANALYSIS & REPORTING                             |
|  - Mutual information estimates                                    |
|  - Recovery probability curves                                     |
|  - Sample-complexity curves                                        |
|  - Noise-sensitivity curves                                        |
|  - Modulus-scaling tables                                          |
|  - DCP vs. EDCP comparison tables                                  |
+====================================================================+
```

### 1.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Modularity** | Each engine is a self-contained Python package with a clear public API. No engine reaches into another engine’s internals. |
| **Reproducibility** | Every experiment is fully parameterised. A saved config file plus a random seed reproduces the exact same results. |
| **Separation of concerns** | Quantum circuit construction is decoupled from classical information processing, which is decoupled from statistical analysis. |
| **State-vector access first** | The default backend is Qiskit Aer state-vector simulation to enable full information-theoretic analysis. Shot-based simulation is a secondary mode. |
| **Extensibility** | New problem variants (e.g., noisy DCP, different EDCP structures) are added by implementing a new Engine subclass, not by modifying existing code. |

---

## 2. Component Architecture

### 2.1 Component Dependency Graph

```text
                    +------------------+
                    |   config         |
                    |  (YAML / dict)   |
                    +--------+---------+
                             |
                             v
+------------+     +---------+---------+     +------------------+
| dcp_engine |<----|  orchestrator     |---->| edcp_engine      |
+-----+------+     +---------+---------+     +--------+---------+
      |                      |                        |
      v                      |                        v
+-----+------+               |               +--------+---------+
| qft_engine |               |               | edcp_qft_engine   |
+-----+------+               |               +--------+---------+
      |                      |                        |
      v                      |                        v
+-----+------+               |               +--------+---------+
| info_engine|               |               | mod_halving_engine|
+-----+------+               |               +--------+---------+
      |                      |                        |
      v                      v                        v
+-----+------+     +---------+---------+     +--------+---------+
| recovery   |     |   stats_engine     |     | recovery         |
| _engine    |     |                    |     | _engine          |
+-----+------+     +---------+---------+     +--------+---------+
      |                        |                        |
      +------------------------+------------------------+
                               |
                               v
                    +----------+---------+
                    |   results_store    |
                    +--------------------+
```

### 2.2 Component Descriptions

---
#### 2.2.1 Orchestrator (`orchestrator.py`)

**Responsibility:** Top-level experiment runner. Reads an experiment configuration, constructs the appropriate pipeline, runs it, and persists results.

**Inputs:**
- `ExperimentConfig` dataclass (or YAML file path)

**Outputs:**
- `ExperimentResult` dataclass written to disk (Parquet row + JSON metadata)

**Interface:**
```python
class Orchestrator:
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        ...

    def run_sweep(self, config: SweepConfig) -> pd.DataFrame:
        # Runs many experiments varying one or more parameters
        ...
```

---
#### 2.2.2 DCP Engine (`dcp_engine.py`)

**Responsibility:** Construct the DCP quantum state for a given modulus N, secret s, and random offset x.

**The DCP State:**

$$|\psi_{x,s}\rangle = \frac{1}{\sqrt{2}} \left( |0\rangle |x\rangle + |1\rangle |x+s\rangle \right)$$

**Inputs:**
- `N: int` — modulus (controls register width: n = ceil(log2(N)) qubits)
- `s: int` — hidden secret, 0 \le s < N
- `x: int` — random offset, 0 \le x < N

**Outputs:**
- `QuantumCircuit` with n+1 qubits (1 flag qubit + n data qubits)
- `Statevector` (if running in state-vector mode)
- `DCPState` dataclass encapsulating both

**Circuit Structure:**
```text
q_flag: |0> ---H---*------------------*---
                     |                  |
q_data: |x> -------+--ADD(s, mod N)---+---

Result: (|0>|x> + |1>|x+s>) / sqrt(2)
```

Where `ADD(s, mod N)` is a modular addition circuit conditioned on the flag qubit.

---
#### 2.2.3 QFT Engine (`qft_engine.py`)

**Responsibility:** Apply the Quantum Fourier Transform to the data register of a DCP (or EDCP) state and extract Fourier information.

**The QFT:**

$$|x\rangle \rightarrow \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{2\pi i x y / N} |y\rangle$$

**Inputs:**
- `DCPState` (or `EDCPState`)
- `apply_to: str` — which register(s) to transform (default: data register)

**Outputs:**
- `QFTResult` dataclass containing:
  - `statevector: Statevector` — the post-QFT quantum state
  - `fourier_distribution: dict[int, float]` — probability distribution over Fourier labels y
  - `phases: dict[int, complex]` — secret-dependent phase values for each y
  - `circuit: QuantumCircuit` — the full circuit (DCP state + QFT)

**Circuit Extension:**
```text
[DCP State Circuit] --- [QFT on data register] --- [Measurement]
```

---
#### 2.2.4 EDCP Engine (`edcp_engine.py`)

**Responsibility:** Construct the generalised EDCP quantum state with multiple terms.

**The EDCP State:**

$$|\psi\rangle = \sum_{j} \chi(j) |j\rangle |x + js \rangle$$

where $\chi(j)$ are complex coefficients (for DCP, there are only two terms with equal weight).

**Inputs:**
- `N: int` — modulus
- `s: int` — hidden secret
- `x: int` — random offset
- `chi: dict[int, complex]` — coefficients {j: chi(j)} defining the EDCP instance structure

**Outputs:**
- `EDCPState` dataclass (circuit + statevector + metadata)

---
#### 2.2.5 Modulus-Halving Engine (`mod_halving_engine.py`)

**Responsibility:** Implement the toy-scale version of Bai et al.’s modulus-halving mechanism for EDCP.

**Pipeline:**
```text
EDCP state
    |
    v
QFT on data register
    |
    v
Extract Fourier labels (y values)
    |
    v
Tensor product states from multiple samples
    |
    v
Set up linear equations over Z_N
    |
    v
Reduce modulus: N -> N/2
    |
    v
Repeat (configurable iterations)
    |
    v
Output: reduced-secret information
```

**Inputs:**
- `list[EDCPState]` — multiple EDCP samples
- `iterations: int` — number of modulus-halving rounds
- `N: int` — initial modulus

**Outputs:**
- `ModHalvingResult` dataclass containing:
  - `reduced_modulus: int`
  - `equations: list[LinearEquation]`
  - `recovered_info: dict`
  - `success: bool`

---
#### 2.2.6 Information Engine (`info_engine.py`)

**Responsibility:** Systematically manipulate Fourier information. This is the heart of the project.

**Operations:**

1. **Full extraction** — return the complete Fourier label y (all n bits)
2. **Bit truncation** — retain only the most significant k bits of y, producing y_k
3. **Bit removal** — remove specific bit positions (e.g., discard LSBs, discard MSBs)
4. **Noise injection** — flip selected bits of y with probability epsilon
5. **Quantisation** — coarsen the Fourier label resolution

**Truncation Convention:**

For a Fourier label y represented as an n-bit string (MSB first):

```text
y = b_{n-1} b_{n-2} ... b_1 b_0

Full:          b_{n-1} b_{n-2} ... b_1 b_0     (k = n)
Truncated k=6: b_{n-1} b_{n-2} ... b_{n-6}     (keep top 6 bits)
Truncated k=4: b_{n-1} b_{n-2} b_{n-3} b_{n-4}  (keep top 4 bits)
Truncated k=2: b_{n-1} b_{n-2}                  (keep top 2 bits)
```

**Inputs:**
- `QFTResult` — Fourier information from the QFT engine
- `k: int` — number of bits to retain (None = full)
- `noise_level: float` — bit-flip probability epsilon (default 0.0)
- `truncation_mode: str` — "msb" (keep top k bits), "lsb" (keep bottom k bits), or "custom"

**Outputs:**
- `InformationResult` dataclass containing:
  - `Y_full: int` — complete Fourier label
  - `Y_truncated: int` — truncated Fourier label y_k
  - `k: int` — bits retained
  - `n: int` — total bits
  - `noise_applied: bool`
  - `bit_flips: list[int]` — positions of any flipped bits (for debugging)

---
#### 2.2.7 Secret Recovery Engine (`recovery_engine.py`)

**Responsibility:** Given (possibly truncated) Fourier information, infer the hidden secret s.

**Recovery Strategies:**

| Strategy | Description | When Used |
|----------|-------------|-----------|
| **Brute-force** | Try all s \in Z_N, pick the one maximising likelihood | Small N (\le 64) |
| **Maximum-likelihood** | Compute P(y_k | s) for each s, select argmax | General case |
| **Phase-matching** | Use secret-dependent phases to narrow candidates | Full Fourier info |
| **Bayesian** | Maintain a posterior distribution over s, update with each sample | Multi-sample experiments |
| **Bit-wise** | Infer individual bits s_i independently | Bit-recovery experiments |

**Inputs:**
- `list[InformationResult]` — Fourier observations from m samples
- `N: int` — modulus
- `strategy: str` — recovery strategy name
- `s_true: int` — ground-truth secret (for evaluation only)

**Outputs:**
- `RecoveryResult` dataclass containing:
  - `s_hat: int` — estimated secret
  - `s_true: int` — actual secret
  - `correct: bool` — whether s_hat == s_true
  - `posterior: dict[int, float]` — full posterior distribution P(s | observations)
  - `bit_correct: list[bool]` — per-bit correctness (for bit-recovery analysis)
  - `confidence: float` — posterior probability assigned to s_hat
  - `strategy: str` — which strategy was used

---
#### 2.2.8 Statistics Engine (`stats_engine.py`)

**Responsibility:** Compute all quantitative metrics and perform statistical analysis.

**Metrics Computed:**

| Metric | Formula | Description |
|--------|---------|-------------|
| Recovery probability | P(s_hat = s) | Fraction of trials where secret is correctly recovered |
| Bit-recovery probability | P(s_hat_i = s_i) | Per-bit accuracy averaged over all bit positions |
| Mutual information (full) | I(S; Y) | MI between secret and complete Fourier observation |
| Mutual information (truncated) | I(S; Y_k) | MI between secret and truncated Fourier observation |
| Information loss ratio | 1 - I(S; Y_k) / I(S; Y) | Fraction of information lost due to truncation |
| Advantage (bit) | P_success - 1/2 | Advantage over random guessing for single-bit recovery |
| Confidence interval | Wilson score interval | 95% CI on recovery probability |

**Inputs:**
- `list[RecoveryResult]` — results from many experiment trials
- `ExperimentConfig` — original experiment parameters

**Outputs:**
- `StatisticsResult` dataclass containing all metrics above, plus:
  - `runtime_seconds: float`
  - `circuit_depth: int`
  - `num_qubits: int`
  - `raw_data: pd.DataFrame` — full trial-level data

---

## 3. Data Flow

### 3.1 Primary DCP Pipeline — Single Experiment

```text
Step 1: DCP Engine
  INPUT:  N=16, s=5, x=11
  OUTPUT: |psi> = (|0>|11> + |1>|0>) / sqrt(2)   [since 11+5=16=0 mod 16]
  ARTIFACT: QuantumCircuit(5 qubits), Statevector

Step 2: QFT Engine
  INPUT:  |psi>
  OUTPUT: Fourier state + distribution + phases
  ARTIFACT: QFTResult {distribution: {0: 0.5, 8: 0.5}, phases: {...}}

Step 3: Information Engine (truncation)
  INPUT:  QFTResult, k=2
  OUTPUT: Y_full=8 (1000), Y_truncated=2 (10)   [keep top 2 of 4 bits]
  ARTIFACT: InformationResult {Y_full=8, Y_truncated=2, k=2, n=4}

Step 4: Secret Recovery Engine
  INPUT:  [InformationResult] (m=1 sample), N=16, strategy="brute_force"
  OUTPUT: s_hat=13, correct=False
  ARTIFACT: RecoveryResult {s_hat=13, correct=False, posterior={...}}

Step 5: Statistics Engine
  INPUT:  [RecoveryResult] (repeated 1000 times with different x)
  OUTPUT: P_success=0.003, I(S;Y_k)=0.02 bits, advantage=0.003
  ARTIFACT: StatisticsResult + saved Parquet row
```

### 3.2 Multi-Sample Pipeline

```text
For m samples (m independent DCP instances with same s, different x_i):

  x_1 --> DCP --> QFT --> Info -->|
  x_2 --> DCP --> QFT --> Info -->|--> Recovery (joint inference) --> Stats
  ...                            |
  x_m --> DCP --> QFT --> Info -->|

The recovery engine receives m InformationResult objects and performs
joint inference (e.g., Bayesian posterior update or majority voting).
```

### 3.3 Parameter Sweep Pipeline

```text
SweepConfig:
  base: {N=16, s=5, m=1, epsilon=0, shots=1000}
  vary:  k = [2, 3, 4]       (truncation sweep)
  vary:  N = [4, 8, 16, 32]  (modulus sweep)
  vary:  m = [1, 2, 4, 8]    (sample-complexity sweep)
  vary:  epsilon = [0, 0.01, 0.05, 0.1, 0.2]  (noise sweep)

Each (k, N, m, epsilon) combination --> 1 ExperimentResult
All results --> aggregated DataFrame --> plots
```

---

## 4. Data Models

### 4.1 Core Data Classes

```python
@dataclass
class ExperimentConfig:
    N: int                    # Modulus
    s: int                    # Hidden secret
    n: int                    # Qubit count = ceil(log2(N))
    m: int                    # Number of DCP samples
    k: int | None             # Bits retained (None = full)
    epsilon: float            # Noise level (bit-flip probability)
    shots: int                # Measurement repetitions (shot-based mode)
    seed: int                 # Random seed for reproducibility
    problem_type: str         # "dcp" or "edcp"
    recovery_strategy: str    # "brute_force", "ml", "phase_match", "bayesian", "bitwise"
    truncation_mode: str      # "msb", "lsb", "custom"
    backend: str              # "statevector", "shots"
    edcp_chi: dict[int, complex] | None  # EDCP coefficients (None for DCP)
    mod_halving_iterations: int  # For EDCP Bai-style pipeline (default 0)

@dataclass
class DCPState:
    circuit: QuantumCircuit
    statevector: Statevector
    N: int
    s: int
    x: int
    n_qubits: int

@dataclass
class QFTResult:
    statevector: Statevector
    circuit: QuantumCircuit
    fourier_distribution: dict[int, float]   # y -> P(y)
    phases: dict[int, complex]               # y -> phase factor
    N: int

@dataclass
class InformationResult:
    Y_full: int                   # Complete Fourier label
    Y_truncated: int              # Truncated Fourier label
    k: int | None                 # Bits retained
    n: int                        # Total bits
    noise_applied: bool
    bit_flips: list[int]          # Positions of flipped bits
    truncation_mode: str

@dataclass
class RecoveryResult:
    s_hat: int                    # Estimated secret
    s_true: int                   # True secret
    correct: bool                 # Full-secret recovery success
    posterior: dict[int, float]   # P(s | observations)
    bit_correct: list[bool]       # Per-bit correctness
    confidence: float             # P(s = s_hat | observations)
    strategy: str
    num_samples_used: int

@dataclass
class StatisticsResult:
    recovery_prob: float          # P(s_hat = s)
    recovery_prob_ci: tuple       # (lower, upper) 95% CI
    bit_recovery_probs: list[float]  # Per-bit P(s_hat_i = s_i)
    bit_advantages: list[float]      # Per-bit advantage over 1/2
    mi_full: float                # I(S; Y)
    mi_truncated: float           # I(S; Y_k)
    info_loss_ratio: float        # 1 - I(S;Y_k)/I(S;Y)
    runtime_seconds: float
    circuit_depth: int
    num_qubits: int
    num_samples: int
    raw_data: pd.DataFrame

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    dcp_state: DCPState | None
    qft_result: QFTResult | None
    info_result: InformationResult | None
    recovery_result: RecoveryResult | None
    statistics: StatisticsResult | None
    timestamp: str
```

### 4.2 EDCP-Specific Data Classes

```python
@dataclass
class EDCPState:
    circuit: QuantumCircuit
    statevector: Statevector
    N: int
    s: int
    x: int
    chi: dict[int, complex]    # j -> chi(j)
    n_qubits: int
    num_terms: int

@dataclass
class ModHalvingResult:
    initial_modulus: int
    reduced_modulus: int
    iterations: int
    equations: list[tuple]      # Linear equations over Z_N
    recovered_info: dict
    success: bool
    intermediate_results: list   # Per-iteration snapshots
```

---

## 5. Directory Structure

```text
dcp-framework/
|
+-- README.md
+-- pyproject.toml                  # Project metadata, dependencies
+-- requirements.txt
+-- setup.cfg
|
+-- configs/                       # Experiment configuration files
|   +-- dcp_base.yaml
|   +-- dcp_truncation_sweep.yaml
|   +-- dcp_sample_sweep.yaml
|   +-- dcp_noise_sweep.yaml
|   +-- dcp_modulus_sweep.yaml
|   +-- edcp_base.yaml
|   +-- edcp_mod_halving.yaml
|
+-- src/
|   +-- __init__.py
|   +-- orchestrator.py             # Top-level experiment runner
|   +-- config.py                   # ExperimentConfig, SweepConfig, YAML loading
|   |
|   +-- engines/
|   |   +-- __init__.py
|   |   +-- dcp_engine.py           # DCP state preparation
|   |   +-- qft_engine.py           # Quantum Fourier Transform
|   |   +-- edcp_engine.py          # EDCP state preparation
|   |   +-- mod_halving_engine.py   # Bai-style modulus halving
|   |   +-- info_engine.py          # Fourier information manipulation
|   |   +-- recovery_engine.py      # Secret recovery strategies
|   |   +-- stats_engine.py         # Statistical analysis & metrics
|   |
|   +-- circuits/
|   |   +-- __init__.py
|   |   +-- dcp_circuit.py          # Low-level DCP circuit construction
|   |   +-- qft_circuit.py          # QFT circuit construction
|   |   +-- edcp_circuit.py         # Low-level EDCP circuit construction
|   |   +-- modular_add.py          # Modular addition sub-circuit
|   |   +-- noise.py                # Noise model application
|   |
|   +-- info/
|   |   +-- __init__.py
|   |   +-- truncation.py           # Bit truncation logic
|   |   +-- mutual_information.py   # MI estimation (histogram / KDE)
|   |   +-- phase_analysis.py       # Phase extraction & analysis
|   |
|   +-- recovery/
|   |   +-- __init__.py
|   |   +-- brute_force.py          # Exhaustive search over Z_N
|   |   +-- maximum_likelihood.py   # ML estimation
|   |   +-- phase_matching.py       # Phase-based candidate filtering
|   |   +-- bayesian.py             # Bayesian posterior inference
|   |   +-- bitwise.py              # Individual bit recovery
|   |
|   +-- utils/
|   |   +-- __init__.py
|   |   +-- logging.py              # Structured logging setup
|   |   +-- serialization.py        # Save/load results
|   |   +-- math_utils.py           # Number-theoretic helpers (gcd, etc.)
|   |   +-- validation.py           # Input validation & assertions
|
+-- scripts/
|   +-- run_single.py               # Run a single experiment
|   +-- run_sweep.py                # Run a parameter sweep
|   +-- run_full_battery.py         # Run all experiment batteries
|   +-- plot_results.py             # Generate all standard plots
|   +-- verify_dcp_state.py         # Verification script for DCP state
|   +-- verify_qft.py               # Verification script for QFT
|   +-- compare_dcp_edcp.py         # DCP vs. EDCP comparison
|
+-- tests/
|   +-- __init__.py
|   +-- test_dcp_engine.py
|   +-- test_qft_engine.py
|   +-- test_edcp_engine.py
|   +-- test_info_engine.py
|   +-- test_recovery_engine.py
|   +-- test_stats_engine.py
|   +-- test_truncation.py
|   +-- test_mutual_information.py
|   +-- test_mod_halving.py
|
+-- results/                       # Generated results (git-ignored)
|   +-- raw/                       # Per-experiment JSON/Parquet files
|   +-- aggregated/                # Sweep results as DataFrames
|   +-- figures/                   # Generated plots
|
+-- docs/
|   +-- architecture.md            # This file
|   +-- implementation.md          # Implementation details
|   +-- phase.md                   # Development phases
|   +-- experiment_catalog.md      # Catalog of all experiments
```

---

## 6. Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Quantum simulation** | Qiskit 1.x + Qiskit Aer | Industry-standard Python quantum SDK; state-vector and shot-based backends |
| **Numerical computing** | NumPy, SciPy | Linear algebra, FFT reference, statistical distributions |
| **Data manipulation** | Pandas | Tabular experiment results, aggregation, grouping |
| **Information theory** | scikit-learn (KDE), custom MI estimators | Histogram-based and KDE-based mutual information estimation |
| **Visualisation** | Matplotlib, Seaborn | Publication-quality plots for recovery curves, MI curves, heatmaps |
| **Configuration** | PyYAML | Human-readable experiment configuration files |
| **Persistence** | Parquet (via pyarrow or pandas), JSON | Efficient columnar storage for large result sets; JSON for metadata |
| **Testing** | pytest | Unit and integration tests for every engine |
| **Logging** | Python stdlib `logging` (structured) | Per-experiment traceability |
| **Package management** | pip + pyproject.toml | Standard Python packaging |

---

## 7. Execution Modes

### 7.1 State-Vector Mode (Default)

```text
Purpose: Full information-theoretic analysis
Backend:  Aer.get_backend('statevector_simulator')
Access:   Complete statevector amplitudes
Use for:  Mutual information computation, phase analysis, exact distributions
Limitation: Exponential memory in qubit count (practical up to ~20 qubits)
```

### 7.2 Shot-Based Mode

```text
Purpose: Simulate realistic measurement statistics
Backend:  Aer.get_backend('qasm_simulator')
Access:   Sampled measurement outcomes only
Use for:  Recovery probability estimation, noise robustness, scaling studies
Parameters: shots = 1000 (default), configurable
```

### 7.3 Noisy Simulation Mode (Extension)

```text
Purpose: Study noise robustness beyond simple bit-flip noise
Backend:  Aer with NoiseModel
Sources:  Depolarising noise, thermal relaxation, gate errors
Use for:  Practical quantum-computing robustness analysis (supporting experiment)
```

---

## 8. Interface Contracts

### 8.1 Engine Base Interface

Every engine implements a common pattern:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

class EngineResult(ABC):
    """Base class for all engine outputs."""
    pass

class BaseEngine(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> EngineResult:
        """Execute the engine’s computation and return a result."""
        ...
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Check that inputs are valid. Raise ValueError if not."""
        ...
```

### 8.2 Inter-Engine Data Passing

```text
DCP Engine  -->  QFT Engine  -->  Info Engine  -->  Recovery Engine  -->  Stats Engine
   |               |               |                  |                    |
   v               v               v                  v                    v
DCPState      QFTResult     InformationResult   RecoveryResult    StatisticsResult
```

Each engine accepts the output dataclass of the previous engine as its primary input. The Orchestrator handles the wiring. No engine imports another engine’s module.

---

## 9. Non-Goals (Architectural Boundaries)

The following are explicitly **out of scope** for this framework’s architecture:

| Not In Scope | Reason |
|-------------|--------|
| Real quantum hardware execution | Not required for core experiments; state-vector simulation is primary |
| Cryptographic-size LWE instances (n > 20, q > 2^10) | Far beyond simulation capability; this is a toy-scale framework |
| Novel quantum algorithm design | We implement known approaches (QFT + Fourier analysis), not new algorithms |
| Formal security proofs | We produce experimental evidence, not mathematical proofs |
| Full Bai et al. cryptographic attack | We implement the conceptual mechanism at toy scale only |
| ML-KEM / Kyber / Dilithium attack code | PQC is context, not an experimental target |
| GUI / Web interface | CLI and configuration-driven execution only |
| Distributed computing | Single-machine execution is sufficient for our parameter ranges |

---

## 10. Key Design Decisions

### 10.1 Why State-Vector First?

Mutual information estimation requires access to the full probability distribution P(Y, S), not just samples. State-vector simulation gives us exact amplitudes, from which we can compute exact distributions. Shot-based simulation is added later for robustness studies, but it cannot replace state-vector mode for the core information-theoretic analysis.

### 10.2 Why Dataclasses Over Dictionaries?

Every inter-engine communication uses typed dataclasses. This provides: (a) IDE autocomplete and type checking, (b) explicit documentation of what each engine produces, (c) validation at the boundary between components, and (d) easy serialisation to disk.

### 10.3 Why YAML Configuration?

Experiments are defined declaratively in YAML. This enables: (a) reproducibility (check in configs alongside results), (b) sweep definitions without code changes, (c) easy sharing between collaborators, and (d) version control of experiment definitions.

### 10.4 Why Modular Addition Instead of Oracle?

Some DCP formulations treat the DCP state as given by an oracle. We construct the actual circuit because: (a) it is pedagogically valuable to see the full quantum circuit, (b) it enables noise injection at the gate level, (c) it allows circuit-depth and gate-count measurements for the scaling study, and (d) it makes the simulation self-contained without relying on external oracles.

### 10.5 Why Separate Info Engine from Recovery Engine?

The information manipulation (truncation, noise) is conceptually distinct from the inference algorithm. Separating them means: (a) we can swap recovery strategies without touching the info pipeline, (b) we can study the information-theoretic properties (MI) independently of any particular recovery algorithm, and (c) we can add new information manipulations (e.g., quantisation, random projection) without modifying recovery code.
