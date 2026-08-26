# Project Context & Phase Implementation Log

> **Project:** DCP/EDCP Quantum Information Analysis Framework  
> **Last Updated:** 2026-08-22  
> **Current Status:** Phase 3 Complete (Analysis, Statistics, and Visualisation)

---

## Phase 1 Implementation Log: Foundation (DCP State Construction & QFT)

### 1. Overview & Objectives
Phase 1 establishes the foundational quantum primitives and software infrastructure for the DCP/EDCP framework:
1. Controlled modular addition quantum circuits over $\mathbb{Z}_N$.
2. DCP state preparation circuits constructing $|\psi_{x,s}\rangle = \frac{1}{\sqrt{2}} (|0\rangle |x\rangle + |1\rangle |(x + s) \bmod N\rangle)$.
3. Quantum Fourier Transform (QFT) applied to the data register and Fourier phase extraction $\exp(2\pi i s y / N)$.
4. Configuration and validation subsystem supporting YAML serialization.
5. Verification tooling and comprehensive unit testing test suite.

---

## Phase 2 Implementation Log: Core Pipeline & Truncation Experiments

### 1. Overview & Objectives
Phase 2 establishes the end-to-end simulation and inference pipeline to address the core research question: how Fourier information truncation ($k$ bits) affects hidden secret recovery ($s$) across moduli $N \in \{4, 8, 16, 32\}$.
1. Information Engine with quantum joint measurement sampling $(y, b)$ and bit truncation.
2. Recovery Engine with brute-force, maximum-likelihood, Bayesian sequential updating, bitwise MAP inference, and phase matching.
3. Orchestrator and persistence layer for Parquet/JSON data storage.
4. Execution of the core truncation sweep (14 configurations, 14,000 trials).

---

## Phase 3 Implementation Log: Analysis, Statistics, and Visualisation

### 1. Overview & Objectives
Phase 3 transforms raw trial simulation datasets into rigorous information-theoretic metrics, aggregated statistical datasets, and publication-quality figures:
1. **Mutual Information Engine (`src/info/mutual_information.py`)**:
   - Exact computation of Shannon Mutual Information $I(S; Y_k, B)$ via complete joint distribution enumeration:
     $$I(S; Y_k, B) = \sum_{s, y_k, b} P(s, y_k, b) \log_2 \left( \frac{P(s, y_k, b)}{P(s) P(y_k, b)} \right)$$
   - Computation of the Information Loss Ratio $1 - I(S; Y_k)/I(S; Y_{\text{full}})$.
   - Profile computation across all truncation levels $k \in [1, \dots, n]$.
2. **Statistics Engine Subsystem (`src/engines/stats_engine.py`)**:
   - Automated group aggregation over multi-parameter trial datasets.
   - Wilson score confidence intervals for recovery probabilities.
   - Per-bit success probabilities and per-bit advantages over random guessing ($0.5$).
   - Multi-format dataset persistence (Parquet, CSV, JSON).
3. **Publication-Quality Plotting Suite (`src/visualization/plots.py`, `scripts/plot_results.py`)**:
   - 300 DPI high-resolution figures:
     - `recovery_vs_truncation.png` (empirical success probability with 95% Wilson CI bands)
     - `mi_vs_truncation.png` (exact theoretical mutual information curves)
     - `information_loss_ratio.png` (monotonic information degradation curves)
     - `bit_recovery_heatmap_N*.png` (2D per-bit accuracy heatmaps for each $N$)
     - `dcp_core_dashboard.png` (comprehensive 4-panel analysis dashboard)
     - `sample_posterior_N16.png` (posterior distribution over candidate secrets)

---

### 2. Complete File Inventory

| File Path | Component | Purpose |
|-----------|-----------|---------|
| [`pyproject.toml`](file:///c:/FYP/fyp2027/pyproject.toml) | Build Config | Package metadata, dependencies, and pytest configuration. |
| [`requirements.txt`](file:///c:/FYP/fyp2027/requirements.txt) | Dependency Manifest | Runtime dependencies (`qiskit`, `qiskit-aer`, `numpy`, `scipy`, `pandas`, `pyyaml`, `pyarrow`, `pytest`, `tqdm`, `matplotlib`, `seaborn`). |
| [`src/config.py`](file:///c:/FYP/fyp2027/src/config.py) | Config Subsystem | `ExperimentConfig` dataclass and YAML loader. |
| [`configs/dcp_base.yaml`](file:///c:/FYP/fyp2027/configs/dcp_base.yaml) | Config | Base single-run configuration. |
| [`configs/dcp_truncation_sweep.yaml`](file:///c:/FYP/fyp2027/configs/dcp_truncation_sweep.yaml) | Config | Truncation sweep configuration across $N \in \{4, 8, 16, 32\}$. |
| [`src/circuits/modular_add.py`](file:///c:/FYP/fyp2027/src/circuits/modular_add.py) | Quantum Arithmetic | Draper adder with flag control. |
| [`src/circuits/dcp_circuit.py`](file:///c:/FYP/fyp2027/src/circuits/dcp_circuit.py) | Quantum State Prep | DCP state preparation circuit. |
| [`src/circuits/qft_circuit.py`](file:///c:/FYP/fyp2027/src/circuits/qft_circuit.py) | QFT | Quantum Fourier Transform circuits. |
| [`src/engines/dcp_engine.py`](file:///c:/FYP/fyp2027/src/engines/dcp_engine.py) | DCP Engine | `DCPEngine` and state verification. |
| [`src/engines/qft_engine.py`](file:///c:/FYP/fyp2027/src/engines/qft_engine.py) | QFT Engine | `QFTEngine` and Fourier phase verification. |
| [`src/engines/info_engine.py`](file:///c:/FYP/fyp2027/src/engines/info_engine.py) | Info Engine | `InformationEngine` quantum joint sampling and truncation processor. |
| [`src/engines/recovery_engine.py`](file:///c:/FYP/fyp2027/src/engines/recovery_engine.py) | Recovery Engine | `RecoveryEngine` supporting 5 inference strategies. |
| [`src/engines/stats_engine.py`](file:///c:/FYP/fyp2027/src/engines/stats_engine.py) | Statistics Engine | `StatisticsEngine` for dataset aggregation and MI metrics. |
| [`src/info/truncation.py`](file:///c:/FYP/fyp2027/src/info/truncation.py) | Truncation Module | MSB/LSB truncation and noise injection. |
| [`src/info/mutual_information.py`](file:///c:/FYP/fyp2027/src/info/mutual_information.py) | MI Module | Exact Shannon MI and information loss ratio calculation. |
| [`src/recovery/brute_force.py`](file:///c:/FYP/fyp2027/src/recovery/brute_force.py) | Recovery Strategy | Likelihood evaluation and exhaustive candidate scoring. |
| [`src/recovery/maximum_likelihood.py`](file:///c:/FYP/fyp2027/src/recovery/maximum_likelihood.py) | Recovery Strategy | Maximum-likelihood estimator. |
| [`src/recovery/bayesian.py`](file:///c:/FYP/fyp2027/src/recovery/bayesian.py) | Recovery Strategy | Bayesian sequential updating. |
| [`src/recovery/bitwise.py`](file:///c:/FYP/fyp2027/src/recovery/bitwise.py) | Recovery Strategy | Bit-level posterior marginalization. |
| [`src/recovery/phase_matching.py`](file:///c:/FYP/fyp2027/src/recovery/phase_matching.py) | Recovery Strategy | Phase matching on complex Fourier relative phases. |
| [`src/utils/math_utils.py`](file:///c:/FYP/fyp2027/src/utils/math_utils.py) | Math Utilities | Bit manipulation, entropy, Wilson CI. |
| [`src/utils/serialization.py`](file:///c:/FYP/fyp2027/src/utils/serialization.py) | Persistence | Parquet and JSON serialization. |
| [`src/orchestrator.py`](file:///c:/FYP/fyp2027/src/orchestrator.py) | Orchestrator | End-to-end pipeline and sweep runner. |
| [`src/visualization/plots.py`](file:///c:/FYP/fyp2027/src/visualization/plots.py) | Visualisation Suite | Plotting functions for publication figures. |
| [`scripts/run_single.py`](file:///c:/FYP/fyp2027/scripts/run_single.py) | CLI Script | Single configuration experiment runner. |
| [`scripts/run_sweep.py`](file:///c:/FYP/fyp2027/scripts/run_sweep.py) | CLI Script | Parameter sweep runner. |
| [`scripts/plot_results.py`](file:///c:/FYP/fyp2027/scripts/plot_results.py) | CLI Script | Publication analysis and plot generator. |
| [`tests/test_config.py`](file:///c:/FYP/fyp2027/tests/test_config.py) | Unit Tests | 7 passing tests. |
| [`tests/test_modular_add.py`](file:///c:/FYP/fyp2027/tests/test_modular_add.py) | Unit Tests | 9 passing tests. |
| [`tests/test_dcp_engine.py`](file:///c:/FYP/fyp2027/tests/test_dcp_engine.py) | Unit Tests | 8 passing tests. |
| [`tests/test_qft_engine.py`](file:///c:/FYP/fyp2027/tests/test_qft_engine.py) | Unit Tests | 10 passing tests. |
| [`tests/test_info_engine.py`](file:///c:/FYP/fyp2027/tests/test_info_engine.py) | Unit Tests | 5 passing tests. |
| [`tests/test_recovery_engine.py`](file:///c:/FYP/fyp2027/tests/test_recovery_engine.py) | Unit Tests | 5 passing tests. |
| [`tests/test_orchestrator.py`](file:///c:/FYP/fyp2027/tests/test_orchestrator.py) | Integration Tests | 2 passing tests. |
| [`tests/test_mutual_information.py`](file:///c:/FYP/fyp2027/tests/test_mutual_information.py) | Unit Tests | 4 passing tests. |
| [`tests/test_stats_engine.py`](file:///c:/FYP/fyp2027/tests/test_stats_engine.py) | Unit Tests | 2 passing tests. |

---

### 3. Key Findings & Quantitative Summary Table

The core research question — *how does Fourier information truncation affect secret recovery?* — is quantitatively characterized in the aggregated results table below:

| $N$ | $n$ | $k$ | $s$ | Shots | $P_{\text{success}}$ | 95% Wilson CI | $I(S; Y_k, B)$ (bits) | Info Loss Ratio | Mean Bit Accuracy |
|:---:|:---:|:---:|:---:|:-----:|:--------------------:|:-------------:|:---------------------:|:---------------:|:-----------------:|
| **4** | 2 | 1 | 3 | 1000 | 0.176 | [0.154, 0.201] | 0.2500 | 50.0% | 0.342 |
| **4** | 2 | 2 | 3 | 1000 | 0.199 | [0.175, 0.225] | 0.5000 | **0.0%** | 0.435 |
| **8** | 3 | 1 | 5 | 1000 | 0.078 | [0.063, 0.096] | 0.1250 | 73.7% | 0.442 |
| **8** | 3 | 2 | 5 | 1000 | 0.000 | [0.000, 0.004] | 0.2277 | 52.0% | 0.367 |
| **8** | 3 | 3 | 5 | 1000 | 0.044 | [0.033, 0.059] | 0.4748 | **0.0%** | 0.471 |
| **16** | 4 | 1 | 11 | 1000 | 0.041 | [0.030, 0.055] | 0.0625 | 86.4% | 0.397 |
| **16** | 4 | 2 | 11 | 1000 | 0.000 | [0.000, 0.004] | 0.1115 | 75.7% | 0.440 |
| **16** | 4 | 3 | 11 | 1000 | 0.050 | [0.038, 0.065] | 0.2172 | 52.7% | 0.416 |
| **16** | 4 | 4 | 11 | 1000 | 0.010 | [0.005, 0.018] | 0.4592 | **0.0%** | 0.369 |
| **32** | 5 | 1 | 19 | 1000 | 0.018 | [0.011, 0.028] | 0.0312 | 93.1% | 0.455 |
| **32** | 5 | 2 | 19 | 1000 | 0.000 | [0.000, 0.004] | 0.0554 | 87.7% | 0.504 |
| **32** | 5 | 3 | 19 | 1000 | 0.000 | [0.000, 0.004] | 0.1067 | 76.3% | 0.481 |
| **32** | 5 | 4 | 19 | 1000 | 0.000 | [0.000, 0.004] | 0.2120 | 53.0% | 0.441 |
| **32** | 5 | 5 | 19 | 1000 | 0.004 | [0.002, 0.010] | 0.4510 | **0.0%** | 0.452 |

---

### 4. Generated Artifacts
- **Aggregated Datasets**:
  - `results/aggregated/dcp_core_summary.parquet`
  - `results/aggregated/dcp_core_summary.csv`
  - `results/aggregated/dcp_core_summary_metadata.json`
- **Publication Figures**:
  - `results/figures/dcp_core/dcp_core_dashboard.png`
  - `results/figures/dcp_core/recovery_vs_truncation.png`
  - `results/figures/dcp_core/mi_vs_truncation.png`
  - `results/figures/dcp_core/information_loss_ratio.png`
  - `results/figures/dcp_core/bit_recovery_heatmap_N*.png`
  - `results/figures/dcp_core/sample_posterior_N16.png`
