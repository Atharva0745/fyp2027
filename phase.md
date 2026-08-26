# Development Phases — DCP/EDCP Quantum Information Analysis Framework

> **Version:** 1.0  
> **Date:** 2026-08-17  
> **Status:** Development Roadmap

---

## Overview

Development proceeds in **six phases**, ordered by dependency. Each phase produces a verified, testable deliverable that the next phase builds on. The core experiment (Phases 1–3) is self-contained and publishable on its own. Phases 4–6 are extensions that broaden the study.

```text
Phase 1        Phase 2          Phase 3           Phase 4         Phase 5         Phase 6
Foundation     Core Pipeline    Analysis &        Sample          Noise &        EDCP &
               (single sample)  Statistics         Complexity      Scaling       Modulus
                                                                        Halving
   |               |                |                 |               |              |
   v               v                v                 v               v              v
 DCP state      QFT + Info       MI, recovery      Multi-sample    Noise curves   EDCP engine
 construction   engine +         probability,      experiments,    Robustness    Bai toy
 + QFT          recovery         plots, tables     m vs P_success  analysis      pipeline
 verification    engine                            DCP vs EDCP
```

**Estimated total timeline: 8–10 weeks** for a single developer working part-time. Adjust proportionally for team size.

---

## Phase 1 — Foundation: DCP State Construction and QFT

### Objective

Build and verify the two quantum primitives: the DCP state and the Quantum Fourier Transform. These are the building blocks for everything else. Nothing in later phases works unless these are correct.

### Scope

| Component | Status | Description |
|-----------|--------|-------------|
| DCP Engine | **Build** | Construct the DCP quantum state ||ψ_{x,s}\u27e9 for arbitrary (N, s, x) |
| QFT Engine | **Build** | Apply QFT to the data register |
| Modular adder circuit | **Build** | Controlled ADD(s, mod N) sub-circuit |
| State verification | **Build** | Automated tests that the constructed states match theory |
| Phase extraction | **Build** | Extract Fourier distribution and secret-dependent phases from post-QFT statevector |
| Configuration system | **Build** | ExperimentConfig dataclass, YAML loading |
| Project scaffold | **Build** | Directory structure, pyproject.toml, test infrastructure |

### Tasks

#### 1.1 Project Scaffolding
- [ ] Create directory structure as specified in architecture.md
- [ ] Set up pyproject.toml with all dependencies
- [ ] Initialise git repository with .gitignore (exclude results/, __pycache__/)
- [ ] Set up pytest configuration (pytest.ini or pyproject.toml section)
- [ ] Create src/__init__.py, src/engines/__init__.py, etc.
- [ ] Verify: `pytest tests/` runs (empty suite, 0 errors)

#### 1.2 Configuration System
- [ ] Implement ExperimentConfig dataclass in src/config.py
- [ ] Implement YAML config loading (load_config function)
- [ ] Implement config validation (validate_config function)
- [ ] Write tests: valid config loads correctly, invalid config raises ValueError
- [ ] Create configs/dcp_base.yaml with sensible defaults

#### 1.3 Modular Addition Circuit
- [ ] Implement modular_adder in src/circuits/modular_add.py
- [ ] Verify for N=4: ADD(1, 4) maps |0>\u2192|1>, |1>\u2192|2>, |2>\u2192|3>, |3>\u2192|0>
- [ ] Verify for N=8, N=16 with multiple values of a
- [ ] Handle the controlled version (flag qubit as control)
- [ ] Write comprehensive unit tests (test_modular_add.py)
- [ ] **Milestone:** All modular-adder tests pass for N \u2208 {4, 8, 16, 32}

#### 1.4 DCP State Construction
- [ ] Implement DCP Engine (src/engines/dcp_engine.py)
- [ ] Implement DCP circuit builder (src/circuits/dcp_circuit.py)
- [ ] Implement verification function (verify_dcp_state)
- [ ] Test for N=4: verify statevector for all (s, x) combinations
- [ ] Test for N=8: spot-check multiple (s, x) pairs
- [ ] Test for N=16, N=32: at least one (s, x) pair each
- [ ] **Milestone:** DCP state verification passes for all tested (N, s, x)

#### 1.5 QFT and Phase Extraction
- [ ] Implement QFT Engine (src/engines/qft_engine.py)
- [ ] Implement QFT circuit construction (src/circuits/qft_circuit.py)
- [ ] Implement phase extraction (extract_fourier_info)
- [ ] Implement phase verification (verify_phases)
- [ ] Test: for known (N, s), verify extracted phases match exp(2\u03c0 i s y / N)
- [ ] Test: Fourier distribution matches theoretical DCP distribution
- [ ] **Milestone:** QFT phase extraction verified for N \u2208 {4, 8, 16, 32}

#### 1.6 End-to-End Verification Script
- [ ] Create scripts/verify_dcp_state.py (CLI: run DCP + QFT + verify)
- [ ] Create scripts/verify_qft.py (CLI: check phases for given parameters)
- [ ] Run: `python scripts/verify_dcp_state.py --N 16 --s 5 --x 11` and confirm pass
- [ ] **Milestone:** End-to-end verification passes for all tested parameters

### Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| DCP Engine | src/engines/dcp_engine.py | Tested, verified DCP state construction |
| QFT Engine | src/engines/qft_engine.py | Tested, verified QFT + phase extraction |
| Modular adder | src/circuits/modular_add.py | Controlled modular addition circuit |
| Config system | src/config.py | ExperimentConfig + YAML loading |
| Verification scripts | scripts/verify_*.py | CLI tools for verifying state correctness |
| Unit tests | tests/test_dcp_engine.py, tests/test_qft_engine.py | Full test coverage for Phase 1 components |

### Exit Criteria

- [ ] All unit tests pass (pytest, 0 failures)
- [ ] DCP statevector matches theoretical prediction for N \u2208 {4, 8, 16, 32}
- [ ] Extracted Fourier phases match exp(2\u03c0 i s y / N) within tolerance 1e-8
- [ ] Verification scripts run cleanly from command line

---

## Phase 2 — Core Pipeline: Information Engine, Recovery Engine, Single-Sample Experiments

### Objective

Build the core experimental pipeline: truncate Fourier information, recover the secret, and run the first controlled experiments measuring how truncation affects recovery. This is where the **central research question** is first addressed.

### Scope

| Component | Status | Description |
|-----------|--------|-------------|
| Information Engine | **Build** | Bit truncation, noise injection, Fourier label sampling |
| Secret Recovery Engine | **Build** | Brute-force and Bayesian recovery strategies |
| Orchestrator | **Build** | Wire all engines together, run single experiments |
| Results persistence | **Build** | Save experiment results to Parquet/JSON |
| Core truncation experiment | **Run** | First real data: P_success vs. k for various N |

### Tasks

#### 2.1 Information Engine
- [ ] Implement truncation logic (src/info/truncation.py)
- [ ] Implement MSB truncation: keep top k bits of n-bit label
- [ ] Implement LSB truncation: keep bottom k bits (secondary mode)
- [ ] Implement noise injection: independent bit-flip with probability epsilon
- [ ] Implement Fourier label sampling from distribution
- [ ] Implement Information Engine (src/engines/info_engine.py)
- [ ] Write unit tests: known truncation inputs produce known outputs
- [ ] Write unit tests: noise injection flips correct number of bits on average
- [ ] **Milestone:** All truncation and noise tests pass

#### 2.2 Secret Recovery Engine
- [ ] Implement brute-force recovery (src/recovery/brute_force.py)
- [ ] Implement maximum-likelihood recovery (src/recovery/maximum_likelihood.py)
- [ ] Implement Bayesian single-sample recovery (src/recovery/bayesian.py)
- [ ] Implement bit-wise recovery (src/recovery/bitwise.py)
- [ ] Implement recovery engine wrapper (src/engines/recovery_engine.py)
- [ ] Critical test: with full info (k=n) and N=4, brute-force should achieve ~100% recovery
- [ ] Critical test: with k=1 and N=16, recovery should be near-random (\u2248 1/16)
- [ ] **Milestone:** Recovery engine works correctly for full and truncated information

#### 2.3 Orchestrator
- [ ] Implement Orchestrator.run(config) (src/orchestrator.py)
- [ ] Wire: DCP Engine \u2192 QFT Engine \u2192 Info Engine \u2192 Recovery Engine
- [ ] Implement single-experiment result collection
- [ ] Implement results persistence (src/utils/serialization.py)
- [ ] Write integration test: full pipeline with N=4, k=2 (full info)
- [ ] **Milestone:** Orchestrator runs a complete experiment end-to-end

#### 2.4 Core Truncation Experiment
- [ ] Create experiment config for truncation sweep: N \u2208 {4, 8, 16, 32}, k from 1 to n, m=1, epsilon=0, shots=1000
- [ ] Create scripts/run_single.py (run one experiment config)
- [ ] Run truncation sweep for N=4
- [ ] Run truncation sweep for N=8
- [ ] Run truncation sweep for N=16
- [ ] Run truncation sweep for N=32
- [ ] Inspect raw results: verify P_success \u2192 1 as k \u2192 n, P_success \u2192 1/N as k \u2192 0
- [ ] **Milestone:** First experimental data demonstrating the information-recovery relationship

### Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Information Engine | src/engines/info_engine.py | Truncation and noise injection |
| Recovery Engine | src/engines/recovery_engine.py | Multiple recovery strategies |
| Orchestrator | src/orchestrator.py | Full pipeline execution |
| First dataset | results/raw/dcp_truncation_core/ | Recovery vs. k for N \u2208 {4, 8, 16, 32} |

### Exit Criteria

- [ ] Full pipeline runs without errors for all tested parameters
- [ ] With full Fourier info (k=n), recovery probability > 0.9 for N \u2264 16
- [ ] With heavy truncation (k=1), recovery probability is near 1/N (random baseline)
- [ ] Results are saved to disk in Parquet format and can be loaded with Pandas

---

## Phase 3 — Analysis, Statistics, and Visualisation

### Objective

Build the statistical analysis engine and produce the publication-quality plots and tables that form the core experimental results. This phase transforms raw experiment outputs into research findings.

### Scope

| Component | Status | Description |
|-----------|--------|-------------|
| Statistics Engine | **Build** | Recovery probability, CI, mutual information, advantage |
| Mutual information module | **Build** | Exact MI computation via joint distribution enumeration |
| Plotting suite | **Build** | All standard plots (recovery, MI, heatmaps) |
| Core analysis | **Run** | Analyse Phase 2 data, produce figures and tables |

### Tasks

#### 3.1 Mutual Information Computation
- [ ] Implement exact MI computation (src/info/mutual_information.py)
- [ ] Compute I(S; Y) for full Fourier information — this is the baseline
- [ ] Compute I(S; Y_k) for truncated information — this is the key quantity
- [ ] Compute information loss ratio: 1 - I(S;Y_k)/I(S;Y)
- [ ] Verify: for N=2, I(S;Y) should equal 1 bit (complete information about 1-bit secret)
- [ ] Verify: for N=4 with k=n=2, I(S;Y_k) should be close to 2 bits
- [ ] Verify: MI is monotonically non-decreasing in k
- [ ] **Milestone:** MI computation verified against known theoretical values

#### 3.2 Statistics Engine
- [ ] Implement Statistics Engine (src/engines/stats_engine.py)
- [ ] Implement recovery probability with Wilson confidence intervals
- [ ] Implement bit-recovery probability and per-bit advantage
- [ ] Implement MI aggregation across trials
- [ ] Implement runtime and circuit-depth tracking
- [ ] Write tests: known input data produces known statistics
- [ ] **Milestone:** Statistics engine produces correct metrics

#### 3.3 Plotting Suite
- [ ] Implement plot_recovery_vs_truncation (P_success vs. k for each N)
- [ ] Implement plot_mi_vs_truncation (I(S;Y_k) vs. k for each N)
- [ ] Implement plot_information_loss (loss ratio vs. k)
- [ ] Implement plot_posterior (posterior distribution over s for specific trials)
- [ ] Implement plot_bit_recovery_heatmap (per-bit accuracy vs. k)
- [ ] Set up consistent visual style (font sizes, colours, figure dimensions)
- [ ] Create scripts/plot_results.py (generate all standard plots from a results directory)
- [ ] **Milestone:** All standard plots generate correctly from Phase 2 data

#### 3.4 Core Analysis Run
- [ ] Load Phase 2 truncation sweep results
- [ ] Compute MI for each (N, k) combination
- [ ] Generate all standard plots
- [ ] Create summary table: N, n, k, P_success, 95% CI, I(S;Y_k), info_loss_ratio
- [ ] Inspect results: confirm the qualitative pattern (more info \u2192 better recovery)
- [ ] Identify any surprising results or edge cases for further investigation
- [ ] **Milestone:** Complete core analysis with publication-quality figures

### Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| MI module | src/info/mutual_information.py | Exact MI computation |
| Statistics Engine | src/engines/stats_engine.py | All quantitative metrics |
| Plotting suite | scripts/plot_results.py | All standard plot generators |
| Core figures | results/figures/dcp_core/ | Recovery, MI, info-loss, posterior, heatmap plots |
| Summary table | results/aggregated/dcp_core_summary.parquet | Comprehensive metrics table |

### Exit Criteria

- [ ] MI computation matches theoretical predictions for N=2, 4, 8
- [ ] All plots render without errors and are visually clear
- [ ] Summary table contains all metrics for all (N, k) combinations
- [ ] The core research question ("how does Fourier info loss affect secret recovery?") is quantitatively answered

---

## Phase 4 — Sample Complexity Experiments

### Objective

Investigate whether multiple independent DCP samples can compensate for information loss due to truncation. This connects our experiments to the sample-complexity concerns in the EDCP/Bai research literature.

### Scope

| Component | Status | Description |
|-----------|--------|-------------|
| Multi-sample recovery | **Extend** | Bayesian recovery with m > 1 samples |
| Sample-complexity sweep | **Run** | Vary m \u2208 {1, 2, 4, 8, 16} for each (N, k) |
| Analysis | **Run** | P_success vs. m curves, MI aggregation |

### Tasks

#### 4.1 Multi-Sample Bayesian Recovery
- [ ] Extend Bayesian recovery to handle m independent observations
- [ ] Implement log-space posterior update for numerical stability
- [ ] Test: with m=2 and full info, recovery should be higher than m=1
- [ ] Test: posterior concentrates around true secret as m increases
- [ ] **Milestone:** Multi-sample recovery works correctly

#### 4.2 Sample-Complexity Sweep
- [ ] Create sweep config: N \u2208 {4, 8, 16}, k \u2208 {1, ..., n}, m \u2208 {1, 2, 4, 8, 16}, epsilon=0, shots=500
- [ ] Run sweep (this is a large computation — may take 30–60 minutes)
- [ ] Implement plot_recovery_vs_samples (P_success vs. m for each (N, k))
- [ ] Analyse: for a fixed target P_success, how many samples are needed at each k?
- [ ] Analyse: does increasing m compensate for decreasing k? By how much?
- [ ] **Milestone:** Sample-complexity curves generated and analysed

### Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Multi-sample recovery | src/recovery/bayesian.py (extended) | Sequential Bayesian update |
| Sample-complexity data | results/raw/dcp_sample_complexity/ | P_success(m) for all (N, k, m) |
| Sample-complexity plots | results/figures/dcp_sample_complexity/ | P_success vs. m curves |

### Exit Criteria

- [ ] Multi-sample recovery produces valid posteriors
- [ ] P_success is non-decreasing in m for fixed (N, k)
- [ ] Sample-complexity plots are generated for all tested parameter combinations

---

## Phase 5 — Noise and Scaling Experiments

### Objective

Add two supporting dimensions: noise robustness and modulus scaling. These broaden the experimental study and provide practical context for the core results.

### Scope

| Component | Status | Description |
|-----------|--------|-------------|
| Noise injection | **Already built** | Bit-flip noise in Information Engine (Phase 2) |
| Noise sweep | **Run** | Vary epsilon \u2208 {0, 0.05, 0.1, 0.2} |
| Modulus sweep | **Run** | Add N=64 to the parameter set |
| Scaling analysis | **Run** | Measure runtime, circuit depth, memory vs. N |

### Tasks

#### 5.1 Noise Robustness Sweep
- [ ] Create noise sweep config: N \u2208 {4, 8, 16}, k \u2208 {2, ..., n}, m=1, epsilon \u2208 {0, 0.05, 0.1, 0.15, 0.2}, shots=500
- [ ] Run noise sweep
- [ ] Implement plot_recovery_vs_noise (P_success vs. epsilon for each (N, k))
- [ ] Analyse: at what noise level does recovery collapse to random?
- [ ] Analyse: is there an interaction between noise and truncation? (Are truncated observations more or less noise-sensitive?)
- [ ] **Milestone:** Noise-robustness curves generated and analysed

#### 5.2 Modulus Scaling
- [ ] Run core truncation experiment with N=64 (n=6 qubits)
- [ ] Measure: circuit depth, number of gates, statevector size, runtime per trial
- [ ] Run for N=4, 8, 16, 32, 64 with k=n (full info) as a scaling baseline
- [ ] Create scaling table: N, n, qubits, circuit_depth, gates, runtime_ms, memory_bytes
- [ ] Implement plot_scaling (log-log plots of runtime and circuit depth vs. N)
- [ ] **Milestone:** Scaling measurements complete up to N=64

#### 5.3 Combined Parameter Sweep
- [ ] Create a comprehensive sweep config varying N, k, m, and epsilon simultaneously
- [ ] Run the full sweep (estimated ~25 minutes for full parameter space)
- [ ] Generate heatmap: P_success as a function of (k, epsilon) for fixed N and m
- [ ] Generate heatmap: P_success as a function of (k, m) for fixed N and epsilon
- [ ] **Milestone:** Full parameter-space characterisation

### Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| Noise data | results/raw/dcp_noise/ | P_success(epsilon) for all parameters |
| Scaling table | results/aggregated/scaling_summary.parquet | Runtime, depth, memory vs. N |
| Scaling plots | results/figures/dcp_scaling/ | Log-log scaling curves |
| Heatmaps | results/figures/dcp_heatmaps/ | 2D parameter-space heatmaps |

### Exit Criteria

- [ ] Noise curves show monotonic degradation of recovery with increasing epsilon
- [ ] Scaling measurements are consistent with theoretical expectations
- [ ] N=64 experiments complete without memory issues

---

## Phase 6 — EDCP and Modulus-Halving Pipeline

### Objective

Extend the framework to EDCP and implement a toy-scale version of Bai et al.'s modulus-halving approach. This connects our DCP experiments to the broader EDCP/LWE research context.

### Scope

| Component | Status | Description |
|-----------|--------|-------------|
| EDCP Engine | **Build** | General EDCP state construction |
| Modulus-halving engine | **Build** | Toy Bai-style pipeline |
| DCP vs. EDCP comparison | **Run** | Compare information-processing behaviour |

### Tasks

#### 6.1 EDCP State Construction
- [ ] Implement EDCP Engine (src/engines/edcp_engine.py)
- [ ] Implement EDCP circuit construction (src/circuits/edcp_circuit.py)
- [ ] Define standard EDCP chi configurations (2-term, 4-term, LWE-like)
- [ ] Verify: DCP is a special case of EDCP (2-term with equal weights)
- [ ] Test EDCP state correctness for small instances
- [ ] **Milestone:** EDCP engine produces correct states

#### 6.2 Toy Bai-Style Modulus Halving
- [ ] Implement modulus-halving engine (src/engines/mod_halving_engine.py)
- [ ] Implement single iteration: EDCP \u2192 QFT \u2192 Fourier labels \u2192 equations \u2192 halve N
- [ ] Implement multi-iteration loop with state tracking
- [ ] Test with known small instances where the expected outcome is verifiable
- [ ] **Milestone:** Modulus halving works correctly for toy parameters

#### 6.3 DCP vs. EDCP Comparison
- [ ] Run truncation experiments for EDCP instances with the same N
- [ ] Compare: I(S;Y_k) for DCP vs. EDCP at the same N and k
- [ ] Compare: P_success for DCP vs. EDCP
- [ ] Analyse: does EDCP's richer structure provide more information per Fourier label?
- [ ] Implement plot_dcp_vs_edcp comparison figure
- [ ] **Milestone:** DCP vs. EDCP comparison complete

### Deliverables

| Deliverable | Location | Description |
|-------------|----------|-------------|
| EDCP Engine | src/engines/edcp_engine.py | General EDCP state construction |
| Mod-halving engine | src/engines/mod_halving_engine.py | Toy Bai-style pipeline |
| Comparison data | results/raw/edcp_comparison/ | EDCP truncation + comparison results |
| Comparison plots | results/figures/edcp_comparison/ | DCP vs. EDCP figures |

### Exit Criteria

- [ ] EDCP state verification passes for tested instances
- [ ] Modulus halving produces correct modulus reduction
- [ ] DCP vs. EDCP comparison plots are generated

---

## Phase Dependency Summary

```text
Phase 1 (Foundation)
   |
   +---> Phase 2 (Core Pipeline)
            |
            +---> Phase 3 (Analysis & Statistics)
            |       |
            |       +---> Phase 4 (Sample Complexity)
            |       |
            |       +---> Phase 5 (Noise & Scaling)
            |
            +---> Phase 6 (EDCP)
                    |
                    +---> Phase 5 (uses EDCP for comparison)
```

Phases 4, 5, and 6 are **independent of each other** and can be developed in parallel once Phase 3 is complete. Phase 6 can begin earlier (after Phase 2) if the EDCP engine is needed sooner.

---

## Timeline Estimation

### Single Developer, Part-Time (~20 hours/week)

| Phase | Duration | Cumulative | Key Milestone |
|-------|----------|------------|---------------|
| **Phase 1** | 2 weeks | Week 2 | DCP state + QFT verified |
| **Phase 2** | 2 weeks | Week 4 | First truncation data |
| **Phase 3** | 1.5 weeks | Week 5.5 | Core figures and MI analysis |
| **Phase 4** | 1 week | Week 6.5 | Sample-complexity curves |
| **Phase 5** | 1 week | Week 7.5 | Noise + scaling complete |
| **Phase 6** | 1.5 weeks | Week 9 | EDCP + comparison |
| **Buffer** | 1 week | Week 10 | Integration, bug fixes, documentation |

### Critical Path

```text
Phase 1 --> Phase 2 --> Phase 3 --> {Phase 4, Phase 5, Phase 6} --> Done
  2 wk        2 wk       1.5 wk      parallel: max 1.5 wk        1 wk buffer
                     
                     Total: ~8-10 weeks
```

### Parallelisation Opportunities

If multiple developers are available:

| Work Stream | Can Start After | Assigned To |
|-------------|-----------------|-------------|
| Phase 1 (DCP + QFT) | Immediately | Developer A |
| Phase 1 (config + infra) | Immediately | Developer B |
| Phase 4 (sample complexity) | Phase 3 | Developer B |
| Phase 5 (noise + scaling) | Phase 3 | Developer C |
| Phase 6 (EDCP) | Phase 2 | Developer C |

With 2 developers, the timeline reduces to approximately **6–7 weeks**.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Modular adder circuit is incorrect for large N | Medium | High | Verify against Qiskit's built-in adders; test exhaustively for small N |
| State-vector simulation too slow for N=64 | Low | Medium | N=64 uses only 7 qubits (128 amplitudes) — well within capability |
| MI estimation is inaccurate | Low | High | Use exact computation (not estimation) for our small N range |
| Recovery engine produces wrong results | Medium | High | Strong testing: verify full-info recovery is near-perfect for small N |
| Phase 6 EDCP is more complex than anticipated | Medium | Medium | Phase 6 is explicitly secondary; core results do not depend on it |
| Sweep computation takes too long | Low | Low | Total ~1.5M trials at ~1ms each = ~25 min; can parallelise across cores |

---

## Minimum Viable Research Product (MVRP)

If time is constrained, the **minimum publishable result** requires only:

- **Phase 1** (verified DCP + QFT)
- **Phase 2** (core pipeline with truncation)
- **Phase 3** (MI analysis + plots)

This produces:

1. Recovery probability vs. truncation level for N \u2208 {4, 8, 16, 32}
2. Mutual information I(S; Y_k) vs. truncation level
3. Information-loss characterisation
4. Posterior distribution visualisations

Phases 4–6 are **valuable extensions** that strengthen the paper but are not required for the core contribution.

---

## What Each Phase Enables

```text
After Phase 1:  "We can construct and verify DCP quantum states and apply QFT."

After Phase 2:  "We can truncate Fourier information and measure its effect on
                secret recovery. Here is our first experimental data."

After Phase 3:  "Here is the quantitative information-theoretic characterisation
                of the information-recovery relationship. Here are the figures."

After Phase 4:  "Here is how sample complexity interacts with information loss.
                More samples partially compensate for less information."

After Phase 5:  "Here is how noise degrades recovery, and here is how the
                experiment scales with modulus size."

After Phase 6:  "Here is how EDCP compares to DCP, and here is a toy
                implementation of the Bai modulus-halving mechanism."
```

The narrative arc of the project is: **build \u2192 measure \u2192 quantify \u2192 extend \u2192 contextualise**.
