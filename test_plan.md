# End-to-End Test & Verification Plan

This document outlines a comprehensive test plan to verify the complete DCP/EDCP Quantum Information Analysis Framework. It covers unit testing, running each phase's master scripts, and verifying the expected numerical and visual outputs.

## 1. Automated Test Suite Verification

**Objective**: Verify the correctness of all core quantum primitives, engines, and utility functions using `pytest`.

**Command**:
```bash
.venv\Scripts\python -m pytest tests/ -v
```

**Expected Result**:
- Pytest should collect exactly 57 items (tests).
- All 57 tests should pass (`100% PASSED`).
- No warnings related to missing modules or incorrect quantum amplitudes should appear.
- Total execution time should be around 10-15 seconds depending on hardware.

## 2. Phase 2 & 3: Core Truncation Sweep and Plotting

**Objective**: Verify the baseline truncation pipeline and generation of core dashboards.

**Command 1 (Run Sweep)**:
```bash
.venv\Scripts\python scripts/run_sweep.py --config configs/dcp_truncation_sweep.yaml
```

**Expected Result 1**:
- The script executes 14 parameter configurations (combinations of $N \in \{4, 8, 16, 32\}$ and corresponding $k$ values).
- Each configuration uses `m=8` (Bayesian recovery) and `shots=1000`.
- Execution should take a few minutes (approx. 2-5 minutes).
- Outputs are saved to:
  - `results/raw/dcp_truncation_core/dcp_truncation_sweep.parquet`
  - `results/raw/dcp_truncation_core/dcp_truncation_sweep_metadata.json`

**Command 2 (Generate Plots)**:
```bash
.venv\Scripts\python scripts/plot_results.py
```

**Expected Result 2**:
- The script reads the parquet file generated above.
- Expected standard output will list the creation of multiple `.png` files under `results/figures/dcp_core/`.
- Expected artifacts generated:
  - `dcp_core_dashboard.png`: A 4-panel dashboard containing recovery probability, theoretical MI, information loss ratio, and average bitwise accuracy.
  - `bit_recovery_heatmap_N16.png` and `_N32.png`: Heatmaps showing sequential recovery degradation of LSBs as $k$ decreases.
  - The plots should empirically show $P_{\text{success}}$ dropping towards random guess baseline ($1/N$) as $k \to 1$, while remaining near $1.0$ for full $k=n$.

## 3. Phase 4: Sample Complexity Sweep

**Objective**: Verify that increasing the number of independent samples ($m$) compensates for information truncation.

**Command**:
```bash
.venv\Scripts\python scripts/run_sweep.py --config configs/dcp_sample_complexity.yaml
```

**Expected Result**:
- The script executes the grid defined for varying $m \in \{1, 2, 4, 8, 16\}$ across truncated $k$ values for $N=16$ and $N=32$.
- Outputs are saved to `results/raw/dcp_sample_complexity/dcp_sample_complexity.parquet`.
- (If `plot_sample_complexity.py` exists and is run): Produces line plots showing $P_{\text{success}}$ monotonically increasing with $m$, crossing the $95\%$ threshold for larger $m$ even at small $k$.

## 4. Phase 5: Noise Robustness, Scaling, and Combined Sweep

**Objective**: Verify Phase 5 implementations including noise injection (`epsilon`), scaling up to $N=64$, and multidimensional combined sweeps.

**Command**:
```bash
.venv\Scripts\python scripts/run_phase5.py
```

**Expected Result**:
- **Phase 5.1 (Noise)**: Executes `dcp_noise_sweep.yaml` (35 jobs) evaluating $P_{\text{success}}$ against $\epsilon \in \{0, 0.05, 0.1, 0.15, 0.2\}$. Saves to `results/raw/dcp_noise/`.
- **Phase 5.2 (Scaling)**: Executes `dcp_scaling_sweep.yaml` up to $N=64$. Outputs measurements for runtime, gates, and circuit depth. Saves to `results/raw/dcp_scaling/`.
- **Phase 5.3 (Combined)**: Executes `dcp_combined_sweep.yaml` (72 jobs varying $N, k, m, \epsilon$). Saves to `results/raw/dcp_combined/`.
- **Visuals generated**:
  - `results/figures/dcp_noise/recovery_vs_noise_N*.png`: Shows curves collapsing as $\epsilon \to 0.25$.
  - `results/figures/dcp_scaling/scaling_curves.png`: Log-log plots showing exponential runtime growth vs $n$ qubits (or polynomial vs $N$).
  - `results/figures/dcp_heatmaps/heatmap_k_vs_eps_N*.png` and `heatmap_k_vs_m_N*.png`: 2D heatmaps illustrating interactive parameter limits.

## 5. Phase 6: EDCP Comparison and Modulus Halving

**Objective**: Verify the generalized Error Distribution over Conjugate Phase (EDCP) engine and the toy Bai-style modulus-halving pipeline.

**Command**:
```bash
.venv\Scripts\python scripts/run_phase6.py
```

**Expected Result**:
- **Phase 6.3 (EDCP vs DCP Sweep)**: Executes `edcp_comparison_sweep.yaml` using the LWE-like 4-term $\chi$ configuration.
  - Automatically loads Phase 2 DCP data to produce the comparison plots.
  - Expected plots: `results/figures/edcp_comparison/dcp_vs_edcp_N16.png` and `dcp_vs_edcp_N32.png`.
  - The plots should overlay DCP (2-term, solid blue) with EDCP (4-term, orange squares). Due to a broader initial superposition in EDCP, information leakage per sample may vary, potentially requiring a higher $k$ or $m$ to achieve the same $P_{\text{success}}$ as DCP.
- **Phase 6.2 (Toy Halving)**: Console output will print 3 sequential halving iterations:
  - Iteration 1: $N=64 \rightarrow 32$, $s$ correctly guessed LSB and halves.
  - Iteration 2: $N=32 \rightarrow 16$, $s$ halves.
  - Iteration 3: $N=16 \rightarrow 8$, $s$ halves.
  - The output strictly verifies that the modulus halving wrapper can iterate structurally.
