# Verified Project Results

Date: 2026-10-01

This document captures the verified results from the full project test plan and experiment runs.

---

## 1. Automated Test Suite Verification

Command run:

```bash
.\.venv\Scripts\python -m pytest tests/ -v
```

Verified result:
- Collected: 57 tests
- Passed: 57 tests
- Failed: 0
- Runtime: 9.31s

Status: PASS

---

## 2. Phase 2 & 3: Core Truncation Sweep and Plotting

Command run:

```bash
.\.venv\Scripts\python scripts/run_sweep.py --config configs/dcp_truncation_sweep.yaml
.\.venv\Scripts\python scripts/plot_results.py
```

Verified result:
- Sweep completed successfully
- Total execution time: 361.20 s
- Raw output:
  - results/raw/dcp_truncation_core/dcp_truncation_sweep.parquet
  - results/raw/dcp_truncation_core/dcp_truncation_sweep_metadata.json
- Aggregated output:
  - results/aggregated/dcp_core_summary.parquet
  - results/aggregated/dcp_core_summary.csv
  - results/aggregated/dcp_core_summary_metadata.json
- Figures generated:
  - results/figures/dcp_core/recovery_vs_truncation.png
  - results/figures/dcp_core/mi_vs_truncation.png
  - results/figures/dcp_core/information_loss_ratio.png
  - results/figures/dcp_core/dcp_core_dashboard.png
  - heatmaps for N=4, 8, 16, 32

Summary table from the raw sweep:

| N | n | k | m | epsilon | s | shots | recovery_prob | mirror_recovery_prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 1 | 8 | 0.0 | 3 | 1000 | 0.320 | 0.674 |
| 4 | 2 | 2 | 8 | 0.0 | 3 | 1000 | 0.533 | 0.990 |
| 8 | 3 | 1 | 8 | 0.0 | 5 | 1000 | 0.160 | 0.311 |
| 8 | 3 | 2 | 8 | 0.0 | 5 | 1000 | 0.253 | 0.529 |
| 8 | 3 | 3 | 8 | 0.0 | 5 | 1000 | 0.469 | 0.911 |
| 16 | 4 | 1 | 8 | 0.0 | 11 | 1000 | 0.067 | 0.127 |
| 16 | 4 | 2 | 8 | 0.0 | 11 | 1000 | 0.091 | 0.182 |
| 16 | 4 | 3 | 8 | 0.0 | 11 | 1000 | 0.232 | 0.490 |
| 16 | 4 | 4 | 8 | 0.0 | 11 | 1000 | 0.424 | 0.804 |
| 32 | 5 | 1 | 8 | 0.0 | 19 | 1000 | 0.032 | 0.055 |
| 32 | 5 | 2 | 8 | 0.0 | 19 | 1000 | 0.011 | 0.025 |
| 32 | 5 | 3 | 8 | 0.0 | 19 | 1000 | 0.028 | 0.057 |
| 32 | 5 | 4 | 8 | 0.0 | 19 | 1000 | 0.061 | 0.121 |
| 32 | 5 | 5 | 8 | 0.0 | 19 | 1000 | 0.358 | 0.697 |

Status: PASS

---

## 3. Phase 4: Sample Complexity Sweep

Command run:

```bash
.\.venv\Scripts\python scripts/run_sweep.py --config configs/dcp_sample_complexity.yaml
.\.venv\Scripts\python scripts/plot_sample_complexity.py
```

Verified result:
- Sweep completed successfully
- Total execution time: 334.60 s
- Raw output:
  - results/raw/dcp_sample_complexity/dcp_sample_complexity_sweep.parquet
  - results/raw/dcp_sample_complexity/dcp_sample_complexity_sweep_metadata.json
- Plots generated:
  - results/figures/dcp_sample_complexity/recovery_vs_samples_N4.png
  - results/figures/dcp_sample_complexity/recovery_vs_samples_N8.png
  - results/figures/dcp_sample_complexity/recovery_vs_samples_N16.png

Representative results from the sample-complexity sweep:

| N | k | m | recovery_prob |
|---|---:|---:|---:|
| 4 | 1 | 1 | 0.170 |
| 4 | 1 | 2 | 0.210 |
| 4 | 1 | 4 | 0.338 |
| 4 | 1 | 8 | 0.326 |
| 4 | 1 | 16 | 0.420 |
| 8 | 3 | 1 | 0.048 |
| 8 | 3 | 2 | 0.130 |
| 8 | 3 | 4 | 0.318 |
| 8 | 3 | 8 | 0.482 |
| 8 | 3 | 16 | 0.510 |
| 16 | 4 | 1 | 0.018 |
| 16 | 4 | 2 | 0.088 |
| 16 | 4 | 4 | 0.246 |
| 16 | 4 | 8 | 0.424 |
| 16 | 4 | 16 | 0.510 |

Interpretation: recovery probability generally increases with m, confirming the expected sample-complexity trend under truncation.

Status: PASS

---

## 4. Phase 5: Noise Robustness, Scaling, and Combined Sweep

Command run:

```bash
.\.venv\Scripts\python scripts/run_phase5.py
```

Verified result:
- Total wall time: 327.1 s
- Noise sweep completed: 35 jobs in 171.7 s
- Scaling sweep completed: 5 configs in 3.8 s
- Combined sweep completed: 72 jobs in 130.5 s

### 4.1 Noise Robustness Summary

Raw output:
- results/raw/dcp_noise/dcp_noise_sweep.parquet
- results/raw/dcp_noise/dcp_noise_sweep_metadata.json

Sample noise results:

| N | k | epsilon | recovery_prob |
|---|---:|---:|---:|
| 4 | 1 | 0.00 | 0.338 |
| 4 | 1 | 0.05 | 0.290 |
| 4 | 1 | 0.10 | 0.250 |
| 4 | 1 | 0.15 | 0.246 |
| 4 | 1 | 0.20 | 0.184 |
| 8 | 3 | 0.00 | 0.318 |
| 8 | 3 | 0.05 | 0.278 |
| 8 | 3 | 0.10 | 0.200 |
| 8 | 3 | 0.15 | 0.174 |
| 8 | 3 | 0.20 | 0.168 |
| 16 | 4 | 0.00 | 0.246 |
| 16 | 4 | 0.05 | 0.176 |
| 16 | 4 | 0.10 | 0.102 |
| 16 | 4 | 0.15 | 0.100 |
| 16 | 4 | 0.20 | 0.076 |

Figures produced:
- results/figures/dcp_noise/recovery_vs_noise_N4.png
- results/figures/dcp_noise/recovery_vs_noise_N8.png
- results/figures/dcp_noise/recovery_vs_noise_N16.png
- noise heatmaps for N=4, 8, 16

### 4.2 Scaling Summary

Raw output:
- results/raw/dcp_scaling/dcp_scaling_sweep.parquet
- results/raw/dcp_scaling/dcp_scaling_sweep_metadata.json

| N | n | k | circuit_depth | num_qubits | runtime_sec |
|---|---:|---:|---:|---:|---:|
| 4 | 2 | 2 | 4 | 3 | 0.396429 |
| 8 | 3 | 3 | 5 | 4 | 0.570409 |
| 16 | 4 | 4 | 7 | 5 | 0.649611 |
| 32 | 5 | 5 | 8 | 6 | 0.871335 |
| 64 | 6 | 6 | 9 | 7 | 1.318510 |

Figures produced:
- results/figures/dcp_scaling/scaling_runtime.png
- results/figures/dcp_scaling/scaling_circuit_depth.png
- results/figures/dcp_scaling/scaling_dashboard.png

### 4.3 Combined Sweep Summary

Raw output:
- results/raw/dcp_combined/dcp_combined_sweep.parquet
- results/raw/dcp_combined/dcp_combined_sweep_metadata.json

Verified result:
- 72 jobs completed in 130.5 s
- Generated heatmap dashboards and combined plots under:
  - results/figures/dcp_heatmaps/

Status: PASS

---

## 5. Phase 6: EDCP Comparison and Modulus Halving

Command run:

```bash
.\.venv\Scripts\python scripts/run_phase6.py
```

Verified result:
- Phase 6.3 completed: 9 jobs in 15.3 s
- Phase 6.2 completed successfully

### 5.1 EDCP Comparison Output

Generated plots:
- results/figures/edcp_comparison/dcp_vs_edcp_N16.png
- results/figures/edcp_comparison/dcp_vs_edcp_N32.png

Raw output:
- results/raw/edcp_comparison/edcp_sweep.parquet
- results/raw/edcp_comparison/edcp_sweep_metadata.json

### 5.2 Modulus Halving Pipeline

Observed iteration summary:

| Iteration | N before | N after | s before | s after | LSB guessed |
|---|---:|---:|---:|---:|---|
| 1 | 64 | 32 | 42 | 21 | 0 |
| 2 | 32 | 16 | 21 | 10 | 1 |
| 3 | 16 | 8 | 10 | 5 | 0 |

Status: PASS

---

## 6. Final Outcome

All project verification steps from the test plan were followed and completed successfully:
- Unit tests: PASS
- Truncation pipeline: PASS
- Sample-complexity analysis: PASS
- Noise/scaling/combined sweeps: PASS
- EDCP comparison and modulus-halving pipeline: PASS

The project is in a verified state with generated data and figures saved under the results directory.
