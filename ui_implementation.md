# Interactive UI Implementation Log

**Project:** DCP/EDCP Quantum Information Analysis Framework
**Feature:** "See the lattice problem being solved" visualizer
**Last updated:** 2026-10-05

## Purpose

This document tracks the UI roadmap and the implementation completed in this
repository. The visualizer calls the existing project engines; it does not
replace or independently implement the DCP/QFT/inference math.

## Verified measurement convention

`InformationEngine` samples the Fourier label and measures the flag qubit in
the Hadamard (X) basis from the post-QFT statevector. For DCP:

```text
P(y, b | s) = (1 / (2N)) * (1 + (-1)^b cos(2πsy/N))
```

It then independently flips the n bits of `y` and the flag bit `b` with
probability `epsilon`, and truncates the noisy `y`. The UI displays this joint
convention and the post-noise, post-truncation observation that the recovery
engine actually receives.

## Phase V1 — Engine trace

**Status: Implemented**

- Added `TraceEvent` and `Orchestrator.run_traced()`.
- Trace stages include the first prepared state and QFT, followed by each
  sample, each sequential Bayesian posterior, and the recovery verdict.
- Trace payloads expose basis amplitudes, Fourier distribution and relative
  phases, sample labels, observed flag bit, y-bit flips, flag-bit flip, and
  posterior probabilities.
- Optional `offset_override` fixes the first random offset for demonstrations
  without changing the measurement random stream.
- The existing `Orchestrator.run()` and traced execution use the same pipeline.
  Intermediate posterior calculations use an independent RNG so tracing cannot
  alter sampled outcomes or secret-recovery tie breaks.
- Tests compare normal and traced recovered outputs under Bayesian, brute-force,
  maximum-likelihood, and bitwise strategies, and compare the final Bayesian
  posterior to the recovery engine's posterior.

## Phase V2 — Single-trial visualizer

**Status: Implemented**

- Added a Streamlit app in `app.py`, using Plotly and the existing engines.
- Sidebar controls: `N`, `s`, first offset `x`, retained bits `k`, sample count
  `m`, bit-flip noise `epsilon`, truncation mode, recovery strategy, and seed.
- Step/replay controls and optional auto-play show the circuit and prepared
  amplitudes, Fourier probabilities and secret-dependent phases, each
  measured/noisy/truncated label, Bayesian posterior evolution, and the final
  verdict with bit correctness.
- The posterior highlights the true secret and its DCP mirror candidate.
- UI dependencies are optional under the `ui` extra so the core simulator does
  not require Streamlit or Plotly.

## Phase V3 — Results dashboard and live validation

**Status: Implemented**

- Interactive charts load the archived truncation and sample-complexity CSV
  summaries from `results/aggregated/`.
- Added a reusable `plot_recovery_vs_samples()` figure helper for sample-
  complexity trial/summary data.
- Golden validation lets the user select one published core configuration,
  rerun it through `Orchestrator`, and compare the live exact-recovery estimate
  to the published 95% Wilson interval.
- An on-demand noise sweep runs five noise levels through `Orchestrator` and
  reports exact-recovery Wilson intervals.
- Live checks are explicitly run by the user; opening the dashboard does not
  start expensive experiment sweeps.

## Phase V4 — Advanced panels

**Status: Not started (optional roadmap items)**

- EDCP-specific trace and coefficient (`chi`) controls.
- Modulus-halving animation and story-mode presets.
- Dedicated mirror-likelihood explorer.

## Run the UI

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Validation

The focused automated checks are in `tests/test_visualizer_trace.py`. They
verify run equivalence, trace payloads, posterior agreement, offset validation,
and trace compatibility across the listed recovery methods. The Streamlit
`AppTest` runner was also used to verify app startup, trace generation, and
single-step playback. The full test suite passed: 71 tests.
