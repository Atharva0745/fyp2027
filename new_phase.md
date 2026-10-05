# DCP/EDCP Information-Recovery Phase Transition --- Research Roadmap

## 0. Research Objective

### Core research question

> **Can we experimentally characterize the information-recovery phase
> transition of DCP/EDCP under partial Fourier observations and noisy
> quantum measurements, and compare empirical recovery thresholds with
> known theoretical information-loss bounds?**

The project should be treated as an **experimental research and
attack-auditing framework**, not merely as another DCP implementation.

The central pipeline is:

``` text
DCP / EDCP instance
        ↓
Quantum state
        ↓
QFT / Fourier sampling
        ↓
Fourier labels
        ↓
Partial observation + noise
        ↓
DCP/Simon-style attack engine
        ↓
Information analysis
        ↓
Secret recovery
        ↓
Phase-transition measurement
        ↓
Comparison with theoretical bounds
```

Your existing framework already contains DCP/EDCP construction, QFT,
Fourier-label truncation, noise injection, secret recovery, Bayesian
recovery, mutual information, experiment sweeps, and statistical
analysis. Therefore, the goal is to **extend and audit the existing
framework rather than rewrite it**.

> Source: current project roadmap and uploaded implementation notes.
> fileciteturn2file0L47-L63

------------------------------------------------------------------------

# Phase 0 --- Freeze and Verify the Existing Baseline

## Goal

Create a stable baseline before adding the new research components.

## Tasks

-   Freeze the current DCP/EDCP implementation.

-   Create a Git tag:

    ``` bash
    git tag v1-dcp-baseline
    ```

-   Run the complete existing test suite.

-   Record:

    -   Python version
    -   Qiskit version
    -   simulator version
    -   dependency versions
    -   operating system
    -   random seeds
    -   experiment configuration

-   Save a baseline experiment configuration.

-   Make all existing tests reproducible.

## Deliverable

``` text
DCP Framework v1
        ↓
Verified baseline
        ↓
Research/Audit Framework v2
```

### Acceptance criterion

All existing tests pass and the baseline results can be reproduced from
a clean environment.

------------------------------------------------------------------------

# Phase 1 --- Reproduce the DCP Input Model

## Goal

Implement the exact DCP sample model used by the target algorithm.

A DCP sample has the form:

\[ \|`\psi`{=tex}`\rangle `{=tex}= `\frac{1}{\sqrt{2}}`{=tex}
`\left`{=tex}( \|0,x`\rangle`{=tex}+
\|1,x+d`\bmod `{=tex}N`\rangle`{=tex} `\right`{=tex}) \]

where:

-   \(N\) = modulus
-   \(d\) = hidden secret
-   \(x\) = random offset

The Simon paper additionally considers faulty samples.

## Implement

``` python
generate_dcp_sample(
    N,
    d,
    x,
    faulty_probability
)
```

Return:

``` python
{
    "state": state,
    "x": x,
    "secret": d,
    "faulty": False
}
```

## Tests

For example:

``` text
N = 16
d = 5
x = 7
```

verify:

\[ \|`\psi`{=tex}`\rangle `{=tex}= `\frac{1}{\sqrt2}`{=tex}
(\|0,7`\rangle `{=tex}+ \|1,12`\rangle`{=tex}) \]

## Deliverable

A validated DCP sample generator supporting both:

-   ideal samples
-   faulty samples

------------------------------------------------------------------------

# Phase 2 --- Reproduce the QFT/Fourier Sampling Stage

## Goal

Verify that the Fourier measurement stage preserves the expected
secret-dependent phase information.

The relevant phase relationship is:

\[ `\phi`{=tex}(y)=e\^{2`\pi `{=tex}i d y/N} \]

## Implement

``` python
fourier_sample(dcp_state)
```

Return:

``` python
{
    "y": y,
    "phase": phase,
    "probability": probability
}
```

## Tests

Verify:

1.  QFT output distribution.
2.  Fourier label range.
3.  Relative phase.
4.  Agreement with theoretical phase.
5.  Reproducibility under fixed random seeds.

## Deliverable

A standalone Fourier-sampling module with unit tests.

------------------------------------------------------------------------

# Phase 3 --- Implement the Subset-Sum Layer

## Goal

Implement the subset-sum processing required by the Simon-style DCP
attack.

The relevant quantity is:

\[ z=`\sum`{=tex}\_i b_i y_i \]

where (b_i) are sample bits and (y_i) are Fourier labels.

## Implement

``` python
class SubsetSumEngine:

    def compute_subset_sum(
        self,
        y_values,
        b_register
    ):
        ...

    def measure_low_bits(
        self,
        z,
        bits_to_keep
    ):
        ...
```

## Track

-   Fourier labels (y_i)
-   sample bits (b_i)
-   subset sum (z)
-   lower measured bits
-   remaining high-order bit(s)

## Deliverable

A verified subset-sum engine whose outputs can be compared against
brute-force calculations for small (N).

------------------------------------------------------------------------

# Phase 4 --- Implement Group Construction

## Goal

Implement the grouping mechanism used by the Simon-style attack.

Groups contain approximately:

\[ c`\log `{=tex}n \]

samples.

## Implement

``` python
create_groups(
    samples,
    group_size
)
```

For each group calculate:

\[ r_j=`\sum`{=tex}\_{i`\in `{=tex}g_j} b_i y_i \]

and extract the relevant measurement information.

## Store

``` text
group_id
sample_indices
r_j
s_j
measurement_result
belongs_to_A
belongs_to_B
```

## Deliverable

A deterministic and testable `GroupAnalysis` component.

------------------------------------------------------------------------

# Phase 5 --- Implement the A/B Partition

## Goal

Implement the partition of groups into the two sets used by the attack.

``` text
All groups
    │
    ├── A
    │
    └── B
```

The analysis must record:

-   (\|A\|)
-   (\|B\|)
-   group distributions
-   measurement outcomes
-   partition rule
-   random seed, if applicable

## Important research rule

**Do not assume the theoretical independence claim is correct.**

The implementation must expose enough information to experimentally test
it.

## Deliverable

A reproducible A/B partition module with complete experiment logging.

------------------------------------------------------------------------

# Phase 6 --- Test the Critical Independence Assumption

## Goal

This is one of the most important research phases.

Test whether the contribution from (B) behaves similarly under the two
values of the relevant (h\^\*) bit.

Compare:

\[ P(B`\mid `{=tex}h\^\*=0) \]

with:

\[ P(B`\mid `{=tex}h\^\*=1) \]

## Metrics

### Total variation distance

\[ D\_{TV}(P_0,P_1) \]

### KL divergence

\[ D\_{KL}(P_0`\Vert `{=tex}P_1) \]

### Mutual information

\[ I(H\^\*;B) \]

## Interpretation

If:

\[ I(H\^\*;B)`\approx0`{=tex} \]

then the observed data is consistent with weak dependence.

If:

\[ I(H\^\*;B)\>0 \]

then the data indicates that (B) contains information about (H\^\*),
requiring further theoretical analysis.

## Experimental design

Repeat the experiment across:

-   different (N)
-   different sample counts
-   different group sizes
-   different faulty-sample rates
-   different Fourier truncation levels

## Deliverable

A statistical report showing whether the observed independence behavior
is consistent across parameter regimes.

------------------------------------------------------------------------

# Phase 7 --- Implement Secret-Bit Recovery

## Goal

Recover one secret bit first.

Implement:

``` python
recover_secret_bit(...)
```

Then implement complete recovery:

``` python
recover_secret(
    samples,
    N
)
```

## Output

``` python
{
    "d_hat": recovered_secret,
    "correct": True,
    "confidence": confidence,
    "iterations": iterations
}
```

## Experiments

Measure:

\[ P(`\hat `{=tex}d=d) \]

for increasing numbers of samples.

## Deliverable

A validated secret-recovery engine.

------------------------------------------------------------------------

# Phase 8 --- Add Partial Fourier Observations

## Goal

Turn your existing Fourier truncation functionality into the central
experimental variable.

Let:

\[ k = `\text{number of retained Fourier bits}`{=tex} \]

Run:

``` text
k = n
k = n-1
k = n-2
...
k = 1
```

Test both:

-   MSB retention
-   LSB retention

## Measure

For every (k):

\[ P\_{`\text{recovery}`{=tex}}(k) \]

\[ I(D;Y_k) \]

\[ R_I = `\frac{I(D;Y_k)}{I(D;Y)}`{=tex} \]

\[ `\text{sample complexity}`{=tex}(k) \]

## Deliverable

A dataset showing how secret recovery changes as Fourier information is
removed.

------------------------------------------------------------------------

# Phase 9 --- Add Noise and Faulty Samples

## Goal

Measure robustness under increasingly noisy observations.

Use two distinct models.

## A. Faulty DCP samples

Sweep the faulty-sample probability:

``` text
0
1/log(n)
1/(2 log(n))
1/(4 log(n))
...
```

## B. Measurement/bit noise

Sweep:

``` text
epsilon = 0
0.01
0.05
0.10
0.20
...
```

## Important distinction

Keep the following experiments separate:

``` text
Partial Fourier information
        ≠
Faulty DCP input samples
        ≠
Quantum measurement/readout noise
        ≠
Classical bit-flip noise
```

This separation is essential for interpreting the results.

## Deliverable

A multidimensional noise/recovery dataset:

\[ P\_{`\text{recovery}`{=tex}}(k,`\epsilon`{=tex},p_f,m,N) \]

------------------------------------------------------------------------

# Phase 10 --- Build the Information-Theoretic Analysis

## Goal

Connect information retention directly to secret recovery.

Calculate:

\[ I(D;Y) \]

for the complete observation.

Then:

\[ I(D;Y_k) \]

for partial Fourier observations.

Calculate the information-retention ratio:

\[ R_I = `\frac{I(D;Y_k)}`{=tex} {I(D;Y)} \]

Compare it with:

\[ P(`\hat `{=tex}D=D) \]

## Core plots

Generate:

1.  Mutual information vs retained Fourier bits.
2.  Recovery probability vs retained Fourier bits.
3.  Recovery probability vs mutual information.
4.  Information-retention ratio vs recovery probability.
5.  Sample complexity vs retained information.

## Deliverable

A reproducible information-theoretic analysis module.

------------------------------------------------------------------------

# Phase 11 --- Locate the Information-Recovery Phase Transition

## Goal

Find the empirical critical point where recovery changes from unreliable
to reliable.

Define:

\[ k_c \]

such that:

\[ P(`\hat `{=tex}d=d)`\geq 0.90`{=tex} \]

or use another threshold that is justified in the paper.

Conceptually:

``` text
Recovery
probability

1.0 |                         ███████
    |                     ████
    |                  ███
0.5 |--------------████
    |           ███
    |        ███
0.0 |████████
    +--------------------------------
       low              high
          Fourier information
                ↑
               k_c
```

## Also determine

\[ `\epsilon`{=tex}\_c \]

for the critical noise level.

And:

\[ p\_{f,c} \]

for the critical faulty-sample probability.

## Repeat for

-   different (N)
-   different sample counts
-   DCP
-   EDCP
-   MSB truncation
-   LSB truncation

## Deliverable

Empirical phase diagrams such as:

\[ (k,`\epsilon`{=tex})`\rightarrow `{=tex}P\_{`\text{recovery}`{=tex}}
\]

and:

\[ (k,m)`\rightarrow `{=tex}P\_{`\text{recovery}`{=tex}} \]

------------------------------------------------------------------------

# Phase 12 --- Compare With Theoretical Information-Loss Bounds

## Goal

This is where the experimental work connects to current theory.

A recent 2026 paper by Gupte, Ragavan, and Zhandry establishes a no-go
theorem for a broad Fourier-sampling/subset-sum template for DCP. It
states that after the subset-sum measurement, discarding
(`\omega`{=tex}(`\log `{=tex}n)) bits from each Fourier label prevents
successful DCP recovery for that class of algorithms. The paper applies
this result to Simon's 2026 algorithm. citeturn0academia6

Your experiment should therefore compare:

``` text
THEORETICAL BOUND
        ↓
Expected information-loss barrier
        │
        │ compare
        ↓
EMPIRICAL THRESHOLD
        ↓
Observed recovery transition
```

## Questions

1.  Does empirical recovery deteriorate before the theoretical barrier?
2.  Does recovery remain possible close to the theoretical boundary?
3.  Does the finite-(N) transition approach the asymptotic theoretical
    prediction?
4.  How does sample count change the transition?
5.  How does noise shift the transition?

## Deliverable

A theory-vs-experiment comparison.

------------------------------------------------------------------------

# Phase 13 --- Finite-Size Scaling

## Goal

Determine whether the observed phase transition is a genuine scalable
phenomenon or merely a small-instance artifact.

Sweep:

``` text
n = 4
n = 6
n = 8
n = 10
n = 12
...
```

where computationally feasible.

For each (n), record:

-   critical (k_c)
-   critical noise (`\epsilon`{=tex}\_c)
-   critical faulty rate
-   sample complexity
-   recovery probability
-   mutual information
-   runtime

## Analyze

Estimate:

\[ k_c(n) \]

and:

\[ m_c(n) \]

where (m_c) is the number of samples required for reliable recovery.

## Important

Do not claim polynomial scaling merely because small experiments run
quickly.

Separate:

-   empirical runtime scaling
-   theoretical complexity claims

## Deliverable

Finite-size scaling plots and fitted empirical models.

------------------------------------------------------------------------

# Phase 14 --- Ideal vs Noisy Quantum Simulation

## Goal

Only after the ideal algorithm and information experiments work,
introduce realistic quantum noise.

Architecture:

``` text
              Attack
                │
        ┌───────┴───────┐
        ↓               ↓
 Ideal simulator    Noisy simulator
        │               │
        │          ┌────┼────┐
        │          ↓    ↓    ↓
        │      depolarizing
        │      readout error
        │      bit-flip error
        │
        └───────┬───────┘
                ↓
          Compare recovery
```

## Measure

-   ideal recovery
-   noisy recovery
-   circuit depth
-   qubit count
-   noise sensitivity
-   information loss due specifically to hardware-style noise

## Deliverable

A realistic robustness study.

------------------------------------------------------------------------

# Phase 15 --- DCP Scaling and Attack Audit

## Goal

Turn the implementation into a neutral attack-auditing framework.

Create a table:

  Claim                        Experiment                  Metric              Result
  ---------------------------- --------------------------- ------------------- --------
  Subset-sum behavior          Distribution experiment     TV/KL               ?
  A/B behavior                 Partition experiment        Statistics          ?
  Independence assumption      Conditional distributions   MI                  ?
  Secret-bit recovery          Recovery experiment         Success rate        ?
  Fault tolerance              Fault sweep                 Critical rate       ?
  Partial Fourier robustness   Truncation sweep            (k_c)               ?
  Information barrier          MI experiment               (I(D;Y_k))          ?
  Scaling                      Size sweep                  Runtime/resources   ?

## Scientific principle

Do **not** design the experiments to prove or disprove Simon.

Design them to answer:

> **Under which parameter regimes does the proposed information flow
> actually preserve enough information for secret recovery?**

The current literature is actively debating Simon's algorithm. Simon's
ePrint entry notes a preprint claiming that the algorithm cannot work,
while Guo and Yang identify a surviving hypothesis concerning
independence of the partition from the measured string.
citeturn0search0turn0academia7

------------------------------------------------------------------------

# Phase 16 --- Extend the Framework to EDCP

## Goal

After DCP experiments are validated, extend the same methodology to
EDCP.

Your current framework already contains the EDCP state:

\[ \|`\psi`{=tex}`\rangle `{=tex}= `\sum`{=tex}\_j
`\chi`{=tex}(j)\|j`\rangle`{=tex} \|x+js`\bmod `{=tex}N`\rangle`{=tex}
\]

## Apply the same experiment matrix

``` text
EDCP
 │
 ├── Full Fourier information
 │
 ├── Partial Fourier information
 │
 ├── Faulty samples
 │
 ├── Measurement noise
 │
 ├── Mutual information
 │
 ├── Secret recovery
 │
 └── Phase transition
```

## Research question

> **Does EDCP exhibit the same information-recovery transition as DCP,
> or does the generalized state structure change the threshold?**

## Deliverable

DCP vs EDCP comparative results.

------------------------------------------------------------------------

# Phase 17 --- Connect Results to Lattice-Based Cryptography

## Goal

Only after the DCP/EDCP experiments are validated should the project
investigate cryptographic implications.

The motivation is that DCP/EDCP have known connections to lattice
problems and LWE. Simon's paper claims consequences for lattice problems
if its DCP algorithm is correct, while earlier work established
important reductions involving DCP/eDCP and lattice problems.

## Pipeline

``` text
DCP
 ↓
EDCP
 ↓
Relevant reduction
 ↓
Lattice problem
 ↓
LWE / SVP implications
```

## Critical rule

Do not claim:

> "Our DCP experiment breaks LWE/ML-KEM."

Instead report:

``` text
DCP experimental result
        ↓
Applicable reduction?
        ↓
Parameter mapping
        ↓
Conditional cryptographic implication
```

## Deliverable

A careful section explaining what the experimental results do and do not
imply for post-quantum cryptography.

------------------------------------------------------------------------

# Phase 18 --- Reproducibility Infrastructure

## Goal

Make every research result reproducible.

Every experiment should record:

``` json
{
  "experiment_id": "...",
  "seed": 12345,
  "N": 64,
  "secret": 17,
  "samples": 128,
  "fourier_bits": 6,
  "noise": 0.05,
  "fault_rate": 0.01,
  "algorithm": "...",
  "simulator": "...",
  "software_version": "...",
  "result": "..."
}
```

## Add

-   deterministic seeds
-   configuration files
-   experiment IDs
-   result versioning
-   raw result storage
-   statistical confidence intervals
-   repeated trials
-   automatic plotting
-   experiment manifests

## Deliverable

A researcher can reproduce any published graph from a configuration
file.

------------------------------------------------------------------------

# Phase 19 --- Production Architecture

Only after the research engine is stable.

## Architecture

``` text
                    Frontend
                       │
                       ▼
                    FastAPI
                       │
              ┌────────┴────────┐
              ↓                 ↓
        Experiment API      Results API
              │                 │
              ▼                 ▼
         Job Queue          PostgreSQL
              │
              ▼
       Quantum Workers
              │
       ┌──────┼──────┐
       ↓      ↓      ↓
      DCP    EDCP   Attack
                     Engine
       └──────┼──────┘
              ↓
       Analysis Engine
              │
      ┌───────┼────────┐
      ↓       ↓        ↓
     MI    Recovery  Statistics
              │
              ▼
        Research Report
```

## Suggested technologies

``` text
Backend       FastAPI
Workers       Celery/RQ/other queue
Database      PostgreSQL
Cache         Redis
Quantum       Qiskit/Aer
Container     Docker
Experiments   Python
Analysis      NumPy/SciPy/Pandas
Plots         Matplotlib
Frontend      React
```

------------------------------------------------------------------------

# Phase 20 --- Final Research Paper

## Suggested paper structure

### 1. Introduction

Explain:

-   DCP/EDCP
-   Fourier sampling
-   information loss
-   motivation for studying partial observations and noise

### 2. Background

Cover:

-   DCP
-   EDCP
-   QFT
-   Fourier labels
-   subset-sum processing
-   secret recovery

### 3. Related Work

Include:

-   Regev's DCP approach
-   Simon's 2026 algorithm
-   Guo & Yang's analysis
-   Gupte, Ragavan & Zhandry's information-loss barrier
-   relevant DCP/EDCP/lattice literature

### 4. Experimental Framework

Describe:

-   simulator
-   DCP generator
-   QFT
-   Fourier truncation
-   noise models
-   recovery engine
-   information metrics

### 5. Experimental Methodology

Define:

\[
N,`\quad `{=tex}k,`\quad `{=tex}m,`\quad `{=tex}`\epsilon`{=tex},`\quad `{=tex}p_f
\]

and explain repetitions, confidence intervals, and statistical tests.

### 6. Information-Recovery Phase Transition

Present:

\[ P\_{`\text{recovery}`{=tex}}(k) \]

and:

\[ I(D;Y_k) \]

### 7. Noise Robustness

Present:

\[ P\_{`\text{recovery}`{=tex}}(`\epsilon`{=tex}) \]

and:

\[ P\_{`\text{recovery}`{=tex}}(p_f) \]

### 8. Theory vs Experiment

Compare:

\[ `\text{Empirical }`{=tex} k_c \]

with the theoretical information-loss boundary.

### 9. DCP vs EDCP

Compare the two problems under equivalent experimental conditions.

### 10. Scaling

Report:

-   sample complexity
-   runtime
-   qubits
-   circuit depth
-   empirical scaling

### 11. Limitations

Explicitly discuss:

-   simulator size limits
-   finite-size effects
-   statistical uncertainty
-   difference between simulation and real quantum hardware
-   difference between DCP experiments and concrete cryptographic
    attacks

### 12. Conclusion

Summarize what the experiments establish and what remains theoretical.

------------------------------------------------------------------------

# Implementation Order --- Do Not Build Everything at Once

## Milestone 1 --- Baseline

``` text
DCP
 ↓
QFT
 ↓
Fourier samples
```

### Success condition

Existing implementation remains correct.

------------------------------------------------------------------------

## Milestone 2 --- Attack Mechanism

``` text
Fourier samples
 ↓
Subset sum
 ↓
Groups
 ↓
A/B partition
```

### Success condition

All intermediate states can be inspected and verified on small
instances.

------------------------------------------------------------------------

## Milestone 3 --- Recovery

``` text
A/B mechanism
 ↓
Secret bit
 ↓
Complete secret
```

### Success condition

Recovery works on small ideal instances.

------------------------------------------------------------------------

## Milestone 4 --- Audit

``` text
A/B mechanism
 ↓
Independence tests
 ↓
MI / TV / KL
```

### Success condition

The critical assumptions can be quantitatively tested.

------------------------------------------------------------------------

## Milestone 5 --- Information Loss

``` text
Full Fourier labels
        ↓
Partial Fourier labels
        ↓
Recovery
        ↓
MI
```

### Success condition

A measurable recovery transition appears.

------------------------------------------------------------------------

## Milestone 6 --- Noise

``` text
Partial information
        +
Faulty samples
        +
Measurement noise
        ↓
Recovery phase diagram
```

### Success condition

Critical thresholds can be estimated with confidence intervals.

------------------------------------------------------------------------

## Milestone 7 --- Theory Comparison

``` text
Known theoretical bound
        ↓
Empirical threshold
        ↓
Finite-size analysis
```

### Success condition

A quantitative theory-vs-experiment comparison is produced.

------------------------------------------------------------------------

## Milestone 8 --- EDCP

``` text
DCP framework
     ↓
EDCP framework
     ↓
Compare transitions
```

### Success condition

The same experimental methodology works for both problems.

------------------------------------------------------------------------

## Milestone 9 --- Scaling

``` text
N ↑
samples ↑
noise ↑
experiments ↑
```

### Success condition

Finite-size scaling and resource measurements are reported.

------------------------------------------------------------------------

## Milestone 10 --- Research Package

``` text
Research engine
      ↓
Reproducible configs
      ↓
Automated experiments
      ↓
Plots
      ↓
Statistical reports
      ↓
Paper
```

------------------------------------------------------------------------

# Minimum Viable Research Contribution

If time is limited, do **not** implement every phase before starting the
research analysis.

The minimum strong version is:

``` text
DCP
 ↓
QFT
 ↓
Partial Fourier observations
 ↓
Noise
 ↓
Secret recovery
 ↓
Mutual information
 ↓
Phase transition
 ↓
Theory comparison
```

The most important experimental result should be a relationship of the
form:

\[ `\boxed{
P_{\mathrm{recovery}}
=
f\left(
\frac{k}{n},
m,
\epsilon,
N
\right)
}`{=tex} \]

together with:

\[ `\boxed{
I(D;Y_k)
\longleftrightarrow
P_{\mathrm{recovery}}
}`{=tex} \]

and a comparison against known theoretical information-loss barriers.

------------------------------------------------------------------------

# Important Novelty Position

Do **not** claim that the theoretical information-loss question itself
is completely unexplored.

A September 30, 2026 paper by Gupte, Ragavan, and Zhandry already proves
a theoretical Fourier-label information-loss barrier for a broad class
of DCP algorithms and applies it to Simon's algorithm.
citeturn0academia6

The potentially novel contribution is the **experimental
finite-parameter characterization**:

> **Measure where secret recovery actually transitions under partial
> Fourier observations, sample limitations, and noise, and compare the
> empirical transition with the theoretical information-loss boundary.**

That should be presented as an **empirical/experimental investigation of
finite-size behavior**, not as a claim that the underlying theoretical
barrier is new.

------------------------------------------------------------------------

# Final Target

The finished project should answer:

1.  **How much Fourier information is required for recovery?**
2.  **Does a sharp recovery phase transition exist?**
3.  **How does the transition depend on sample count?**
4.  **How does noise move the transition?**
5.  **How does faulty-sample probability move the transition?**
6.  **Does the empirical transition approach the theoretical
    information-loss boundary?**
7.  **Does EDCP behave differently from DCP?**
8.  **Which assumptions of the target attack remain empirically
    supported?**
9.  **What are the computational scaling limits?**
10. **What, if anything, follows for lattice-based cryptography?**

The end product is therefore:

``` text
                DCP / EDCP
                    │
                    ▼
             Fourier Information
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
      Information          Noise
        loss                 │
          │                   │
          └─────────┬─────────┘
                    ↓
             Secret Recovery
                    │
                    ▼
          Phase Transition
                    │
                    ▼
        Theory vs Experiment
                    │
                    ▼
             DCP vs EDCP
                    │
                    ▼
        Cryptographic Implications
```

## First thing to implement

**Start with Phase 1 → Phase 2 → Phase 3.**

Do not build the frontend, database, Docker production stack, or EDCP
experiments yet.

The first concrete research checkpoint is:

> **Given a known DCP secret (d), can our implementation reproduce the
> Fourier labels and subset-sum behavior expected by the target
> algorithm on small instances?**

Once that passes, proceed to the A/B partition and independence
experiment.
