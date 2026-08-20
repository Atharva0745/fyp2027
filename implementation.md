# Implementation — DCP/EDCP Quantum Information Analysis Framework

> **Version:** 1.0  
> **Date:** 2026-08-17  
> **Status:** Implementation Specification

---

## 1. Prerequisites and Environment Setup

### 1.1 Python Version

Python 3.11+ (required for modern type hints, `tomllib`, and performance improvements).

### 1.2 Core Dependencies

```text
qiskit>=1.0              # Quantum circuit construction and simulation
qiskit-aer>=0.14         # Aer simulator backends (statevector + qasm)
numpy>=1.26              # Numerical computing
scipy>=1.12              # Statistical distributions, signal processing
pandas>=2.2              # Tabular data management
matplotlib>=3.8           # Plotting
seaborn>=0.13            # Statistical visualisation
pyyaml>=6.0              # YAML configuration loading
pyarrow>=14.0            # Parquet file support (optional, pandas fallback)
pytest>=8.0              # Testing framework
tqdm>=4.66               # Progress bars for long sweeps
```

### 1.3 Installation

```bash
cd dcp-framework
pip install -r requirements.txt
pip install -e .
```

---

## 2. DCP State Construction — Detailed Implementation

### 2.1 Mathematical Foundation

The DCP state for modulus N, secret s, and random offset x is:

$$|\psi_{x,s}\rangle = \frac{1}{\sqrt{2}} \left( |0\rangle |x\rangle + |1\rangle |x+s \pmod{N}\rangle \right)$$

The data register requires n = ceil(log2(N)) qubits. The flag register is 1 qubit. Total: n+1 qubits.

### 2.2 Circuit Construction Algorithm

The DCP circuit is built in three stages:

**Stage 1 — Initialise data register to |x>:**

```python
def apply_x_initialisation(circuit, data_qubits, x, n):
    """Encode integer x into the data register using X gates."""
    for i in range(n):
        if (x >> i) & 1:
            circuit.x(data_qubits[i])
```

**Stage 2 — Create superposition on flag qubit:**

```python
circuit.h(flag_qubit)  # |0> -> (|0> + |1>) / sqrt(2)
```

**Stage 3 — Conditional modular addition:**

Apply ADD(s, mod N) to the data register, controlled by the flag qubit being |1>. This transforms the state to:

```
(|0>|x> + |1>|x+s mod N>) / sqrt(2)
```

### 2.3 Modular Addition Circuit

The modular addition ADD(a, N) computes (x + a) mod N on an n-qubit register.

**Algorithm:**

1. Construct a standard in-place adder for a on n+1 qubits (one overflow qubit)
2. Subtract N from the result (using two's complement addition)
3. If the result underflowed (overflow bit is 1), add N back
4. Uncompute the overflow qubit

**Implementation using Qiskit:**

```python
def modular_adder(circuit, data_qubits, ancilla_qubit, a, N):
    """
    In-place modular addition: |x> -> |(x + a) mod N>
    
    Args:
        circuit: QuantumCircuit to append gates to
        data_qubits: list of n qubits holding x
        ancilla_qubit: 1 ancillary qubit (for overflow detection)
        a: integer to add (0 <= a < N)
        N: modulus
    """
    n = len(data_qubits)
    
    # Step 1: Standard addition |x> -> |x + a> (with overflow)
    # Use ripple-carry adder
    for i in range(n):
        if (a >> i) & 1:
            for j in range(i, n):
                # CNOT chain for carry propagation
                circuit.cnot(data_qubits[j], ancilla_qubit)
                circuit.cx(ancilla_qubit, data_qubits[j])
    
    # Step 2: Subtract N |x+a> -> |x+a-N>
    # ... (two's complement subtraction, similar structure)
    
    # Step 3: Conditional add-back
    # If overflow qubit indicates negative result, add N back
    # ... (controlled addition)
    
    # Step 4: Uncompute overflow
    # ...
```

> **Note:** For production code, use Qiskit's built-in `DraperQFTAdder` or implement the circuit using the QFT-based addition approach which is more efficient. The ripple-carry approach above is shown for conceptual clarity. The final implementation will use the QFT-based adder from Qiskit's circuit library where available, and a hand-optimised version otherwise.

### 2.4 Conditional Application

The modular adder is applied only when the flag qubit is |1>:

```python
# For each gate in the modular adder, wrap with control on flag_qubit
# Qiskit supports: circuit.append(gate.control(1), [flag] + target_qubits)
```

### 2.5 Verification

After constructing the circuit, verify correctness by:

1. Running state-vector simulation
2. Checking that non-zero amplitudes occur only at indices corresponding to |0,x> and |1,(x+s) mod N>
3. Checking that both amplitudes have magnitude exactly 1/sqrt(2)
4. Checking relative phase is 0 (both amplitudes are real and positive)

```python
def verify_dcp_state(state: Statevector, x: int, s: int, N: int, n: int) -> bool:
    """Verify that the statevector matches the expected DCP state."""
    expected = np.zeros(2 * N)
    idx_0 = 0 * N + x                    # |0>|x>
    idx_1 = 1 * N + ((x + s) % N)       # |1>|x+s>
    expected[idx_0] = 1.0 / np.sqrt(2)
    expected[idx_1] = 1.0 / np.sqrt(2)
    
    actual = np.abs(state.data)**2
    # Check support
    for i in range(2 * N):
        if i not in (idx_0, idx_1):
            assert actual[i] < 1e-10, f"Unexpected amplitude at index {i}"
    # Check magnitudes
    assert abs(actual[idx_0] - 0.5) < 1e-10
    assert abs(actual[idx_1] - 0.5) < 1e-10
    return True
```

---

## 3. QFT Implementation

### 3.1 Standard QFT Circuit

The QFT on n qubits applies the transformation:

$$|x\rangle \rightarrow \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{2\pi i x y / N} |y\rangle$$

**Gate decomposition:**

```python
from qiskit.circuit.library import QFT as QFTGate

def apply_qft(circuit, qubits, inverse=False):
    """Apply QFT (or inverse QFT) to the given qubits."""
    n = len(qubits)
    qft = QFTGate(num_qubits=n, inverse=inverse, do_swaps=True)
    circuit.append(qft, qubits)
```

### 3.2 Applying QFT to the DCP State

After QFT on the data register only, the DCP state becomes:

$$\frac{1}{\sqrt{2}} \left( |0\rangle \frac{1}{\sqrt{N}} \sum_y e^{2\pi i x y / N} |y\rangle + |1\rangle \frac{1}{\sqrt{N}} \sum_y e^{2\pi i (x+s) y / N} |y\rangle \right)$$

$$= \frac{1}{\sqrt{2N}} \sum_y e^{2\pi i x y / N} \left( |0\rangle + e^{2\pi i s y / N} |1\rangle \right) |y\rangle$$

**Key observation:** The phase factor $e^{2\pi i s y / N}$ encodes information about the secret s in the relative phase between |0> and |1> for each Fourier label y.

### 3.3 Extracting Fourier Information

After constructing the full circuit (DCP + QFT), we extract information from the statevector:

```python
def extract_fourier_info(statevector: Statevector, n: int, N: int) -> QFTResult:
    """
    Extract the Fourier label distribution and secret-dependent phases
    from the post-QFT statevector.
    
    The statevector has 2*N amplitudes indexed as |flag, data>.
    For each y in 0..N-1:
        P(y) = |<0,y|psi>|^2 + |<1,y|psi>|^2
        phase(y) = arg( <1,y|psi> / <0,y|psi> )  [when both are non-zero]
    """
    distribution = {}
    phases = {}
    
    for y in range(N):
        amp_0y = statevector.data[0 * N + y]   # <0, y|psi>
        amp_1y = statevector.data[1 * N + y]   # <1, y|psi>
        
        prob_y = abs(amp_0y)**2 + abs(amp_1y)**2
        distribution[y] = prob_y
        
        if prob_y > 1e-12:
            # Extract the secret-dependent relative phase
            if abs(amp_0y) > 1e-12 and abs(amp_1y) > 1e-12:
                phase = amp_1y / amp_0y
                phases[y] = phase  # Should be close to exp(2*pi*i*s*y/N)
    
    return QFTResult(
        statevector=statevector,
        fourier_distribution=distribution,
        phases=phases,
        N=N
    )
```

### 3.4 Phase Verification

For the DCP state, the extracted phase for Fourier label y should equal:

$$\phi(y) = \frac{2\pi s y}{N} \pmod{2\pi}$$

Verification code:

```python
def verify_phases(phases: dict[int, complex], s: int, N: int, tol=1e-8):
    """Verify that extracted phases match the theoretical prediction."""
    for y, phase in phases.items():
        expected = np.exp(2j * np.pi * s * y / N)
        assert abs(phase - expected) < tol, \
            f"Phase mismatch at y={y}: got {phase}, expected {expected}"
```

---

## 4. Information Engine — Detailed Implementation

### 4.1 Bit Truncation

Given a Fourier label y with n bits, truncation to k bits keeps the k most significant bits:

```python
def truncate_label(y: int, n: int, k: int, mode: str = "msb") -> int:
    """
    Truncate an n-bit Fourier label to k bits.
    
    Args:
        y: full Fourier label (0 <= y < 2^n)
        n: total number of bits
        k: number of bits to retain (k <= n)
        mode: "msb" keeps top k bits, "lsb" keeps bottom k bits
    
    Returns:
        Truncated label y_k as an integer
    """
    if k >= n:
        return y
    if k == 0:
        return 0
    
    if mode == "msb":
        # Keep the top k bits: right-shift by (n - k)
        return y >> (n - k)
    elif mode == "lsb":
        # Keep the bottom k bits: mask with (2^k - 1)
        return y & ((1 << k) - 1)
    else:
        raise ValueError(f"Unknown truncation mode: {mode}")
```

**Example (n=8, y=182 = 10110110):**

| k | Mode | Operation | Result (binary) | Result (decimal) |
|---|------|-----------|-----------------|------------------|
| 8 | msb  | Full      | 10110110        | 182              |
| 6 | msb  | y >> 2    | 00101101        | 45               |
| 4 | msb  | y >> 4    | 00001011        | 11               |
| 2 | msb  | y >> 6    | 00000010        | 2                |

### 4.2 Noise Injection

```python
def inject_noise(y: int, n: int, epsilon: float, rng: np.random.Generator) -> tuple[int, list[int]]:
    """
    Flip each bit of y independently with probability epsilon.
    
    Args:
        y: Fourier label
        n: bit width
        epsilon: bit-flip probability
        rng: numpy random generator (for reproducibility)
    
    Returns:
        (noisy_y, list of flipped bit positions)
    """
    flipped = []
    y_noisy = y
    for i in range(n):
        if rng.random() < epsilon:
            y_noisy ^= (1 << i)
            flipped.append(i)
    return y_noisy, flipped
```

### 4.3 Sampling Fourier Labels from the Distribution

In state-vector mode, we can sample y values from the exact Fourier distribution:

```python
def sample_fourier_label(distribution: dict[int, float], 
                         rng: np.random.Generator) -> int:
    """Sample a Fourier label y from P(y)."""
    labels = list(distribution.keys())
    probs = np.array([distribution[y] for y in labels])
    probs /= probs.sum()  # Normalise (handle floating-point drift)
    idx = rng.choice(len(labels), p=probs)
    return labels[idx]
```

---

## 5. Secret Recovery — Detailed Implementation

### 5.1 Brute-Force Recovery (Primary Strategy)

For small N (up to 64), we can enumerate all candidate secrets:

```python
def brute_force_recovery(y_k: int, k: int, n: int, N: int, 
                         x: int, mode: str = "msb") -> tuple[int, dict[int, float]]:
    """
    For each candidate secret s_candidate in 0..N-1,
    compute the likelihood of observing y_k given s_candidate,
    and return the MAP estimate.
    
    The likelihood model:
        P(y_k | s) = sum over all full y that truncate to y_k of P(y | s)
    
    where P(y | s) comes from the theoretical DCP Fourier distribution.
    """
    likelihoods = {}
    
    for s_candidate in range(N):
        # Theoretical Fourier distribution for DCP with secret s_candidate
        # P(y | s) = (1/N) * |1 + exp(2*pi*i*s*y/N)|^2 / 2
        # For DCP: this simplifies to cos^2(pi*s*y/N) for even/odd structure
        
        total_likelihood = 0.0
        for y_full in range(N):
            y_trunc = truncate_label(y_full, n, k, mode)
            if y_trunc == y_k:
                # Compute P(y_full | s_candidate)
                phase = np.exp(2j * np.pi * s_candidate * y_full / N)
                prob = (1.0 / N) * (1.0 + np.cos(2 * np.pi * s_candidate * y_full / N))
                total_likelihood += prob
        
        likelihoods[s_candidate] = total_likelihood
    
    # Normalise to get posterior (uniform prior)
    total = sum(likelihoods.values())
    posterior = {s: l / total for s, l in likelihoods.items()}
    
    # MAP estimate
    s_hat = max(posterior, key=posterior.get)
    
    return s_hat, posterior
```

### 5.2 Multi-Sample Bayesian Recovery

When m > 1 samples are available, we update the posterior sequentially:

```python
def bayesian_recovery_multi_sample(
    observations: list[tuple[int, int]],  # list of (y_k, x_i) pairs
    k: int, n: int, N: int, mode: str = "msb"
) -> tuple[int, dict[int, float]]:
    """
    Bayesian secret recovery using multiple independent DCP samples.
    
    Prior: P(s) = 1/N for all s (uniform)
    Update: P(s | y_k^{(1)}, ..., y_k^{(m)}) proportional to 
            prod_i P(y_k^{(i)} | s)
    """
    # Initialise uniform prior
    log_posterior = {s: -np.log(N) for s in range(N)}
    
    for y_k, x_i in observations:
        for s in range(N):
            # Compute log P(y_k | s) for this observation
            log_lik = compute_log_likelihood(y_k, s, k, n, N, x_i, mode)
            log_posterior[s] += log_lik
    
    # Normalise (log-sum-exp for numerical stability)
    max_log = max(log_posterior.values())
    posterior = {s: np.exp(lp - max_log) for s, lp in log_posterior.items()}
    total = sum(posterior.values())
    posterior = {s: p / total for s, p in posterior.items()}
    
    s_hat = max(posterior, key=posterior.get)
    return s_hat, posterior
```

### 5.3 Bit-Wise Recovery

For individual bit recovery, we marginalise the posterior over all other bits:

```python
def bitwise_recovery(posterior: dict[int, float], n: int) -> list[bool]:
    """
    For each bit position i, compute P(s_i = 1 | data) and
    return the MAP bit estimate.
    """
    bit_estimates = []
    for i in range(n):
        # P(s_i = 1) = sum of posterior over all s with bit i set
        p_bit_1 = sum(p for s, p in posterior.items() if (s >> i) & 1)
        bit_estimates.append(p_bit_1 > 0.5)
    return bit_estimates
```

---

## 6. Mutual Information Estimation

### 6.1 Theoretical Setup

We want to estimate:

$$I(S; Y_k) = H(Y_k) - H(Y_k | S)$$

where:
- S is the secret (uniform over Z_N)
- Y_k is the truncated Fourier observation

### 6.2 Estimation via Joint Distribution

For small N (which is our regime), we can compute the exact joint distribution P(S, Y_k) by enumeration:

```python
def compute_mutual_information(N: int, k: int, n: int, 
                                mode: str = "msb") -> float:
    """
    Compute I(S; Y_k) exactly by enumerating the joint distribution.
    
    P(S, Y_k) = (1/N) * P(Y_k | S)  [uniform prior on S]
    P(Y_k) = sum_S P(S, Y_k)
    """
    # Build joint distribution matrix: joint[s][y_k] = P(s, y_k)
    # Y_k can take 2^k distinct values (0 to 2^k - 1)
    y_max = 1 << k  # 2^k
    joint = np.zeros((N, y_max))
    
    for s in range(N):
        for y_full in range(N):
            # P(y_full | s) for DCP
            phase = 2 * np.pi * s * y_full / N
            p_y_given_s = (1.0 / N) * (1.0 + np.cos(phase)) / 2.0
            
            # Map to truncated label
            y_k = truncate_label(y_full, n, k, mode)
            joint[s][y_k] += p_y_given_s
        
        # Multiply by prior P(s) = 1/N
        joint[s] /= N
    
    # Marginals
    p_s = np.sum(joint, axis=1)       # P(S), should be uniform
    p_yk = np.sum(joint, axis=0)      # P(Y_k)
    
    # Mutual information: I(S; Y_k) = sum_{s,y} P(s,y) * log(P(s,y) / (P(s)*P(y)))
    mi = 0.0
    for s in range(N):
        for yk in range(y_max):
            if joint[s][yk] > 1e-15:
                mi += joint[s][yk] * np.log2(joint[s][yk] / (p_s[s] * p_yk[yk]))
    
    return mi
```

### 6.3 Information Loss Ratio

```python
def information_loss_ratio(mi_full: float, mi_truncated: float) -> float:
    """
    Compute the fraction of information lost due to truncation.
    
    ratio = 1 - I(S; Y_k) / I(S; Y)
    
    Returns 0 when no information is lost (k = n).
    Returns 1 when all information is lost (k = 0 or MI = 0).
    """
    if mi_full <= 0:
        return 1.0
    return max(0.0, 1.0 - mi_truncated / mi_full)
```

### 6.4 Confidence Intervals

For recovery probability estimates from finite trials:

```python
from scipy.stats import beta

def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    from statsmodels.stats.proportion import proportion_confint
    return proportion_confint(successes, total, alpha=alpha, method='wilson')
```

---

## 7. EDCP Implementation

### 7.1 EDCP State Construction

The general EDCP state:

$$|\psi\rangle = \sum_{j} \chi(j) |j\rangle |x + js \pmod{N}\rangle$$

For a DCP instance, j in {0, 1} and chi(0) = chi(1) = 1/sqrt(2).

For a general EDCP instance, we need to specify the chi coefficients. A common choice for the LWE connection has j ranging over a larger set.

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

def build_edcp_state(N: int, s: int, x: int, 
                     chi: dict[int, complex]) -> tuple[QuantumCircuit, Statevector]:
    """
    Build the EDCP quantum state.
    
    Args:
        N: modulus
        s: hidden secret
        x: random offset
        chi: coefficients {j: chi(j)} defining the EDCP structure
    
    Returns:
        (circuit, statevector)
    """
    j_values = sorted(chi.keys())
    j_max = max(j_values)
    n_j = max(1, (j_max).bit_length())   # qubits for j-register
    n_x = max(1, (N - 1).bit_length())   # qubits for data register
    
    total_qubits = n_j + n_x
    circuit = QuantumCircuit(total_qubits)
    j_qubits = list(range(n_j))
    x_qubits = list(range(n_j, n_j + n_x))
    
    # Prepare superposition: sum_j chi(j) |j>
    # First create uniform superposition, then apply amplitude encoding
    # For small instances, we can use initialize
    j_state = np.zeros(2**n_j, dtype=complex)
    for j, coeff in chi.items():
        j_state[j] = coeff
    circuit.initialize(j_state, j_qubits)
    
    # Initialise data register to |x>
    for i in range(n_x):
        if (x >> i) & 1:
            circuit.x(x_qubits[i])
    
    # For each j > 0, apply conditional modular addition of j*s
    # |j>|x> -> |j>|x + j*s mod N>
    # This requires controlled addition for each j
    for j in j_values:
        if j == 0:
            continue
        js = (j * s) % N
        if js == 0:
            continue
        # Apply ADD(js, N) controlled on |j> register
        _apply_controlled_mod_add(circuit, j_qubits, x_qubits, js, N, j)
    
    statevector = Statevector.from_instruction(circuit)
    return circuit, statevector
```

### 7.2 Toy Bai-Style Modulus Halving

The conceptual pipeline:

```python
def bai_toy_pipeline(N: int, s: int, samples: int, 
                     iterations: int, chi: dict[int, complex],
                     rng: np.random.Generator) -> ModHalvingResult:
    """
    Toy-scale implementation of Bai et al. modulus-halving approach.
    
    For each iteration:
    1. Generate EDCP samples
    2. Apply QFT
    3. Extract Fourier labels
    4. Set up linear equations
    5. Reduce modulus by factor of 2
    6. Repeat with reduced parameters
    
    This is a SIMPLIFIED version for illustration.
    The full Bai algorithm has additional structure we reproduce only at toy scale.
    """
    current_N = N
    equations = []
    intermediate = []
    
    for it in range(iterations):
        # Generate samples at current modulus
        fourier_labels = []
        for _ in range(samples):
            x = rng.integers(0, current_N)
            circuit, sv = build_edcp_state(current_N, s, x, chi)
            # Apply QFT to data register
            # Extract Fourier labels
            # ... (similar to DCP QFT extraction)
            # fourier_labels.append(y)
        
        # Set up linear equations from Fourier labels
        # y * s = phase_info (mod current_N)
        # ... (equation construction)
        
        # Halve the modulus
        new_N = current_N // 2
        intermediate.append({
            'iteration': it,
            'modulus': current_N,
            'new_modulus': new_N,
            'num_equations': len(equations)
        })
        current_N = new_N
    
    return ModHalvingResult(
        initial_modulus=N,
        reduced_modulus=current_N,
        iterations=iterations,
        equations=equations,
        recovered_info={},
        success=(current_N <= 2),
        intermediate_results=intermediate
    )
```

---

## 8. Experiment Configuration

### 8.1 YAML Configuration Format

```yaml
# configs/dcp_base.yaml
experiment:
  name: "dcp_truncation_core"
  description: "Core DCP Fourier information truncation experiment"

parameters:
  N: 16                      # Modulus
  s: 5                       # Hidden secret
  m: 1                       # Number of samples
  k: [2, 3, 4, 5, 6, 7, 8]  # Truncation levels (sweep)
  epsilon: 0.0               # Noise level
  shots: 1000                # Repetitions per configuration
  seed: 42                   # Random seed
  problem_type: "dcp"
  recovery_strategy: "brute_force"
  truncation_mode: "msb"
  backend: "statevector"

output:
  dir: "results/raw/dcp_truncation_core"
  format: "parquet"
  save_circuits: false
  save_statevectors: false
```

### 8.2 Sweep Configuration

```yaml
# configs/dcp_truncation_sweep.yaml
sweep:
  name: "dcp_comprehensive_sweep"
  description: "Full parameter sweep across N, k, m, epsilon"

base_parameters:
  s: 5
  epsilon: 0.0
  shots: 500
  seed: 42
  problem_type: "dcp"
  recovery_strategy: "brute_force"
  truncation_mode: "msb"
  backend: "statevector"

axes:
  N: [4, 8, 16, 32]
  k: null  # Special handling: k ranges from 1 to n for each N
  m: [1, 2, 4, 8]
  epsilon: [0.0, 0.05, 0.1, 0.2]

output:
  dir: "results/aggregated/dcp_comprehensive"
  format: "parquet"
```

### 8.3 Configuration Loading

```python
import yaml
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    N: int
    s: int
    n: int = 0  # Auto-computed from N
    m: int = 1
    k: int | None = None
    epsilon: float = 0.0
    shots: int = 1000
    seed: int = 42
    problem_type: str = "dcp"
    recovery_strategy: str = "brute_force"
    truncation_mode: str = "msb"
    backend: str = "statevector"
    edcp_chi: dict[int, complex] | None = None
    mod_halving_iterations: int = 0
    
    def __post_init__(self):
        if self.n == 0:
            self.n = max(1, (self.N - 1).bit_length())

def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data['parameters'])
```

---

## 9. Results Persistence

### 9.1 Per-Experiment Output

Each experiment run produces:

```text
results/raw/<experiment_name>/
  +-- <timestamp>_<N>_<k>_<m>_<epsilon>.parquet   # Trial-level data
  +-- <timestamp>_<N>_<k>_<m>_<epsilon>_meta.json  # Configuration + summary stats
```

### 9.2 Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| trial_id | int | Unique trial index (0 to shots-1) |
| N | int | Modulus |
| s | int | True secret |
| s_hat | int | Estimated secret |
| correct | bool | Whether s_hat == s |
| k | int | Bits retained |
| m | int | Samples used |
| epsilon | float | Noise level |
| y_full | int | Full Fourier label |
| y_truncated | int | Truncated Fourier label |
| confidence | float | Posterior confidence |
| runtime_s | float | Wall-clock time per trial |

### 9.3 Aggregated Results

Sweep results are stored as a single Parquet file with all trials across all parameter combinations, enabling efficient analysis with Pandas:

```python
df = pd.read_parquet("results/aggregated/dcp_comprehensive/sweep_results.parquet")

# Recovery probability by k
success_by_k = df.groupby('k')['correct'].mean()

# MI by k
mi_by_k = df.groupby('k')['mi_truncated'].mean()

# Heatmap: recovery vs (k, N)
pivot = df.pivot_table(values='correct', index='k', columns='N', aggfunc='mean')
```

---

## 10. Plotting and Visualisation

### 10.1 Standard Plot Suite

| Plot | X-axis | Y-axis | Purpose |
|------|--------|--------|---------|
| Recovery vs. truncation | k | P_success | Core result: how truncation affects recovery |
| MI vs. truncation | k | I(S; Y_k) | Information-theoretic view of truncation |
| Information loss ratio | k | 1 - I(S;Y_k)/I(S;Y) | Fractional information loss |
| Recovery vs. samples | m | P_success | Sample-complexity experiment |
| Recovery vs. noise | epsilon | P_success | Noise robustness |
| Scaling | N | P_success, runtime | Modulus scaling behaviour |
| Bit-recovery heatmap | (bit position, k) | P(s_hat_i = s_i) | Per-bit recovery analysis |
| DCP vs. EDCP comparison | k | P_success (both) | Compare DCP and EDCP information content |
| Posterior visualisation | s | P(s | data) | Show posterior distribution shape |

### 10.2 Plotting Implementation

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Font setup for any CJK characters in labels
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_recovery_vs_truncation(df: pd.DataFrame, N: int, 
                                  save_path: str | None = None):
    """Plot recovery probability as a function of retained bits k."""
    subset = df[df['N'] == N]
    grouped = subset.groupby('k').agg(
        p_success=('correct', 'mean'),
        ci_lower=('correct', lambda x: wilson_ci(x.sum(), len(x))[0]),
        ci_upper=('correct', lambda x: wilson_ci(x.sum(), len(x))[1]),
        n_trials=('correct', 'count')
    ).reset_index()
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(grouped['k'], grouped['p_success'], 'o-', linewidth=2, markersize=8)
    ax.fill_between(grouped['k'], grouped['ci_lower'], grouped['ci_upper'], 
                     alpha=0.2)
    ax.axhline(1.0/N, color='red', linestyle='--', label=f'Random (1/{N})')
    ax.set_xlabel('Retained bits (k)')
    ax.set_ylabel('Recovery probability $P(\hat{{s}} = s)$')
    ax.set_title(f'DCP Secret Recovery vs. Fourier Information (N={N})')
    ax.legend()
    ax.set_xticks(grouped['k'])
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

Each engine has a corresponding test file:

| Test File | What It Tests |
|-----------|--------------|
| `test_dcp_engine.py` | DCP state correctness for N=4,8,16 with various (s,x) pairs |
| `test_qft_engine.py` | QFT output matches theoretical Fourier distribution and phases |
| `test_info_engine.py` | Truncation produces correct labels; noise injection flips correct bits |
| `test_recovery_engine.py` | Recovery with full info (k=n) achieves ~100% success for small N |
| `test_stats_engine.py` | MI computation matches known theoretical values for trivial cases |
| `test_mutual_information.py` | I(S;Y) for DCP with N=2 should equal 1 bit (complete information) |
| `test_mod_halving.py` | Modulus halving correctly reduces N across iterations |

### 11.2 Integration Tests

```python
def test_full_dcp_pipeline():
    """End-to-end test: DCP state -> QFT -> truncate -> recover -> stats."""
    config = ExperimentConfig(N=4, s=1, m=1, k=2, shots=100, seed=0)
    result = Orchestrator().run(config)
    
    # With full info (k=n=2) and N=4, recovery should be very high
    assert result.statistics.recovery_prob > 0.8

def test_dcp_truncation_degrades_recovery():
    """Test that more truncation = worse recovery."""
    results = []
    for k in [1, 2]:  # n=2 for N=4
        config = ExperimentConfig(N=4, s=1, m=1, k=k, shots=500, seed=42)
        result = Orchestrator().run(config)
        results.append((k, result.statistics.recovery_prob))
    
    # k=2 (full) should be better than k=1 (truncated)
    assert results[1][1] >= results[0][1]
```

### 11.3 Verification Scripts

Standalone scripts for mathematical verification:

```bash
# Verify DCP state construction
python scripts/verify_dcp_state.py --N 16 --s 5 --x 11

# Verify QFT produces correct phases
python scripts/verify_qft.py --N 16 --s 5

# Run full test suite
pytest tests/ -v
```

---

## 12. Logging and Reproducibility

### 12.1 Structured Logging

```python
import logging
import json

def setup_experiment_logger(name: str, log_file: str | None = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

### 12.2 Experiment Provenance

Every result file includes:

```json
{
  "timestamp": "2026-08-17T10:30:00Z",
  "config": { ... },
  "git_commit": "a1b2c3d",
  "python_version": "3.11.5",
  "qiskit_version": "1.2.0",
  "random_seed": 42,
  "hostname": "...",
  "runtime_seconds": 12.5
}
```

---

## 13. Performance Considerations

### 13.1 State-Vector Memory

For n qubits, the statevector has 2^n complex numbers (16 bytes each in double precision):

| n (data qubits) | Total qubits (n+1) | Statevector size | Memory |
|-----------------|---------------------|-------------------|--------|
| 2 | 3 | 8 | 128 B |
| 4 | 5 | 32 | 512 B |
| 8 | 9 | 512 | 8 KB |
| 10 | 11 | 2,048 | 32 KB |
| 16 | 17 | 131,072 | 2 MB |
| 20 | 21 | 2,097,152 | 32 MB |

Our maximum N=64 requires n=6 data qubits + 1 flag = 7 total, using only 2 KB. State-vector simulation is entirely feasible for our entire parameter range.

### 13.2 Brute-Force Recovery Cost

For each trial, brute-force recovery evaluates N candidate secrets, each requiring O(N) work to compute the likelihood. Total: O(N^2) per trial. For N=64, this is 4096 operations — negligible.

### 13.3 Mutual Information Computation

The exact MI computation enumerates the N x 2^k joint distribution: O(N * 2^k). For our maximum parameters (N=64, k=6), this is 64 * 64 = 4096 entries. Trivial.

### 13.4 Sweep Scaling

A comprehensive sweep with N in {4,8,16,32,64}, k from 1 to n, m in {1,2,4,8,16}, epsilon in {0,0.05,0.1,0.2}, with 500 shots each produces approximately:

5 (N values) * ~30 (k values total) * 5 (m) * 4 (epsilon) * 500 (shots) = **1.5 million trials**

Each trial takes ~1ms (dominated by circuit construction), so the full sweep completes in approximately **25 minutes**. This is manageable as a batch job.

---

## 14. Error Handling

### 14.1 Input Validation

```python
def validate_config(config: ExperimentConfig):
    """Validate experiment configuration parameters."""
    assert config.N >= 2, "Modulus N must be >= 2"
    assert 0 <= config.s < config.N, f"Secret s must be in [0, {config.N})"
    assert config.k is None or 0 <= config.k <= config.n, \
        f"k must be in [0, {config.n}]"
    assert config.m >= 1, "Must have at least 1 sample"
    assert 0.0 <= config.epsilon <= 1.0, "Noise must be in [0, 1]"
    assert config.shots >= 1, "Must have at least 1 shot"
```

### 14.2 Numerical Stability

- Log-space computations for likelihoods and posteriors (log-sum-exp trick)
- Normalise probability distributions after every operation
- Tolerance thresholds for phase comparisons (1e-8 default)
- Guard against division by zero in MI computation (skip terms with P < 1e-15)
