# How to use the qiskit / pennylane training code on this branch

The original `README.md` covers the upstream EXAQC project. This file
covers what's new on the `Quantum-Congressman-Vogt-Cooking` branch:
the qiskit autodiff training pipeline, the parameter-shift rule
addition, and how the existing pennylane backend hooks in alongside
it. Read this if you just want to know how to run the code.

## Setup

Python 3.10 or 3.11. Numpy has to be pinned `<2` because the torch 2.2
wheels were built against numpy 1.x; you'll get an obscure
`_ARRAY_API not found` import error otherwise.

```bash
cd /path/to/exaqc
python3.10 -m venv .venv
.venv/bin/pip install \
    pennylane torch \
    qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms \
    matplotlib scikit-learn pytest loguru \
    "numpy<2"
```

Run things with `PYTHONPATH=. .venv/bin/python -m <module>`.

## The single entry point

Both backends go through the same function:

```python
from src.objectives.genome_objectives import train_genome_objective

train_genome_objective(
    genome,
    dataset=[train_ds, test_ds],
    backend="pennylane",     # or "qiskit"
    loss="ce",
    encoding="angle",
    epochs=30,
    lr=0.05,
    n_classes=3,
    batch_size=8,
    noise_model=None,        # or a PennyLaneNoiseModel / QiskitNoiseModel
    qiskit_config=None,      # only used when backend="qiskit"
)
```

After it returns, `genome.fitness` is populated and the trained
weights are written back into the genome's gate parameters.

## The pennylane backend

`backend="pennylane"` runs the existing `_train_with_pennylane` path.
It uses `default.qubit` (or `default.mixed` if a noise model is
provided) with `diff_method="backprop"` through the torch interface.
Adam optimizer, mini-batch with `BalancedBatchSampler`. Pretty much
unchanged from before this branch except for one safety fix in the
training loop: NaN gradients are now zeroed before `clip_grad_norm_`
and `opt.step()`, because `clip_grad_norm_` doesn't sanitize NaN —
its norm is NaN, the scale factor is NaN, and the gradients stay NaN
through the clip. After one optimizer step every weight on the genome
becomes NaN, and from there every forward pass returns NaN. The new
sanitizer block catches that.

If you don't pass `qiskit_config`, the pennylane path is what runs.
That's how it always worked.

## The qiskit backend

`backend="qiskit"` runs a new `_train_with_qiskit` in
`src/objectives/qiskit_train.py`. Same shape as the pennylane one
from the outside — fitness goes onto the genome, weights get written
back. Inside, the qiskit path:

1. Builds a parametric qiskit circuit from the genome via
   `CircuitGenome.generate_qiskit_circuit_parametric()`. Trainable
   gate angles become qiskit `Parameter` symbols (cached on each
   gate so the same symbol is reused across rebuilds).
2. Prepends a feature map for angle encoding (`RY(pi*x_i)` per input
   qubit using a qiskit `ParameterVector`).
3. Builds joint-distribution Pauli projector observables on the
   output qubits via `src/utils/qiskit_observables.bitstring_projector`,
   in PennyLane's MSB convention so the QNN forward output has the
   same shape and indexing as `qml.probs(wires=output)`.
4. Wires that into `EstimatorQNN` and wraps with `TorchConnector` so
   the same Adam loop and the same NaN sanitizer from the pennylane
   path apply unchanged.

To use it:

```python
train_genome_objective(
    genome,
    dataset=[train_ds, test_ds],
    backend="qiskit",
    loss="ce",
    encoding="angle",
    epochs=30,
    lr=0.05,
    n_classes=3,
    batch_size=8,
    qiskit_config={"gradient_method": "reverse"},   # or "param_shift"
)
```

## The parameter-shift rule addition

This is the main new thing. The qiskit backend takes a
`gradient_method` config field with two valid values:

### `gradient_method="reverse"` (default)

Uses `qiskit_algorithms.gradients.ReverseEstimatorGradient`. This is
classical statevector backprop — the gradient is computed by
reverse-mode automatic differentiation through the simulator's tensor
operations. Properties:

- Fast. Roughly 2x the cost of a forward pass regardless of how many
  parameters the circuit has, like any backprop.
- No shot noise (statevector is exact).
- **Simulator only.** Real quantum hardware never exposes a
  statevector to differentiate through, so this path cannot run on a
  real QPU.

Use this when you're doing pure simulation work and don't plan to
take the circuit to hardware.

### `gradient_method="param_shift"`

Uses `qiskit_algorithms.gradients.ParamShiftEstimatorGradient`. This
is the parameter-shift rule, an analytical identity that says for any
Pauli rotation:

```
d<H>/d(theta) = ( f(theta + pi/2) - f(theta - pi/2) ) / 2
```

So computing the gradient with respect to one parameter requires two
extra circuit evaluations. With P trainable parameters, gradient cost
is roughly `2*P` forward passes. Properties:

- Slower than backprop on a simulator. For a 50-parameter circuit,
  expect ~50x slower training compared to `reverse`.
- The identity is exact — not a finite-difference approximation. On a
  noiseless statevector, `param_shift` and `reverse` agree to ~1e-16.
- **Works on any backend that can run the circuit.** Real IBM
  hardware, noisy Aer simulators, anything. The gradient method
  treats the circuit as a black box.

Use this when you eventually want to run on real hardware (which is
where backprop simply isn't an option), or when you want a sanity
check that your simulator gradients match the hardware-realistic
gradients.

### Auto-switching when noise is on

If you pass `noise_model` to `train_genome_objective` with
`backend="qiskit"`, the code auto-switches `gradient_method` to
`"param_shift"` and logs a warning. The reason: `ReverseEstimatorGradient`
does its own classical statevector tracking and ignores whatever
estimator is passed in. So combining it with
`AerEstimator(noise_model=...)` would silently give noise-free
gradients with a noisy forward — a real bug, not just a slowdown.
Easier to force the right gradient method than to leave the foot-gun
in place.

If you want to silence the warning, just pass `gradient_method="param_shift"`
explicitly when you also pass a noise model.

## Picking a gradient method (tldr)

| your situation | use |
|---|---|
| pure simulation, no hardware plans | `reverse` |
| eventually want to run on a real IBM QPU | `param_shift` |
| running anything noisy | `param_shift` (auto-selected) |
| sanity-checking gradients between methods | run both, compare |

The two methods are computing the same mathematical quantity by
different algorithms — so any disagreement on a noiseless statevector
beyond ~1e-12 is a bug, not a tolerance issue.

## Quick example, end to end

A minimal Iris training that doesn't involve evolution, just trains
one fixed genome on each backend and prints the fitness:

```python
import math
import numpy as np
import torch

from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers
from src.datasets.classification import IrisDataset
from src.objectives.genome_objectives import train_genome_objective


def build_genome(target):
    g = CircuitGenome(
        genome_number=0,
        target=target,
        input_qubits=expand_registers({"input": 4}),
        output_qubits=expand_registers({"output": 2}),
    )
    # PennyLane uses 'phi' for RY's angle param; qiskit uses 'theta'.
    key = "theta" if target == "qiskit" else "phi"
    depth = 0.05
    for q_name, q_idx in g.qubits:
        g.add_gate(depth=depth, method_name="ry",
                   qubits=[(q_name, q_idx)],
                   parameters={key: 0.1 * (q_idx + 1)})
        depth += 0.04
    g.add_gate(depth=depth, method_name="cx",
               qubits=[g.qubits[0], g.qubits[1]])
    return g


train_ds = IrisDataset(split="train")
test_ds = IrisDataset(split="test")

for backend in ("pennylane", "qiskit"):
    np.random.seed(0); torch.manual_seed(0)
    genome = build_genome(backend)
    genome.hyperparameters = {"epochs": 5, "learning_rate": 0.05,
                              "batch_size": 8, "encoding": "angle",
                              "log_every": 1}
    train_genome_objective(
        genome, dataset=[train_ds, test_ds],
        backend=backend, loss="ce", encoding="angle",
        epochs=5, lr=0.05, n_classes=3, log_every=1, batch_size=8,
        qiskit_config={"gradient_method": "reverse"} if backend == "qiskit" else None,
    )
    print(backend, genome.fitness)
```

For the existing MPI / evolutionary search examples
(`src/examples/pl_classification.py`,
`src/examples/pl_classification_noisy.py`), nothing changes — they
were never qiskit-aware and the pennylane path they use is unchanged.

## Files added or modified for this work

```
src/circuits/gate.py                          (modified — parametric mode + cache)
src/circuits/circuit.py                       (modified — generate_qiskit_circuit_parametric)
src/objectives/genome_objectives.py           (modified — qiskit dispatch + NaN sanitizer)
src/objectives/qiskit_train.py                (NEW — _train_with_qiskit)
src/utils/qiskit_observables.py               (NEW — bitstring projector)
src/noise/PennyLaneNoiseModel.py              (modified — thermal-relaxation clamps)
src/examples/pl_classification_noisy.py       (modified — eval NaN sanitizer)
src/analysis/train_genome.py                  (modified — eval NaN sanitizer)
```

## src/Ryan_cookin/ — encoder-learning experiment

A separate piece of code on the same branch that compares fixed vs
trainable classical-to-quantum encoders at varying input qubit
counts. Useful if you want to see whether the encoder choice matters
for a particular dataset, or if you want a starting point for
extending the encoder side of a VQC.

Files inside `src/Ryan_cookin/`:

- **`encoders.py`** — three encoder classes behind a uniform interface:
  - `FixedAngleEncoder` — `RY(pi*x_i)` per qubit, no trainable
    parameters. If `N > D` (more qubits than features), features
    cycle: qubit q sees `x[q mod D]`.
  - `FixedAmplitudeEncoder` — `AmplitudeEmbedding` with zero-padding
    and L2 normalization. Also no trainable parameters. Hard ceiling
    at `N = ceil(log2(D))`; past that, padding pushes the state into
    a small subspace and signal collapses.
  - `LearnedAngleEncoder` — 2 layers of `RY(a*x[q mod D] + b)` per
    qubit, then a linear CNOT chain. The `(a, b)` per (layer, qubit)
    are trainable. `4N` encoder parameters total. Initialised with
    slope `~pi` and bias `~0` so the first forward is essentially
    `FixedAngleEncoder`, and any divergence is the encoder learning.
  Each encoder exposes `apply(x, theta_enc)` (appends quantum ops
  inside a QNode body) and `n_params` (length of the theta_enc tensor
  the runner allocates).
- **`stage_a.py`** — runner + CLI. For each `(dataset, N, encoder)`
  cell it builds a QNode (encoder followed by a fixed 2-layer
  RY+CNOT-chain ansatz), trains via Adam on cross-entropy with the
  same NaN sanitizer used in the main training paths, evaluates, and
  appends one row per run to a CSV. Pennylane backprop on
  `default.qubit`, no noise.
- **`stage_b.py`, `stage_c.py`** — placeholders that raise
  `NotImplementedError`. Stage B would let the encoder structure
  mutate as a second CircuitGenome partition; Stage C would also
  evolve N itself. Build Stage A first.
- **`results/`** — CSV + log output from sweeps. Created on first
  run.

How to run Stage A. Full sweep, ~5–6 hours on local CPU:

```bash
PYTHONPATH=. .venv/bin/python -m src.Ryan_cookin.stage_a \
    --datasets iris wine seeds breast_cancer \
    --n_qubits 4 6 8 10 \
    --encoders fixed_angle fixed_amplitude learned \
    --epochs 30 --batch_size 8 --lr 0.05 \
    --out src/Ryan_cookin/results/stage_a.csv
```

Single-cell smoke test (~25 sec) to check the runner works end to end:

```bash
PYTHONPATH=. .venv/bin/python -m src.Ryan_cookin.stage_a \
    --datasets iris --n_qubits 4 --encoders learned \
    --epochs 3 --batch_size 16 \
    --out /tmp/smoke.csv
```

Rows are appended one at a time, so partial sweeps accumulate into
the same CSV without losing anything. Each row has the configuration
that produced it (dataset, encoder, n_qubits, seed, epochs, lr) along
with `train_loss`, `train_acc`, `test_loss`, `test_acc`,
`n_encoder_params`, `n_ansatz_params`, and `elapsed_s`. To re-run a
single cell, delete that row first or write to a different `--out`.

`--help` lists every flag.
