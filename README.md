# EXAQC

**Evolutionary Exploration of Augmenting Quantum Circuits.**

EXAQC evolves quantum circuits for machine-learning tasks. Rather than fixing a
circuit ansatz up front and only training its rotation angles, EXAQC treats the
*circuit itself* as the thing being searched: gates are added, disabled,
reordered and recombined across a population of candidate circuits, while each
candidate is also trained with gradient descent.

A candidate is a **genome**, and a genome is a hybrid model with three stages:

```
inputs -> [ classical encoder ] -> [ quantum circuit ] -> [ classical decoder ] -> outputs
```

Either classical stage may be absent. Quantum-teacher imitation uses neither, so
the search is over the circuit alone.

Circuits run on either [PennyLane](https://pennylane.ai/) or
[Qiskit](https://www.ibm.com/quantum/qiskit), and the whole model is a
[PyTorch](https://pytorch.org/) module, so the circuit's gate parameters and the
classical layers train together with ordinary backpropagation.

---

## Table of Contents

- [Installation](#installation)
- [How a run is put together](#how-a-run-is-put-together)
- [EXAQC: the evolutionary search](#exaqc-the-evolutionary-search)
  - [How the search works](#how-the-search-works)
  - [Search command-line arguments](#search-command-line-arguments)
  - [Circuit and genome arguments](#circuit-and-genome-arguments)
  - [Choosing search hyperparameters](#choosing-search-hyperparameters)
- [Population strategies](#population-strategies)
  - [steady_state](#steady_state)
  - [islands](#islands)
  - [Choosing a population strategy](#choosing-a-population-strategy)
- [Trainers](#trainers)
  - [SupervisedTrainer](#supervisedtrainer)
  - [Reinforcement-learning trainers](#reinforcement-learning-trainers)
    - [REINFORCE](#reinforce)
    - [Actor-critic (a2c)](#actor-critic-a2c)
    - [PPO](#ppo)
    - [Q-learning and SARSA](#q-learning-and-sarsa)
  - [Choosing trainer hyperparameters](#choosing-trainer-hyperparameters)
- [Entry points](#entry-points)
  - [classification](#classification)
  - [teacher](#teacher)
  - [reinforcement_learning](#reinforcement_learning)
  - [refine_genome](#refine_genome)
  - [evaluate](#evaluate)
  - [visualize_rl](#visualize_rl)
  - [classical_image_classification](#classical_image_classification)
  - [reinforcement_learning_fixed](#reinforcement_learning_fixed)
- [Analysis](#analysis)
- [Reproducing the PPSN results](#reproducing-the-ppsn-results)
- [Contributing](#contributing)

---

## Installation

To run EXAQC, first create a python3.12 virtual environment (currently the newest
version which will have the appropriate torch, qiskit and pennylane dependencies):

```
python3.12 -m venv </path/to/exaqc/environment/>
```

Then load that environment:

```
source </path/to/exaqc/environment/bin/activate/>
```

Then dependencies can be installed with (from the EXAQC project root directory):

```
python3 -m pip install -e .
```

Please note you will need to have some version of MPI installed (probably openmpi).  If you are on OSX you can install with:

```
brew install openmpi
```

Or on linux with `apt` (replace with your favorite application manager):

```
sudo apt-get install openmpi
```

---

## How a run is put together

Every evolutionary entry point wires up the same four pieces, and the
command-line arguments group the same way:

| Piece | What it does | Where its arguments come from |
|---|---|---|
| **EXAQC** | Generates new genomes by mutation and crossover | [Search arguments](#search-command-line-arguments) |
| **Population strategy** | Decides which genomes survive and become parents | [`steady_state`](#steady_state) / [`islands`](#islands) sub-command |
| **Trainer** | Trains each genome once it is generated | [Trainers](#trainers) |
| **Objective** | Trains a genome and writes its `fitness` | The entry point itself |

Runs are parallelised with **MPI**: rank 0 is the master that generates genomes
and owns the population, and every other rank is a worker that trains them.

> **A run needs at least 2 MPI ranks.** With `-n 1` there are no workers, and the
> master blocks forever waiting for results. Use `mpiexec -n <ranks>`, where
> `<ranks>` is one master plus however many genomes you want trained
> concurrently.

```
mpiexec -n 12 python3 -m src.examples.classification ... steady_state --max_population_size 30
```

Because each genome is trained independently, the search scales close to
linearly with the number of workers.

---

## EXAQC: the evolutionary search

`src/evolution/exaqc.py` owns genome *generation*. It holds the allowed gate
set, the initial encoder/decoder, the hyperparameters stamped onto each genome,
and the strategies controlling how children are produced.

### How the search works

Until the population is full, EXAQC seeds it by mutating an initial empty
genome. After that, each new child is produced by one of four operators, chosen
by the crossover rates:

| Operator | Selected with | What it does |
|---|---|---|
| **Binary crossover** | `--binary_crossover_rate` | Recombines two parents |
| **N-ary crossover** | `--n_ary_crossover_rate` | Recombines several parents (count from `--parent_strategy`) |
| **Exponential crossover** | `--exponential_crossover_rate` | Splices two parents at a random circuit depth |
| **Mutation** | whatever fraction remains | Applies `--mutation_strategy` mutations to one parent |

Mutation itself picks from a weighted set of operators: adding a gate (~55%),
reordering a gate, swapping which qubits a gate acts on, enabling or disabling a
gate, cloning, and perturbing some or all gate weights. Only gates that have
been validated for the chosen backend are ever added.

A child is rejected and regenerated if its inputs cannot reach its outputs
through enabled gates, so every evaluated circuit is functionally connected.

Every genome EXAQC generates is stamped with the `task` and `task_target` it was
evolved for, which is what lets [`refine_genome`](#refine_genome) reload one
later without being told anything about it.

### Search command-line arguments

These are shared by [`classification`](#classification), [`teacher`](#teacher)
and [`reinforcement_learning`](#reinforcement_learning), because all three call
`EXAQC.initialize_parser()`.

| Argument | Default | Description |
|---|---|---|
| `--mutation_strategy`, `-ms` | *required* | How many mutations each child gets: `uniform <min> <max>` (integers, min ≥ 1) or `exponential <scale>` |
| `--parent_strategy`, `-ps` | *required* | How many parents an n-ary crossover uses: `uniform <min> <max>` (integers, min ≥ 2) or `exponential <scale>` |
| `--binary_crossover_rate` | `0.0` | Fraction of children made by two-parent crossover |
| `--n_ary_crossover_rate` | `0.2` | Fraction of children made by multi-parent crossover |
| `--exponential_crossover_rate` | `0.1` | Fraction of children made by depth-spliced crossover |
| `--number_genomes` | `2000` (RL: `500`) | Total genomes to evaluate before stopping |

The three crossover rates must sum to at most `1.0`; the remainder is the
mutation rate. With the defaults, 70% of children come from mutation.

### Circuit and genome arguments

| Argument | Default | Description |
|---|---|---|
| `--target` | `pennylane` | Backend: [`pennylane`](https://docs.pennylane.ai/en/stable/) or [`qiskit`](https://quantum.cloud.ibm.com/docs/en/guides) |
| `--input_qubits` | *required* | Qubits the inputs are encoded onto |
| `--output_qubits` | *required* | Qubits measured for the output |
| `--quantum_input_mode`, `-qim` | `u3` | How classical values become circuit inputs: `u3`, `rx`, `ry`, `rz`, `basis`, `amplitude` |
| `--quantum_output_mode`, `-qom` | `probs` | Readout: `probs` (2^output_qubits values) or `expval` (one Pauli-Z expectation per output qubit) |
| `--encoding` | `linear` | Classical encoder: `identity`, `linear`, `cnn` |
| `--decoding` | `linear` | Classical decoder: `linear`, `clipped` |
| `--quantum_dropout` | off | Master switch for dropping gates during training |
| `--quantum_dropout_type`, `-qdt` | `none` | `gate`, `rotation`, `entangling`, `qubit`, `innovation` |
| `--quantum_dropout_rate`, `-qdr` | `0.0` | Dropout probability |

The input mode determines how many values the circuit consumes: `rx`/`ry`/`rz`
take one per input qubit, `u3` takes three, and `amplitude` takes up to
2^input_qubits (it pads and normalises, so it packs many features into few
qubits). `expval` is implemented only on the PennyLane backend.

### Choosing search hyperparameters

- **Start small.** Simulating a quantum circuit costs time exponential in qubit
  count. 4–8 qubits is a practical range on a laptop; every extra qubit roughly
  doubles the cost of every forward pass.
- **`--input_qubits` with `amplitude` encoding.** Amplitude embedding fits `2^n`
  features into `n` qubits, so 30-feature breast cancer data needs only 5–8
  qubits. With `u3`/`ry` you need roughly one qubit per feature, so pair those
  with a `linear` encoder that compresses first.
- **`--output_qubits`.** For classification with `probs`, `2^output_qubits`
  should be ≥ the number of classes; the decoder maps that vector onto class
  scores. One or two output qubits is typical.
- **Mutation vs. crossover.** Mutation-heavy settings (the defaults) explore
  circuit *topologies*; raising the crossover rates exploits combinations of
  parents that already work. `-ms uniform 1 3` is a good default: mostly small
  edits, occasionally a bigger jump.
- **`--number_genomes`** is the real budget knob. The published runs use 1000–2000.

Background on the underlying ideas:
[variational quantum circuits](https://pennylane.ai/qml/glossary/variational_circuit),
[quantum embeddings](https://pennylane.ai/qml/glossary/quantum_embedding),
[barren plateaus](https://pennylane.ai/qml/demos/tutorial_barren_plateaus) (why
deep, wide circuits can become untrainable), and
[neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution).

---

## Population strategies

The population strategy decides which genomes are kept and which become parents.
It is chosen with a **sub-command**, which must come *after* the other
arguments:

```
python3 -m src.examples.classification <options...> steady_state --max_population_size 30
python3 -m src.examples.classification <options...> islands --n_islands 10 --max_island_size 10
```

### steady_state

A single population sorted by fitness. A new genome is inserted if it is better
than the worst member; once the population is full, the worst is dropped. There
are no generations — insertion is continuous, which suits the asynchronous
MPI master/worker design, since workers finish at different times.

| Argument | Default | Description |
|---|---|---|
| `--max_population_size` | `30` | Genomes retained in the population |

### islands

Several independent steady-state populations ("islands") evolved in parallel.
Islands mostly breed within themselves, which preserves distinct solution
lineages that a single population would wash out. Periodically the worst islands
suffer an **extinction event**: they are cleared and repopulated from the best
island, spreading good material without collapsing diversity.

| Argument | Default | Description |
|---|---|---|
| `--n_islands` | `10` | Number of islands |
| `--max_island_size` | `10` | Genomes retained per island |
| `--genomes_before_extinction` | `100` | Genomes inserted before the first extinction event |
| `--genomes_for_next_extinction` | `200` | Genomes inserted between later extinction events |
| `--islands_to_extinct` | `1` | Worst islands cleared and repopulated each event |
| `--primary_parent` | `best` | Which parent leads a crossover: `best` (highest fitness first) or `island` (the target island's genome first) |
| `--intra_island_crossover_rate` | `0.5` | Fraction of an island's children bred within that island |

### Choosing a population strategy

- **`steady_state` is the simpler default** and is what the published
  classification and RL results use. A population of 30–50 works well.
- **`islands` helps when the search converges prematurely** — many genomes with
  near-identical fitness and structure. `--islands_to_extinct 0` disables
  extinction entirely, giving fully independent parallel searches; raising it
  increases how aggressively good material is shared.
- **Total capacity is `n_islands × max_island_size`.** Keep that in the same
  range as a steady-state population you would otherwise use.
- **Match population size to worker count.** A population much smaller than the
  number of workers means workers keep training children of the same few
  parents.

Background: island models are a standard technique in
[evolutionary algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm)
for maintaining population diversity.

---

## Trainers

Once EXAQC generates a genome, a trainer trains it. Every trainer reads its
hyperparameters from `genome.hyperparameters`, which is stamped onto the genome
by the entry point, so hyperparameters travel with a genome and can be evolved.

### SupervisedTrainer

`src/trainer/supervised_trainer.py`. Used by both
[`classification`](#classification) and [`teacher`](#teacher). It is task
agnostic: it drives `genome.forward` over dataloaders and hands each batch's
predictions and targets to the caller's loss function and metrics, passing
targets through untouched. Classification supplies integer class labels with
[cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html);
teacher imitation supplies float target vectors with a distribution measure.

Training uses [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html),
snapshots the best weights by validation loss, restores them at the end, and
stops early after `improvement_cutoff` epochs without improvement.

| Hyperparameter | Argument | Description |
|---|---|---|
| `epochs` | `--epochs` | Maximum training epochs per genome |
| `learning_rate` | `--learning_rate`, `-lr` | Adam learning rate |
| `weight_decay` | `--weight_decay` | Adam L2 regularisation |
| `improvement_cutoff` | `--improvement_cutoff` | Epochs without validation improvement before stopping |
| `batch_size` | `--batch_size` | Samples per gradient step |

### Reinforcement-learning trainers

`src/trainer/reinforcement_trainer.py` provides the shared scaffold — the
environment abstraction, greedy evaluation, best-weight snapshotting — and each
algorithm subclasses it. Choose one with `--algo`.

Terminology used throughout: a **step** is one interaction with the environment,
an **episode** is a full rollout, and an **epoch** is one weight update.

| `--algo` | Class | Continuous actions | Extra decoder outputs |
|---|---|---|---|
| `reinforce` | `ReinforceTrainer` | yes | 0 |
| `actor_critic` / `a2c` | `ActorCriticTrainer` | yes | 1 |
| `ppo` | `PPOTrainer` | yes | 1 |
| `q_learning` | `QLearningTrainer` | no | 0 |
| `sarsa` | `QLearningTrainer` (on-policy target) | no | 0 |

Advantage methods need a state value, and they get it from one **extra decoder
output** rather than a separate head — so the value function is part of the
genome and is evolved and serialized along with the policy. Because those
outputs must be unconstrained, use `--decoding linear` (not `clipped`) with
`actor_critic`, `a2c` and `ppo`.

Shared arguments: `--episodes`, `--eval_episodes`, `--max_steps`, `--gamma`,
`--learning_rate`, `--entropy_coef`, `--log_every`, `--improvement_cutoff`,
`--ema_alpha`.

#### REINFORCE

Monte-Carlo policy gradient: run one episode, compute discounted returns, take
one gradient step. Simplest and highest variance.

| Argument | Default | Description |
|---|---|---|
| `--baseline` | `mean` | `mean` subtracts the batch-mean return to reduce variance; `none` disables it |

#### Actor-critic (a2c)

Adds a learned state-value baseline read from the extra decoder output, which
lowers variance relative to REINFORCE.

| Argument | Default | Description |
|---|---|---|
| `--value_coef` | `0.5` | Weight on the value loss relative to the policy loss |

#### PPO

[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347). The only
algorithm whose outer iteration spans several episodes: it collects a rollout,
computes [GAE](https://arxiv.org/abs/1506.02438) advantages, then performs many
minibatch updates under a clipped objective.

| Argument | Default | Description |
|---|---|---|
| `--rollout_steps` | `512` | Environment steps collected per rollout |
| `--ppo_passes` | `4` | Passes over each rollout (PPO papers call these "epochs") |
| `--ppo_minibatch` | `128` | Transitions per weight update |
| `--ppo_clip` | `0.2` | Probability-ratio clip range |
| `--gae_lambda` | `0.95` | GAE bias/variance trade-off |

#### Q-learning and SARSA

Semi-gradient temporal-difference learning that treats the circuit's outputs as
action values `Q(s, ·)`, updating at *every environment step*. Actions are
chosen ε-greedily. `q_learning` bootstraps off-policy (`max_a' Q`), `sarsa`
on-policy (the ε-greedy action actually taken). Both enumerate actions, so they
are **discrete-only** and refuse continuous environments.

| Argument | Default | Description |
|---|---|---|
| `--epsilon` | `0.2` | Initial exploration probability |
| `--epsilon_min` | `0.05` | Floor for exploration |
| `--epsilon_decay` | `0.995` | Per-episode multiplicative decay |

### Choosing trainer hyperparameters

- **Circuit simulation dominates the cost.** `episodes × max_steps` is the
  number of forward passes per genome, and that is multiplied by
  `--number_genomes`. Start with `--episodes 60 --max_steps 500` and shrink if a
  run is too slow.
- **Learning rate.** Quantum gate parameters are angles, so they tolerate larger
  steps than deep classical nets: `1e-2` is the RL default, while supervised
  classification defaults to `5e-4`. If loss oscillates, lower it; if nothing
  moves, raise it.
- **`--gamma`** near `0.99` suits long-horizon control; lower it (`0.9`–`0.95`)
  for short episodes.
- **`--entropy_coef`** above `0` (try `0.01`) if the policy collapses to one
  action early.
- **`--improvement_cutoff`** is the biggest time saver: it abandons genomes that
  have stopped improving so the search spends its budget elsewhere.
- **Algorithm choice.** `reinforce` is the simplest baseline; `ppo` is usually
  the most sample-efficient and is the best first choice for continuous control;
  `q_learning`/`sarsa` suit small discrete environments like FrozenLake.

Background: [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
is an excellent primer on these algorithms and their hyperparameters, and
[Gymnasium's documentation](https://gymnasium.farama.org/) describes each
environment's reward scale and episode length.

---

## Entry points

All entry points live in `src/examples/`. The three evolutionary ones
(`classification`, `teacher`, `reinforcement_learning`) share the search and
population arguments described above; the rest are single-genome tools.

Common to the evolutionary entry points:

| Argument | Default | Description |
|---|---|---|
| `--out_dir` | `artifacts` | Where per-genome JSON, diagrams, plots and logs are written |
| `--save_training_plot` | off | Also write a training-history plot beside each saved diagram |
| `--device` | `cpu` | PyTorch device |
| `--seed` | `0` | Random seed |
| `--logging_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### classification

Evolves hybrid circuits to classify tabular or image datasets.

```
mpiexec -n 12 python3 -m src.examples.classification \
    --dataset iris --input_qubits 4 --output_qubits 2 --batch_size 3 \
    --number_genomes 1000 \
    -ms uniform 1 3 -ps uniform 5 5 \
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 \
    -qim amplitude -qom probs --encoding identity --decoding clipped \
    --out_dir ./artifacts/iris \
    steady_state --max_population_size 30
```

**Datasets.** Tabular: `iris`, `wine`, `breast_cancer` (loaded via
[scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html)) and
`seeds` ([UCI](https://archive.ics.uci.edu/dataset/236/seeds)). Image:
`mnist`, `fashion_mnist`, `cifar10` (via
[torchvision](https://docs.pytorch.org/vision/stable/datasets.html)).

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *required* | One of the datasets above |
| `--epochs` | `30` | Training epochs per genome |
| `--learning_rate`, `-lr` | `5e-4` | Adam learning rate |
| `--weight_decay` | `0.0` | Adam L2 regularisation |
| `--improvement_cutoff` | `2` | Epochs without validation improvement before stopping |
| `--batch_size` | `1` | Use `1` for small tabular data, larger for images |
| `--validation_batch_size` | = `--batch_size` | Validation batch size |
| `--validation_fraction` | `0.1` | Held-out fraction when no fixed split exists |
| `--normalization` | `minmax` | `none`, `zscore`, `minmax` |
| `--training_samples` / `--validation_samples` | all | Cap the splits for quick runs |
| `--data_dir` | `data` | Where datasets are stored |
| `--download_dataset` | on | Download image datasets if missing |
| `--num_workers` / `--pin_memory` | `0` / off | Dataloader settings |
| `--encoder_config` | `configs` | JSON file of encoder options |
| `--cnn_channels` | `[16, 32]` | Channels for the two CNN encoder conv layers |
| `--cnn_pooled_size` | `4` | Spatial size the CNN pools down to |
| `--cnn_dropout` | `0.0` | Dropout inside the CNN encoder |

**Guidance.** Image datasets need `--encoding cnn`, which convolves and pools
before the circuit; tabular data must *not* use it. `--encoding identity` passes
features straight through, so the feature count must equal the circuit's input
width — pair it with `-qim amplitude`, which absorbs many features into few
qubits. `--decoding clipped` normalises circuit outputs into class scores and
suits `probs`. Fitness records `loss` and `target_metric` (mean class accuracy).

### teacher

Evolves **purely quantum** circuits to reproduce the outputs of a known
reference ("teacher") circuit. There is nothing classical to learn, so genomes
carry no encoder or decoder and there are no `--encoding`/`--decoding` options.
The dataset is generated: random input angles are drawn and the teacher's
outputs become the targets.

```
mpiexec -n 12 python3 -m src.examples.teacher \
    --teacher half_adder --input_qubits 2 --output_qubits 2 \
    --number_genomes 1000 --loss fidelity --epochs 30 --batch_size 8 \
    -ms uniform 1 3 -ps uniform 5 5 \
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 \
    -qim ry -qom probs \
    --out_dir ./artifacts/half_adder \
    steady_state --max_population_size 30
```

**Teachers.** `identity`, `x_out4`, `bell_out`, `copy_in_to_out`,
`parity012_to_out4`, `input_controlled_bell`, `2layer_out_block`, `grover`,
`half_adder`. Each is itself a circuit genome, so it runs on either backend, and
each reports clearly if the requested wires cannot express it.

Input wires are the first `--input_qubits` wires and readout wires are the
`--output_qubits` wires after them, so a teacher spans
`--input_qubits + --output_qubits` wires and the two sets never overlap.

| Argument | Default | Description |
|---|---|---|
| `--teacher` | *required* | Reference circuit to imitate |
| `--loss` | `fidelity` | `fidelity`, `angle`, `kl`, `mse` |
| `--quantum_input_mode`, `-qim` | `ry` | `rx`, `ry`, `rz` (one value per input wire) |
| `--n_training_samples` | `64` | Generated training samples |
| `--n_validation_samples` | `64` | Generated validation samples |
| `--batch_size` | `8` | Samples per gradient step |
| `--epochs` | `30` | Training epochs per genome |
| `--learning_rate`, `-lr` | `5e-3` | Adam learning rate |
| `--improvement_cutoff` | `5` | Epochs without validation improvement before stopping |

**Losses.** All four are reported every epoch regardless of which is optimized,
so runs stay comparable.

| Loss | Range | Meaning |
|---|---|---|
| `fidelity` | 0–1, higher better | Classical ([Bhattacharyya](https://en.wikipedia.org/wiki/Bhattacharyya_distance)) overlap of the distributions; optimized as `1 − fidelity` |
| `angle` | 0–π/2, lower better | [Bures angle](https://en.wikipedia.org/wiki/Bures_metric), `arccos(√fidelity)` — a true distance |
| `kl` | ≥ 0, lower better | [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) from target to prediction |
| `mse` | ≥ 0, lower better | Mean squared error; makes no distribution assumption |

`fidelity`, `angle` and `kl` compare probability distributions and therefore
require `-qom probs`. `mse` also works with `-qom expval`.

**Guidance.** Start with `bell_out` or `copy_in_to_out` to confirm a setup —
they are shallow and reachable in a few gates. `grover` is limited to 2 or 3
total wires, because a wider multi-controlled Z would need a gate the search is
not allowed to use. `fidelity` is the most forgiving objective; `kl` punishes
missing probability mass hardest and can dominate the reported scale early on.

### reinforcement_learning

Evolves hybrid circuits as control policies for
[Gymnasium](https://gymnasium.farama.org/) environments.

```
mpiexec -n 12 python3 -m src.examples.reinforcement_learning \
    --env cartpole --algo ppo \
    --input_qubits 4 --output_qubits 2 --episodes 100 --number_genomes 1000 \
    --mutation_strategy uniform 1 3 --parent_strategy uniform 5 5 \
    --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 \
    -qim amplitude -qom probs --encoding identity --decoding linear \
    --out_dir ./artifacts/cartpole \
    steady_state --max_population_size 50
```

**Environments.**

| `--env` | Gymnasium id | Actions |
|---|---|---|
| `cartpole` | [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | discrete |
| `acrobot` | [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) | discrete |
| `mountaincar` | [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) | discrete |
| `frozenlake` | [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) | discrete |
| `mountaincar_continuous` | [MountainCarContinuous-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) | continuous |
| `pendulum` | [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/) | continuous |
| `hopper` | [Hopper-v5](https://gymnasium.farama.org/environments/mujoco/hopper/) | continuous |
| `walker2d` | [Walker2d-v5](https://gymnasium.farama.org/environments/mujoco/walker2d/) | continuous |
| `halfcheetah` | [HalfCheetah-v5](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | continuous |
| `ant` | [Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/) | continuous |
| `humanoid` | [Humanoid-v5](https://gymnasium.farama.org/environments/mujoco/humanoid/) | continuous |

The MuJoCo environments require `gymnasium[mujoco]` (see the
[MuJoCo docs](https://mujoco.readthedocs.io/en/stable/)). Continuous
environments work only with `reinforce`, `actor_critic`/`a2c` and `ppo`.

| Argument | Default | Description |
|---|---|---|
| `--env` | *required* | Environment above |
| `--algo` | *required* | `reinforce`, `actor_critic`, `a2c`, `ppo`, `q_learning`, `sarsa` |
| `--number_genomes` | `500` | Genomes to evaluate |
| `--input_qubits` | `4` | Input qubits |
| `--output_qubits` | from the environment | Readout qubits; defaults to the smallest register that fits the policy, `ceil(log2(n_policy_outputs))` — where a discrete policy needs one output per action and a continuous one two per action dimension |
| `--episodes` | `60` | Training episodes per genome |
| `--eval_episodes` | `10` | Greedy episodes used to score a genome |
| `--max_steps` | `500` | Step cap per episode |
| `--log_every` | `10` | Evaluate and log every N episodes |
| `--improvement_cutoff` | `30` | Episodes without an improved evaluation before stopping |
| `--ema_alpha` | `0.05` | Smoothing for the reported training return |
| `--train_vs_validation_bias`, `-tvb` | `0.01` | Weight of training return vs. evaluation return in fitness |
| `--map_name` / `--is_slippery` | `4x4` / off | FrozenLake only |

Plus the per-algorithm arguments in [Trainers](#trainers).

**Guidance.** The decoder must produce one output per action (plus one more for
`actor_critic`/`a2c`/`ppo`), which the entry point sizes automatically from the
environment — so use `--decoding linear` for those algorithms. `cartpole` is the
fastest environment to sanity-check a configuration. Fitness records
`train_return_mean`, `eval_return_mean` and `best_episode_return`.

### refine_genome

Reloads one saved genome and trains it further — useful for giving the best
genome of a search a longer run than the search could afford.

```
python3 -m src.examples.refine_genome --genome ./artifacts/iris/all_genomes/genome_11.json
```

Genome files are self-describing: EXAQC stamps the `task` and `task_target` onto
every genome it generates, so nothing but the path is needed. The stored
hyperparameters are reused unchanged unless overridden.

| Argument | Default | Description |
|---|---|---|
| `--genome` | *required* | Genome JSON written by the search |
| `--set KEY=VALUE` | — | Override a stored hyperparameter; repeatable |
| `--out_dir` | `artifacts` | Where the refined genome and diagram are written |
| `--save_circuit` | on | Also write the architecture diagram |
| `--save_training_plot` | off | Also write a training-history plot |

```
python3 -m src.examples.refine_genome --genome best_genome.json \
    --out_dir ./refined --set epochs=200 --set learning_rate=0.01
```

Each override is coerced to the type the hyperparameter already had, and an
unknown key is rejected rather than silently added. The refined genome is
written back out as JSON, still self-describing, so it can be refined again.
Genomes saved before task recording are refused with an explanatory message.

### evaluate

Scores a saved classification genome on an image dataset's official test split
(the search itself only ever sees training and validation data).

| Argument | Default | Description |
|---|---|---|
| `--genome` | *required* | Genome JSON |
| `--dataset` | *required* | `mnist`, `fashion_mnist`, `cifar10` |
| `--data_dir` | `data` | Dataset location |
| `--batch_size` | `32` | Evaluation batch size |
| `--download_dataset` | on | Download if missing |

### visualize_rl

Replays a trained RL genome in its environment so you can *watch* the evolved
circuit control it, optionally saving an animated GIF.

```
python3 -m src.examples.visualize_rl --genome_json ./artifacts/cartpole/all_genomes/genome_42.json \
    --episodes 3 --output_file cartpole.gif
```

| Argument | Default | Description |
|---|---|---|
| `--genome_json` | *required* | Genome JSON from the RL entry point |
| `--env` | from the genome | Override the environment |
| `--episodes` | `3` | Episodes to play |
| `--max_steps` | from the genome, else `500` | Step cap |
| `--seed` | random | Base environment seed |
| `--output_file` | — | Save the rollout as a GIF instead of rendering to screen |
| `--fps` | `30` | GIF frame rate |
| `--map_name` / `--is_slippery` | `4x4` / off | FrozenLake only |

### classical_image_classification

A **classical baseline**, with no quantum circuit and no evolution: it trains a
standard image model so quantum results have something to be compared against.

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *required* | `mnist`, `fashion_mnist`, `cifar10` |
| `--model` | *required* | `linear`, `mlp`, `cnn`, or a [torchvision](https://docs.pytorch.org/vision/stable/models.html) architecture (`resnet18`…`resnet152`, `densenet*`, `efficientnet*`, `mobilenet_v3_*`, `convnext_*`, `vgg*`, `regnet_y_*`) |
| `--model_config` | — | JSON file of model options |
| `--epochs` | `100` | Training epochs |
| `--learning_rate` | `1e-3` | Adam learning rate |
| `--batch_size` | `64` | Samples per gradient step |
| `--validation_batch_size` | = `--batch_size` | Validation batch size |
| `--data_dir` | `data` | Dataset location |
| `--out_dir` | `artifacts/classical` | Where results are written |
| `--num_workers` / `--pin_memory` | `0` / off | Dataloader settings |
| `--seed` | `0` | Random seed |
| `--device` | auto | PyTorch device (CUDA when available) |

### reinforcement_learning_fixed

A variant of the RL entry point that trains a **fixed classical MLP**
(`ClassicalModel`, two 64-unit `tanh` layers) instead of an evolved quantum
circuit. It accepts the same RL hyperparameters and serves as the classical
control for RL experiments.

---

## Analysis

`src/analysis/analyze_genome_generation.py` aggregates finished runs into tables
of mutation/crossover effectiveness and statistics on the best genomes found.

```
python3 -m src.analysis.analyze_genome_generation \
    --input_directories ./artifacts/classification/* \
    --groups iris seeds wine breast_cancer --metric target_metric
```

`--metric` names a key in each genome's `fitness` dict — `target_metric` for
classification and teacher runs, `eval_return_mean` for RL runs.

---

## Reproducing the PPSN results

The classification benchmarks use amplitude encoding, the `probs` output mode
with a `clipped` decoder, and an identity encoder:

```
mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset breast_cancer --number_genomes 1000 --input_qubits 8 --output_qubits 1 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/breast_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset iris --number_genomes 1000 --input_qubits 4 --output_qubits 2 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/iris_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset seeds --number_genomes 1000 --input_qubits 6 --output_qubits 2 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/seeds_i30_per_class_1 steady_state --max_population_size 30

mpiexec -n 12 python3 -m src.examples.classification --logging_level INFO --dataset wine --number_genomes 1000 --input_qubits 4 --output_qubits 2 --batch_size 3 -ms uniform 1 3 -ps uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim amplitude -qom probs --encoding identity --decoding clipped --out_dir ./2026_ppsn_exaqc/wine_i30_per_class_1 steady_state --max_population_size 30
```

Repeated experiments can be run with the scripts in [`./scripts`](./scripts)
(each creates 10 repeats):

```
sh scripts/run_iris.sh 1 10 per_class ./2026_ppsn_exaqc/classification
sh scripts/run_seeds.sh 1 10 per_class ./2026_ppsn_exaqc/classification
sh scripts/run_breast_cancer.sh 1 10 per_class ./2026_ppsn_exaqc/classification
sh scripts/run_wine.sh 1 10 per_class ./2026_ppsn_exaqc/classification
```

Reinforcement learning experiments:

```
sh scripts/run_cartpole.sh 1 10 per_class ./2026_ppsn_exaqc/rl
sh scripts/run_frozenlake.sh 1 10 per_class ./2026_ppsn_exaqc/rl
sh scripts/run_walker2d.sh 1 10 per_class ./2026_ppsn_exaqc/rl
sh scripts/run_mountaincar_continuous.sh 1 10 per_class ./2026_ppsn_exaqc/rl
```

Quantum teacher imitation:

```
sh scripts/run_teacher.sh 1 10 half_adder ./2026_ppsn_exaqc/teacher
```

The results can then be processed into the tables of mutation and crossover
rates and statistics on the best found genomes:

```
python3 -m src.analysis.analyze_genome_generation --input_directories ./2026_ppsn_exaqc/classification/* --groups iris seeds wine breast_cancer --metric target_metric
python3 -m src.analysis.analyze_genome_generation --input_directories ./2026_ppsn_exaqc/rl/* --groups cartpole frozenlake walker2d mountaincar_continuous --metric target_metric
```

---

## Contributing

### Docstring Formatting

We use Google format for docstrings. See https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html

> If you use an editor like `PyCharm` you can enable auto doc string comments by going to
> Settings -> Tools -> Python Integrated Tools -> Docstrings -> Select Google
