"""Evolve quantum genomes for reinforcement learning with EXAQC.

This is the reinforcement-learning counterpart to
:mod:`src.examples.classification`, refactored to reuse the same modular
building blocks:

* the genome's ``initialize_model`` / ``forward`` hybrid-model interface,
* the existing ``LinearEncoder`` / ``LinearDecoder`` (and identity/clipped
  variants) for embedding observations into the circuit and mapping circuit
  outputs to per-action values,
* an :class:`~src.evolution.objective.Objective` that wraps a *trainer* and
  sets genome fitness,
* the same ``master_worker`` evolutionary driver.

The RL algorithms live in :mod:`src.trainer.reinforcement_trainer` as
pluggable trainer classes (REINFORCE, actor-critic, PPO, Q-learning), exactly
mirroring how ``SupervisedTrainer`` is a pluggable component of the
classification objective. The environment is described by a pluggable
:class:`~src.trainer.reinforcement_trainer.RLEnvironment`, which the
:class:`ReinforcementLearningObjective` receives together with a trainer.

Example (single-process smoke run is driven programmatically via the
objective; the ``__main__`` block wires everything into ``master_worker`` for
an MPI evolutionary search)::

    mpirun -n 4 python -m src.examples.reinforcement_learning \\
        --env cartpole --algo ppo -ms "uniform 1 3" -ps "uniform 2 3" \\
        --target pennylane --batch_placeholder steady_state
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import gymnasium as gym
from loguru import logger

from src.circuits.circuit import (
    CircuitGenome,
    QUANTUM_INPUT_MODES,
    QUANTUM_OUTPUT_MODES,
)
from src.circuits.decoder import initialize_decoder, DECODING_OPTIONS
from src.circuits.encoder import initialize_encoder, ENCODING_OPTIONS
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications

from src.evolution.master_worker import master_worker
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation

from src.trainer.reinforcement_trainer import (
    RLEnvironment,
    ReinforcementLearningTrainer,
    box_observation_encoder,
    onehot_observation_encoder,
)
from src.trainer.rl_trainer_registry import TRAINER_REGISTRY

# ---------------------------------------------------------------------
# Environments (pluggable, analogous to get_dataset/get_dataloaders)
# ---------------------------------------------------------------------


#: The single source of truth mapping every supported friendly ``--env`` name
#: to its Gymnasium id, in the order the names are offered on the command line.
#: Both the discrete and continuous tasks live here, so :data:`ENV_CHOICES`
#: (the CLI choices) and ``src.examples.visualize_rl``'s reverse ``env_id ->
#: name`` lookup are derived from this one mapping.
ENV_IDS: dict[str, str] = {
    "cartpole": "CartPole-v1",
    "acrobot": "Acrobot-v1",
    "mountaincar": "MountainCar-v0",
    "mountaincar_continuous": "MountainCarContinuous-v0",
    "frozenlake": "FrozenLake-v1",
    "pendulum": "Pendulum-v1",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
    "halfcheetah": "HalfCheetah-v5",
    "ant": "Ant-v5",
    "humanoid": "Humanoid-v5",
}

#: The subset of :data:`ENV_IDS` that are continuous (``Box``-action) tasks.
#: MountainCarContinuous and Pendulum are classic control; the rest are MuJoCo
#: tasks (require
#: ``gymnasium[mujoco]``). Their observation size, action dimensionality, and
#: action bounds are read from the environment at build time by
#: :func:`make_continuous_environment` rather than hardcoded, since these
#: differ across Gymnasium versions (e.g. Ant/Humanoid observation sizes).
CONTINUOUS_ENVS: frozenset[str] = frozenset(
    {
        "mountaincar_continuous",
        "pendulum",
        "hopper",
        "walker2d",
        "halfcheetah",
        "ant",
        "humanoid",
    }
)

#: All environment names understood by :func:`make_environment`, in the order
#: they are offered on the command line (derived from :data:`ENV_IDS`).
ENV_CHOICES: tuple[str, ...] = tuple(ENV_IDS)


def make_continuous_environment(env_id: str, **env_kwargs) -> RLEnvironment:
    """Builds a continuous-action :class:`RLEnvironment` by probing the env.

    The environment is instantiated once to read its observation size, action
    dimensionality, and per-dimension action bounds directly from its Gym
    spaces -- so the ``RLEnvironment`` is always consistent with the installed
    Gymnasium/MuJoCo version rather than relying on hardcoded dimensions.

    Args:
        env_id: Gymnasium id of a continuous ``Box``-action environment.
        **env_kwargs: Extra keyword arguments forwarded to ``gym.make`` (also
            stored on the returned environment so trainers re-create it
            identically).

    Returns:
        A configured continuous :class:`RLEnvironment`.
    """

    probe = gym.make(env_id, **env_kwargs)
    try:
        n_observation_features = int(np.prod(probe.observation_space.shape))
        action_space = probe.action_space
        action_dim = int(np.prod(action_space.shape))
        action_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    finally:
        probe.close()

    return RLEnvironment(
        env_id=env_id,
        n_actions=action_dim,
        n_observation_features=n_observation_features,
        obs_encoder=box_observation_encoder(),
        env_kwargs=env_kwargs or None,
        continuous=True,
        action_low=action_low,
        action_high=action_high,
    )


def make_environment(name: str, **kwargs) -> RLEnvironment:
    """Builds an :class:`RLEnvironment` for one of the supported tasks.

    Adding a new environment is a matter of returning another
    ``RLEnvironment`` here (or constructing one directly), so this function
    is the single place that maps a human-friendly name to a fully-specified,
    trainer-ready environment.

    Args:
        name: Environment name; one of :data:`ENV_CHOICES`. The discrete tasks
            are ``"cartpole"``, ``"acrobot"``, ``"mountaincar"`` and
            ``"frozenlake"``; the continuous (``Box``-action) tasks are the
            members of :data:`CONTINUOUS_ENVS` (``"mountaincar_continuous"``,
            ``"pendulum"``, ``"hopper"``, ``"walker2d"``, ``"halfcheetah"``,
            ``"ant"``, ``"humanoid"``).
        **kwargs: Environment-specific options (e.g. ``map_name`` and
            ``is_slippery`` for FrozenLake).

    Returns:
        A configured :class:`RLEnvironment`.

    Raises:
        ValueError: If ``name`` is not a supported environment.
    """

    if name not in ENV_IDS:
        raise ValueError(f"Unknown environment: {name!r}")

    env_id = ENV_IDS[name]

    if name in CONTINUOUS_ENVS:
        # Continuous Box-action tasks (Pendulum + MuJoCo); only the policy-
        # gradient trainers (reinforce, actor_critic, ppo) support these.
        return make_continuous_environment(env_id)

    if name == "cartpole":
        # 4 continuous observation features, 2 discrete actions.
        return RLEnvironment(
            env_id=env_id,
            n_actions=2,
            n_observation_features=4,
            obs_encoder=box_observation_encoder(scales=np.array([2.4, 3.0, 0.21, 3.0])),
        )

    if name == "acrobot":
        # 6 continuous observation features, 3 discrete actions.
        return RLEnvironment(
            env_id=env_id,
            n_actions=3,
            n_observation_features=6,
            obs_encoder=box_observation_encoder(),
        )

    if name == "mountaincar":
        # 2 continuous observation features, 3 discrete actions.
        return RLEnvironment(
            env_id=env_id,
            n_actions=3,
            n_observation_features=2,
            obs_encoder=box_observation_encoder(scales=np.array([1.2, 0.07])),
        )

    # frozenlake (the only remaining discrete task)
    map_name = kwargs.get("map_name", "4x4")
    is_slippery = kwargs.get("is_slippery", False)
    n_states = 16 if map_name == "4x4" else 64
    # Discrete integer observation -> one-hot; 4 discrete actions.
    # A non-slippery FrozenLake (fixed map, fixed start, deterministic
    # transitions) is fully deterministic, so greedy evaluation only needs
    # a single episode.
    return RLEnvironment(
        env_id=env_id,
        n_actions=4,
        n_observation_features=n_states,
        obs_encoder=onehot_observation_encoder(n_states),
        env_kwargs={"map_name": map_name, "is_slippery": is_slippery},
        deterministic=not is_slippery,
    )


def build_trainer(algo: str, **overrides) -> ReinforcementLearningTrainer:
    """Constructs a trainer for the requested algorithm.

    Args:
        algo: Algorithm name; one of the keys of
            ``src.trainer.reinforcement_trainer.TRAINER_REGISTRY``.
        **overrides: Constructor keyword arguments forwarded to the trainer
            (algorithm defaults; per-genome values from
            ``genome.hyperparameters`` still take precedence at train time).

    Returns:
        An instantiated :class:`ReinforcementLearningTrainer` subclass.

    Raises:
        ValueError: If ``algo`` is not a known trainer.
    """

    if algo not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown algorithm {algo!r}; choices: {sorted(TRAINER_REGISTRY)}"
        )

    trainer_class = TRAINER_REGISTRY[algo]

    # SARSA is the on-policy variant of the value-based trainer.
    if algo == "sarsa":
        overrides.setdefault("sarsa", True)

    return trainer_class(**overrides)


# ---------------------------------------------------------------------
# Objective and single-objective comparison
# ---------------------------------------------------------------------


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """Sorts genomes by fitness ``loss`` (lower is better).

    Fitness ``loss`` is set to the negative mean evaluation return, so
    sorting ascending by loss is equivalent to sorting descending by return
    -- matching the convention used by the classification example.

    Args:
        genome1: First genome.
        genome2: Second genome.

    Returns:
        Negative if ``genome1`` should sort first, positive if ``genome2``
        should sort first, and 0 if equivalent.
    """

    loss1 = float(genome1.fitness["loss"])
    loss2 = float(genome2.fitness["loss"])
    if loss1 < loss2:
        return -1
    if loss1 > loss2:
        return 1
    return 0


class ReinforcementLearningObjective(Objective):
    """Objective that trains a genome on an environment with a trainer.

    This mirrors ``ClassificationObjective``: it holds a pluggable trainer
    (the RL algorithm) and a pluggable environment (the RL task), trains the
    genome, and writes its ``fitness``.

    Args:
        environment: The target reinforcement-learning environment.
        trainer: The reinforcement-learning trainer (algorithm) to use.
    """

    def __init__(
        self,
        environment: RLEnvironment,
        trainer: ReinforcementLearningTrainer,
    ):
        self.environment = environment
        self.trainer = trainer

    def __call__(self, genome: CircuitGenome):
        """Trains and evaluates a genome, setting its fitness.

        Args:
            genome: The genome to train and evaluate. Its ``fitness``
                attribute is populated on return.
        """

        self.trainer.train(genome, self.environment)

        training_metrics = genome.metadata["best_training_metrics"]
        validation_metrics = genome.metadata["best_validation_metrics"]

        mean_return = (
            validation_metrics["return_mean"] + training_metrics["return_mean"]
        ) / 2.0
        # mean_return = validation_metrics["return_mean"]
        # mean_return = training_metrics["return_mean"]

        # "loss" (lower is better) drives population sorting via compare();
        # the remaining keys mirror the RL fields used by save_circuit's tag
        # fallback and by downstream analysis.
        genome.fitness = {
            "loss": -mean_return,
            "target_metric": validation_metrics["return_mean"],
            "eval_return_mean": validation_metrics["return_mean"],
            "eval_return_std": validation_metrics["return_std"],
            "train_return_mean": training_metrics["return_mean"],
            "best_episode_return": training_metrics["best_episode_return"],
            "env_id": self.environment.env_id,
        }

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"train_return_mean={genome.fitness['train_return_mean']:.2f} "
            f"best_episode_return={genome.fitness['best_episode_return']:.2f} "
            f"eval_return_mean={genome.fitness['eval_return_mean']:.2f} "
            f"eval_return_std={genome.fitness['eval_return_std']:.2f} "
            f"env={self.environment.env_id}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env",
        choices=list(ENV_CHOICES),
        required=True,
    )
    p.add_argument(
        "--algo",
        choices=sorted(TRAINER_REGISTRY.keys()),
        required=True,
        default="reinforce",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs",
    )

    p.add_argument(
        "--mutation_strategy",
        "-ms",
        type=str,
        nargs="+",
        required=True,
    )
    p.add_argument(
        "--parent_strategy",
        "-ps",
        type=str,
        nargs="+",
        required=True,
    )
    p.add_argument(
        "--binary_crossover_rate",
        type=float,
        default=0.00,
        help="Fraction of genomes generated via binary crossover once the population is initialized.",
    )
    p.add_argument(
        "--n_ary_crossover_rate",
        type=float,
        default=0.20,
        help="Fraction of genomes generated via n-ary crossover once the population is initialized.",
    )
    p.add_argument(
        "--exponential_crossover_rate",
        type=float,
        default=0.10,
        help="Fraction of genomes generated via exponential crossover once the population is initialized.",
    )

    subparsers = p.add_subparsers(
        dest="population_strategy",
        help="Specify how genomes will be handled.",
        required=True,
    )

    steady_state_parser = subparsers.add_parser(
        "steady_state", help="Use a single steady state population."
    )
    steady_state_parser.add_argument("--max_population_size", type=int, default=30)

    islands_parser = subparsers.add_parser(
        "islands", help="Use multiple islands of steady state populations."
    )
    islands_parser.add_argument("--n_islands", type=int, default=10)
    islands_parser.add_argument("--max_island_size", type=int, default=10)
    islands_parser.add_argument("--genomes_before_extinction", type=int, default=100)
    islands_parser.add_argument("--genomes_for_next_extinction", type=int, default=200)
    islands_parser.add_argument("--islands_to_extinct", type=int, default=1)
    islands_parser.add_argument("--primary_parent", type=str, default="best")
    islands_parser.add_argument(
        "--intra_island_crossover_rate", type=float, default=0.5
    )

    # Evolution
    p.add_argument("--number_genomes", type=int, default=500)
    p.add_argument("--input_qubits", type=int, default=4)
    p.add_argument("--output_qubits", type=int, default=None)

    p.add_argument(
        "--target", type=str, choices=["pennylane", "qiskit"], default="pennylane"
    )

    p.add_argument(
        "--quantum_input_mode",
        "-qim",
        choices=QUANTUM_INPUT_MODES,
        type=str,
        default="u3",
        help="Initial gate types whose parameters are set from the encoded observation.",
    )
    p.add_argument(
        "--quantum_output_mode",
        "-qom",
        choices=QUANTUM_OUTPUT_MODES,
        type=str,
        default="probs",
        help="Output mode from the quantum circuit.",
    )
    p.add_argument(
        "--encoding",
        choices=ENCODING_OPTIONS,
        type=str,
        default="linear",
        help="Observation-to-circuit encoding.",
    )
    p.add_argument(
        "--decoding",
        choices=DECODING_OPTIONS,
        type=str,
        default="linear",
        help="Circuit-output-to-action decoding.",
    )
    p.add_argument(
        "--quantum_dropout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Master switch for quantum dropout during training. Disabled by "
            "default; when enabled, dropout is sampled per training episode "
            "from --quantum_dropout_type and --quantum_dropout_rate and is "
            "never applied during greedy evaluation."
        ),
    )
    p.add_argument(
        "--quantum_dropout_type",
        "-qdt",
        type=str,
        default="none",
        choices=["gate", "rotation", "entangling", "qubit", "innovation"],
        help="Quantum dropout type (used only when --quantum_dropout is set).",
    )
    p.add_argument(
        "--quantum_dropout_rate",
        "-qdr",
        type=float,
        default=0.0,
        help="Quantum dropout rate (used only when --quantum_dropout is set).",
    )

    # RL hyperparameters (become genome.hyperparameters, mutable by the search)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning_rate", "-lr", type=float, default=1e-2)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--baseline", choices=["mean", "none"], default="mean")
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument(
        "--save_training_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also save a line plot of return and loss per episode next to each "
            "saved genome's diagram."
        ),
    )
    p.add_argument(
        "--ema_alpha",
        type=float,
        default=0.05,
        help="Smoothing factor for the exponential moving average of episode "
        "returns reported as the training return mean.",
    )

    # PPO extras
    p.add_argument("--rollout_steps", type=int, default=512)
    p.add_argument(
        "--ppo_passes",
        type=int,
        default=4,
        help="Passes over each PPO rollout (PPO literature calls these 'epochs').",
    )
    p.add_argument("--ppo_minibatch", type=int, default=128)
    p.add_argument("--ppo_clip", type=float, default=0.2)
    p.add_argument("--gae_lambda", type=float, default=0.95)

    # Value-based extras
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_min", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.995)

    # FrozenLake options
    p.add_argument("--map_name", choices=["4x4", "8x8"], default="4x4")
    p.add_argument("--is_slippery", action="store_true")

    p.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="DEBUG/INFO/WARNING/ERROR/CRITICAL",
    )

    args = p.parse_args()

    logger.remove()
    os.makedirs(args.out_dir, exist_ok=True)
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"))

    # -----------------------------------------------------------------
    # Environment + trainer + objective
    # -----------------------------------------------------------------
    environment = make_environment(
        args.env,
        map_name=args.map_name,
        is_slippery=args.is_slippery,
    )

    if environment.deterministic and args.eval_episodes > 1:
        logger.warning(
            f"environment {environment.env_id} is deterministic, so greedy "
            f"evaluation yields identical episodes; --eval_episodes="
            f"{args.eval_episodes} will be reduced to 1 during evaluation."
        )

    trainer = build_trainer(
        args.algo,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        log_every=args.log_every,
        ema_alpha=args.ema_alpha,
        entropy_coef=args.entropy_coef,
        baseline=args.baseline,
        value_coef=args.value_coef,
        gae_lambda=args.gae_lambda,
        rollout_steps=args.rollout_steps,
        ppo_passes=args.ppo_passes,
        ppo_minibatch=args.ppo_minibatch,
        ppo_clip=args.ppo_clip,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        quantum_dropout=args.quantum_dropout,
    )

    # Value-based trainers (q_learning / sarsa) enumerate discrete actions and
    # cannot drive a continuous Box-action environment; fail fast with a clear
    # message rather than deep inside the first weight update.
    if environment.continuous and not trainer.supports_continuous:
        p.error(
            f"algorithm {args.algo!r} does not support the continuous "
            f"environment {args.env!r}; use reinforce, actor_critic, or ppo."
        )

    objective = ReinforcementLearningObjective(environment=environment, trainer=trainer)

    # These become each genome's hyperparameters, so the evolutionary search
    # can carry/mutate them per genome (mirroring the classification example).
    hyperparameters = {
        "quantum_input_mode": args.quantum_input_mode,
        "quantum_output_mode": args.quantum_output_mode,
        "quantum_dropout_type": args.quantum_dropout_type,
        "quantum_dropout_rate": args.quantum_dropout_rate,
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "baseline": args.baseline,
        "value_coef": args.value_coef,
        "gae_lambda": args.gae_lambda,
        "rollout_steps": args.rollout_steps,
        "ppo_passes": args.ppo_passes,
        "ppo_minibatch": args.ppo_minibatch,
        "ppo_clip": args.ppo_clip,
        "epsilon": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "seed": args.seed,
        "log_every": args.log_every,
        "ema_alpha": args.ema_alpha,
    }

    target = args.target

    # -----------------------------------------------------------------
    # Encoder / decoder sizing (reuses the existing linear encoder/decoder)
    # -----------------------------------------------------------------
    n_input_registers = args.input_qubits
    if args.encoding == "identity":
        # The identity encoder passes its input straight through, so its output
        # size must equal its input size (the observation feature count) -- it
        # does not resize or clip to the qubit count.
        n_encoder_outputs = environment.n_observation_features
    else:
        n_encoder_outputs = n_input_registers
        if args.quantum_input_mode == "u3":
            n_encoder_outputs *= 3

    # The policy occupies environment.n_policy_outputs decoder outputs: one per
    # action for a discrete space, or a mean + log-std per action dimension for
    # a continuous space. Size the quantum output register so it has at least as
    # many features as the policy needs.
    n_output_registers = (
        int(args.output_qubits)
        if args.output_qubits is not None
        else max(1, int(np.ceil(np.log2(environment.n_policy_outputs))))
    )
    n_decoder_inputs = n_output_registers
    if args.quantum_output_mode == "probs":
        n_decoder_inputs = 2**n_output_registers

    # advantage methods (actor-critic, PPO) ask the decoder for one extra
    # output holding the scalar state value, so the value function is part of
    # the genome (evolved by crossover, preserved by serialization) rather
    # than a separate head.
    n_decoder_outputs = environment.n_policy_outputs + trainer.n_value_outputs

    # encoder: encoded observation (n_observation_features) -> quantum inputs
    initial_encoder = initialize_encoder(
        target=target,
        encoding_str=args.encoding,
        n_inputs=environment.n_observation_features,
        n_outputs=n_encoder_outputs,
        quantum_input_mode=args.quantum_input_mode,
        n_input_qubits=n_input_registers,
    )
    # decoder: quantum outputs -> per-action values (policy logits / Q-values),
    # plus an optional trailing state-value output for advantage methods.
    initial_decoder = initialize_decoder(
        target=target,
        decoding_str=args.decoding,
        n_inputs=n_decoder_inputs,
        n_outputs=n_decoder_outputs,
    )

    # -----------------------------------------------------------------
    # Population strategy
    # -----------------------------------------------------------------
    if args.population_strategy == "steady_state":
        population = SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
            save_training_plot=args.save_training_plot,
        )
    elif args.population_strategy == "islands":
        population = SteadyStateIslands(
            n_islands=args.n_islands,
            max_island_size=args.max_island_size,
            genomes_before_extinction=args.genomes_before_extinction,
            genomes_for_next_extinction=args.genomes_for_next_extinction,
            islands_to_extinct=args.islands_to_extinct,
            primary_parent=args.primary_parent,
            compare=compare,
            out_dir=args.out_dir,
            save_training_plot=args.save_training_plot,
        )
    else:
        raise ValueError(args.population_strategy)

    gate_specifications = (
        pennylane_gate_specifications
        if target == "pennylane"
        else qiskit_gate_specifications
    )

    logger.info(
        f"env={environment.env_id} algo={args.algo} target={target} "
        f"input_registers={{'input': {n_input_registers}}} "
        f"output_registers={{'input': {n_output_registers}}}"
    )

    master_worker(
        gate_specifications=gate_specifications,
        population=population,
        objective=objective,
        initial_encoder=initial_encoder,
        initial_decoder=initial_decoder,
        hyperparameters=hyperparameters,
        mutation_strategy=args.mutation_strategy,
        parent_strategy=args.parent_strategy,
        binary_crossover_rate=args.binary_crossover_rate,
        n_ary_crossover_rate=args.n_ary_crossover_rate,
        exponential_crossover_rate=args.exponential_crossover_rate,
        run_for=args.number_genomes,
        input_registers={"input": n_input_registers},
        output_registers={"input": n_output_registers},
        target=target,
    )
