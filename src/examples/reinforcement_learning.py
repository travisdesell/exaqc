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
* the same ``run_evolution`` evolutionary driver.

The RL algorithms live in :mod:`src.trainer.reinforcement_trainer` as
pluggable trainer classes (REINFORCE, actor-critic, PPO, Q-learning), exactly
mirroring how ``SupervisedTrainer`` is a pluggable component of the
classification objective. The environment is described by a pluggable
:class:`~src.trainer.reinforcement_trainer.RLEnvironment`, which the
:class:`ReinforcementLearningObjective` receives together with a trainer.

Example (single-process runs execute serially via ``EXAQC.run_for``; with more
than one MPI rank :func:`main` runs a master/worker search -- both selected
automatically by :func:`~src.evolution.master_worker.run_evolution`)::

    mpirun -n 4 python -m src.examples.reinforcement_learning \\
        --env cartpole --algo ppo -ms uniform 1 3 -ps uniform 2 3 \\
        --target pennylane steady_state
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import gymnasium as gym
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import initialize_decoder
from src.circuits.encoder import initialize_encoder
from src.circuits.gate_specifications import GateSpecifications
from src.evolution.exaqc import EXAQC
from src.evolution.master_worker import run_evolution
from src.evolution.objective import Objective
from src.evolution.population_strategy import PopulationStrategy

from src.trainer.reinforcement_trainer import (
    RLEnvironment,
    ReinforcementLearningTrainer,
    box_observation_encoder,
    onehot_observation_encoder,
)
from src.trainer.ppo_trainer import PPOTrainer
from src.trainer.q_learning_trainer import QLearningTrainer
from src.trainer.reinforce_trainer import ReinforceTrainer
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


def build_trainer(algo: str) -> ReinforcementLearningTrainer:
    """Constructs a trainer for the requested algorithm.

    Every training hyperparameter (including the quantum-dropout master switch)
    is carried per genome in ``genome.hyperparameters`` and resolved by the
    trainer at train time, so no hyperparameters are passed here -- the only
    construction-time input is the algorithm choice itself, which also selects
    the on-policy SARSA variant of the value-based trainer.

    Args:
        algo: Algorithm name; one of the keys of
            ``src.trainer.rl_trainer_registry.TRAINER_REGISTRY``.

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

    # SARSA is the on-policy variant of the value-based trainer, selected by a
    # constructor flag rather than a per-genome hyperparameter.
    if algo == "sarsa":
        return trainer_class(sarsa=True)

    return trainer_class()


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
        train_vs_validation_bias: float = 0.1,
    ):
        self.environment = environment
        self.trainer = trainer
        self.train_vs_validation_bias = train_vs_validation_bias

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
            self.train_vs_validation_bias * validation_metrics["return_mean"]
        ) + ((1.0 - self.train_vs_validation_bias) * training_metrics["return_mean"])

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
# Command-line interface
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the reinforcement-learning experiment.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description="Evolve quantum genomes for reinforcement learning with EXAQC."
    )
    parser.add_argument(
        "--env",
        choices=list(ENV_CHOICES),
        required=True,
        help="Gymnasium environment to evolve policies on.",
    )

    parser.add_argument(
        "--algo",
        choices=sorted(TRAINER_REGISTRY.keys()),
        required=True,
        default="reinforce",
        help="Reinforcement-learning algorithm used to train each genome.",
    )

    # The evolutionary search's own flags -- mutation/parent strategies,
    # crossover rates, the genome budget, --out_dir and --save_training_plot --
    # are owned by EXAQC so every entry point stays in sync.
    EXAQC.initialize_parser(parser)

    # The choice of population strategy (and each strategy's own flags) is owned
    # by PopulationStrategy.
    PopulationStrategy.initialize_parser(parser)

    # The backend (--target) and optional gate-set restriction (--use_only) are
    # owned by GateSpecifications.
    GateSpecifications.initialize_parser(parser)

    # The circuit-genome flags (qubit counts, quantum input/output modes,
    # encoder/decoder, quantum dropout) are owned by CircuitGenome so every
    # entry point stays in sync.
    CircuitGenome.initialize_parser(parser)

    # The training-loop hyperparameters are owned by the RL trainer classes so
    # each flag lives with the code that reads it: the base trainer owns the
    # knobs common to every algorithm (plus the policy-gradient entropy/value
    # coefficients), and each algorithm's extras come from its own class. All of
    # them are registered regardless of --algo so the full flag set is available.
    ReinforcementLearningTrainer.initialize_parser(parser)
    ReinforceTrainer.initialize_parser(parser)
    PPOTrainer.initialize_parser(parser)
    QLearningTrainer.initialize_parser(parser)

    # FrozenLake options
    parser.add_argument(
        "--map_name",
        choices=["4x4", "8x8"],
        default="4x4",
        help="FrozenLake grid size (used only for the frozenlake environment).",
    )

    parser.add_argument(
        "--is_slippery",
        action="store_true",
        help="Enable stochastic (slippery) transitions for the frozenlake environment.",
    )

    parser.add_argument(
        "--train_vs_validation_bias",
        "-tvb",
        type=float,
        default=0.01,
        help="Weights how the loss is calculated as (<tvb> * train_return) + ((1.0 - <tvb>) * validation_return)).",
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="DEBUG/INFO/WARNING/ERROR/CRITICAL",
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Runs a reinforcement-learning experiment."""

    parser = build_parser()
    args = parser.parse_args()

    # The output directory is created by the EXAQC constructor; loguru creates
    # the run.log parent directory as needed when the file sink is added.
    logger.remove()
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

    # All training hyperparameters are carried per genome (see the
    # ``hyperparameters`` dict below) and resolved by the trainer at train time,
    # so the trainer itself is constructed with only the algorithm choice.
    trainer = build_trainer(args.algo)

    # Value-based trainers (q_learning / sarsa) enumerate discrete actions and
    # cannot drive a continuous Box-action environment; fail fast with a clear
    # message rather than deep inside the first weight update.
    if environment.continuous and not trainer.supports_continuous:
        parser.error(
            f"algorithm {args.algo!r} does not support the continuous "
            f"environment {args.env!r}; use reinforce, actor_critic, or ppo."
        )

    # The objective is built on every rank because worker ranks evaluate genomes
    # with it; only the search machinery in build_exaqc is master/serial-only.
    objective = ReinforcementLearningObjective(
        environment=environment,
        trainer=trainer,
        train_vs_validation_bias=args.train_vs_validation_bias,
    )

    target = args.target

    def build_exaqc() -> EXAQC:
        """Builds the EXAQC search for the serial run or the MPI master.

        Worker ranks never call this, so the encoder/decoder sizing, population
        strategy, gate set and the rest of the search machinery are only
        constructed where they are actually driven.

        Returns:
            The fully-configured :class:`~src.evolution.exaqc.EXAQC` search,
            wrapping the ``objective`` and ``environment`` built above.
        """

        # These become each genome's hyperparameters, so the evolutionary search
        # can carry/mutate them per genome (mirroring the classification example).
        hyperparameters = {
            "quantum_input_mode": args.quantum_input_mode,
            "quantum_output_mode": args.quantum_output_mode,
            "algo": args.algo,
            "quantum_dropout": args.quantum_dropout,
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
            "improvement_cutoff": args.improvement_cutoff,
        }

        # -----------------------------------------------------------------
        # Encoder / decoder sizing (reuses the existing linear encoder/decoder)
        # -----------------------------------------------------------------
        n_input_registers = args.input_qubits
        if args.encoding == "identity":
            # The identity encoder passes its input straight through, so its
            # output size must equal its input size (the observation feature
            # count) -- it does not resize or clip to the qubit count.
            n_encoder_outputs = environment.n_observation_features
        else:
            n_encoder_outputs = n_input_registers
            if args.quantum_input_mode == "u3":
                n_encoder_outputs *= 3

        # The policy occupies environment.n_policy_outputs decoder outputs: one
        # per action for a discrete space, or a mean + log-std per action
        # dimension for a continuous space. The output register must be wide
        # enough to carry them; --output_qubits is required, so it is used as-is.
        n_output_registers = int(args.output_qubits)
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
        # decoder: quantum outputs -> per-action values (policy logits /
        # Q-values), plus an optional trailing state-value output for advantage
        # methods.
        initial_decoder = initialize_decoder(
            target=target,
            decoding_str=args.decoding,
            n_inputs=n_decoder_inputs,
            n_outputs=n_decoder_outputs,
        )

        logger.info(
            f"env={environment.env_id} algo={args.algo} target={target} "
            f"input_registers={{'input': {n_input_registers}}} "
            f"output_registers={{'input': {n_output_registers}}}"
        )

        # The gate set and population strategy are built from `args` by their own
        # factories (which every entry point shares); only the task-specific
        # encoder/decoder sizing, hyperparameters and register layout are
        # computed here.
        return EXAQC(
            gate_specifications=GateSpecifications.from_args(args),
            population=PopulationStrategy.from_args(args, compare),
            objective=objective,
            initial_encoder=initial_encoder,
            initial_decoder=initial_decoder,
            hyperparameters=hyperparameters,
            mutation_strategy=args.mutation_strategy,
            parent_strategy=args.parent_strategy,
            binary_crossover_rate=args.binary_crossover_rate,
            n_ary_crossover_rate=args.n_ary_crossover_rate,
            exponential_crossover_rate=args.exponential_crossover_rate,
            input_registers={"input": n_input_registers},
            output_registers={"input": n_output_registers},
            task="reinforcement_learning",
            task_target=args.env,
        )

    run_evolution(
        objective=objective,
        build_exaqc=build_exaqc,
        run_for=args.number_genomes,
    )


if __name__ == "__main__":
    main()
