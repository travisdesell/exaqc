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
classification objective. The environments, the trainer factory and the
objective live in :mod:`src.objectives.reinforcement_learning_objective`, which
the single-genome tools share; this module only adds the command line and the
search around them.

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

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import initialize_decoder
from src.circuits.encoder import initialize_encoder
from src.circuits.gate_specifications import GateSpecifications
from src.evolution.exaqc import EXAQC
from src.evolution.master_worker import run_evolution
from src.evolution.population_strategy import PopulationStrategy
from src.objectives.reinforcement_learning_objective import (
    ENV_CHOICES,
    ReinforcementLearningObjective,
    add_environment_knob_arguments,
    build_trainer,
    compare,
    environment_knob_kwargs,
    make_environment,
)
from src.utils import restart

from src.trainer.reinforcement_trainer import ReinforcementLearningTrainer
from src.trainer.ppo_trainer import PPOTrainer
from src.trainer.q_learning_trainer import QLearningTrainer
from src.trainer.reinforce_trainer import ReinforceTrainer
from src.utils.genome_archive import GenomeArchive
from src.trainer.rl_trainer_registry import TRAINER_REGISTRY

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
    # crossover rates and the genome budget -- are owned by EXAQC so every entry
    # point stays in sync.
    EXAQC.initialize_parser(parser)

    # Where and how the run's outputs are written (--out_dir,
    # --shared_file_system) is owned by GenomeArchive.
    GenomeArchive.initialize_parser(parser)

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

    # MuJoCo reward / termination / reset knobs
    add_environment_knob_arguments(parser)

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
        default=0.1,
        help="Weights how the loss is calculated: -((<tvb> * train_return) + ((1.0 - <tvb>) * validation_return)).",
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

    # Decided before anything is built, and on every rank: a restarted run takes
    # its configuration from the run it continues, so the objective a worker
    # evaluates with matches the search the master restores.
    args, restart_state, run_for = restart.prepare(args, parser.error)

    # The output directory is created by GenomeArchive.from_args (on the serial
    # run or MPI master); loguru creates the run.log parent directory as needed
    # when the file sink is added.
    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    if args.save_run_log:
        logger.add(os.path.join(args.out_dir, "run.log"), level=args.logging_level)

    # -----------------------------------------------------------------
    # Environment + trainer + objective
    # -----------------------------------------------------------------
    try:
        env_kwargs = environment_knob_kwargs(args, args.env)
    except ValueError as error:
        parser.error(str(error))

    environment = make_environment(
        args.env,
        env_kwargs=env_kwargs,
        map_name=args.map_name,
        is_slippery=args.is_slippery,
    )

    if env_kwargs:
        logger.info(f"environment knobs overriding {args.env} defaults: {env_kwargs}")

    # All training hyperparameters are carried per genome (see the
    # ``hyperparameters`` dict below) and resolved by the trainer at train time,
    # so the trainer itself is constructed with only the algorithm choice.
    trainer = build_trainer(args.algo)

    resolved_eval_policy = (
        trainer.natural_eval_policy if args.eval_policy == "match" else args.eval_policy
    )

    # Identical episodes are only a concern for a greedy rollout: a stochastic
    # policy varies episode to episode even in a deterministic environment.
    if (
        environment.deterministic
        and args.eval_episodes > 1
        and resolved_eval_policy == "greedy"
    ):
        logger.warning(
            f"environment {environment.env_id} is deterministic, so greedy "
            f"evaluation yields identical episodes; --eval_episodes="
            f"{args.eval_episodes} will be reduced to 1 during evaluation."
        )

    logger.info(
        f"evaluating genomes under the {resolved_eval_policy!r} policy "
        f"(--eval_policy {args.eval_policy}, --algo {args.algo})"
    )

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
            "eval_policy": args.eval_policy,
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
            "eval_seed": args.eval_seed,
            # the environment a genome was evolved against is part of what its
            # fitness means, so the single-genome tools can rebuild it exactly
            "env_kwargs": env_kwargs,
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
        population = (
            PopulationStrategy.from_args(args, compare)
            if restart_state is None
            else restart.restored_strategy(restart_state, args, compare)
        )

        search = EXAQC(
            gate_specifications=GateSpecifications.from_args(args),
            population=population,
            archive=GenomeArchive.from_args(args, restarting=restart_state is not None),
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
            restarting=restart_state is not None,
        )

        if restart_state is not None:
            restart.resume(search, restart_state, args)

        return search

    run_evolution(
        objective=objective,
        build_exaqc=build_exaqc,
        run_for=run_for,
    )


if __name__ == "__main__":
    main()
