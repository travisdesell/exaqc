"""Evolve quantum circuits to imitate a reference ("teacher") circuit.

This is the quantum-teacher counterpart to :mod:`src.examples.classification`
and :mod:`src.examples.reinforcement_learning`, and reuses the same building
blocks: the genome's ``initialize_model`` / ``forward`` interface, the shared
:class:`~src.trainer.supervised_trainer.SupervisedTrainer`, an
:class:`~src.evolution.objective.Objective` that trains a genome and sets its
fitness, the same population strategies, and the same ``run_evolution``
evolutionary driver.

What differs from classification is that there is nothing classical to learn.
A teacher is itself a :class:`~src.circuits.circuit.CircuitGenome`
(:mod:`src.circuits.teacher_circuits`), and the students evolved to imitate it
carry **no encoder and no decoder**: inputs are fed straight into the circuit
through ``quantum_input_mode`` and the outputs are the raw circuit readout. The
search is therefore over the circuit alone. Because of that this entry point has
no ``--encoding`` / ``--decoding`` options.

The dataset is generated rather than loaded: random input angles are drawn and
the teacher's outputs for them become the targets
(:mod:`src.datasets.teacher_loaders`).

Example::

    mpiexec -n 4 python3 -m src.examples.teacher --teacher half_adder \\
        --input_qubits 2 --output_qubits 2 --loss fidelity \\
        -ms uniform 1 3 -ps uniform 2 3 --binary_crossover_rate 0.1 \\
        --out_dir ./artifacts steady_state --max_population_size 30
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.gate_specifications import GateSpecifications
from src.circuits.teacher_circuits import DEFAULT_REGISTER_NAME, TEACHER_NAMES

from src.datasets.teacher_loaders import (
    TEACHER_INPUT_MODES,
    get_teacher_dataloaders,
)

from src.evolution.exaqc import EXAQC
from src.evolution.master_worker import run_evolution
from src.evolution.objective import Objective
from src.evolution.population_strategy import PopulationStrategy

from src.metrics.teacher_losses import TEACHER_LOSS_NAMES, get_teacher_loss
from src.metrics.teacher_metrics import build_teacher_metrics
from src.trainer.supervised_trainer import SupervisedTrainer


def compare(
    genome1: CircuitGenome,
    genome2: CircuitGenome,
) -> int:
    """Compares genomes using the minimized loss objective.

    Args:
        genome1: First genome.
        genome2: Second genome.

    Returns:
        Negative when ``genome1`` is better, positive when ``genome2`` is
        better, and zero when they are equal.
    """

    return genome1.fitness["loss"] - genome2.fitness["loss"]


class TeacherObjective(Objective):
    """Teacher-imitation objective backed by :class:`SupervisedTrainer`."""

    def __init__(
        self,
        training_dataloader,
        validation_dataloader,
        loss_name: str,
        device: str | None = None,
    ) -> None:
        """Initializes the teacher-imitation objective.

        Args:
            training_dataloader: Generated training loader.
            validation_dataloader: Generated validation loader.
            loss_name: Which measure to optimize; one of
                :data:`~src.metrics.teacher_losses.TEACHER_LOSS_NAMES`.
            device: PyTorch device to train on, or ``None`` to auto-select.
        """

        self.loss_name = loss_name
        loss_function = get_teacher_loss(loss_name)

        # Every measure is reported each epoch, whichever one is optimized, so
        # runs using different losses stay comparable after the fact.
        self.trainer = SupervisedTrainer(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            training_loss_function=loss_function,
            validation_loss_function=loss_function,
            metrics=build_teacher_metrics(),
            device=device,
        )

    def __call__(self, genome: CircuitGenome) -> None:
        """Trains a genome and assigns teacher-imitation fitness.

        Args:
            genome: Genome to train and evaluate.

        Returns:
            None. Sets ``genome.fitness`` with a minimized ``"loss"`` (the mean
            of the best training and validation loss) and a ``"target_metric"``
            holding the corresponding mean fidelity, matching the fitness keys
            the classification objective writes so the analysis tooling reads
            both the same way.
        """

        self.trainer.train(genome)

        training = genome.metadata["best_training_metrics"]
        validation = genome.metadata["best_validation_metrics"]

        genome.fitness = {
            "loss": (float(training["loss"]) + float(validation["loss"])) / 2.0,
            "target_metric": (
                float(training["fidelity"]["mean"])
                + float(validation["fidelity"]["mean"])
            )
            / 2.0,
        }


def teacher_wires(
    n_input_qubits: int,
    n_output_qubits: int,
) -> tuple[list[int], list[int]]:
    """Lays out the input and output wires a teacher circuit spans.

    Teachers read out wires that are *disjoint* from the ones the inputs drive
    (unlike the classification entry point, where the readout qubits are a
    prefix of the input register), so the inputs take the first wires and the
    outputs take the ones after them.

    Args:
        n_input_qubits: How many wires the classical inputs drive.
        n_output_qubits: How many wires are read out.

    Returns:
        A tuple of the input wire indices and the output wire indices.
    """

    input_wires = list(range(n_input_qubits))
    output_wires = list(range(n_input_qubits, n_input_qubits + n_output_qubits))
    return input_wires, output_wires


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the teacher-imitation experiment.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description=("Evolve quantum circuits to imitate a reference teacher circuit.")
    )

    parser.add_argument(
        "--teacher",
        choices=list(TEACHER_NAMES),
        required=True,
        help="Reference circuit the evolved circuits are trained to imitate.",
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
    # quantum dropout) are owned by CircuitGenome so every entry point stays in
    # sync. A teacher search is purely quantum: it seeds no encoder or decoder
    # (so --encoding/--decoding are omitted) and feeds inputs straight in through
    # a single-axis rotation.
    CircuitGenome.initialize_parser(
        parser,
        include_encoding_decoding=False,
        quantum_input_mode_choices=list(TEACHER_INPUT_MODES),
        quantum_input_mode_default="ry",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=list(TEACHER_LOSS_NAMES),
        default="fidelity",
        help=(
            "Measure optimized during training. 'fidelity', 'angle' and 'kl' "
            "treat the outputs as probability distributions, so they require "
            "-qom probs; 'mse' works with either output mode. Every measure is "
            "reported each epoch regardless of which is optimized."
        ),
    )

    parser.add_argument(
        "--n_training_samples",
        type=int,
        default=64,
        help="Number of teacher-labelled training samples to generate.",
    )

    parser.add_argument(
        "--n_validation_samples",
        type=int,
        default=64,
        help="Number of teacher-labelled validation samples to generate.",
    )

    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
        help="Batch size for validation; defaults to --batch_size when unset.",
    )

    # The supervised-training flags (epochs, learning rate, weight decay,
    # improvement cutoff, batch size) are owned by SupervisedTrainer so the
    # classification and teacher entry points stay in sync.
    SupervisedTrainer.initialize_parser(parser)

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "PyTorch device to use for training, e.g. 'cpu', 'cuda', or "
            "'cuda:0'. Defaults to CUDA when available."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the generated teacher dataset.",
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    return parser


def main() -> None:
    """Runs a quantum-teacher imitation experiment."""

    parser = build_parser()
    args = parser.parse_args()

    # The output directory is created by the EXAQC constructor; loguru creates
    # the run.log parent directory as needed when the file sink is added.
    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"))

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using PyTorch device: {}", device)

    if args.input_qubits < 1 or args.output_qubits < 1:
        parser.error("--input_qubits and --output_qubits must both be at least 1.")

    # The distribution measures compare probability vectors, which only the
    # probs readout produces.
    if args.loss != "mse" and args.quantum_output_mode != "probs":
        parser.error(
            f"--loss {args.loss} compares probability distributions and requires "
            "-qom probs; use --loss mse for the expval readout."
        )

    input_wires, output_wires = teacher_wires(args.input_qubits, args.output_qubits)

    # A teacher that cannot be built from these wires reports why.
    try:
        training_loader, validation_loader = get_teacher_dataloaders(
            teacher_name=args.teacher,
            input_wires=input_wires,
            output_wires=output_wires,
            target=args.target,
            quantum_input_mode=args.quantum_input_mode,
            quantum_output_mode=args.quantum_output_mode,
            n_training_samples=args.n_training_samples,
            n_validation_samples=args.n_validation_samples,
            batch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    # The objective is built on every rank because worker ranks evaluate genomes
    # with it; only the search machinery below is master/serial-only.
    objective = TeacherObjective(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        loss_name=args.loss,
        device=args.device,
    )

    def build_exaqc() -> EXAQC:
        """Builds the EXAQC search for the serial run or the MPI master.

        Worker ranks never call this, so the population strategy, gate set and
        the rest of the search machinery are only constructed where they are
        actually driven.

        Returns:
            The fully-configured :class:`~src.evolution.exaqc.EXAQC` search,
            wrapping the ``objective`` built above.
        """

        hyperparameters = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "improvement_cutoff": args.improvement_cutoff,
            "batch_size": args.batch_size,
            "quantum_input_mode": args.quantum_input_mode,
            "quantum_output_mode": args.quantum_output_mode,
            "quantum_dropout": args.quantum_dropout,
            "quantum_dropout_type": args.quantum_dropout_type,
            "quantum_dropout_rate": args.quantum_dropout_rate,
        }

        logger.info(
            "Imitating teacher '{}' on {} with input wires {} and output wires {} "
            "({} -> {}), optimizing {}.",
            args.teacher,
            args.target,
            input_wires,
            output_wires,
            args.quantum_input_mode,
            args.quantum_output_mode,
            args.loss,
        )

        # The gate set and population strategy are built from `args` by their own
        # factories (which every entry point shares). A teacher-imitation genome
        # is purely quantum: there is nothing classical to learn, so it carries
        # no encoder and no decoder, and its input/output wires are given as
        # explicit, disjoint qubit lists.
        return EXAQC(
            gate_specifications=GateSpecifications.from_args(args),
            population=PopulationStrategy.from_args(args, compare),
            objective=objective,
            initial_encoder=None,
            initial_decoder=None,
            hyperparameters=hyperparameters,
            mutation_strategy=args.mutation_strategy,
            parent_strategy=args.parent_strategy,
            binary_crossover_rate=args.binary_crossover_rate,
            n_ary_crossover_rate=args.n_ary_crossover_rate,
            exponential_crossover_rate=args.exponential_crossover_rate,
            input_qubits=[(DEFAULT_REGISTER_NAME, wire) for wire in input_wires],
            output_qubits=[(DEFAULT_REGISTER_NAME, wire) for wire in output_wires],
            task="teacher",
            task_target=args.teacher,
        )

    run_evolution(
        objective=objective,
        build_exaqc=build_exaqc,
        run_for=args.number_genomes,
    )


if __name__ == "__main__":
    main()
