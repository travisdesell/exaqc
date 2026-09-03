"""Evolve quantum circuits to imitate a reference ("teacher") circuit.

This is the quantum-teacher counterpart to :mod:`src.examples.classification`
and :mod:`src.examples.reinforcement_learning`, and reuses the same building
blocks: the genome's ``initialize_model`` / ``forward`` interface, the shared
:class:`~src.trainer.supervised_trainer.SupervisedTrainer`, an
:class:`~src.evolution.objective.Objective` that trains a genome and sets its
fitness, the same population strategies, and the same ``master_worker``
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
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.circuits.teacher_circuits import DEFAULT_REGISTER_NAME, TEACHER_NAMES

from src.datasets.teacher_loaders import (
    TEACHER_INPUT_MODES,
    TEACHER_OUTPUT_MODES,
    get_teacher_dataloaders,
)

from src.evolution.exaqc import EXAQC
from src.evolution.master_worker import master_worker
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation

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
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Directory to write per-genome artifacts (diagrams, plots, logs) into.",
    )

    # The evolutionary search's own flags (mutation/parent strategies and
    # crossover rates) are owned by EXAQC so every entry point stays in sync.
    EXAQC.initialize_parser(parser)

    populations = parser.add_subparsers(
        dest="population_strategy",
        required=True,
        help="Specify how genomes will be handled.",
    )
    # Each population strategy owns the flags for its own constructor.
    SteadyStatePopulation.initialize_parser(
        populations.add_parser(
            "steady_state", help="Use a single steady state population."
        )
    )
    SteadyStateIslands.initialize_parser(
        populations.add_parser(
            "islands", help="Use multiple islands of steady state populations."
        )
    )

    parser.add_argument(
        "--input_qubits",
        type=int,
        required=True,
        help=(
            "Number of wires the generated inputs drive. With no encoder these "
            "are fed straight into the circuit, so this is also the number of "
            "input values per sample."
        ),
    )
    parser.add_argument(
        "--output_qubits",
        type=int,
        required=True,
        help=(
            "Number of wires read out. These are disjoint from the input wires, "
            "so the circuit spans --input_qubits + --output_qubits wires."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["pennylane", "qiskit"],
        default="pennylane",
        help="Quantum backend used to build and simulate the evolved circuits.",
    )
    parser.add_argument(
        "--quantum_input_mode",
        "-qim",
        type=str,
        choices=list(TEACHER_INPUT_MODES),
        default="ry",
        help=(
            "How each input value is encoded onto its wire. The teacher and the "
            "evolved circuits always share this encoding."
        ),
    )
    parser.add_argument(
        "--quantum_output_mode",
        "-qom",
        type=str,
        choices=list(TEACHER_OUTPUT_MODES),
        default="probs",
        help="Choose the output mode from the quantum circuit.",
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
        "--quantum_dropout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Master switch for quantum dropout during training. Disabled by "
            "default; when enabled, dropout is applied per training batch using "
            "--quantum_dropout_type and --quantum_dropout_rate."
        ),
    )
    parser.add_argument(
        "--quantum_dropout_type",
        "-qdt",
        type=str,
        default="none",
        choices=["gate", "rotation", "entangling", "qubit", "innovation"],
        help="Choose the dropout type for quantum gates (used only when --quantum_dropout is set).",
    )
    parser.add_argument(
        "--quantum_dropout_rate",
        "-qdr",
        type=float,
        default=0.0,
        help="Choose the dropout rate for quantum gates (used only when --quantum_dropout is set).",
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
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
        help="Batch size for validation; defaults to --batch_size when unset.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs per genome.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=5e-3,
        help="Adam learning rate used when training each genome.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Adam weight decay (L2 regularization) used when training each genome.",
    )
    parser.add_argument(
        "--improvement_cutoff",
        type=int,
        default=5,
        help="Stop training a genome after this many epochs without validation improvement.",
    )
    parser.add_argument(
        "--number_genomes",
        type=int,
        default=2000,
        help="Total number of genomes to evolve and evaluate before stopping.",
    )

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
        "--save_training_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also save a line plot of loss and fidelity per epoch next to each "
            "saved genome's diagram."
        ),
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    return parser


def main() -> None:
    """Runs the quantum-teacher imitation experiment."""

    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
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

    objective = TeacherObjective(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        loss_name=args.loss,
        device=args.device,
    )

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

    gate_specifications = (
        pennylane_gate_specifications
        if args.target == "pennylane"
        else qiskit_gate_specifications
    )

    if args.population_strategy == "steady_state":
        population = SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
            save_training_plot=args.save_training_plot,
        )
    else:
        population = SteadyStateIslands(
            n_islands=args.n_islands,
            max_island_size=args.max_island_size,
            genomes_before_extinction=args.genomes_before_extinction,
            genomes_for_next_extinction=args.genomes_for_next_extinction,
            islands_to_extinct=args.islands_to_extinct,
            primary_parent=args.primary_parent,
            intra_island_crossover_rate=args.intra_island_crossover_rate,
            compare=compare,
            out_dir=args.out_dir,
            save_training_plot=args.save_training_plot,
        )

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

    master_worker(
        gate_specifications=gate_specifications,
        population=population,
        objective=objective,
        # A teacher-imitation genome is purely quantum: there is nothing
        # classical to learn, so it carries no encoder and no decoder.
        initial_encoder=None,
        initial_decoder=None,
        hyperparameters=hyperparameters,
        mutation_strategy=args.mutation_strategy,
        parent_strategy=args.parent_strategy,
        binary_crossover_rate=args.binary_crossover_rate,
        n_ary_crossover_rate=args.n_ary_crossover_rate,
        exponential_crossover_rate=args.exponential_crossover_rate,
        run_for=args.number_genomes,
        input_qubits=[(DEFAULT_REGISTER_NAME, wire) for wire in input_wires],
        output_qubits=[(DEFAULT_REGISTER_NAME, wire) for wire in output_wires],
        target=args.target,
        task="teacher",
        task_target=args.teacher,
    )


if __name__ == "__main__":
    main()
