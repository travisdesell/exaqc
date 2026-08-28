"""Run multi-objective EXAQC classification experiments.

This example extends the standard EXAQC classification workflow with
multi-objective optimization using NSGA-II or NSGA-III. Both algorithms
may also be used with the steady-state island model.

All optimization objectives are internally represented as minimization
objectives. Objectives that should be maximized use ``sign=-1.0``.

Objective names can reference:

1. Metrics configured in the ``metrics`` dictionary, such as
   ``mean_class_accuracy``.
2. ``loss``.
3. Structural circuit objectives:
   ``n_parameters`` and ``n_gates``.

Examples:
    Run NSGA-II maximizing mean class accuracy while minimizing the
    number of trainable quantum parameters::

        mpiexec -n 4 python -m src.examples.classification_moo \
            --dataset iris \
            --input_qubits 4 \
            --output_qubits 2 \
            --number_genomes 500 \
            --objectives mean_class_accuracy:-1 n_parameters:1 \
            -ms uniform 1 3 \
            -ps uniform 2 3 \
            nsga2 \
            --max_population_size 30

    Run island-based NSGA-II::

        mpiexec -n 8 python -m src.examples.classification_moo \
            --dataset iris \
            --input_qubits 4 \
            --output_qubits 2 \
            --number_genomes 1000 \
            --objectives mean_class_accuracy:-1 n_parameters:1 \
            -ms uniform 1 3 \
            -ps uniform 2 3 \
            nsga2_islands \
            --n_islands 4 \
            --max_island_size 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.circuits.circuit import (
    CircuitGenome,
    QUANTUM_INPUT_MODES,
    QUANTUM_OUTPUT_MODES,
)
from src.circuits.decoder import (
    DECODING_OPTIONS,
    initialize_decoder,
)
from src.circuits.encoder import (
    ENCODING_OPTIONS,
    initialize_encoder,
)
from src.circuits.pennylane_gate_specifications import (
    pennylane_gate_specifications,
)
from src.circuits.qiskit_gate_specifications import (
    qiskit_gate_specifications,
)
from src.datasets.classification_loaders import (
    CLASSIFICATION_DATASETS,
    IMAGE_DATASETS,
    get_image_dataloaders,
    get_uci_dataloaders,
)
from src.evolution.master_worker import master_worker
from src.evolution.moo.islands import (
    MultiObjectiveSteadyStateIslands,
)
from src.evolution.moo.nsga2 import NSGA2
from src.evolution.moo.nsga3 import NSGA3
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.objective import Objective
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.metrics.metric import Metric
from src.trainer.supervised_trainer import SupervisedTrainer


STRUCTURAL_OBJECTIVES = {
    "n_parameters",
    "n_gates",
}


class MultiObjectiveClassificationObjective(Objective):
    """Classification objective for multi-objective evolution.

    The classification model is trained using ``SupervisedTrainer``.
    Fitness values are then constructed dynamically from the objectives
    selected by the user.

    Metric-based objectives are read from the trainer's best training and
    validation metrics. Structural objectives are calculated directly from
    the evolved circuit.

    Args:
        training_dataloader: Batched training data loader.
        validation_dataloader: Batched validation data loader.
        training_loss_function: Training loss function.
        validation_loss_function: Validation loss function.
        metrics: Metrics calculated during training and validation.
        objectives: Objectives selected for multi-objective optimization.
        device: PyTorch device used for model training.
    """

    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        training_loss_function: Any,
        validation_loss_function: Any,
        metrics: dict[str, Metric],
        objectives: list[ObjectiveSpec],
        device: str | None = None,
    ) -> None:
        """Initialize the classification objective."""
        self.metrics = metrics
        self.objectives = objectives

        self.trainer = SupervisedTrainer(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            training_loss_function=training_loss_function,
            validation_loss_function=validation_loss_function,
            metrics=metrics,
            device=device,
        )

    @staticmethod
    def _scalar_metric_value(
        value: Any,
    ) -> float:
        """Convert a trainer metric result into a scalar value.

        Some EXAQC metrics return dictionaries containing aggregate values
        such as ``{"mean": ...}``, while others may directly return scalar
        values.

        Args:
            value: Metric result produced by the trainer.

        Returns:
            Scalar representation of the metric.

        Raises:
            ValueError: If a dictionary metric does not contain ``mean``.
        """
        if isinstance(value, dict):
            if "mean" not in value:
                raise ValueError(
                    "Cannot use metric as a MOO objective because its "
                    "result does not contain a 'mean' value."
                )

            return float(value["mean"])

        return float(value)

    def _metric_objective(
        self,
        name: str,
        training: dict[str, Any],
        validation: dict[str, Any],
    ) -> float:
        """Calculate the fitness value for a trainer metric.

        The objective is the average of the best training and validation
        values, matching the behavior of the standard classification
        objective.

        Args:
            name: Name of the metric.
            training: Best training metrics stored on the genome.
            validation: Best validation metrics stored on the genome.

        Returns:
            Mean training and validation metric value.
        """
        training_value = self._scalar_metric_value(training[name])

        validation_value = self._scalar_metric_value(validation[name])

        return (training_value + validation_value) / 2.0

    def __call__(
        self,
        genome: CircuitGenome,
    ) -> None:
        """Train a genome and assign its multi-objective fitness.

        Only objectives explicitly selected by the user are stored in
        ``genome.fitness``.

        Args:
            genome: Genome to train and evaluate.

        Raises:
            ValueError: If an unsupported objective is encountered.
        """
        self.trainer.train(genome)

        training = genome.metadata["best_training_metrics"]

        validation = genome.metadata["best_validation_metrics"]

        enabled_gates = [gate for gate in genome.gates if gate.enabled]

        fitness: dict[str, float] = {}

        for objective in self.objectives:
            name = objective.name

            if name == "loss":
                fitness[name] = (
                    float(training["loss"]) + float(validation["loss"])
                ) / 2.0

            elif name == "n_parameters":
                fitness[name] = float(
                    sum(len(gate.parameters) for gate in enabled_gates)
                )

            elif name == "n_gates":
                fitness[name] = float(len(enabled_gates))

            elif name in self.metrics:
                fitness[name] = self._metric_objective(
                    name=name,
                    training=training,
                    validation=validation,
                )

            else:
                raise ValueError(f"Unsupported MOO objective '{name}'.")

        genome.fitness = fitness


def add_population_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add arguments shared by non-island MOO populations.

    Args:
        parser: Population subparser.
    """
    parser.add_argument(
        "--max_population_size",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--tournament_size",
        type=int,
        default=2,
    )


def add_island_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add arguments shared by MOO island populations.

    Args:
        parser: Island population subparser.
    """
    parser.add_argument(
        "--n_islands",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_island_size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tournament_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--intra_island_crossover_rate",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--genomes_before_extinction",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--genomes_for_next_extinction",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--islands_to_extinct",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--primary_parent",
        type=str,
        choices=[
            "best",
            "island",
        ],
        default="best",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the multi-objective classification command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-objective EXAQC classification using NSGA-II, "
            "NSGA-III, or steady-state island variants."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=CLASSIFICATION_DATASETS,
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
    )

    parser.add_argument(
        "--mutation_strategy",
        "-ms",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--parent_strategy",
        "-ps",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--objectives",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Multi-objective fitness specifications in NAME:SIGN form. "
            "Metric names must correspond to entries in the metrics "
            "dictionary. Built-in structural objectives are "
            "'n_parameters' and 'n_gates'. "
            "Use -1 to maximize and 1 to minimize. Example: "
            "--objectives mean_class_accuracy:-1 n_parameters:1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--improvement_cutoff",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--number_genomes",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--input_qubits",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output_qubits",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--target",
        type=str,
        choices=[
            "pennylane",
            "qiskit",
        ],
        default="pennylane",
    )

    parser.add_argument(
        "--quantum_input_mode",
        "-qim",
        choices=QUANTUM_INPUT_MODES,
        default="u3",
        help=(
            "Choose the initial gate type whose parameters are set "
            "from classical inputs."
        ),
    )

    parser.add_argument(
        "--quantum_output_mode",
        "-qom",
        type=str,
        choices=QUANTUM_OUTPUT_MODES,
        default="probs",
        help="Choose the output mode from the quantum circuit.",
    )

    parser.add_argument(
        "--quantum_dropout_type",
        "-qdt",
        type=str,
        default="none",
        choices=[
            "none",
            "gate",
            "rotation",
            "entangling",
            "qubit",
            "innovation",
        ],
        help="Choose the dropout type for quantum gates.",
    )

    parser.add_argument(
        "--quantum_dropout_rate",
        "-qdr",
        type=float,
        default=0.0,
        help="Choose the dropout rate for quantum gates.",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        choices=ENCODING_OPTIONS,
        default="linear",
        help="Choose the classical-to-quantum encoding.",
    )

    parser.add_argument(
        "--encoder_config",
        type=str,
        default="configs",
    )

    parser.add_argument(
        "--decoding",
        type=str,
        choices=DECODING_OPTIONS,
        default="linear",
        help="Choose the quantum-to-classical decoding.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Batch size for every dataset. Use 1 for per-sample UCI "
            "execution and larger values for image datasets."
        ),
    )

    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )

    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--training_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--validation_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "PyTorch device to use for training, e.g. " "'cpu', 'cuda', or 'cuda:0'."
        ),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--cnn_channels",
        type=int,
        nargs=2,
        default=[16, 32],
    )

    parser.add_argument(
        "--cnn_pooled_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--cnn_dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=[
            "none",
            "zscore",
            "minmax",
        ],
        default="minmax",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help=(
            "Logging level shown on the terminal. "
            "Use DEBUG to show all logging messages."
        ),
    )

    populations = parser.add_subparsers(
        dest="population_strategy",
        required=True,
    )

    nsga2 = populations.add_parser(
        "nsga2",
        help="Use a single NSGA-II population.",
    )

    add_population_arguments(nsga2)

    nsga3 = populations.add_parser(
        "nsga3",
        help="Use a single NSGA-III population.",
    )

    add_population_arguments(nsga3)

    nsga3.add_argument(
        "--reference_divisions",
        type=int,
        default=8,
    )

    nsga2_islands = populations.add_parser(
        "nsga2_islands",
        help=("Use the steady-state island model with NSGA-II " "within each island."),
    )

    add_island_arguments(nsga2_islands)

    nsga3_islands = populations.add_parser(
        "nsga3_islands",
        help=("Use the steady-state island model with NSGA-III " "within each island."),
    )

    add_island_arguments(nsga3_islands)

    nsga3_islands.add_argument(
        "--reference_divisions",
        type=int,
        default=8,
    )

    return parser


def load_data(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    """Load tabular or image classification data.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Training and validation data loaders.
    """
    if args.dataset in IMAGE_DATASETS:
        training_loader, validation_loader = get_image_dataloaders(
            args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            validation_batch_size=(args.validation_batch_size),
            validation_fraction=(args.validation_fraction),
            training_samples=(args.training_samples),
            validation_samples=(args.validation_samples),
            seed=args.seed,
            download=(args.download_dataset),
            num_workers=(args.num_workers),
            pin_memory=(args.pin_memory),
        )

        return (
            training_loader,
            validation_loader,
        )

    training_loader, validation_loader = get_uci_dataloaders(
        args.dataset,
        normalize=args.normalization,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    return (
        training_loader,
        validation_loader,
    )


def load_encoder_config(
    path: str | None,
) -> dict:
    """Load an encoder configuration file.

    Args:
        path: JSON configuration path or ``None``.

    Returns:
        Parsed encoder configuration.

    Raises:
        ValueError: If the configuration does not contain a JSON object.
    """
    if path is None:
        return {}

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError("Encoder configuration must contain a JSON object.")

    return config


def build_metrics(
    training_loader: DataLoader,
) -> dict[str, Metric]:
    """Construct metrics available to the classification trainer.

    Add additional EXAQC metrics here to automatically make them available
    as MOO objectives through ``--objectives``.

    Args:
        training_loader: Training data loader containing label metadata.

    Returns:
        Mapping of metric names to metric objects.
    """
    return {
        "mean_class_accuracy": MeanClassAccuracy(training_loader.n_labels),
    }


def build_objectives(
    args: argparse.Namespace,
    metrics: dict[str, Metric],
) -> list[ObjectiveSpec]:
    """Construct user-defined MOO objectives.

    Each objective is specified as ``NAME:SIGN``. Metric objective names
    must correspond to entries in the existing ``metrics`` dictionary.
    ``loss``, ``n_parameters``, and ``n_gates`` are also supported.

    A sign of ``1`` preserves minimization. A sign of ``-1`` converts a
    maximization objective into minimization form.

    Args:
        args: Parsed command-line arguments.
        metrics: Metrics configured for the supervised trainer.

    Returns:
        Objective specifications.

    Raises:
        ValueError: If fewer than two objectives are specified.
        ValueError: If an objective specification is malformed.
        ValueError: If an objective name is unavailable.
        ValueError: If an objective sign is not ``1`` or ``-1``.
        ValueError: If the same objective is specified multiple times.
    """
    available_objectives = set(metrics.keys()) | STRUCTURAL_OBJECTIVES | {"loss"}

    objectives: list[ObjectiveSpec] = []
    objective_names: set[str] = set()

    for specification in args.objectives:
        try:
            name, sign_string = specification.rsplit(
                ":",
                maxsplit=1,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid objective specification '{specification}'. "
                "Expected NAME:SIGN, for example "
                "'mean_class_accuracy:-1'."
            ) from exc

        name = name.strip()

        if name not in available_objectives:
            raise ValueError(
                f"Unknown objective '{name}'. Available objectives are: "
                f"{sorted(available_objectives)}."
            )

        if name in objective_names:
            raise ValueError(f"Objective '{name}' was specified more than once.")

        try:
            sign = float(sign_string)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sign '{sign_string}' for objective '{name}'. "
                "Expected 1 or -1."
            ) from exc

        if sign not in {
            -1.0,
            1.0,
        }:
            raise ValueError(
                f"Objective '{name}' has sign {sign}. "
                "Expected 1 for minimization or -1 for maximization."
            )

        objectives.append(
            ObjectiveSpec(
                name=name,
                sign=sign,
            )
        )

        objective_names.add(name)

    if len(objectives) < 2:
        raise ValueError(
            "Multi-objective optimization requires at least " "two objectives."
        )

    return objectives


def build_population(
    args: argparse.Namespace,
    objectives: list[ObjectiveSpec],
):
    """Construct the requested MOO population strategy.

    Args:
        args: Parsed command-line arguments.
        objectives: User-selected MOO objectives.

    Returns:
        Configured NSGA-II, NSGA-III, or island-based population.

    Raises:
        ValueError: If the population strategy is unsupported.
    """
    if args.population_strategy == "nsga2":
        return NSGA2(
            max_population_size=(args.max_population_size),
            objectives=objectives,
            tournament_size=(args.tournament_size),
            out_dir=args.out_dir,
            seed=args.seed,
        )

    if args.population_strategy == "nsga3":
        return NSGA3(
            max_population_size=(args.max_population_size),
            objectives=objectives,
            tournament_size=(args.tournament_size),
            reference_divisions=(args.reference_divisions),
            out_dir=args.out_dir,
            seed=args.seed,
        )

    if args.population_strategy == "nsga2_islands":
        return MultiObjectiveSteadyStateIslands(
            population_class=NSGA2,
            objectives=objectives,
            n_islands=args.n_islands,
            max_island_size=(args.max_island_size),
            tournament_size=(args.tournament_size),
            intra_island_crossover_rate=(args.intra_island_crossover_rate),
            genomes_before_extinction=(args.genomes_before_extinction),
            genomes_for_next_extinction=(args.genomes_for_next_extinction),
            islands_to_extinct=(args.islands_to_extinct),
            primary_parent=(args.primary_parent),
            out_dir=args.out_dir,
        )

    if args.population_strategy == "nsga3_islands":
        return MultiObjectiveSteadyStateIslands(
            population_class=NSGA3,
            objectives=objectives,
            n_islands=args.n_islands,
            max_island_size=(args.max_island_size),
            tournament_size=(args.tournament_size),
            intra_island_crossover_rate=(args.intra_island_crossover_rate),
            genomes_before_extinction=(args.genomes_before_extinction),
            genomes_for_next_extinction=(args.genomes_for_next_extinction),
            islands_to_extinct=(args.islands_to_extinct),
            primary_parent=(args.primary_parent),
            out_dir=args.out_dir,
            population_kwargs={
                "reference_divisions": (args.reference_divisions),
            },
        )

    raise ValueError("Unknown population strategy: " f"{args.population_strategy}")


def main() -> None:
    """Run the multi-objective classification experiment."""
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    logger.remove()

    logger.add(
        sys.stdout,
        level=args.logging_level,
    )

    logger.add(
        os.path.join(
            args.out_dir,
            "run.log",
        )
    )

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    logger.info(
        "Using PyTorch device: {}",
        device,
    )

    logger.info(
        "Population strategy: {}",
        args.population_strategy,
    )

    # ------------------------------------------------------------------
    # Load dataset.
    # ------------------------------------------------------------------

    training_loader, validation_loader = load_data(args)

    if not training_loader.is_image and args.encoding == "cnn":
        parser.error("CNN encoding is only valid for image datasets.")

    # ------------------------------------------------------------------
    # Metrics and MOO objectives.
    # ------------------------------------------------------------------

    metrics = build_metrics(training_loader)

    try:
        objectives = build_objectives(
            args,
            metrics,
        )
    except ValueError as exc:
        parser.error(str(exc))

    logger.info("Multi-objective optimization objectives:")

    for objective_spec in objectives:
        direction = "maximize" if objective_spec.sign == -1.0 else "minimize"

        logger.info(
            "  {}: {} (sign={})",
            objective_spec.name,
            direction,
            objective_spec.sign,
        )

    # ------------------------------------------------------------------
    # Training objective.
    # ------------------------------------------------------------------

    objective = MultiObjectiveClassificationObjective(
        training_dataloader=(training_loader),
        validation_dataloader=(validation_loader),
        training_loss_function=(
            torch.nn.CrossEntropyLoss(
                weight=(training_loader.label_weights),
                reduction="mean",
            )
        ),
        validation_loss_function=(
            torch.nn.CrossEntropyLoss(
                weight=(validation_loader.label_weights),
                reduction="mean",
            )
        ),
        metrics=metrics,
        objectives=objectives,
        device=device,
    )

    # ------------------------------------------------------------------
    # Encoder dimensions.
    # ------------------------------------------------------------------

    n_encoder_outputs = args.input_qubits

    if args.quantum_input_mode == "u3":
        n_encoder_outputs *= 3

    # ------------------------------------------------------------------
    # Decoder dimensions.
    # ------------------------------------------------------------------

    n_decoder_inputs = args.output_qubits

    if args.quantum_output_mode == "probs":
        n_decoder_inputs = 2**args.output_qubits

    # ------------------------------------------------------------------
    # Build optional CNN encoder configuration.
    # ------------------------------------------------------------------

    encoder_config = None

    if training_loader.is_image and args.encoding == "cnn":
        channels, height, width = training_loader.input_shape

        encoder_config = load_encoder_config(args.encoder_config)

        encoder_config.update(
            {
                "input_channels": channels,
                "input_height": height,
                "input_width": width,
                "hidden_channels": (args.cnn_channels),
                "pooled_size": (args.cnn_pooled_size),
                "dropout": (args.cnn_dropout),
            }
        )

    # ------------------------------------------------------------------
    # Encoder and decoder.
    # ------------------------------------------------------------------

    initial_encoder = initialize_encoder(
        target=args.target,
        encoding_str=args.encoding,
        n_inputs=(training_loader.n_features),
        n_outputs=n_encoder_outputs,
        config=encoder_config,
    )

    initial_decoder = initialize_decoder(
        target=args.target,
        decoding_str=args.decoding,
        n_inputs=n_decoder_inputs,
        n_outputs=(training_loader.n_labels),
    )

    # ------------------------------------------------------------------
    # MOO population.
    # ------------------------------------------------------------------

    population = build_population(
        args,
        objectives,
    )

    # ------------------------------------------------------------------
    # Genome training hyperparameters.
    # ------------------------------------------------------------------

    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": (args.learning_rate),
        "weight_decay": (args.weight_decay),
        "improvement_cutoff": (args.improvement_cutoff),
        "batch_size": (args.batch_size),
        "quantum_input_mode": (args.quantum_input_mode),
        "quantum_output_mode": (args.quantum_output_mode),
        "quantum_dropout_type": (args.quantum_dropout_type),
        "quantum_dropout_rate": (args.quantum_dropout_rate),
    }

    gate_specifications = (
        pennylane_gate_specifications
        if args.target == "pennylane"
        else qiskit_gate_specifications
    )

    # ------------------------------------------------------------------
    # Run asynchronous EXAQC evolution.
    # ------------------------------------------------------------------

    master_worker(
        gate_specifications=(gate_specifications),
        population=population,
        objective=objective,
        initial_encoder=(initial_encoder),
        initial_decoder=(initial_decoder),
        hyperparameters=(hyperparameters),
        mutation_strategy=(args.mutation_strategy),
        parent_strategy=(args.parent_strategy),
        run_for=(args.number_genomes),
        input_registers={
            "input": args.input_qubits,
        },
        output_registers={
            "input": args.output_qubits,
        },
        target=args.target,
    )


if __name__ == "__main__":
    main()
