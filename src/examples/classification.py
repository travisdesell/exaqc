from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import json
from pathlib import Path

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
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import (
    SteadyStatePopulation,
)
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.metrics.metric import Metric
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


class ClassificationObjective(Objective):
    """Classification objective backed by :class:`SupervisedTrainer`."""

    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        training_loss_function: Any,
        validation_loss_function: Any,
        metrics: dict[str, Metric],
        device: str | None = None,
    ) -> None:
        """Initializes the classification objective.

        Args:
            training_dataloader: Batched training loader.
            validation_dataloader: Batched validation loader.
            training_loss_function: Training loss function.
            validation_loss_function: Validation loss function.
            metrics: Evaluation metrics.
        """
        self.trainer = SupervisedTrainer(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            training_loss_function=training_loss_function,
            validation_loss_function=validation_loss_function,
            metrics=metrics,
            device=device,
        )

    def __call__(self, genome: CircuitGenome) -> None:
        """Trains a genome and assigns classification fitness.

        Args:
            genome: Genome to train and evaluate.
        """
        self.trainer.train(genome)

        training = genome.metadata["best_training_metrics"]
        validation = genome.metadata["best_validation_metrics"]

        genome.fitness = {
            "loss": (float(training["loss"]) + float(validation["loss"])) / 2.0,
            "target_metric": (
                float(training["mean_class_accuracy"]["mean"])
                + float(validation["mean_class_accuracy"]["mean"])
            )
            / 2.0,
        }


def build_parser() -> argparse.ArgumentParser:
    """Creates the classification command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=("Run batched EXAQC classification on UCI and image datasets.")
    )
    parser.add_argument(
        "--dataset",
        choices=CLASSIFICATION_DATASETS,
        required=True,
    )
    parser.add_argument("--out_dir", type=str, default="artifacts")
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

    populations = parser.add_subparsers(
        dest="population_strategy",
        required=True,
    )
    steady = populations.add_parser("steady_state")
    steady.add_argument(
        "--max_population_size",
        type=int,
        default=30,
    )

    islands = populations.add_parser("islands")
    islands.add_argument("--n_islands", type=int, default=10)
    islands.add_argument(
        "--max_island_size",
        type=int,
        default=10,
    )
    islands.add_argument(
        "--genomes_before_extinction",
        type=int,
        default=100,
    )
    islands.add_argument(
        "--genomes_for_next_extinction",
        type=int,
        default=200,
    )
    islands.add_argument(
        "--islands_to_extinct",
        type=int,
        default=2,
    )
    islands.add_argument(
        "--primary_parent",
        default="best",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
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
        choices=["pennylane", "qiskit"],
        default="pennylane",
    )
    parser.add_argument(
        "--quantum_input_mode",
        "-qim",
        choices=QUANTUM_INPUT_MODES,
        default="u3",
        help="Choose initial gate types whose parameteres will be  set from classical inputs",
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
        choices=["gate", "rotation", "entangling", "qubit", "innovation"],
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
        help="Choose the kind of encoding",
    )
    parser.add_argument("--encoder_config", type=str, default="configs")
    parser.add_argument(
        "--decoding",
        type=str,
        choices=DECODING_OPTIONS,
        default="linear",
        help="Choose the kind of decoding",
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

    parser.add_argument("--data_dir", type=str, default="data")
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
            "PyTorch device to use for training, e.g. "
            "'cpu', 'cuda', or 'cuda:0'. "
            "Defaults to CUDA when available."
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
        choices=["none", "zscore", "minmax"],
        default="minmax",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )
    return parser


def load_data(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    """Loads tabular or image dataloaders.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Training and validation dataloaders.
    """
    if args.dataset in IMAGE_DATASETS:
        training_loader, validation_loader = get_image_dataloaders(
            args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            validation_fraction=args.validation_fraction,
            training_samples=args.training_samples,
            validation_samples=args.validation_samples,
            seed=args.seed,
            download=args.download_dataset,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        return training_loader, validation_loader

    training_loader, validation_loader = get_uci_dataloaders(
        args.dataset,
        normalize=args.normalization,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return training_loader, validation_loader


def load_encoder_config(
    path: str | None,
) -> dict:
    """Loads an encoder configuration file.

    Args:
        path: JSON configuration path or ``None``.

    Returns:
        Parsed encoder configuration.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    if path is None:
        return {}

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Encoder configuration must contain a JSON object.")

    return config


def main() -> None:
    """Runs the classification experiment."""
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

    training_loader, validation_loader = load_data(args)

    # if training_loader.is_image and args.encoding != "cnn":
    #     parser.error("Image datasets require --encoding cnn in this implementation.")
    if not training_loader.is_image and args.encoding == "cnn":
        parser.error("CNN encoding is only valid for image datasets.")

    metrics: dict[str, Metric] = {
        "mean_class_accuracy": MeanClassAccuracy(training_loader.n_labels)
    }
    objective = ClassificationObjective(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        training_loss_function=torch.nn.CrossEntropyLoss(
            weight=training_loader.label_weights,
            reduction="mean",
        ),
        validation_loss_function=torch.nn.CrossEntropyLoss(
            weight=validation_loader.label_weights,
            reduction="mean",
        ),
        metrics=metrics,
        device=args.device,
    )

    n_encoder_outputs = args.input_qubits
    if args.quantum_input_mode == "u3":
        n_encoder_outputs *= 3

    n_decoder_inputs = args.output_qubits
    if args.quantum_output_mode == "probs":
        n_decoder_inputs = 2**args.output_qubits

    encoder_config = None
    if training_loader.is_image and args.encoding == "cnn":
        channels, height, width = training_loader.input_shape

        encoder_config = load_encoder_config(args.encoder_config)

        encoder_config.update(
            {
                "input_channels": channels,
                "input_height": height,
                "input_width": width,
                "hidden_channels": args.cnn_channels,
                "pooled_size": args.cnn_pooled_size,
                "dropout": args.cnn_dropout,
            }
        )

    initial_encoder = initialize_encoder(
        target=args.target,
        encoding_str=args.encoding,
        n_inputs=training_loader.n_features,
        n_outputs=n_encoder_outputs,
        config=encoder_config,
    )
    initial_decoder = initialize_decoder(
        target=args.target,
        decoding_str=args.decoding,
        n_inputs=n_decoder_inputs,
        n_outputs=training_loader.n_labels,
    )

    if args.population_strategy == "steady_state":
        population = SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
        )
    else:
        population = SteadyStateIslands(
            n_islands=args.n_islands,
            max_island_size=args.max_island_size,
            genomes_before_extinction=(args.genomes_before_extinction),
            genomes_for_next_extinction=(args.genomes_for_next_extinction),
            islands_to_extinct=args.islands_to_extinct,
            primary_parent=args.primary_parent,
            compare=compare,
            out_dir=args.out_dir,
        )

    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "improvement_cutoff": args.improvement_cutoff,
        "batch_size": args.batch_size,
        "quantum_input_mode": args.quantum_input_mode,
        "quantum_output_mode": args.quantum_output_mode,
        "quantum_dropout_type": args.quantum_dropout_type,
        "quantum_dropout_rate": args.quantum_dropout_rate,
    }

    gate_specifications = (
        pennylane_gate_specifications
        if args.target == "pennylane"
        else qiskit_gate_specifications
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
        run_for=args.number_genomes,
        input_registers={"input": args.input_qubits},
        output_registers={"input": args.output_qubits},
        target=args.target,
    )


if __name__ == "__main__":
    main()
