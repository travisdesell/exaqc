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

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import initialize_decoder
from src.circuits.encoder import initialize_encoder
from src.circuits.gate_specifications import GateSpecifications
from src.datasets.classification_loaders import (
    CLASSIFICATION_DATASETS,
    IMAGE_DATASETS,
    get_image_dataloaders,
    get_uci_dataloaders,
)
from src.evolution.exaqc import EXAQC
from src.evolution.master_worker import run_evolution
from src.evolution.objective import Objective
from src.evolution.population_strategy import PopulationStrategy
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

        Quantum dropout is not configured here: it is carried per genome via
        the ``quantum_dropout`` hyperparameter and read by the trainer at train
        time, so the evolutionary search can carry and mutate it per genome.

        Args:
            training_dataloader: Batched training loader.
            validation_dataloader: Batched validation loader.
            training_loss_function: Training loss function.
            validation_loss_function: Validation loss function.
            metrics: Evaluation metrics.
            device: PyTorch device to train on, or ``None`` to auto-select.
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
        help="Dataset to evolve classifiers on (a UCI tabular or image dataset).",
    )

    # The evolutionary search's own flags -- mutation/parent strategies,
    # crossover rates, the genome budget, --out_dir and --save_training_plot --
    # are owned by EXAQC so every entry point stays in sync.
    EXAQC.initialize_parser(parser)

    # The choice of population strategy (and each strategy's own flags) is owned
    # by PopulationStrategy.
    PopulationStrategy.initialize_parser(parser)

    # The supervised-training flags (epochs, learning rate, weight decay,
    # improvement cutoff, batch size) are owned by SupervisedTrainer so the
    # classification and teacher entry points stay in sync.
    SupervisedTrainer.initialize_parser(parser)

    # The backend (--target) and optional gate-set restriction (--use_only) are
    # owned by GateSpecifications.
    GateSpecifications.initialize_parser(parser)

    # The circuit-genome flags (qubit counts, quantum input/output modes,
    # quantum dropout, encoder/decoder) are owned by CircuitGenome so every
    # entry point stays in sync.
    CircuitGenome.initialize_parser(parser)

    parser.add_argument(
        "--encoder_config",
        type=str,
        default="configs",
        help="Path to a JSON file of encoder configuration options (e.g. CNN activation, batch norm).",
    )

    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
        help="Batch size for validation/testing; defaults to --batch_size when unset.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory where datasets are stored (and downloaded to).",
    )

    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the dataset if it is not already present in --data_dir.",
    )

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="Fraction of the training data held out for validation when no fixed split exists.",
    )

    parser.add_argument(
        "--training_samples",
        type=int,
        default=None,
        help="Cap the training set to this many samples (use all when unset).",
    )

    parser.add_argument(
        "--validation_samples",
        type=int,
        default=None,
        help="Cap the validation set to this many samples (use all when unset).",
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
        help="Number of worker processes used by the PyTorch DataLoaders.",
    )

    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pinned (page-locked) host memory in the DataLoaders for faster GPU transfers.",
    )

    parser.add_argument(
        "--cnn_channels",
        type=int,
        nargs=2,
        default=[16, 32],
        help="Output channel counts for the two CNN encoder convolution layers (image datasets).",
    )

    parser.add_argument(
        "--cnn_pooled_size",
        type=int,
        default=4,
        help="Spatial size (height = width) the CNN encoder pools its feature maps down to.",
    )

    parser.add_argument(
        "--cnn_dropout",
        type=float,
        default=0.0,
        help="Dropout rate applied inside the CNN encoder.",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "zscore", "minmax"],
        default="minmax",
        help="Feature normalization applied to tabular datasets before encoding.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible dataset splits and evolution.",
    )

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
    """Runs a classification experiment."""
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

    def build_exaqc() -> EXAQC:
        """Builds the EXAQC search machinery for this run.

        This constructs the encoder/decoder, population strategy, gate set and
        the :class:`EXAQC` object. It is only invoked on the serial run and the
        MPI master (via :func:`run_evolution`), so worker ranks never build any
        of it.

        Returns:
            The fully-configured :class:`EXAQC` search wrapping ``objective``.
        """

        if args.encoding == "identity":
            # The identity encoder passes its input straight through, so its
            # output size must equal its input size (the raw feature count) --
            # it does not resize or clip to the qubit count.
            n_encoder_outputs = training_loader.n_features
        else:
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
            quantum_input_mode=args.quantum_input_mode,
            n_input_qubits=args.input_qubits,
        )
        initial_decoder = initialize_decoder(
            target=args.target,
            decoding_str=args.decoding,
            n_inputs=n_decoder_inputs,
            n_outputs=training_loader.n_labels,
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

        # The gate set and population strategy are built from `args` by their own
        # factories (which every entry point shares); only the task-specific
        # encoder/decoder, hyperparameters and register layout are computed here.
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
            input_registers={"input": args.input_qubits},
            output_registers={"input": args.output_qubits},
            task="classification",
            task_target=args.dataset,
        )

    # Every rank builds the objective (workers evaluate genomes with it); only
    # the serial run and the MPI master build the EXAQC search, via build_exaqc.
    run_evolution(
        objective=objective,
        build_exaqc=build_exaqc,
        run_for=args.number_genomes,
    )


if __name__ == "__main__":
    main()
