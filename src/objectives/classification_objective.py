"""The classification task: its objective, genome ordering and data loading.

Shared by the ``src.examples.classification`` search and by the single-genome
tools (``src.examples.refine_genome``). It lives outside the entry point so the
tools can build a classification objective without importing the search driver,
and with it MPI.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.circuits.circuit import CircuitGenome
from src.datasets.classification_loaders import (
    IMAGE_DATASETS,
    get_image_dataloaders,
    get_uci_dataloaders,
)
from src.evolution.objective import Objective
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
        test_dataloader: DataLoader | None = None,
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
            test_dataloader: Optional held-out test loader. When given, each
                genome's best weights are scored on it after training and
                recorded as ``test_loss`` / ``test_accuracy``; selection never
                uses them.

        Returns:
            None. Sets ``trainer``.
        """
        self.trainer = SupervisedTrainer(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            training_loss_function=training_loss_function,
            validation_loss_function=validation_loss_function,
            metrics=metrics,
            device=device,
            testing_dataloader=test_dataloader,
            testing_loss_function=(
                None
                if test_dataloader is None
                else torch.nn.CrossEntropyLoss(
                    # class-balanced like the validation loss, when the loader
                    # carries its class weights
                    weight=getattr(test_dataloader, "label_weights", None),
                    reduction="mean",
                )
            ),
        )

    def __call__(self, genome: CircuitGenome) -> None:
        """Trains a genome and assigns classification fitness.

        Args:
            genome: Genome to train and evaluate.

        Returns:
            None. Sets ``genome.fitness`` with a minimized ``"loss"`` (the mean
            of the best training and validation loss) and a ``"target_metric"``
            holding the corresponding mean class accuracy, and
            ``genome.metadata["initial_fitness"]``: the same fitness for the
            inherited weights, before any training. With a test set, the
            fitness also records ``test_loss`` and ``test_accuracy`` (mean class
            accuracy) of the best weights on it, and the full metrics go to
            ``genome.metadata["testing_metrics"]``.
        """
        self.trainer.train(genome)

        genome.fitness = self.fitness_from_metrics(
            genome.metadata["best_training_metrics"],
            genome.metadata["best_validation_metrics"],
        )
        if self.trainer.testing_dataloader is not None:
            # scored once, on the best weights the trainer restored; recorded
            # only, never part of what selection compares
            test = self.trainer.test(genome)
            genome.fitness["test_loss"] = float(test["loss"])
            genome.fitness["test_accuracy"] = float(test["mean_class_accuracy"]["mean"])
        initial_training = genome.metadata.get("initial_training_metrics")
        initial_validation = genome.metadata.get("initial_validation_metrics")
        if initial_training is not None and initial_validation is not None:
            genome.metadata["initial_fitness"] = self.fitness_from_metrics(
                initial_training, initial_validation
            )

    @staticmethod
    def fitness_from_metrics(
        training: dict[str, Any], validation: dict[str, Any]
    ) -> dict[str, float]:
        """Computes classification fitness from one epoch's metrics.

        Args:
            training: The training-data metrics (``loss`` and
                ``mean_class_accuracy``).
            validation: The validation-data metrics, with the same keys.

        Returns:
            The fitness: ``loss`` is the mean of the training and validation
            loss, and ``target_metric`` the mean of their mean class accuracy.
        """

        return {
            "loss": (float(training["loss"]) + float(validation["loss"])) / 2.0,
            "target_metric": (
                float(training["mean_class_accuracy"]["mean"])
                + float(validation["mean_class_accuracy"]["mean"])
            )
            / 2.0,
        }


def load_data(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    """Loads tabular or image dataloaders.

    Args:
        args: Parsed command-line arguments (or a namespace carrying the same
            dataset-loading fields).

    Returns:
        Training and validation dataloaders. For a UCI dataset with a
        ``test_fraction`` above 0, the training loader's ``test_loader``
        attribute holds the held-out test loader (``None`` otherwise).

    Raises:
        ValueError: If a test set is asked for on an image dataset (they are
            not split for one yet).
    """
    test_fraction = float(getattr(args, "test_fraction", 0.0) or 0.0)
    if args.dataset in IMAGE_DATASETS:
        if test_fraction > 0.0:
            raise ValueError(
                "--test_fraction is only supported for the UCI datasets so far."
            )
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
        training_loader.test_loader = None
        return training_loader, validation_loader

    training_loader, validation_loader = get_uci_dataloaders(
        args.dataset,
        normalize=args.normalization,
        batch_size=args.batch_size,
        seed=args.seed,
        test_fraction=test_fraction,
        validation_fraction=args.validation_fraction,
    )
    return training_loader, validation_loader
