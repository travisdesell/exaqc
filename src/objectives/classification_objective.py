"""The classification task: its objective, genome ordering and data loading.

Shared by the ``src.examples.classification`` search and by the single-genome
tools (``src.examples.refine_genome``). It lives outside the entry point so the
tools can build a classification objective without importing the search driver,
and with it MPI.
"""

from __future__ import annotations

import argparse
from typing import Any

from torch.utils.data import DataLoader

from src.circuits.circuit import CircuitGenome
from src.datasets.classification_loaders import (
    IMAGE_DATASETS,
    get_image_dataloaders,
    get_uci_dataloaders,
    get_uci_test_dataloader,
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
        testing_dataloader: DataLoader | None = None,
        testing_loss_function: Any | None = None,
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
            testing_dataloader: Batched held-out test loader, if there is one.
            testing_loss_function: Test loss function; defaults to the validation
                loss function.

        Returns:
            None. Sets ``trainer``.
        """
        self.trainer = SupervisedTrainer(
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            training_loss_function=training_loss_function,
            validation_loss_function=validation_loss_function,
            metrics=metrics,
            testing_dataloader=testing_dataloader,
            testing_loss_function=testing_loss_function,
            device=device,
        )

    def __call__(self, genome: CircuitGenome) -> None:
        """Trains a genome and assigns classification fitness.

        Args:
            genome: Genome to train and evaluate.

        Returns:
            None. Sets ``genome.fitness`` with a minimized ``"loss"`` (the mean
            of the best training and validation loss) and a ``"target_metric"``
            holding the corresponding mean class accuracy.
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

    def test(self, genome: CircuitGenome) -> dict[str, Any] | None:
        """Scores a trained genome on the held-out test set.

        The model is rebuilt from the weights stored on the genome, so this gives
        the same answer for a genome held in memory, received over MPI or read
        back from an archive.

        Args:
            genome: The trained genome to score. Its weights are not changed.

        Returns:
            The test loss and metrics, or ``None`` when there is no test set.
        """
        if self.trainer.testing_dataloader is None:
            return None

        genome.initialize_model()
        genome.hybrid_model.to(self.trainer.device)
        return self.trainer.test(genome)


def load_data(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    """Loads tabular or image dataloaders.

    Args:
        args: Parsed command-line arguments (or a namespace carrying the same
            dataset-loading fields).

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
        split=args.uci_split,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return training_loader, validation_loader


def load_test_data(args: argparse.Namespace) -> DataLoader | None:
    """Loads the held-out test set a search scores its best genomes on.

    Only UCI datasets have one: an image dataset's official test split is scored
    separately, by ``src.examples.evaluate``.

    Args:
        args: Parsed command-line arguments, as given to :func:`load_data`.

    Returns:
        The test dataloader, or ``None`` when the dataset has no test split here.
    """
    if args.dataset in IMAGE_DATASETS:
        return None

    return get_uci_test_dataloader(
        args.dataset,
        normalize=args.normalization,
        split=args.uci_split,
        batch_size=args.batch_size,
        seed=args.seed,
    )
