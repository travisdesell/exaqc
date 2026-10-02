"""Evaluate a saved image-classification genome on its dataset's test split.

The search only ever trains and validates genomes, so this scores one genome --
loaded from a JSON file, or from a run's ``genomes.sqlar`` archive by its genome
number -- on the dataset's official test split::

    python3 -m src.examples.evaluate --genome_json best_fitness.json --dataset mnist
    python3 -m src.examples.evaluate --archive ./artifacts/mnist --genome_number 42 --dataset mnist
"""

from __future__ import annotations

import argparse

import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.datasets.classification_loaders import (
    get_image_test_dataloader,
)
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.trainer.supervised_trainer import SupervisedTrainer
from src.utils.genome_archive import (
    add_genome_source_arguments,
    check_genome_source_arguments,
    load_genome_dict,
)


def main() -> None:
    """Evaluates a saved classification genome on the official test split.

    Returns:
        None. Logs the genome's test metrics.
    """
    parser = argparse.ArgumentParser()
    # --genome_json or --archive (with --genome_number) chooses the genome.
    add_genome_source_arguments(
        parser,
        json_help="Path to a genome JSON file to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10"],
        required=True,
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    check_genome_source_arguments(parser, args)

    try:
        serialized = load_genome_dict(
            args.genome_json, args.archive, args.genome_number
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    genome = CircuitGenome.from_dict(serialized)
    genome.initialize_model()

    testing_loader = get_image_test_dataloader(
        args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=args.download_dataset,
    )

    metrics = {"mean_class_accuracy": MeanClassAccuracy(testing_loader.n_labels)}

    testing_loss = torch.nn.CrossEntropyLoss(
        weight=testing_loader.label_weights,
        reduction="mean",
    )

    trainer = SupervisedTrainer(
        training_dataloader=testing_loader,
        validation_dataloader=testing_loader,
        testing_dataloader=testing_loader,
        training_loss_function=testing_loss,
        validation_loss_function=testing_loss,
        testing_loss_function=testing_loss,
        metrics=metrics,
    )

    test_metrics = trainer.test(genome)

    logger.info("Test metrics: {}", test_metrics)


if __name__ == "__main__":
    main()
