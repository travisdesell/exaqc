from __future__ import annotations

import argparse
import json
import sys
import os

import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.datasets.classification_loaders import (
    get_image_test_dataloader,
)
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.trainer.supervised_trainer import SupervisedTrainer


def main() -> None:
    """Evaluates a saved classification genome on the official test split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
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
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )
    args = parser.parse_args()

    out_dir = "/".join(args.genome.split("/")[:-1])

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(out_dir, "test.log"))

    with open(args.genome, "r", encoding="utf-8") as file:
        serialized = json.load(file)

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
