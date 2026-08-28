from __future__ import annotations

import argparse
import json
import torch
import sys

from loguru import logger

from src.circuits.circuit import CircuitGenome

from src.examples.classification import (
    get_dataset,
    get_dataloaders,
    ClassificationObjective,
)

from src.metrics.mean_class_accuracy import MeanClassAccuracy

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", choices=["iris", "wine", "seeds", "breast_cancer"], required=True
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning_rate", "-lr", type=float, default=5e-4)

    p.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Use mini-batch training with the given batch size, if provided",
    )

    p.add_argument(
        "--json_filename",
        type=str,
        required=True,
        help="The filename for the genome to continue to train.",
    )

    p.add_argument(
        "--logging_level",
        type=str,
        required=False,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    args = p.parse_args()

    # remove the old logging handler.
    logger.remove()
    # create a new logging handler at the appropriate level
    logger.add(sys.stdout, level=args.logging_level)

    # get the dataset samples and targets
    x, y = get_dataset(args.dataset)

    # create the dataloaders from the dataset
    training_dataloader, validation_dataloader = get_dataloaders(
        x=x,
        y=y,
        normalize="minmax",
        batch_size=args.batch_size,
    )

    metrics = {}
    metrics["mean_class_accuracy"] = MeanClassAccuracy(training_dataloader.n_labels)

    # set up the objective function
    objective = ClassificationObjective(
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        training_loss_function=torch.nn.CrossEntropyLoss(
            weight=training_dataloader.label_weights
        ),
        validation_loss_function=torch.nn.CrossEntropyLoss(
            weight=validation_dataloader.label_weights
        ),
        metrics=metrics,
    )

    # load genome from provided JSON
    with open(args.json_filename) as json_file:
        data = json.load(json_file)

        print(json.dumps(data, indent=2))

        genome = CircuitGenome.from_dict(data)

    # set new training parameteres from command line arguments
    # so we can further train the genome
    genome.hyperparameters["epochs"] = args.epochs
    genome.hyperparameters["learning_rate"] = args.learning_rate
    genome.hyperparameters["batch_size"] = args.batch_size

    objective(genome)
