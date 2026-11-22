"""Evaluate a saved classification genome on its dataset's test split.

The search only ever trains and validates genomes, so this scores one genome --
loaded from a JSON file, or from a run's ``genomes.sqlar`` archive by its genome
number -- on data it never saw: an image dataset's official test split, or the
test fraction of a UCI dataset's ``--uci_split``::

    python3 -m src.examples.evaluate --archive ./artifacts/iris --genome_number 42
    python3 -m src.examples.evaluate --genome_json genome_42.json --dataset mnist

A UCI test split is only the same samples the run held out when it is rebuilt
with the run's split, normalization and seed. Given ``--archive``, those are
read from the arguments the run recorded; given ``--genome_json``, they come
from the flags below (or their defaults), which must then match the run's.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.datasets.classification_loaders import (
    CLASSIFICATION_DATASETS,
    DEFAULT_UCI_SPLIT,
    LEGACY_UCI_SPLIT,
    UCI_DATASETS,
)
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.objectives.classification_objective import load_test_data
from src.trainer.supervised_trainer import SupervisedTrainer
from src.utils.genome_archive import (
    GenomeArchive,
    add_genome_source_arguments,
    check_genome_source_arguments,
    load_genome_dict,
)

#: The UCI test-split settings, and what each one is when neither the command
#: line nor the run's archive gives it (the classification entry point's
#: defaults).
UCI_SPLIT_DEFAULTS: dict[str, Any] = {
    "uci_split": list(DEFAULT_UCI_SPLIT),
    "normalization": "minmax",
    "seed": 0,
}


def build_parser() -> argparse.ArgumentParser:
    """Creates the evaluation command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Score a saved classification genome on its dataset's test split."
    )
    # --genome_json or --archive (with --genome_number) chooses the genome.
    add_genome_source_arguments(
        parser,
        json_help="Path to a genome JSON file to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        choices=CLASSIFICATION_DATASETS,
        default=None,
        help="Dataset to score on; defaults to the one the genome was evolved on.",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--uci_split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=None,
        help=(
            "UCI datasets: the run's training, validation and test fractions. Read "
            f"from --archive when given, otherwise {' '.join(map(str, DEFAULT_UCI_SPLIT))}."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "zscore", "minmax"],
        default=None,
        help="UCI datasets: the run's normalization. Read from --archive when given, otherwise minmax.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="UCI datasets: the run's split seed. Read from --archive when given, otherwise 0.",
    )
    return parser


def recorded_arguments(archive: str | None) -> dict[str, Any]:
    """Reads the arguments a run recorded, to rebuild the data it used.

    Args:
        archive: The run's archive or directory, or ``None`` for a genome file.

    Returns:
        The recorded arguments; empty when there is no archive, or the run
        recorded none.
    """
    if archive is None:
        return {}
    with GenomeArchive.open_readonly(archive) as reader:
        return dict(reader.run_info().get("arguments") or {})


def uci_split_settings(
    args: argparse.Namespace, recorded: dict[str, Any]
) -> dict[str, Any]:
    """Decides the split, normalization and seed a UCI test split is rebuilt with.

    A value given on the command line wins, then the one the run recorded, then
    the classification entry point's default. A run recorded before
    ``--uci_split`` existed held out no test samples, so it gets the 80/20 split
    it ran with.

    Args:
        args: Parsed command-line arguments.
        recorded: The arguments the run recorded (see :func:`recorded_arguments`).

    Returns:
        ``uci_split``, ``normalization`` and ``seed``.
    """
    settings: dict[str, Any] = {}
    for name, default in UCI_SPLIT_DEFAULTS.items():
        given = getattr(args, name)
        if given is not None:
            if name in recorded and recorded[name] != given:
                logger.warning(
                    "--{} {} differs from the run's recorded {}; the test split will "
                    "not be the samples the run held out.",
                    name,
                    given,
                    recorded[name],
                )
            settings[name] = given
        elif name in recorded:
            settings[name] = recorded[name]
        elif name == "uci_split" and recorded:
            settings[name] = list(LEGACY_UCI_SPLIT)
        else:
            settings[name] = default
    return settings


def evaluate(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    """Scores the chosen genome on its dataset's test split.

    Args:
        args: Parsed command-line arguments.
        parser: The parser, whose ``error`` reports unusable arguments.

    Returns:
        The genome's test loss and metrics.

    Raises:
        SystemExit: Via ``parser.error``, if the genome cannot be loaded, its
            dataset is unknown, or the run held out no test samples.
    """
    check_genome_source_arguments(parser, args)

    try:
        serialized = load_genome_dict(
            args.genome_json, args.archive, args.genome_number
        )
        recorded = recorded_arguments(args.archive)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    genome = CircuitGenome.from_dict(serialized)
    dataset = args.dataset or genome.task_target
    if dataset not in CLASSIFICATION_DATASETS:
        parser.error(
            f"the genome does not record a classification dataset ({dataset!r}); "
            "pass --dataset."
        )

    settings = uci_split_settings(args, recorded)
    testing_loader = load_test_data(
        SimpleNamespace(
            dataset=dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            validation_batch_size=None,
            download_dataset=args.download_dataset,
            num_workers=0,
            pin_memory=False,
            **settings,
        )
    )
    if testing_loader is None:
        parser.error(
            f"--uci_split {' '.join(map(str, settings['uci_split']))} holds out no "
            f"test samples of {dataset}, so there is nothing to score the genome on."
        )
    if dataset in UCI_DATASETS:
        logger.info(
            "Scoring on {} test samples of {} (split {}, {} normalization, seed {}).",
            len(testing_loader.dataset),
            dataset,
            settings["uci_split"],
            settings["normalization"],
            settings["seed"],
        )

    genome.initialize_model()
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
        metrics={"mean_class_accuracy": MeanClassAccuracy(testing_loader.n_labels)},
        device="cpu",
    )
    return trainer.test(genome)


def main() -> None:
    """Evaluates a saved classification genome on its dataset's test split.

    Returns:
        None. Logs the genome's test metrics.
    """
    parser = build_parser()
    test_metrics = evaluate(parser.parse_args(), parser)
    logger.info("Test metrics: {}", test_metrics)


if __name__ == "__main__":
    main()
