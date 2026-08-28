from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from src.datasets.classification_loaders import (
    IMAGE_DATASETS,
    get_image_dataloaders,
    get_image_test_dataloader,
)
from src.models.classical_image_classifier import (
    CLASSICAL_IMAGE_MODELS,
    create_classical_image_model,
)
from src.trainer.classical_supervised_trainer import (
    ClassicalSupervisedTrainer,
)


def load_model_config(path: str | None) -> dict[str, Any]:
    """Loads a classical model configuration file.

    Args:
        path: Optional JSON configuration path.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ValueError: If the configuration root is not an object.
    """
    if path is None:
        return {}

    with Path(path).open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Model configuration must contain a JSON object.")

    return config


def set_seed(seed: int) -> None:
    """Sets common random seeds.

    Args:
        seed: Experiment seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    """Creates the classical baseline argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train classical image-classification baselines using the "
            "same EXAQC image dataloaders."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(IMAGE_DATASETS),
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=CLASSICAL_IMAGE_MODELS,
        required=True,
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
    )

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/classical",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
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
        "--testing_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--improvement_cutoff",
        type=int,
        default=10,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
    )

    return parser


def main() -> None:
    """Runs a classical image-classification baseline."""
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.out_dir) / args.dataset / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(run_dir / "run.log")

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

    testing_loader = get_image_test_dataloader(
        args.dataset,
        data_dir=args.data_dir,
        testing_samples=args.testing_samples,
        batch_size=(
            args.test_batch_size or args.validation_batch_size or args.batch_size
        ),
        download=args.download_dataset,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model_config = load_model_config(args.model_config)

    model = create_classical_image_model(
        model_name=args.model,
        input_shape=training_loader.input_shape,
        n_classes=training_loader.n_labels,
        config=model_config,
    )

    logger.info("Model:\n{}", model)
    logger.info(
        "Trainable parameters: {}",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    trainer = ClassicalSupervisedTrainer(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        testing_dataloader=testing_loader,
        device=args.device,
    )

    results = trainer.train(
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        improvement_cutoff=args.improvement_cutoff,
        label_smoothing=args.label_smoothing,
    )

    trainer.save_checkpoint(
        model,
        run_dir / "best_model.pt",
    )

    with (run_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    logger.info("Best epoch: {}", results["best_epoch"])
    logger.info(
        "Final training metrics: {}",
        results["training_metrics"],
    )
    logger.info(
        "Final validation metrics: {}",
        results["validation_metrics"],
    )

    if "testing_metrics" in results:
        logger.info(
            "Final testing metrics: {}",
            results["testing_metrics"],
        )


if __name__ == "__main__":
    main()
