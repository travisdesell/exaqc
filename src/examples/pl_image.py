from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable, Optional

import numpy as np
import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.datasets import QuantumDataset
from src.datasets.image_embeddings import (
    CIFAR10EmbeddingDataset,
    FashionMNISTEmbeddingDataset,
    MNISTEmbeddingDataset,
)
from src.evolution.master_worker import master_worker
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.objectives.genome_objectives import train_genome_objective
from src.utils.helpers import genome_to_torch_params
from src.utils.losses import LOSS_REGISTRY, ce_onehot_on_probs


@torch.no_grad()
def predict_from_probs(
    probs_full: torch.Tensor,
    *,
    n_classes: int,
    eps: float = 1e-12,
) -> tuple[int, torch.Tensor]:
    """Convert full output probabilities into a prediction over n_classes."""
    probs = probs_full[:n_classes]
    probs = probs / (probs.sum() + eps)
    pred = int(torch.argmax(probs).item())
    return pred, probs


@torch.no_grad()
def eval_probs_ce_and_acc(
    genome: CircuitGenome,
    dataset: Iterable[tuple[torch.Tensor, torch.Tensor, str]],
    *,
    n_classes: int,
    loss: Optional[str] = None,
) -> dict[str, float]:
    """Evaluate loss and accuracy from probability outputs."""
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")

    loss_fn = LOSS_REGISTRY[loss]
    params = genome_to_torch_params(genome)

    losses: list[torch.Tensor] = []
    probas: list[torch.Tensor] = []
    y_onehots: list[torch.Tensor] = []
    correct = 0
    total = 0
    per_class_correct: dict[str, int] = {}

    beta = (len(dataset) - 1) / len(dataset)
    alpha = (1.0 - beta) / (
        1.0 - np.power(beta, np.array(dataset.counts, dtype=np.float32))
    )
    alpha = torch.as_tensor(alpha / alpha.mean(), dtype=torch.float32)

    for x, y, cls in dataset:
        per_class_correct.setdefault(cls, 0)

        probs_full = genome.circuit(x, params)
        probs_full = torch.as_tensor(probs_full, dtype=torch.float32)

        pred, probs = predict_from_probs(probs_full, n_classes=n_classes)
        loss_value = ce_onehot_on_probs(probs, y, alpha_per_class=alpha)

        losses.append(loss_value)
        probas.append(probs)
        y_onehots.append(y)

        true = int(torch.argmax(y).item())
        correct += int(pred == true)
        total += 1

        if pred == true:
            per_class_correct[cls] += 1

    if loss_fn.__name__ != "class_avg_ce_onehot_on_probs":
        avg_loss = float(torch.stack(losses).mean().item()) if losses else 0.0
    else:
        probs_tensor = torch.stack([p.to(torch.float32) for p in probas], dim=0)
        y_tensor = torch.stack([y.to(torch.float32) for y in y_onehots], dim=0)
        avg_loss = float(loss_fn(probs_tensor, y_tensor))

    acc = float(correct / max(total, 1))

    log_parts = []
    for cls_name, count in dataset.class_counts.items():
        cls_acc = per_class_correct.get(cls_name, 0) / max(count, 1)
        log_parts.append(
            f"[{cls_name}] Accuracy: {cls_acc:.4f} "
            f"({per_class_correct.get(cls_name, 0)}/{count})"
        )
    logger.info(" | ".join(log_parts))

    return {"loss": avg_loss, "acc": acc}


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """Compare two genomes for population sorting."""
    return genome1.fitness["test_loss"] - genome2.fitness["test_loss"]


class ClassificationObjective(Objective):
    """Classification objective for precomputed image embeddings."""

    def __init__(
        self,
        train_data: QuantumDataset,
        test_data: QuantumDataset,
        input_size: int,
        n_classes: int,
        loss: str = "ce",
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.input_size = input_size
        self.n_classes = n_classes
        self.loss = loss
        self.target = "pennylane"

    def __call__(self, genome: CircuitGenome) -> None:
        """Train and evaluate a genome."""
        hyperparameters = genome.hyperparameters
        learning_rate = hyperparameters["learning_rate"]
        epochs = hyperparameters["epochs"]
        batch_size = hyperparameters["batch_size"]
        log_every = hyperparameters["log_every"]
        encoding = hyperparameters["encoding"]

        torch_params = genome_to_torch_params(genome)
        if len(torch_params) > 0:
            train_genome_objective(
                genome,
                dataset=[self.train_data, self.test_data],
                backend=self.target,
                encoding=encoding,
                loss=self.loss,
                epochs=epochs,
                lr=learning_rate,
                n_classes=self.n_classes,
                log_every=log_every,
                batch_size=batch_size,
            )

        train_metrics = eval_probs_ce_and_acc(
            genome,
            self.train_data,
            n_classes=self.n_classes,
            loss=self.loss,
        )
        test_metrics = eval_probs_ce_and_acc(
            genome,
            self.test_data,
            n_classes=self.n_classes,
            loss=self.loss,
        )

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
        }

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"train loss={train_metrics['loss']:.4f} "
            f"train acc={train_metrics['acc']:.4f} "
            f"test loss={test_metrics['loss']:.4f} "
            f"test acc={test_metrics['acc']:.4f}"
        )


def build_objective(
    dataset_name: str,
    embedding_root: str,
    loss: str,
) -> ClassificationObjective:
    """Construct the EXAQC classification objective for an embedding dataset."""
    if dataset_name == "mnist":
        train_data = MNISTEmbeddingDataset(
            split="train",
            embedding_root=embedding_root,
        )
        test_data = MNISTEmbeddingDataset(
            split="test",
            embedding_root=embedding_root,
        )
    elif dataset_name == "fashion_mnist":
        train_data = FashionMNISTEmbeddingDataset(
            split="train",
            embedding_root=embedding_root,
        )
        test_data = FashionMNISTEmbeddingDataset(
            split="test",
            embedding_root=embedding_root,
        )
    elif dataset_name == "cifar10":
        train_data = CIFAR10EmbeddingDataset(
            split="train",
            embedding_root=embedding_root,
        )
        test_data = CIFAR10EmbeddingDataset(
            split="test",
            embedding_root=embedding_root,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    input_size = int(train_data.X.shape[1])
    n_classes = int(train_data.num_classes)

    logger.info(
        f"Loaded dataset={dataset_name} | "
        f"input_size={input_size} | n_classes={n_classes} | "
        f"train={len(train_data)} | test={len(test_data)}"
    )

    return ClassificationObjective(
        train_data=train_data,
        test_data=test_data,
        input_size=input_size,
        n_classes=n_classes,
        loss=loss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10"],
        required=True,
    )
    parser.add_argument(
        "--embedding_root",
        type=str,
        default="src/embeddings/images",
        help="Root directory containing saved embedding datasets.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store run artifacts.",
    )
    parser.add_argument(
        "--loss",
        default="ce",
        choices=["per_class", "bce", "focal", "ce", "mse", "kl", "fidelity"],
    )

    subparsers = parser.add_subparsers(
        dest="population_strategy",
        help="Specify how genomes will be handled.",
        required=True,
    )

    steady_state_parser = subparsers.add_parser(
        "steady_state",
        help="Use a single steady state population.",
    )
    steady_state_parser.add_argument("--max_population_size", type=int, default=30)

    islands_parser = subparsers.add_parser(
        "islands",
        help="Use multiple islands of steady state populations.",
    )
    islands_parser.add_argument("--n_islands", type=int, default=10)
    islands_parser.add_argument("--max_island_size", type=int, default=10)
    islands_parser.add_argument("--genomes_before_extinction", type=int, default=100)
    islands_parser.add_argument("--genomes_for_next_extinction", type=int, default=200)
    islands_parser.add_argument("--islands_to_extinct", type=int, default=2)
    islands_parser.add_argument(
        "--intra_island_crossover_rate",
        type=float,
        default=0.5,
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4)
    parser.add_argument("--number_genomes", type=int, default=2000)
    parser.add_argument(
        "--input_qubits",
        type=int,
        default=15,
        help="Input qubits to allocate. Final used size is min(input_qubits, embedding_dim).",
    )
    parser.add_argument(
        "--encoding",
        choices=["basis", "angle", "amplitude"],
        type=str,
        default="angle",
        help="Choose the encoding type.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="Logging level for terminal output.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"), level="DEBUG")

    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "log_every": 15,
        "batch_size": args.batch_size,
        "encoding": args.encoding,
    }

    objective = build_objective(
        dataset_name=args.dataset,
        embedding_root=args.embedding_root,
        loss=args.loss,
    )

    if args.population_strategy == "steady_state":
        population = SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
        )
    elif args.population_strategy == "islands":
        population = SteadyStateIslands(
            n_islands=args.n_islands,
            max_island_size=args.max_island_size,
            genomes_before_extinction=args.genomes_before_extinction,
            genomes_for_next_extinction=args.genomes_for_next_extinction,
            islands_to_extinct=args.islands_to_extinct,
            compare=compare,
            out_dir=args.out_dir,
        )
    else:
        raise ValueError(args.population_strategy)

    logger.info(
        f"Starting EXAQC run | dataset={args.dataset} | "
        f"population_strategy={args.population_strategy} | "
        f"input_size={objective.input_size} | n_classes={objective.n_classes}"
    )

    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=population,
        objective=objective,
        hyperparameters=hyperparameters,
        run_for=args.number_genomes,
        input_registers={"input": min(args.input_qubits, objective.input_size)},
        output_registers={"output": math.ceil(math.log(objective.n_classes, 2))},
        target="pennylane",
    )