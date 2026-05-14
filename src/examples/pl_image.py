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
from src.datasets.classification import ImageDataset
from src.evolution.master_worker import master_worker
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.models import LinearImageEncoder
from src.objectives.genome_objectives import train_genome_objective
from src.utils.helpers import genome_to_torch_params
from src.utils.losses import LOSS_REGISTRY, ce_onehot_on_probs


@torch.no_grad()
def predict_from_probs(
    probs_full: torch.Tensor,
    *,
    n_classes: int,
    eps: float = 1e-8,
) -> tuple[int, torch.Tensor]:
    """Convert full output probabilities into a class prediction."""
    probs = torch.as_tensor(probs_full, dtype=torch.float32).flatten()
    probs = probs[:n_classes]
    probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps)
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum().clamp_min(eps)
    pred = int(torch.argmax(probs).item())
    return pred, probs


@torch.no_grad()
def eval_probs_ce_and_acc(
    genome: CircuitGenome,
    dataset: Iterable[tuple[torch.Tensor, torch.Tensor, str]],
    *,
    n_classes: int,
    loss: Optional[str] = None,
    encoding: str = "angle",
    embedding_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    """Evaluate loss and accuracy from quantum probability outputs."""
    if embedding_model is not None:
        embedding_model.eval()

    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(
            return_probs=True,
            input_mode=encoding,
        )

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

        if embedding_model is not None:
            x = embedding_model(x)

        probs_full = genome.circuit(x, params)
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


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> float:
    """Compare two genomes by test loss."""
    return genome1.fitness["test_loss"] - genome2.fitness["test_loss"]


class ImageClassificationObjective(Objective):
    """Image classification objective with a learnable classical encoder."""

    def __init__(
        self,
        train_data: QuantumDataset,
        test_data: QuantumDataset,
        image_input_dim: int,
        input_size: int,
        n_classes: int,
        hidden_dims: list[int],
        loss: str = "ce",
        activation: str = "tanh",
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.image_input_dim = image_input_dim
        self.input_size = input_size
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.loss = loss
        self.target = "pennylane"
        self.activation = activation

    def build_embedding_model(self) -> torch.nn.Module:
        """Create a fresh image encoder for one genome evaluation."""
        return LinearImageEncoder(
            input_dim=self.image_input_dim,
            embedding_dim=self.input_size,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
        )

    def __call__(self, genome: CircuitGenome) -> None:
        """Train and evaluate one genome."""
        hp = genome.hyperparameters

        learning_rate = hp["learning_rate"]
        epochs = hp["epochs"]
        batch_size = hp["batch_size"]
        log_every = hp["log_every"]
        encoding = hp["encoding"]

        embedding_model = self.build_embedding_model()

        # torch_params = genome_to_torch_params(genome)
        # if len(torch_params) > 0:
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
            embedding_model=embedding_model,
        )

        train_metrics = eval_probs_ce_and_acc(
            genome,
            self.train_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=encoding,
            embedding_model=embedding_model,
        )

        test_metrics = eval_probs_ce_and_acc(
            genome,
            self.test_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=encoding,
            embedding_model=embedding_model,
        )

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "encoder_hidden_dims": self.hidden_dims,
        }

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"train loss={train_metrics['loss']:.4f} "
            f"train acc={train_metrics['acc']:.4f} "
            f"test loss={test_metrics['loss']:.4f} "
            f"test acc={test_metrics['acc']:.4f}"
        )


def build_objective(
    *,
    dataset_name: str,
    data_root: str,
    input_qubits: int,
    hidden_dims: list[int],
    loss: str,
    max_train_samples: int | None,
    max_test_samples: int | None,
    encoding: str = "angle",
    activation: str = "tanh",
) -> ImageClassificationObjective:
    """Construct image objective from raw image datasets."""
    train_data = ImageDataset(
        dataset=dataset_name,
        root=data_root,
        split="train",
        max_samples=max_train_samples,
    )

    test_data = ImageDataset(
        dataset=dataset_name,
        root=data_root,
        split="test",
        max_samples=max_test_samples,
    )

    encoder_output_dim = 3 * input_qubits if encoding == "u3" else input_qubits

    if dataset_name == "cifar10":
        image_input_dim = 3 * 32 * 32
    elif dataset_name in {"mnist", "fashion_mnist"}:
        image_input_dim = 1 * 28 * 28
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    n_classes = 10

    logger.info(
        f"Loaded raw image dataset={dataset_name} | "
        f"image_input_dim={image_input_dim} | "
        f"embedding_dim={input_qubits} | "
        f"hidden_dims={hidden_dims} | "
        f"n_classes={n_classes} | "
        f"train={len(train_data)} | test={len(test_data)}"
    )

    return ImageClassificationObjective(
        train_data=train_data,
        test_data=test_data,
        image_input_dim=image_input_dim,
        input_size=encoder_output_dim,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        loss=loss,
        activation=activation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10"],
        required=True,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for raw image datasets.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
    )

    parser.add_argument(
        "--loss",
        default="mse",
        choices=["per_class", "bce", "focal", "ce", "mse", "kl", "fidelity"],
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--number_genomes", type=int, default=500)
    parser.add_argument("--input_qubits", type=int, default=15)

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Hidden layer sizes for image encoder. "
            "Use no values for pure linear projection."
        ),
    )

    parser.add_argument(
        "--use_input_u3_layer",
        action="store_true",
        help=(
            "If set, add an innovation-tracked trainable U3 layer on all input "
            "qubits after angle encoding and before evolved genome gates."
        ),
    )

    parser.add_argument(
        "--encoding",
        choices=["basis", "angle", "amplitude", "u3"],
        type=str,
        default="angle",
    )

    parser.add_argument(
        "--activation",
        choices=["tanh", "sigmoid"],
        type=str,
        default="tanh",
    )

    parser.add_argument("--batch_size", type=int, required=True)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument("--logging_level", type=str, default="INFO")

    subparsers = parser.add_subparsers(
        dest="population_strategy",
        required=True,
    )

    steady_state_parser = subparsers.add_parser("steady_state")
    steady_state_parser.add_argument("--max_population_size", type=int, default=30)

    islands_parser = subparsers.add_parser("islands")
    islands_parser.add_argument("--n_islands", type=int, default=10)
    islands_parser.add_argument("--max_island_size", type=int, default=10)
    islands_parser.add_argument("--genomes_before_extinction", type=int, default=100)
    islands_parser.add_argument("--genomes_for_next_extinction", type=int, default=200)
    islands_parser.add_argument("--islands_to_extinct", type=int, default=2)
    islands_parser.add_argument(
        "--intra_island_crossover_rate", type=float, default=0.5
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
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "use_input_u3_layer": args.use_input_u3_layer,
    }

    objective = build_objective(
        dataset_name=args.dataset,
        data_root=args.data_root,
        input_qubits=args.input_qubits,
        hidden_dims=args.hidden_dims,
        loss=args.loss,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        encoding=args.encoding,
        activation=args.activation,
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
        f"Starting EXAQC image run | dataset={args.dataset} | "
        f"population_strategy={args.population_strategy} | "
        f"input_qubits={args.input_qubits} | "
        f"encoding={args.encoding} | "
        f"use_input_u3_layer={args.use_input_u3_layer} | "
        f"hidden_dims={args.hidden_dims} | "
        f"n_classes={objective.n_classes}"
    )

    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=population,
        objective=objective,
        hyperparameters=hyperparameters,
        run_for=args.number_genomes,
        input_registers={"input": args.input_qubits},
        output_registers={"output": math.ceil(math.log(objective.n_classes, 2))},
        target="pennylane",
    )
