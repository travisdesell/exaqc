from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Iterable
from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import (
    pennylane_gate_specifications,
)
from src.circuits.qiskit_gate_specifications import (
    qiskit_gate_specifications,
)
from src.datasets import QuantumDataset
from src.datasets.classification import (
    BreastCancerDataset,
    ImageDataset,
    IrisDataset,
    SeedsDataset,
    WineDataset,
)
from src.evolution.master_worker import master_worker
from src.evolution.multi_objective import (
    NSGA2Population,
    NSGA3Population,
    ObjectiveSpec,
)
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.models import (
    CNNImageEncoder,
    LinearImageEncoder,
    ResNetImageEncoder,
)
from src.objectives.genome_objectives import train_genome_objective
from src.utils.helpers import genome_to_torch_params
from src.utils.losses import (
    LOSS_REGISTRY,
    ce_onehot_on_probs,
)


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------


@torch.no_grad()
def predict_from_probs(
    probs_full: torch.Tensor,
    *,
    n_classes: int,
    eps: float = 1e-8,
) -> tuple[int, torch.Tensor]:
    """Convert quantum probabilities into class probabilities.

    Args:
        probs_full: Full quantum output probability vector.
        n_classes: Number of classes.
        eps: Numerical stability value.

    Returns:
        Predicted class index and normalized class probabilities.
    """
    probs = torch.as_tensor(
        probs_full,
        dtype=torch.float32,
    ).flatten()

    probs = probs[:n_classes]
    probs = torch.nan_to_num(
        probs,
        nan=eps,
        posinf=1.0,
        neginf=eps,
    )
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
    readout_head: torch.nn.Module | None = None,
    pl_device: str = "default.qubit",
    diff_method: str | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate loss and accuracy for one dataset split.

    Args:
        genome: Evaluated circuit genome.
        dataset: Dataset yielding input, one-hot label, and class name.
        n_classes: Number of output classes.
        loss: Name of the configured loss function.
        encoding: Quantum input encoding type.
        embedding_model: Optional classical input encoder.
        readout_head: Optional classical output head.
        pl_device: PennyLane device name.
        diff_method: PennyLane differentiation method.
        device: Torch device.

    Returns:
        Dictionary containing loss and accuracy.
    """
    torch_device = torch.device(device)

    if embedding_model is not None:
        embedding_model.to(
            device=torch_device,
            dtype=torch.float32,
        )
        embedding_model.eval()

    if readout_head is not None:
        readout_head.to(
            device=torch_device,
            dtype=torch.float32,
        )
        readout_head.eval()

    if (
        getattr(genome, "circuit", None) is None
        or not callable(genome.circuit)
    ):
        genome.generate_pennylane_circuit(
            return_probs=True,
            input_mode=encoding,
            device_name=pl_device,
            diff_method=diff_method,
            n_classes=n_classes,
        )

    loss_fn = LOSS_REGISTRY[loss]

    params = {
        key: torch.nn.Parameter(
            value.detach().to(torch_device).clone(),
            requires_grad=False,
        )
        for key, value in genome_to_torch_params(genome).items()
    }

    losses: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    onehot_labels: list[torch.Tensor] = []

    correct = 0
    total = 0
    per_class_correct: dict[str, int] = {}

    beta = (len(dataset) - 1) / max(len(dataset), 1)

    alpha = (1.0 - beta) / (
        1.0
        - np.power(
            beta,
            np.asarray(dataset.counts, dtype=np.float32),
        )
    )

    alpha = torch.as_tensor(
        alpha / alpha.mean(),
        dtype=torch.float32,
        device=torch_device,
    )

    for x, y, cls in dataset:
        per_class_correct.setdefault(cls, 0)

        x = x.to(
            device=torch_device,
            dtype=torch.float32,
        )
        y = y.to(
            device=torch_device,
            dtype=torch.float32,
        )

        if embedding_model is not None:
            x = embedding_model(x)

        q_probs = genome.circuit(x, params)

        if isinstance(q_probs, torch.Tensor):
            q_probs = q_probs.to(
                device=torch_device,
                dtype=torch.float32,
            )
        else:
            q_probs = torch.stack(
                [
                    value.to(
                        device=torch_device,
                        dtype=torch.float32,
                    )
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(
                        value,
                        dtype=torch.float32,
                        device=torch_device,
                    )
                    for value in q_probs
                ]
            )

        q_probs = q_probs.flatten()

        logits = None

        if readout_head is not None:
            logits = readout_head(q_probs)
            probs = torch.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs).item())
        else:
            pred, probs = predict_from_probs(
                q_probs,
                n_classes=n_classes,
            )
            probs = probs.to(torch_device)

        if readout_head is not None:
            target = torch.argmax(y).long().unsqueeze(0)

            loss_value = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                target,
                weight=alpha,
            )
        elif loss == "per_class":
            loss_value = ce_onehot_on_probs(
                probs,
                y,
                alpha_per_class=alpha,
            )
        else:
            try:
                loss_value = loss_fn(
                    probs,
                    y,
                    alpha_per_class=alpha,
                )
            except TypeError:
                loss_value = loss_fn(probs, y)

        losses.append(loss_value)
        probabilities.append(probs)
        onehot_labels.append(y)

        true = int(torch.argmax(y).item())

        correct += int(pred == true)
        total += 1

        if pred == true:
            per_class_correct[cls] += 1

    if loss_fn.__name__ != "class_avg_ce_onehot_on_probs":
        average_loss = (
            float(torch.stack(losses).mean().item())
            if losses
            else 0.0
        )
    else:
        probs_tensor = torch.stack(
            [
                value.to(
                    device=torch_device,
                    dtype=torch.float32,
                )
                for value in probabilities
            ],
            dim=0,
        )

        labels_tensor = torch.stack(
            [
                value.to(
                    device=torch_device,
                    dtype=torch.float32,
                )
                for value in onehot_labels
            ],
            dim=0,
        )

        average_loss = float(
            loss_fn(
                probs_tensor,
                labels_tensor,
            ).item()
        )

    accuracy = float(correct / max(total, 1))

    class_log = []

    for class_name, count in dataset.class_counts.items():
        class_accuracy = (
            per_class_correct.get(class_name, 0)
            / max(count, 1)
        )

        class_log.append(
            f"[{class_name}] Accuracy: {class_accuracy:.4f} "
            f"({per_class_correct.get(class_name, 0)}/{count})"
        )

    logger.info(" | ".join(class_log))

    return {
        "loss": average_loss,
        "acc": accuracy,
    }


# ---------------------------------------------------------------------
# Single-objective comparison
# ---------------------------------------------------------------------


def compare(
    genome1: CircuitGenome,
    genome2: CircuitGenome,
) -> float:
    """Compare two genomes by test loss.

    This is used only by the existing steady-state and island strategies.
    NSGA-II and NSGA-III do not use this comparator.
    """
    return (
        float(genome1.fitness["test_loss"])
        - float(genome2.fitness["test_loss"])
    )


# ---------------------------------------------------------------------
# Structural multi-objective metrics
# ---------------------------------------------------------------------


def add_circuit_objectives(
    genome: CircuitGenome,
) -> None:
    """Add circuit-complexity metrics to genome fitness.

    Args:
        genome: Evaluated circuit genome.
    """
    if genome.fitness is None:
        genome.fitness = {}

    enabled_gates = [
        gate
        for gate in genome.gates
        if gate.enabled
    ]

    genome.fitness.update(
        {
            "n_gates": float(len(enabled_gates)),
            "n_parameters": float(
                sum(
                    len(gate.parameters)
                    for gate in enabled_gates
                )
            ),
            "max_gate_depth": float(
                max(
                    (
                        gate.depth
                        for gate in enabled_gates
                    ),
                    default=0.0,
                )
            ),
            "n_single_qubit_gates": float(
                sum(
                    len(gate.qubits) == 1
                    for gate in enabled_gates
                )
            ),
            "n_two_qubit_gates": float(
                sum(
                    len(gate.qubits) == 2
                    for gate in enabled_gates
                )
            ),
            "n_multi_qubit_gates": float(
                sum(
                    len(gate.qubits) > 1
                    for gate in enabled_gates
                )
            ),
        }
    )


# ---------------------------------------------------------------------
# Classification objective
# ---------------------------------------------------------------------


class SupervisedClassificationObjective(Objective):
    """Supervised classification objective for tabular and image data."""

    def __init__(
        self,
        train_data: QuantumDataset,
        test_data: QuantumDataset,
        raw_input_dim: int,
        input_size: int,
        n_classes: int,
        hidden_dims: list[int],
        loss: str = "ce",
        activation: str = "tanh",
        dataset_name: str = "",
        encoder_type: str = "linear",
        conv_channels: list[int] | None = None,
        resnet_model: str = "resnet18",
        resnet_pretrained: bool = True,
        freeze_resnet: bool = True,
        target: str = "pennylane",
    ) -> None:
        """Initialize the supervised objective."""
        self.train_data = train_data
        self.test_data = test_data
        self.raw_input_dim = raw_input_dim
        self.input_size = input_size
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.loss = loss
        self.target = target
        self.activation = activation
        self.dataset_name = dataset_name
        self.encoder_type = encoder_type
        self.conv_channels = conv_channels or [16, 32]
        self.resnet_model = resnet_model
        self.resnet_pretrained = resnet_pretrained
        self.freeze_resnet = freeze_resnet

    def build_embedding_model(self) -> torch.nn.Module:
        """Create a fresh classical encoder for one genome."""
        if self.encoder_type == "linear":
            return LinearImageEncoder(
                input_dim=self.raw_input_dim,
                embedding_dim=self.input_size,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
            )

        if self.encoder_type == "cnn":
            if self.dataset_name == "cifar10":
                in_channels = 3
                image_size = 32
            elif self.dataset_name in {
                "mnist",
                "fashion_mnist",
            }:
                in_channels = 1
                image_size = 28
            else:
                raise ValueError(
                    "CNN encoder is only supported for image datasets."
                )

            return CNNImageEncoder(
                in_channels=in_channels,
                image_size=image_size,
                embedding_dim=self.input_size,
                conv_channels=self.conv_channels,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
            )

        if self.encoder_type == "resnet":
            if self.dataset_name not in {
                "mnist",
                "fashion_mnist",
                "cifar10",
            }:
                raise ValueError(
                    "ResNet encoder is only supported for image datasets."
                )

            return ResNetImageEncoder(
                embedding_dim=self.input_size,
                model_name=self.resnet_model,
                pretrained=self.resnet_pretrained,
                freeze_backbone=self.freeze_resnet,
                activation=self.activation,
            )

        raise ValueError(
            f"Unknown encoder_type={self.encoder_type!r}."
        )

    def __call__(
        self,
        genome: CircuitGenome,
    ) -> None:
        """Train and evaluate one genome.

        Args:
            genome: Circuit genome to train and evaluate.
        """
        hp = genome.hyperparameters

        embedding_model = self.build_embedding_model()

        n_output_qubits = math.ceil(
            math.log2(self.n_classes)
        )
        quantum_output_dim = 2**n_output_qubits

        readout_head = torch.nn.Linear(
            quantum_output_dim,
            self.n_classes,
        )

        train_genome_objective(
            genome,
            dataset=[
                self.train_data,
                self.test_data,
            ],
            backend=self.target,
            encoding=hp["encoding"],
            loss=self.loss,
            epochs=hp["epochs"],
            lr=hp["learning_rate"],
            n_classes=self.n_classes,
            log_every=hp["log_every"],
            batch_size=hp["batch_size"],
            embedding_model=embedding_model,
            readout_head=readout_head,
            pl_device=hp.get(
                "pl_device",
                "default.qubit",
            ),
            diff_method=hp.get(
                "diff_method",
                None,
            ),
            device=hp.get(
                "device",
                "cpu",
            ),
        )

        train_metrics = eval_probs_ce_and_acc(
            genome,
            self.train_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=hp["encoding"],
            embedding_model=embedding_model,
            readout_head=readout_head,
            pl_device=hp.get(
                "pl_device",
                "default.qubit",
            ),
            diff_method=hp.get(
                "diff_method",
                None,
            ),
            device=hp.get(
                "device",
                "cpu",
            ),
        )

        test_metrics = eval_probs_ce_and_acc(
            genome,
            self.test_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=hp["encoding"],
            embedding_model=embedding_model,
            readout_head=readout_head,
            pl_device=hp.get(
                "pl_device",
                "default.qubit",
            ),
            diff_method=hp.get(
                "diff_method",
                None,
            ),
            device=hp.get(
                "device",
                "cpu",
            ),
        )

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "raw_input_dim": float(self.raw_input_dim),
            "encoder_output_dim": float(self.input_size),
        }

        add_circuit_objectives(genome)

        # checkpoint_dir = os.path.join(
        #     hp["out_dir"],
        #     "model_checkpoints",
        # )
        # os.makedirs(
        #     checkpoint_dir,
        #     exist_ok=True,
        # )

        # checkpoint_path = os.path.join(
        #     checkpoint_dir,
        #     f"genome_{genome.genome_number}_hybrid_model.pt",
        # )

        # torch.save(
        #     {
        #         "genome": genome.to_dict(),
        #         "embedding_model_class": (
        #             embedding_model.__class__.__name__
        #         ),
        #         "embedding_model_state_dict": {
        #             key: value.detach().cpu()
        #             for key, value
        #             in embedding_model.state_dict().items()
        #         },
        #         "readout_head_state_dict": {
        #             key: value.detach().cpu()
        #             for key, value
        #             in readout_head.state_dict().items()
        #         },
        #         "fitness": genome.fitness,
        #         "hyperparameters": hp,
        #     },
        #     checkpoint_path,
        # )

        # genome.metadata["hybrid_checkpoint"] = checkpoint_path

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"n_gates={genome.fitness['n_gates']:.0f} "
            f"n_parameters={genome.fitness['n_parameters']:.0f}"
        )


# ---------------------------------------------------------------------
# Dataset and objective builder
# ---------------------------------------------------------------------


def build_objective(
    dataset_name: str,
    data_root: str,
    input_qubits: int,
    hidden_dims: list[int],
    conv_channels: list[int],
    encoder_type: str,
    loss: str,
    max_train_samples: int | None,
    max_test_samples: int | None,
    encoding: str = "angle",
    activation: str = "tanh",
    target: str = "pennylane",
    resnet_model: str = "resnet18",
    resnet_pretrained: bool = False,
    freeze_resnet: bool = False,
) -> SupervisedClassificationObjective:
    """Construct the requested classification objective."""
    if dataset_name in {
        "mnist",
        "fashion_mnist",
        "cifar10",
    }:
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

        raw_input_dim = (
            3 * 32 * 32
            if dataset_name == "cifar10"
            else 1 * 28 * 28
        )
        n_classes = 10

    elif dataset_name == "iris":
        train_data = IrisDataset(split="train")
        test_data = IrisDataset(split="test")
        raw_input_dim = 4
        n_classes = 3

    elif dataset_name == "wine":
        train_data = WineDataset(split="train")
        test_data = WineDataset(split="test")
        raw_input_dim = 13
        n_classes = 3

    elif dataset_name == "seeds":
        train_data = SeedsDataset(split="train")
        test_data = SeedsDataset(split="test")
        raw_input_dim = 7
        n_classes = 3

    elif dataset_name == "breast_cancer":
        train_data = BreastCancerDataset(split="train")
        test_data = BreastCancerDataset(split="test")
        raw_input_dim = 30
        n_classes = 2

    else:
        raise ValueError(
            f"Unsupported dataset={dataset_name!r}."
        )

    image_datasets = {
        "mnist",
        "fashion_mnist",
        "cifar10",
    }

    if (
        encoder_type in {"cnn", "resnet"}
        and dataset_name not in image_datasets
    ):
        raise ValueError(
            f"--encoder_type {encoder_type} is only supported "
            "for image datasets."
        )

    encoder_output_dim = (
        3 * input_qubits
        if encoding == "u3"
        else input_qubits
    )

    logger.info(
        f"Loaded dataset={dataset_name} | "
        f"raw_input_dim={raw_input_dim} | "
        f"encoder_output_dim={encoder_output_dim} | "
        f"input_qubits={input_qubits} | "
        f"hidden_dims={hidden_dims} | "
        f"n_classes={n_classes} | "
        f"train={len(train_data)} | "
        f"test={len(test_data)}"
    )

    return SupervisedClassificationObjective(
        train_data=train_data,
        test_data=test_data,
        raw_input_dim=raw_input_dim,
        input_size=encoder_output_dim,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        loss=loss,
        activation=activation,
        dataset_name=dataset_name,
        encoder_type=encoder_type,
        conv_channels=conv_channels,
        target=target,
        resnet_model=resnet_model,
        resnet_pretrained=resnet_pretrained,
        freeze_resnet=freeze_resnet,
    )


# ---------------------------------------------------------------------
# Multi-objective helpers
# ---------------------------------------------------------------------


def build_objective_specs(
    objective_names: list[str],
    maximize_objectives: list[str],
) -> list[ObjectiveSpec]:
    """Build minimization-form objective specifications.

    Every objective is minimized internally. Objectives listed in
    ``maximize_objectives`` receive a sign of ``-1.0``.

    Args:
        objective_names: Fitness keys to optimize.
        maximize_objectives: Fitness keys whose raw values should be maximized.

    Returns:
        Objective specifications.
    """
    maximize_set = set(maximize_objectives)

    unknown_maximize = maximize_set.difference(
        objective_names
    )

    if unknown_maximize:
        raise ValueError(
            "All --maximize_objectives values must also appear in "
            f"--objective_names. Unknown values: "
            f"{sorted(unknown_maximize)}"
        )

    return [
        ObjectiveSpec(
            name=name,
            sign=(
                -1.0
                if name in maximize_set
                else 1.0
            ),
        )
        for name in objective_names
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Run supervised EXAQC classification."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=[
            "pennylane",
            "qiskit",
        ],
        default="pennylane",
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "mnist",
            "fashion_mnist",
            "cifar10",
            "iris",
            "wine",
            "seeds",
            "breast_cancer",
        ],
        required=True,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
    )

    parser.add_argument(
        "--loss",
        default="ce",
        choices=[
            "per_class",
            "bce",
            "focal",
            "ce",
            "mse",
            "kl",
            "fidelity",
        ],
    )

    parser.add_argument(
        "--mutation_strategy",
        "-ms",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--parent_strategy",
        "-ps",
        type=str,
        nargs="+",
        default=["mutation"],
    )

    parser.add_argument(
        "--use_only",
        default=None,
        nargs="*",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--number_genomes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--input_qubits",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--encoder_type",
        choices=[
            "linear",
            "cnn",
            "resnet",
        ],
        default="linear",
    )

    parser.add_argument(
        "--resnet_model",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
        ],
        default="resnet18",
    )

    parser.add_argument(
        "--resnet_pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--freeze_resnet",
        action="store_true",
    )

    parser.add_argument(
        "--conv_channels",
        type=int,
        nargs="*",
        default=[
            16,
            32,
        ],
    )

    parser.add_argument(
        "--activation",
        choices=[
            "tanh",
            "sigmoid",
        ],
        default="tanh",
    )

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--encoding",
        choices=[
            "basis",
            "angle",
            "amplitude",
            "u3",
        ],
        default="angle",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "gpu",
        ],
        default="cpu",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
    )

    parser.add_argument(
        "--objective_names",
        nargs="+",
        default=[
            "test_loss",
            "n_gates",
        ],
        help=(
            "Fitness keys used by NSGA-II or NSGA-III. "
            "All are minimized internally."
        ),
    )

    parser.add_argument(
        "--maximize_objectives",
        nargs="*",
        default=[],
        help=(
            "Objectives whose raw fitness values should be maximized. "
            "Their values are negated internally."
        ),
    )

    parser.add_argument(
        "--tournament_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    subparsers = parser.add_subparsers(
        dest="population_strategy",
        required=True,
    )

    steady_state_parser = subparsers.add_parser(
        "steady_state",
    )
    steady_state_parser.add_argument(
        "--max_population_size",
        type=int,
        default=30,
    )

    islands_parser = subparsers.add_parser(
        "islands",
    )
    islands_parser.add_argument(
        "--n_islands",
        type=int,
        default=10,
    )
    islands_parser.add_argument(
        "--max_island_size",
        type=int,
        default=10,
    )
    islands_parser.add_argument(
        "--genomes_before_extinction",
        type=int,
        default=100,
    )
    islands_parser.add_argument(
        "--genomes_for_next_extinction",
        type=int,
        default=200,
    )
    islands_parser.add_argument(
        "--islands_to_extinct",
        type=int,
        default=2,
    )
    islands_parser.add_argument(
        "--intra_island_crossover_rate",
        type=float,
        default=0.5,
    )

    nsga2_parser = subparsers.add_parser(
        "nsga2",
    )
    nsga2_parser.add_argument(
        "--max_population_size",
        type=int,
        default=50,
    )

    nsga3_parser = subparsers.add_parser(
        "nsga3",
    )
    nsga3_parser.add_argument(
        "--max_population_size",
        type=int,
        default=50,
    )
    nsga3_parser.add_argument(
        "--reference_divisions",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    logger.remove()
    logger.add(
        sys.stdout,
        level=args.logging_level,
    )
    logger.add(
        os.path.join(
            args.out_dir,
            "run.log",
        ),
        level="DEBUG",
    )

    pl_device = "default.qubit"
    diff_method = "backprop"

    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested GPU execution but CUDA is unavailable."
            )

        local_rank = int(
            os.environ.get(
                "SLURM_LOCALID",
                os.environ.get(
                    "OMPI_COMM_WORLD_LOCAL_RANK",
                    0,
                ),
            )
        )

        gpu_id = (
            local_rank
            % torch.cuda.device_count()
        )

        torch.cuda.set_device(gpu_id)
        torch_device = f"cuda:{gpu_id}"

        # The classical encoder/readout run on GPU, while default.qubit
        # executes the quantum state simulation.
        pl_device = "default.qubit"
        diff_method = "backprop"
    else:
        torch_device = "cpu"

    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "log_every": 15,
        "batch_size": args.batch_size,
        "encoding": args.encoding,
        "conv_channels": args.conv_channels,
        "encoder_type": args.encoder_type,
        "resnet_model": args.resnet_model,
        "resnet_pretrained": args.resnet_pretrained,
        "freeze_resnet": args.freeze_resnet,
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "device": torch_device,
        "pl_device": pl_device,
        "diff_method": diff_method,
        "out_dir": args.out_dir,
    }

    objective = build_objective(
        dataset_name=args.dataset,
        data_root=args.data_root,
        input_qubits=args.input_qubits,
        hidden_dims=args.hidden_dims,
        conv_channels=args.conv_channels,
        encoder_type=args.encoder_type,
        loss=args.loss,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        encoding=args.encoding,
        activation=args.activation,
        target=args.target,
        resnet_model=args.resnet_model,
        resnet_pretrained=args.resnet_pretrained,
        freeze_resnet=args.freeze_resnet,
    )

    hyperparameters["n_classes"] = objective.n_classes

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
            genomes_before_extinction=(
                args.genomes_before_extinction
            ),
            genomes_for_next_extinction=(
                args.genomes_for_next_extinction
            ),
            islands_to_extinct=args.islands_to_extinct,
            intra_island_crossover_rate=(
                args.intra_island_crossover_rate
            ),
            compare=compare,
            out_dir=args.out_dir,
        )

    elif args.population_strategy == "nsga2":
        objective_specs = build_objective_specs(
            args.objective_names,
            args.maximize_objectives,
        )

        population = NSGA2Population(
            max_population_size=args.max_population_size,
            objectives=objective_specs,
            tournament_size=args.tournament_size,
            out_dir=args.out_dir,
            seed=args.seed,
        )

    elif args.population_strategy == "nsga3":
        objective_specs = build_objective_specs(
            args.objective_names,
            args.maximize_objectives,
        )

        population = NSGA3Population(
            max_population_size=args.max_population_size,
            objectives=objective_specs,
            tournament_size=args.tournament_size,
            reference_divisions=args.reference_divisions,
            out_dir=args.out_dir,
            seed=args.seed,
        )

    else:
        raise ValueError(
            f"Unknown population strategy "
            f"{args.population_strategy!r}."
        )

    logger.info(
        f"Starting EXAQC supervised run | "
        f"dataset={args.dataset} | "
        f"population_strategy={args.population_strategy} | "
        f"input_qubits={args.input_qubits} | "
        f"encoding={args.encoding} | "
        f"encoder_type={args.encoder_type} | "
        f"hidden_dims={args.hidden_dims} | "
        f"n_classes={objective.n_classes} | "
        f"torch_device={torch_device} | "
        f"pl_device={pl_device} | "
        f"diff_method={diff_method} | "
        f"objective_names={args.objective_names} | "
        f"maximize_objectives={args.maximize_objectives}"
    )

    gate_spec_map = {
        "pennylane": pennylane_gate_specifications,
        "qiskit": qiskit_gate_specifications,
    }

    gate_specs = gate_spec_map[args.target]

    if args.use_only is not None:
        gate_specs = gate_specs.use_only(
            args.use_only
        )

    master_worker(
        gate_specifications=gate_specs,
        population=population,
        objective=objective,
        hyperparameters=hyperparameters,
        mutation_strategy=args.mutation_strategy,
        parent_strategy=args.parent_strategy,
        run_for=args.number_genomes,
        input_registers={
            "input": args.input_qubits,
        },
        output_registers={
            "output": math.ceil(
                math.log2(objective.n_classes)
            ),
        },
        target=args.target,
    )


if __name__ == "__main__":
    main()