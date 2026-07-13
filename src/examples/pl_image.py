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
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.datasets import QuantumDataset
from src.datasets.classification import (
    ImageDataset,
    IrisDataset,
    WineDataset,
    SeedsDataset,
    BreastCancerDataset,
)
from src.evolution.master_worker import master_worker
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.models import LinearImageEncoder, CNNImageEncoder, ResNetImageEncoder
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
    readout_head: torch.nn.Module | None = None,
    pl_device: str = "default.qubit",
    diff_method: str | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate loss and accuracy from quantum probability outputs."""
    device = torch.device(device)

    if embedding_model is not None:
        embedding_model.to(device)
        embedding_model.eval()

    if readout_head is not None:
        readout_head.to(device)
        readout_head.eval()

    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(
            return_probs=True,
            input_mode=encoding,
            device_name=pl_device,
            diff_method=diff_method,
            n_classes=n_classes,
        )

    loss_fn = LOSS_REGISTRY[loss]
    # params = genome_to_torch_params(genome)
    params = {
        k: torch.nn.Parameter(v.detach().to(device).clone(), requires_grad=False)
        for k, v in genome_to_torch_params(genome).items()
    }

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
    alpha = torch.as_tensor(alpha / alpha.mean(), dtype=torch.float32, device=device)

    for x, y, cls in dataset:
        per_class_correct.setdefault(cls, 0)

        x = x.to(device)
        y = y.to(device)

        if embedding_model is not None:
            x = embedding_model(x)

        probs_full = genome.circuit(x, params)
        probs_full = probs_full.to(dtype=torch.float32)
        logits = None
        if readout_head is not None:
            logits = readout_head(probs_full)
            probs = torch.softmax(logits, dim=-1)
            pred = int(torch.argmax(probs).item())
            
        else:
            pred, probs = predict_from_probs(probs_full, n_classes=n_classes)

        probs = probs.to(device)

        if loss == "per_class":
            loss_value = ce_onehot_on_probs(probs, y, alpha_per_class=alpha)
        else:
            if readout_head is None:
                try:
                    loss_value = loss_fn(probs, y, alpha_per_class=alpha)
                except TypeError:
                    loss_value = loss_fn(probs, y)
            else:
                target = torch.argmax(y).long().unsqueeze(0)
                loss_value = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0),
                    target,
                    weight=alpha,
                )

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


class SupervisedClassificationObjective(Objective):
    """Classification objective with a learnable classical feature encoder."""

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
        """Create a fresh encoder for one genome evaluation."""
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
            elif self.dataset_name in {"mnist", "fashion_mnist"}:
                in_channels = 1
                image_size = 28
            else:
                raise ValueError("CNN encoder is only supported for image datasets.")

            return CNNImageEncoder(
                in_channels=in_channels,
                image_size=image_size,
                embedding_dim=self.input_size,
                conv_channels=self.conv_channels,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
            )

        if self.encoder_type == "resnet":
            if self.dataset_name not in {"mnist", "fashion_mnist", "cifar10"}:
                raise ValueError("ResNet encoder is only supported for image datasets.")

            return ResNetImageEncoder(
                embedding_dim=self.input_size,
                model_name=self.resnet_model,
                pretrained=self.resnet_pretrained,
                freeze_backbone=self.freeze_resnet,
                activation=self.activation,
            )

        raise ValueError(self.encoder_type)

    def __call__(self, genome: CircuitGenome) -> None:
        """Train and evaluate one genome."""
        hp = genome.hyperparameters

        embedding_model = self.build_embedding_model()

        q_out_dim = 2 ** math.ceil(math.log2(self.n_classes))

        readout_head = torch.nn.Linear(
            q_out_dim,
            self.n_classes,
        )

        train_genome_objective(
            genome,
            dataset=[self.train_data, self.test_data],
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
            pl_device=hp.get("pl_device", "default.qubit"),
            diff_method=hp.get("diff_method", None),
            device=hp.get("device", "cpu"),
        )

        train_metrics = eval_probs_ce_and_acc(
            genome,
            self.train_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=hp["encoding"],
            embedding_model=embedding_model,
            readout_head=readout_head,
            pl_device=hp.get("pl_device", "default.qubit"),
            diff_method=hp.get("diff_method", None),
            device=hp.get("device", "cpu"),
        )

        test_metrics = eval_probs_ce_and_acc(
            genome,
            self.test_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=hp["encoding"],
            embedding_model=embedding_model,
            readout_head=readout_head,
            pl_device=hp.get("pl_device", "default.qubit"),
            diff_method=hp.get("diff_method", None),
            device=hp.get("device", "cpu"),
        )

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "encoder_hidden_dims": self.hidden_dims,
            "raw_input_dim": self.raw_input_dim,
            "encoder_output_dim": self.input_size,
        }

        # checkpoint = {
        #     "genome": genome.to_dict(),
        #     "embedding_model_state_dict": (
        #         embedding_model.state_dict()
        #         if embedding_model is not None
        #         else None
        #     ),
        #     "readout_head_state_dict": (
        #         readout_head.state_dict()
        #         if readout_head is not None
        #         else None
        #     ),
        #     "fitness": genome.fitness,
        #     "hyperparameters": hp,
        # }

        # torch.save(
        #     checkpoint,
        #     os.path.join(
        #         hp["out_dir"],
        #         f"genome{genome.genome_number}_model_checkpoint.pt",
        #     ),
        # )

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"train loss={train_metrics['loss']:.4f} "
            f"train acc={train_metrics['acc']:.4f} "
            f"test loss={test_metrics['loss']:.4f} "
            f"test acc={test_metrics['acc']:.4f}"
        )


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
    resnet_model: str = "resnet50",
    resnet_pretrained: bool = False,
    freeze_resnet: bool = False,
) -> SupervisedClassificationObjective:
    """Construct objective for image or tabular datasets."""

    if dataset_name in {"mnist", "fashion_mnist", "cifar10"}:
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

        if dataset_name == "cifar10":
            raw_input_dim = 3 * 32 * 32
        else:
            raw_input_dim = 1 * 28 * 28

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
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    image_datasets = {"mnist", "fashion_mnist", "cifar10"}

    if encoder_type == "cnn" and dataset_name not in image_datasets:
        raise ValueError(
            "--encoder_type cnn is only supported for image datasets. "
            "Use --encoder_type linear for iris/wine/seeds/breast_cancer."
        )

    encoder_output_dim = 3 * input_qubits if encoding == "u3" else input_qubits

    logger.info(
        f"Loaded dataset={dataset_name} | "
        f"raw_input_dim={raw_input_dim} | "
        f"encoder_output_dim={encoder_output_dim} | "
        f"input_qubits={input_qubits} | "
        f"hidden_dims={hidden_dims} | "
        f"n_classes={n_classes} | "
        f"train={len(train_data)} | test={len(test_data)}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=["pennylane", "qiskit"],
        type=str,
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
        help="Root directory for raw image datasets.",
    )

    parser.add_argument("--out_dir", type=str, default="artifacts")

    parser.add_argument(
        "--loss",
        default="ce",
        choices=["per_class", "bce", "focal", "ce", "mse", "kl", "fidelity"],
    )

    parser.add_argument(
        "--mutation_strategy",
        "-ms",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--use_only",
        default=None,
        nargs="*",
        help="List all the required gates",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--number_genomes", type=int, default=500)
    parser.add_argument("--input_qubits", type=int, default=15)

    parser.add_argument(
        "--encoder_type",
        choices=["linear", "cnn", "resnet"],
        default="linear",
    )

    parser.add_argument(
        "--resnet_model",
        choices=["resnet18", "resnet34", "resnet50"],
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
        default=[16, 32],
        help=(
            "CNN convolution channel widths. "
            "Only used when --encoder_type cnn."
        ),
    )

    parser.add_argument(
        "--activation",
        choices=["tanh", "sigmoid"],
        type=str,
        default="tanh",
    )

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Hidden layer sizes for the encoder. "
            "Use no values for pure linear projection."
        ),
    )

    # parser.add_argument(
    #     "--use_input_u3_layer",
    #     action="store_true",
    #     help=(
    #         "If set, add an innovation-tracked trainable U3 layer on all input "
    #         "qubits after encoding and before evolved genome gates."
    #     ),
    # )

    parser.add_argument(
        "--encoding",
        choices=["basis", "angle", "amplitude", "u3"],
        type=str,
        default="angle",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        type=str,
        default="cpu",
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
    islands_parser.add_argument("--intra_island_crossover_rate", type=float, default=0.5)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"), level="DEBUG")

    pl_device = "default.qubit"
    diff_method = "backprop" # "adjoint" if args.device == "gpu" else "backprop"
    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device gpu but torch.cuda.is_available() is False.")
        
        local_rank = int(
            os.environ.get("SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        )
        gpu_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)

        device = f"cuda:{gpu_id}"
        # pl_device = "lightning.gpu"
    else:
        device = "cpu"

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
        # "use_input_u3_layer": args.use_input_u3_layer,

        "device": device,
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
        f"Starting EXAQC supervised run | dataset={args.dataset} | "
        f"population_strategy={args.population_strategy} | "
        f"input_qubits={args.input_qubits} | "
        f"encoding={args.encoding} | "
        # f"use_input_u3_layer={args.use_input_u3_layer} | "
        f"hidden_dims={args.hidden_dims} | "
        f"n_classes={objective.n_classes} | "
        f"device={args.device} | "
        f"pl_device={pl_device} | "
        f"diff_method={diff_method} | "
    )

    gate_spec_map = {
        "pennylane": pennylane_gate_specifications,
        "qiskit": qiskit_gate_specifications,
    }

    if args.target not in gate_spec_map:
        raise ValueError(
            f"Unsupported target backend: {args.target}. "
            "Supported backends are: pennylane, qiskit."
        )

    gate_specs = gate_spec_map[args.target]
    if args.use_only is not None:
        gate_specs = gate_specs.use_only(args.use_only)

    hyperparameters["n_classes"] = objective.n_classes

    master_worker(
        gate_specifications=gate_specs,
        population=population,
        objective=objective,
        hyperparameters=hyperparameters,
        mutation_strategy=args.mutation_strategy,
        run_for=args.number_genomes,
        input_registers={"input": args.input_qubits},
        output_registers={"output": math.ceil(math.log(objective.n_classes, 2))},
        target=args.target,
    )