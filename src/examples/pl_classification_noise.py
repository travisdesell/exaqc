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
from src.datasets.classification import (
    BreastCancerDataset,
    IrisDataset,
    SeedsDataset,
    WineDataset,
)
from src.evolution.master_worker import master_worker
from src.evolution.objective import Objective
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.noise import PennyLaneNoiseModel
from src.objectives.genome_objectives import train_genome_objective
from src.utils.helpers import genome_to_torch_params
from src.utils.losses import LOSS_REGISTRY, ce_onehot_on_probs


# ---------------------------------------------------------------------
# Prediction + evaluation helpers
# ---------------------------------------------------------------------


@torch.no_grad()
def predict_from_probs(
    probs_full: torch.Tensor,
    *,
    n_classes: int,
    eps: float = 1e-12,
) -> tuple[int, torch.Tensor]:
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
    encoding: str = "angle",
    alpha: torch.Tensor = None,
    noise_model: Optional[PennyLaneNoiseModel] = None,
) -> dict[str, float]:
    """Evaluate a genome using probability readout.

    Assumes genome.circuit returns qml.probs over output wires.
    """

    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(
            return_probs=True,
            input_mode=encoding,
            noise_model=noise_model,
        )

    loss_fn = LOSS_REGISTRY[loss]

    params = genome_to_torch_params(genome)
    losses = []
    probas = []
    y_onehots = []
    correct = 0
    total = 0
    per_class_pred = {}

    for x, y, cls in dataset:
        if cls not in per_class_pred:
            per_class_pred[cls] = 0

        probs_full = genome.circuit(x, params)
        probs_full = torch.as_tensor(probs_full, dtype=torch.float32)

        pred, probs = predict_from_probs(probs_full, n_classes=n_classes)

        # Keep CE reporting consistent.
        L = ce_onehot_on_probs(probs, y, alpha_per_class=alpha)

        losses.append(L)
        true = int(torch.argmax(y).item())
        correct += int(pred == true)
        total += 1

        if pred == true:
            per_class_pred[cls] += 1

        probas.append(probs)
        y_onehots.append(y)

    if loss_fn.__name__ != "class_avg_ce_onehot_on_probs":
        avg_loss = float(torch.stack(losses).mean().item()) if losses else 0.0
    else:
        probs = torch.stack([p.to(torch.float32) for p in probas], dim=0)
        y_onehots = torch.stack([p.to(torch.float32) for p in y_onehots], dim=0)
        avg_loss = float(loss_fn(probs, y_onehots))

    acc = float(correct / max(total, 1))

    log = ""
    for k, v in dataset.class_counts.items():
        log += (
            f"[{k}] Accuracy: {per_class_pred[k] / v:.4f} "
            f"({per_class_pred[k]}/{v}) | "
        )
    logger.info(log)

    return {"loss": avg_loss, "acc": acc}


# ---------------------------------------------------------------------
# Objective and comparison
# ---------------------------------------------------------------------


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> float:
    """Sort genomes by test loss; lower is better."""
    return genome1.fitness["test_loss"] - genome2.fitness["test_loss"]


class ClassificationObjective(Objective):
    def __init__(
        self,
        train_data: QuantumDataset,
        test_data: QuantumDataset,
        input_size: int,
        n_classes: int,
        loss: str = "ce",
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.input_size = input_size
        self.n_classes = n_classes
        self.loss = loss
        self.target = "pennylane"

    def __call__(self, genome: CircuitGenome):
        hp = genome.hyperparameters

        learning_rate = hp["learning_rate"]
        epochs = hp["epochs"]
        batch_size = hp["batch_size"]
        log_every = hp["log_every"]
        encoding = hp["encoding"]

        noise_model = PennyLaneNoiseModel.from_hyperparameters(hp)

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
                noise_model=noise_model,
            )

        # setting Alpha from https://arxiv.org/pdf/1901.05555
        beta = (len(self.train_data) - 1) / len(self.train_data)
        alpha = (1.0 - beta) / (
            1.0 - np.power(beta, np.array(self.train_data.counts, dtype=np.float32))
        )
        alpha = torch.as_tensor(alpha / alpha.mean(), dtype=torch.float32)

        train_metrics = eval_probs_ce_and_acc(
            genome,
            self.train_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=encoding,
            alpha=alpha,
            noise_model=noise_model,
        )

        test_metrics = eval_probs_ce_and_acc(
            genome,
            self.test_data,
            n_classes=self.n_classes,
            loss=self.loss,
            encoding=encoding,
            alpha=alpha,
            noise_model=noise_model,
        )

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "noise_type": hp.get("noise_type", "none"),
            "noise_p": float(hp.get("noise_p", 0.0)),
            "noise_p_1q": hp.get("noise_p_1q", None),
            "noise_p_2q": hp.get("noise_p_2q", None),
            "noise_gamma": float(hp.get("noise_gamma", 0.0)),
        }

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"train loss={train_metrics['loss']:.4f} "
            f"train acc={train_metrics['acc']:.4f} "
            f"test loss={test_metrics['loss']:.4f} "
            f"test acc={test_metrics['acc']:.4f} "
            f"noise={hp.get('noise_type', 'none')}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        choices=["iris", "wine", "seeds", "breast_cancer"],
        required=True,
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs.",
    )

    p.add_argument(
        "--loss",
        default="ce",
        choices=["per_class", "bce", "focal", "ce", "mse", "kl", "fidelity"],
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning_rate", "-lr", type=float, default=5e-4)
    p.add_argument("--number_genomes", type=int, default=2000)
    p.add_argument("--input_qubits", type=int, default=6)

    p.add_argument(
        "--encoding",
        choices=["basis", "angle", "amplitude"],
        type=str,
        default="angle",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Mini-batch size.",
    )

    p.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
    )

    # -------------------------
    # Noise arguments
    # -------------------------

    p.add_argument(
        "--noise_type",
        choices=[
            "none",
            "depolarizing",
            "bit_flip",
            "phase_flip",
            "amplitude_damping",
            "phase_damping",
            "thermal_relaxation",
            "mixed",
        ],
        default="none",
    )

    p.add_argument("--noise_p", type=float, default=0.0)
    p.add_argument("--noise_p_1q", type=float, default=None)
    p.add_argument("--noise_p_2q", type=float, default=None)
    p.add_argument("--noise_gamma", type=float, default=0.0)

    p.add_argument("--noise_after_encoding", action="store_true")
    p.add_argument("--noise_after_gates", action="store_true")
    p.add_argument("--noise_before_measurement", action="store_true")

    # -------------------------
    # Population strategy
    # -------------------------

    subparsers = p.add_subparsers(
        dest="population_strategy",
        required=True,
    )

    steady_state_parser = subparsers.add_parser(
        "steady_state",
        help="Use a single steady-state population.",
    )
    steady_state_parser.add_argument("--max_population_size", type=int, default=30)

    islands_parser = subparsers.add_parser(
        "islands",
        help="Use multiple islands of steady-state populations.",
    )
    islands_parser.add_argument("--n_islands", type=int, default=10)
    islands_parser.add_argument("--max_island_size", type=int, default=10)
    islands_parser.add_argument("--genomes_before_extinction", type=int, default=100)
    islands_parser.add_argument("--genomes_for_next_extinction", type=int, default=200)
    islands_parser.add_argument("--islands_to_extinct", type=int, default=2)
    islands_parser.add_argument(
        "--intra_island_crossover_rate", type=float, default=0.5
    )

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"))

    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "log_every": 15,
        "batch_size": args.batch_size,
        "encoding": args.encoding,
        "noise_type": args.noise_type,
        "noise_p": args.noise_p,
        "noise_p_1q": args.noise_p_1q,
        "noise_p_2q": args.noise_p_2q,
        "noise_gamma": args.noise_gamma,
        "noise_after_encoding": args.noise_after_encoding,
        "noise_after_gates": args.noise_after_gates,
        "noise_before_measurement": args.noise_before_measurement,
    }

    if args.dataset == "iris":
        objective = ClassificationObjective(
            train_data=IrisDataset(split="train"),
            test_data=IrisDataset(split="test"),
            loss=args.loss,
            input_size=4,
            n_classes=3,
        )

    elif args.dataset == "wine":
        objective = ClassificationObjective(
            train_data=WineDataset(split="train"),
            test_data=WineDataset(split="test"),
            loss=args.loss,
            input_size=13,
            n_classes=3,
        )

    elif args.dataset == "seeds":
        objective = ClassificationObjective(
            train_data=SeedsDataset(split="train"),
            test_data=SeedsDataset(split="test"),
            loss=args.loss,
            input_size=7,
            n_classes=3,
        )

    elif args.dataset == "breast_cancer":
        objective = ClassificationObjective(
            train_data=BreastCancerDataset(split="train"),
            test_data=BreastCancerDataset(split="test"),
            loss=args.loss,
            input_size=30,
            n_classes=2,
        )

    else:
        raise ValueError(args.dataset)

    logger.info(
        f"Running dataset={args.dataset}, loss={args.loss}, "
        f"noise_type={args.noise_type}, noise_p={args.noise_p}, "
        f"noise_p_1q={args.noise_p_1q}, noise_p_2q={args.noise_p_2q}, "
        f"noise_gamma={args.noise_gamma}"
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
