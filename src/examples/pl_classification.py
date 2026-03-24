from __future__ import annotations

import argparse
import math
import os
import torch
import sys
from typing import Iterable, Optional

from loguru import logger
import numpy as np

from src.evolution.master_worker import master_worker

# from src.evolution.exaqc import EXAQC
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.evolution.objective import Objective
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.circuit import CircuitGenome
from src.objectives.genome_objectives import (
    train_genome_objective,
)
from src.utils.helpers import genome_to_torch_params
from src.utils.losses import LOSS_REGISTRY
from src.quantum_datasets import (
    IrisDataset,
    WineDataset,
    SeedsDataset,
    BreastCancerDataset,
    QuantumDataset,
)

# ---------------------------------------------------------------------
# Prediction + evaluation helpers
# ---------------------------------------------------------------------


@torch.no_grad()
def predict_from_probs(
    probs_full: torch.Tensor, *, n_classes: int = 3, eps: float = 1e-12
) -> tuple[int, torch.Tensor]:
    """
    probs_full: float tensor [2**n_output_qubits] (e.g., 4 if output has 2 qubits)
    """
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
    """
    Assumes genome.circuit returns qml.probs(wires=output_wires) (real-valued).
    dataset yields: (x: float[4], y_onehot: float[3])
    """
    # Ensure qnode exists
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        # IMPORTANT: we want probs readout for classification
        genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")

    loss_fn = LOSS_REGISTRY[loss]

    params = genome_to_torch_params(genome)  # empty dict if no params
    losses = []
    correct = 0
    total = 0
    per_class_pred = {}

    # setting Alpha from https://arxiv.org/pdf/1901.05555
    beta = (len(dataset) - 1) / len(dataset)
    alpha = (1.0 - beta) / (
        1.0 - np.power(beta, np.array(dataset.counts, dtype=np.float32))
    )
    alpha = torch.as_tensor(alpha / alpha.mean(), dtype=torch.float32)

    for x, y, cls in dataset:
        if cls not in per_class_pred:
            per_class_pred[cls] = 0

        probs_full = genome.circuit(x, params)
        probs_full = torch.as_tensor(probs_full, dtype=torch.float32)

        pred, probs = predict_from_probs(probs_full, n_classes=n_classes)
        L = loss_fn(probs, y, alpha_per_class=alpha)

        losses.append(L)
        true = int(torch.argmax(y).item())
        correct += int(pred == true)
        total += 1

        if pred == true:
            per_class_pred[cls] += 1

    avg_loss = float(torch.stack(losses).mean().item()) if losses else 0.0
    acc = float(correct / max(total, 1))

    log = ""
    for k, v in dataset.class_counts.items():
        log += (
            f"[{k}] Accuracy: {per_class_pred[k] / v:.4f} ({per_class_pred[k]}/{v}) | "
        )
    logger.info(f"{log}")

    return {"loss": avg_loss, "acc": acc}


# ---------------------------------------------------------------------
# Objective and single objective comparison
# ---------------------------------------------------------------------


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """
    Used to sort genomes by fitness, even if there are multiple objectives, for population
    management and crossover methods.

    Returns: 0 if the two genomes have equivalent fitnesses, a ngeative value if genome1 should be
        sorted before genome2, and a positive value if genome2 should be sorted before genome1
    """

    # this will return 0 if the losses are the same, negative if genome1 should be before
    # genome2 (genome1's fitness would be lower), and positive if genome2 should be before
    # genome1 (genome2's fitness would be lower)
    return genome1.fitness["test_loss"] - genome2.fitness["test_loss"]
    # return genome2.fitness["test_acc"] - genome1.fitness["test_acc"]


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
        """
        Trains the circuit given the specified hyperparameters and sets its
        loss values.

        Args:
            genome: is the CircuitGenome to train and evaluate. It will have
                a dict of hyperparameters specified by EXAQC, and should
                set its `fitness` attribute with a dict of fitness values
                after training and evaluation.
        """

        hyperparameters = genome.hyperparameters
        learning_rate = hyperparameters["learning_rate"]
        epochs = hyperparameters["epochs"]
        batch_size = hyperparameters["batch_size"]
        log_every = hyperparameters["log_every"]
        encoding = hyperparameters["encoding"]

        # If there are trainable params, train. If not, just forward/eval.
        torch_params = genome_to_torch_params(genome)
        if len(torch_params) > 0:
            train_genome_objective(
                genome,
                dataset=[self.train_data, self.test_data],  # train split only
                backend=self.target,
                encoding=encoding,
                loss=self.loss,  # e.g., "ce"
                epochs=epochs,
                lr=learning_rate,
                n_classes=self.n_classes,
                log_every=log_every,
                batch_size=batch_size,
            )

        # Compute fresh train/test metrics from probs (works for both param & no-param cases)
        train_metrics = eval_probs_ce_and_acc(
            genome, self.train_data, n_classes=self.n_classes, loss=self.loss
        )
        test_metrics = eval_probs_ce_and_acc(
            genome, self.test_data, n_classes=self.n_classes, loss=self.loss
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
            f"test acc={test_metrics['acc']:.4f} "
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", choices=["iris", "wine", "seeds", "breast_cancer"], required=True
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs",
    )

    p.add_argument(
        "--loss", default="ce", choices=["bce", "focal", "ce", "mse", "kl", "fidelity"]
    )

    subparsers = p.add_subparsers(
        dest="population_strategy",
        help="Specify how genomes will be handled.",
        required=True,
    )

    steady_state_parser = subparsers.add_parser(
        "steady_state", help="Use a single steady state population."
    )
    steady_state_parser.add_argument("--max_population_size", type=int, default=30)

    islands_parser = subparsers.add_parser(
        "islands", help="Use multiple islands of steady state opulations."
    )
    islands_parser.add_argument("--n_islands", type=int, default=10)
    islands_parser.add_argument("--max_island_size", type=int, default=10)
    islands_parser.add_argument("--genomes_before_extinction", type=int, default=200)
    islands_parser.add_argument("--islands_to_extinct", type=int, default=1)
    islands_parser.add_argument(
        "--intra_island_crossover_rate", type=float, default=0.5
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
        help="Choose the kind of encoding",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Use mini-batch training with the given batch size, if provided",
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
    logger.add(os.path.join(args.out_dir, "run.log"))

    # specify hyperparameter options for genome evaluation
    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "log_every": 15,
        "batch_size": args.batch_size,
        "encoding": args.encoding,
    }

    # set up the objective function
    objective = None
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

    population = None
    print(f"args.population_strategy: {args.population_strategy}")

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
            islands_to_extinct=args.islands_to_extinct,
            compare=compare,
            out_dir=args.out_dir,
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
