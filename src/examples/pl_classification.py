from __future__ import annotations

import argparse
import math
import os
import torch
import sys
from typing import Iterable

import pennylane as qml
from loguru import logger
import matplotlib.pyplot as plt

from src.evolution.master_worker import master_worker

# from src.evolution.exaqc import EXAQC
from src.evolution.steady_state_population import SteadyStatePopulation
from src.evolution.objective import Objective
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.circuit import CircuitGenome
from src.objectives.genome_objectives import (
    train_genome_objective,
    genome_to_torch_params,
)
from src.utils.helpers import register_wire_map

from src.quantum_datasets import (
    IrisDataset,
    WineDataset,
    SeedsDataset,
    BreastCancerDataset,
    QuantumDataset,
)

logger.add("run.log", level="INFO")

# best_fitness = float("inf")
# best_genome: CircuitGenome | None = None


# ---------------------------------------------------------------------
# Prediction + evaluation helpers
# ---------------------------------------------------------------------


def ce_onehot_on_probs(
    probs: torch.Tensor, y_onehot: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    probs: float tensor [K] (already marginal over output wires)
    y_onehot: float tensor [K]
    """
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum()
    y_onehot = y_onehot.to(dtype=probs.dtype, device=probs.device)
    return -(y_onehot * torch.log(probs)).sum()


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
    dataset: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    n_classes: int,
) -> dict[str, float]:
    """
    Assumes genome.circuit returns qml.probs(wires=output_wires) (real-valued).
    dataset yields: (x: float[4], y_onehot: float[3])
    """
    # Ensure qnode exists
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        # IMPORTANT: we want probs readout for classification
        genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")

    params = genome_to_torch_params(genome)  # empty dict if no params
    losses = []
    correct = 0
    total = 0

    for x, y in dataset:
        probs_full = genome.circuit(x, params)
        probs_full = torch.as_tensor(probs_full, dtype=torch.float32)

        pred, probs = predict_from_probs(probs_full, n_classes=n_classes)
        L = ce_onehot_on_probs(probs, y)

        losses.append(L)
        correct += int(pred == int(torch.argmax(y).item()))
        total += 1

    avg_loss = float(torch.stack(losses).mean().item()) if losses else 0.0
    acc = float(correct / max(total, 1))
    return {"loss": avg_loss, "acc": acc}


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------


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

    def compare(self, genome1: CircuitGenome, genome2: CircuitGenome) -> int:
        """
        Used to sort genomes by fitness, even if there are multiple objectives, for population
        management and crossover methods.

        Returns: 0 if the two genomes have equivalent fitnesses, a ngeative value if genome1 should be
            sorted before genome2, and a positive value if genome2 should be sorted before genome1
        """

        # this will return 0 if the losses are the same, negative if genome1 should be before
        # genome2 (genome1's fitness would be lower), and positive if genome2 should be before
        # genome1 (genome2's fitness would be lower)
        return genome1.fitness['test_loss'] - genome2.fitness['test_loss']


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
        steps = hyperparameters["steps"]
        batch_size = hyperparameters["batch_size"]
        log_every = hyperparameters["log_every"]

        # If there are trainable params, train. If not, just forward/eval.
        torch_params = genome_to_torch_params(genome)
        if len(torch_params) > 0:
            genome = train_genome_objective(
                genome,
                dataset=[self.train_data, self.test_data],  # train split only
                backend=self.target,
                loss=self.loss,  # e.g., "ce"
                steps=steps,
                lr=learning_rate,
                n_classes=self.n_classes,
                log_every=log_every,
                bath_size=batch_size,
            )

        # Compute fresh train/test metrics from probs (works for both param & no-param cases)
        train_metrics = eval_probs_ce_and_acc(
            genome, self.train_data, n_classes=self.n_classes
        )
        test_metrics = eval_probs_ce_and_acc(
            genome, self.test_data, n_classes=self.n_classes
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
            f"test loss={test_metrics['loss']:.4f}"
            f"test acc={test_metrics['acc']:.4f}"
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
    p.add_argument("--loss", default="ce", choices=["ce", "mse", "kl", "fidelity"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max_population_size", type=int, default=50)
    p.add_argument("--number_genomes", type=int, default=500)
    p.add_argument("--input_qubits", type=int, default=6)
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
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
    logger.add(os.path.join(args.out_dir, args.dataset, "run.log"))

    # specify hyperparameter options for genome evaluation
    hyperparameters = {
        'steps': 20,
        'learning_rate': 5e-3,
        'log_every': 10,
        'batch_size': args.batch_size,
    }

    # set up the objective function
    objective = None
    if args.dataset == "iris":
        objective = ClassificationObjective(
            train_data=IrisDataset(split="train"),
            test_data=IrisDataset(split="test"), input_size=4, n_classes=3,
        )

    elif args.dataset == "wine":
        objective = ClassificationObjective(
            train_data=WineDataset(split="train"),
            test_data=WineDataset(split="test"), input_size=13, n_classes=3,
        )

    elif args.dataset == "seeds":
        objective = ClassificationObjective(
            train_data=SeedsDataset(split="train"),
            test_data=SeedsDataset(split="test"), input_size=7, n_classes=3,
        )

    elif args.dataset == "breast_cancer":
        objective = ClassificationObjective(
            train_data=BreastCancerDataset(split="train"),
            test_data=BreastCancerDataset(split="test"), input_size=30, n_classes=2,
        )

    else:
        raise ValueError(dataset_name)

    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=objective.compare,
            out_dir=args.out_dir,
        ),
        objective=objective,
        hyperparameters=hyperparameters,
        run_for=args.number_genomes,
        input_registers={"input": min(args.input_qubits, objective.input_size)},
        output_registers={"output": math.ceil(math.log(objective.n_classes, 2))},
        target="pennylane",
    )
