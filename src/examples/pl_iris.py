# src/examples/iris_evolve_pennylane.py
from __future__ import annotations

import argparse
import os
from typing import Iterable

import torch
from loguru import logger

import pennylane as qml
import matplotlib.pyplot as plt

from src.utils.helpers import register_wire_map
from src.evolution.exaqc import EXAQC
from src.population.steady_state_population import SteadyStatePopulation
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.circuit import CircuitGenome
from src.objectives.genome_objectives import (
    train_genome_objective,
    genome_to_torch_params,
)
from src.quantum_datasets import IrisDataset

logger.add("iris_evolve.log", level="INFO")


# -----------------------
# Global dataset (avoid re-loading per objective call)
# -----------------------
TRAIN_DS = IrisDataset(split="train", train_frac=0.8, seed=0)
TEST_DS = IrisDataset(split="test", train_frac=0.8, seed=0)


best_fitness = float("inf")
best_genome: CircuitGenome | None = None


# -----------------------
# Helpers: CE + accuracy on PROBS
# -----------------------
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
    n_classes: int = 3,
) -> dict[str, float]:
    """
    Assumes genome.circuit returns qml.probs(wires=output_wires) (real-valued).
    dataset yields: (x: float[4], y_onehot: float[3])
    """
    # Ensure qnode exists
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        # IMPORTANT: we want probs readout for classification
        genome.generate_pennylane_circuit(measure_registers=True, input_mode="angle")

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


# -----------------------
# Save best circuit diagram
# -----------------------
def save_best_circuit_png(
    genome: CircuitGenome,
    *,
    out_dir: str,
    tag: str,
):
    """
    Saves a PNG circuit diagram for the current best genome.

    Requires that genome.circuit is a PennyLane QNode.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Make sure circuit exists and is in "probs readout" mode for consistent drawing
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")

    # Use a deterministic sample input and current params for drawing
    x0, _ = TRAIN_DS[0]
    params = genome_to_torch_params(genome)

    try:
        fig, ax = qml.draw_mpl(genome.circuit)(x0, params)
        ax.set_title(f"Genome {genome.genome_number} ({tag})")
        path = os.path.join(out_dir, f"best_genome_{genome.genome_number}_{tag}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved circuit diagram: {path}")
    except Exception as e:
        logger.warning(
            f"Failed to save circuit diagram for genome {genome.genome_number}: {e}"
        )


# -----------------------
# Objective
# -----------------------
def iris_objective(
    genome: CircuitGenome,
    target: str = "pennylane",
    loss: str = "ce",
    *,
    steps: int = 250,
    lr: float = 1e-3,
    log_every: int = 50,
    save_dir: str = "artifacts/iris_best_circuits",
):
    """
    Fitness = TRAIN CE (lower is better)
    Also logs TEST accuracy.
    Saves a circuit diagram each time a new best is found.
    """
    global best_fitness, best_genome

    # Ensure circuit is created in a way that supports classification readout
    # (we want probabilities on the output register)
    genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")

    # If there are trainable params, train. If not, just forward/eval.
    torch_params = genome_to_torch_params(genome)
    if len(torch_params) > 0:
        genome = train_genome_objective(
            genome,
            dataset=[TRAIN_DS, TEST_DS],  # train split only
            backend=target,
            loss=loss,  # e.g., "ce"
            steps=steps,
            lr=lr,
            log_every=log_every,
        )

    # Compute fresh train/test metrics from probs (works for both param & no-param cases)
    train_metrics = eval_probs_ce_and_acc(genome, TRAIN_DS, n_classes=3)
    test_metrics = eval_probs_ce_and_acc(genome, TEST_DS, n_classes=3)

    # Fitness is train loss (lower better)
    avg_loss = float(train_metrics["loss"])
    genome.fitness = {
        "train_loss": avg_loss,
        "train_acc": float(train_metrics["acc"]),
        "loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
    }

    if avg_loss < best_fitness:
        best_fitness = avg_loss
        best_genome = genome
        msg = (
            f"🎯 New best genome {genome.genome_number} "
            f"train_loss={avg_loss:.6f} train_acc={genome.fitness['train_acc']:.3f} "
            f"test_acc={genome.fitness['test_acc']:.3f}"
        )
        print(msg)
        logger.info(msg)

        # Save diagram for every new best genome
        tag = f"trainloss_{avg_loss:.4f}_testacc_{genome.fitness['test_acc']:.3f}"
        save_best_circuit_png(genome, out_dir=save_dir, tag=tag)

    return genome


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_population_size", "-ms", type=int, default=50)
    p.add_argument("--number_genomes", "-ng", type=int, default=500)
    p.add_argument("--allowed_gates", "-g", nargs="+", default=None)
    p.add_argument(
        "--loss", "-l", type=str, default="ce", choices=["ce", "mse", "kl", "fidelity"]
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_dir", type=str, default="artifacts/iris_best_circuits")
    args = p.parse_args()

    gates = pennylane_gate_specifications
    if args.allowed_gates is not None:
        gates = gates.use_only(args.allowed_gates)

    # Wrap objective to pass CLI params (EXAQC only calls objective(genome, target, loss))
    def _obj(genome: CircuitGenome, target: str = "pennylane", loss: str = "ce"):
        return iris_objective(
            genome,
            target=target,
            loss=loss,
            steps=args.steps,
            lr=args.lr,
            log_every=args.log_every,
            save_dir=args.save_dir,
        )

    qubits = {"input": 4, "output": 2}
    register_map = register_wire_map(qubits)
    logger.info(f"register map: {register_map}")

    exaqc = EXAQC(
        gate_specifications=gates,
        population=SteadyStatePopulation(
            max_population_size=args.max_population_size, loss=args.loss
        ),
        registers=qubits,
        objective_function=_obj,
        output_qubits=register_map["output"],
        target="pennylane",
        loss=args.loss,
    )

    exaqc.run_for(args.number_genomes)
