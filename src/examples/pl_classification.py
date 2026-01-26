from __future__ import annotations

import argparse
import os
import torch
import sys
from typing import Iterable

import pennylane as qml
from loguru import logger
import matplotlib.pyplot as plt

from src.evolution.master_worker import master_worker

# from src.evolution.exaqc import EXAQC
from src.population.steady_state_population import SteadyStatePopulation
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
)

logger.add("run.log", level="INFO")

best_fitness = float("inf")
best_genome: CircuitGenome | None = None


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
    n_classes: int = 3,
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
# Circuit saving
# ---------------------------------------------------------------------


def save_best_circuit(genome: CircuitGenome, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)

    genome.generate_pennylane_circuit(return_probs=True, measure_registers=False, input_mode="angle")

    # --- Text gate list ---
    txt_path = os.path.join(out_dir, f"genome_{genome.genome_number}.txt")
    with open(txt_path, "w") as f:
        genome.sort_gates()
        f.write(f"Genome {genome.genome_number}\n")
        f.write(f"Qubits: {genome.qubits}\n\n")
        for g in genome.gates:
            if getattr(g, "enabled", True):
                f.write(f"{g.depth:.3f}  {g.method_name}  {g.qubits}  {g.parameters}\n")

    # --- PennyLane draw ---
    try:
        params = genome_to_torch_params(genome)
        x0 = torch.zeros(len(genome.input_indexes))
        fig, ax = qml.draw_mpl(genome.circuit)(x0, params)
        ax.set_title(f"Genome {genome.genome_number} ({tag})")
        path = os.path.join(out_dir, f"best_genome_{genome.genome_number}_{tag}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not draw circuit: {e}")


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------


def make_objective(
    dataset_name: str,
    loss: str = "ce",
    steps: int = 250,
    lr: float = 1e-3,
    log_every: int = 50,
    batch_size: int = None,
):

    if dataset_name == "iris":
        train_data = IrisDataset(split="train")
        test_data = IrisDataset(split="test")
        input_size = 4
        n_classes = 3
    elif dataset_name == "wine":
        train_data = WineDataset(split="train")
        test_data = WineDataset(split="test")
        input_size = 13
        n_classes = 3
    elif dataset_name == "seeds":
        train_data = SeedsDataset(split="train")
        test_data = SeedsDataset(split="test")
        input_size = 7
        n_classes = 3
    elif dataset_name == "breast_cancer":
        train_data = BreastCancerDataset(split="train")
        test_data = BreastCancerDataset(split="test")
        input_size = 30
        n_classes = 2
    else:
        raise ValueError(dataset_name)

    def objective(
        genome: CircuitGenome, target="pennylane", loss=loss, batch_size=batch_size
    ):
        global best_fitness, best_genome
        # nonlocal train_data, test_data

        train_ds = train_data
        test_ds = test_data

        # If there are trainable params, train. If not, just forward/eval.
        torch_params = genome_to_torch_params(genome)
        if len(torch_params) > 0:
            genome = train_genome_objective(
                genome,
                dataset=[train_ds, test_ds],  # train split only
                backend=target,
                loss=loss,  # e.g., "ce"
                steps=steps,
                lr=lr,
                n_classes=n_classes,
                log_every=log_every,
                bath_size=batch_size,
            )

        # Compute fresh train/test metrics from probs (works for both param & no-param cases)
        train_metrics = eval_probs_ce_and_acc(genome, train_ds, n_classes=n_classes)
        test_metrics = eval_probs_ce_and_acc(genome, test_ds, n_classes=n_classes)

        # fit = genome.fitness or {}
        # avg_loss = float(fit.get("loss", fit.get("ce", float("inf"))))
        avg_loss = float(train_metrics["loss"])

        # genome.generate_pennylane_circuit(
        #     return_probs=True, input_mode="angle"
        # )

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
        }

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"loss={avg_loss:.4f} train={train_metrics['acc']:.3f} test={test_metrics['acc']:.3f}"
        )

        # if avg_loss < best_fitness:
        #     best_fitness = avg_loss
        #     best_genome = genome
        #     logger.info(
        #         f"🎯 New best genome {genome.genome_number} "
        #         f"loss={avg_loss:.4f} test_acc={test_metrics['acc']:.3f}"
        #     )
        #     tag = f"trainloss_{avg_loss:.4f}_testacc_{genome.fitness['test_acc']:.3f}"
        #     save_best_circuit(genome, f"artifacts/{dataset_name}_best", tag)

        return genome

    return objective, input_size


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", choices=["iris", "wine", "seeds", "breast_cancer"], required=True
    )
    p.add_argument("--loss", default="ce", choices=["ce", "mse", "kl", "fidelity"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max_population_size", type=int, default=50)
    p.add_argument("--number_genomes", type=int, default=500)
    p.add_argument("--input_qubits", type=int, default=6)
    p.add_argument("--out_qubits", type=int, default=2)  # 1 for breast cancer
    p.add_argument("--mini_batch", action="store_true", help="Run minibatch training")
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training; available only when mini_batch is set",
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

    bs = args.batch_size if args.mini_batch else None

    objective_fn, input_size = make_objective(
        args.dataset, loss=args.loss, steps=args.steps, lr=args.lr, batch_size=bs
    )

    qubits = {
        "input": min(args.input_qubits, input_size),
        "output": args.out_qubits,  # 2 readout qubits → 4 outcomes ≥ 3 classes
    }
    register_map = register_wire_map(qubits)
    logger.info(f"register map: {register_map}")

    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=args.max_population_size,
            loss="loss",  # weird that the genome loss vs the objective function loss are different
            dataset=args.dataset,
        ),
        objective_function=objective_fn,
        run_for=args.number_genomes,
        input_registers={"input": min(args.input_qubits, input_size)},
        output_registers={"output": args.out_qubits},
        target="pennylane",
        loss=args.loss,
        batch_size=bs,
    )

    """
    exaqc = EXAQC(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=args.max_population_size,
            loss="loss",  # weird that the genome loss vs the objective function loss are different
        ),
        objective_function=objective_fn,
        input_registers={"input": min(args.input_qubits, input_size)},
        output_registers={"output": args.out_qubits},
        target="pennylane",
        loss=args.loss,
        batch_size=bs,
    )

    exaqc.run_for(args.number_genomes)
    """
