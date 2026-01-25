from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

import torch
import pennylane as qml
import matplotlib.pyplot as plt
from loguru import logger

from src.evolution.exaqc import EXAQC
from src.population.steady_state_population import SteadyStatePopulation
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.circuit import CircuitGenome
from src.objectives.genome_objectives import (
    train_genome_objective,
    genome_to_torch_params,
)
from src.utils.helpers import register_wire_map
from src.quantum_datasets import QuantumTeacherDataset

logger.add("run_teacher.log", level="INFO")

best_fitness = float("inf")
best_genome: CircuitGenome | None = None


# ---------------------------------------------------------------------
# Synthetic input splits for teacher mode
# ---------------------------------------------------------------------
def make_teacher_inputs(
    *,
    n_train: int,
    n_test: int,
    input_dim: int,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    X_train = torch.rand(n_train, input_dim, generator=g)
    X_test = torch.rand(n_test, input_dim, generator=g)
    return list(X_train), list(X_test)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
@torch.no_grad()
def fidelity_only(
    phi: torch.Tensor, psi: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    # robust fidelity for complex statevectors
    phi = torch.as_tensor(phi)
    psi = torch.as_tensor(psi)
    if not torch.is_complex(phi):
        phi = phi.to(
            torch.complex128 if phi.dtype == torch.float64 else torch.complex64
        )
    if not torch.is_complex(psi):
        psi = psi.to(
            torch.complex128 if psi.dtype == torch.float64 else torch.complex64
        )

    phi = phi / (torch.linalg.norm(phi) + eps)
    psi = psi / (torch.linalg.norm(psi) + eps)
    overlap = torch.vdot(phi, psi)
    return (overlap.conj() * overlap).real  # |<phi|psi>|^2


@torch.no_grad()
def eval_teacher_metrics(
    genome: CircuitGenome,
    teacher_qnode,
    inputs: Iterable[torch.Tensor],
) -> dict[str, float]:
    """
    Returns avg metrics for teacher imitation on a set of inputs:
      - loss: mean(1 - fidelity)
      - fidelity: mean(fidelity)
      - angle: mean(arccos(sqrt(fid)))  (optional, nice scalar)
    """
    # Ensure student circuit exists and returns state
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(measure_registers=False, input_mode="angle")

    params = genome_to_torch_params(genome)

    fids: list[torch.Tensor] = []
    for x in inputs:
        phi = teacher_qnode(x)
        psi = genome.circuit(x, params)
        fid = fidelity_only(phi, psi)
        fids.append(fid)

    if len(fids) == 0:
        return {"loss": 0.0, "fidelity": 0.0, "angle": 0.0}

    fid_mean = torch.stack(fids).mean()
    loss_mean = 1.0 - fid_mean
    angle_mean = torch.arccos(torch.clamp(torch.sqrt(fid_mean), 0.0, 1.0))
    return {
        "loss": float(loss_mean.item()),
        "fidelity": float(fid_mean.item()),
        "angle": float(angle_mean.item()),
    }


# ---------------------------------------------------------------------
# Circuit saving
# ---------------------------------------------------------------------
def save_best_circuit(genome: CircuitGenome, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)

    genome.generate_pennylane_circuit(measure_registers=False, input_mode="angle")

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
        x0 = torch.zeros(len(genome.qubits), dtype=torch.float32)
        fig, ax = qml.draw_mpl(genome.circuit)(x0, params)
        ax.set_title(f"Genome {genome.genome_number} ({tag})")
        path = os.path.join(out_dir, f"best_genome_{genome.genome_number}_{tag}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not draw circuit: {e}")


# ---------------------------------------------------------------------
# Objective factory (teacher mode)
# ---------------------------------------------------------------------
def make_teacher_objective(
    *,
    teacher_name: str,
    input_mode: str,
    steps: int,
    lr: float,
    log_every: int,
    batch_size: Optional[int],
    n_train_inputs: int,
    n_test_inputs: int,
    seed: int,
    n_wires: int,
    input_wires: list[int],
    output_wires: list[int],
):

    # Create teacher once
    teacher = QuantumTeacherDataset.make_teacher_qnode(
        n_wires=n_wires,
        input_wires=input_wires,
        output_wires=output_wires,
        teacher_name=teacher_name,
        input_mode=input_mode,
    )

    # Create synthetic splits once
    X_train, X_test = QuantumTeacherDataset.make_teacher_inputs(
        n_train=n_train_inputs,
        n_test=n_test_inputs,
        input_dim=len(input_wires),
        seed=seed,
    )

    def objective(
        genome: CircuitGenome,
        target="pennylane",
        loss="fidelity",
        batch_size=batch_size,
    ):
        global best_fitness, best_genome

        # Always train against teacher (even if you have 0 params, it'll just eval)
        genome = train_genome_objective(
            genome,
            dataset=[X_train, X_test],  # your API expects [train, test]
            backend=target,
            loss="fidelity",  # teacher imitation uses fidelity
            steps=steps,
            lr=lr,
            log_every=log_every,
            bath_size=batch_size,
            # IMPORTANT: your train_genome_objective must accept these two
            teacher_qnode=teacher,
            # teacher_mode=True,
        )

        # Fresh eval metrics (train/test)
        train_metrics = eval_teacher_metrics(genome, teacher, X_train)
        test_metrics = eval_teacher_metrics(genome, teacher, X_test)

        avg_loss = float(train_metrics["loss"])

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_fidelity": float(train_metrics["fidelity"]),
            "train_angle": float(train_metrics["angle"]),
            "loss": float(test_metrics["loss"]),
            "test_fidelity": float(test_metrics["fidelity"]),
            "test_angle": float(test_metrics["angle"]),
        }

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"loss={avg_loss:.4f} "
            f"train_fid={train_metrics['fidelity']:.3f} "
            f"test_fid={test_metrics['fidelity']:.3f}"
        )

        if avg_loss < best_fitness:
            best_fitness = avg_loss
            best_genome = genome
            logger.info(
                f"🎯 New best genome {genome.genome_number} "
                f"loss={avg_loss:.4f} test_fid={test_metrics['fidelity']:.3f}"
            )
            tag = f"trainloss_{avg_loss:.4f}_testfid_{test_metrics['fidelity']:.3f}"
            save_best_circuit(genome, f"artifacts/teacher_{teacher_name}_best", tag)

        return genome

    return objective


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher",
        required=True,
        choices=[
            "identity",
            "x_out0",
            "bell_out",
            "copy_in_to_out",
            "parity012_to_out0",
            "input_controlled_bell",
            "2layer_out_block",
        ],
    )
    p.add_argument("--input_mode", default="angle", choices=["angle", "basis"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--loss", default="fidelity", choices=["ce", "mse", "kl", "fidelity"]
    )

    p.add_argument("--max_population_size", type=int, default=50)
    p.add_argument("--number_genomes", type=int, default=500)

    p.add_argument("--input_qubits", type=int, default=4)  # teacher input dim
    p.add_argument("--out_qubits", type=int, default=2)  # teacher output block size

    p.add_argument("--n_train_inputs", type=int, default=64)
    p.add_argument("--n_test_inputs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--mini_batch", action="store_true", help="Run minibatch training")
    p.add_argument("--batch_size", type=int, default=16)

    args = p.parse_args()

    bs = args.batch_size if args.mini_batch else None

    # Registers for student genomes
    qubits = {"input": args.input_qubits, "output": args.out_qubits}
    register_map = register_wire_map(qubits)
    logger.info(f"register map: {register_map}")

    # Teacher wiring consistent with your setup: input wires first then output wires
    n_wires = args.input_qubits + args.out_qubits
    input_wires = register_map["input"]
    output_wires = register_map["output"]

    objective_fn = make_teacher_objective(
        teacher_name=args.teacher,
        input_mode=args.input_mode,
        steps=args.steps,
        lr=args.lr,
        log_every=args.log_every,
        batch_size=bs,
        n_train_inputs=args.n_train_inputs,
        n_test_inputs=args.n_test_inputs,
        seed=args.seed,
        n_wires=n_wires,
        input_wires=input_wires,
        output_wires=output_wires,
    )

    exaqc = EXAQC(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=args.max_population_size,
            loss="fidelity",
        ),
        input_registers={"input": args.input_qubits},
        output_registers={"output": args.out_qubits},
        objective_function=objective_fn,
        target="pennylane",
        loss="fidelity",
        batch_size=bs,
    )

    exaqc.run_for(args.number_genomes)
