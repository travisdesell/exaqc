from __future__ import annotations

import argparse
import os
import sys

from datetime import datetime
from typing import Iterable

import torch
from loguru import logger

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
from src.quantum_datasets import QuantumTeacherDataset


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


class TeacherObjective(Objective):
    def __init__(
        self,
        teacher_name: str,
        input_mode: str,
        n_train_inputs: int,
        n_test_inputs: int,
        input_wires: list[int],
        output_wires: list[int],
        seed: int,
    ):
        self.teacher_name = teacher_name
        self.n_wires = len(input_wires) + len(output_wires)

        # Create teacher once
        self.teacher = QuantumTeacherDataset.make_teacher_qnode(
            n_wires=self.n_wires,
            input_wires=input_wires,
            output_wires=output_wires,
            teacher_name=self.teacher_name,
            input_mode=input_mode,
        )

        # Create synthetic splits once
        self.X_train, self.X_test = QuantumTeacherDataset.make_teacher_inputs(
            n_train=n_train_inputs,
            n_test=n_test_inputs,
            input_dim=len(input_wires),
            seed=seed,
        )

    def __call__(
        self,
        genome: CircuitGenome,
    ):
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

        # Always train against teacher (even if you have 0 params, it'll just eval)
        train_genome_objective(
            genome,
            dataset=[self.X_train, self.X_test],  # your API expects [train, test]
            backend="pennylane",
            loss="fidelity",  # teacher imitation uses fidelity
            steps=steps,
            lr=learning_rate,
            log_every=log_every,
            batch_size=batch_size,
            teacher_qnode=self.teacher,
        )

        # Fresh eval metrics (train/test)
        train_metrics = eval_teacher_metrics(genome, self.teacher, self.X_train)
        test_metrics = eval_teacher_metrics(genome, self.teacher, self.X_test)

        genome.fitness = {
            "train_loss": float(train_metrics["loss"]),
            "train_fidelity": float(train_metrics["fidelity"]),
            "train_angle": float(train_metrics["angle"]),
            "test_loss": float(test_metrics["loss"]),
            "test_fidelity": float(test_metrics["fidelity"]),
            "test_angle": float(test_metrics["angle"]),
        }

        avg_loss = float(train_metrics["loss"])
        logger.info(
            f"[{genome.genome_number:04d}] "
            f"avg_loss={avg_loss:.4f} "
            f"train_fid={train_metrics['fidelity']:.3f} "
            f"test_fid={test_metrics['fidelity']:.3f}"
        )


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
            "x_out4",
            "bell_out",
            "copy_in_to_out",
            "parity012_to_out4",
            "input_controlled_bell",
            "2layer_out_block",
            "grover",
            "half_adder",
        ],
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs",
    )
    p.add_argument("--input_mode", default="angle", choices=["angle", "basis"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--learning_rate", "-lr", type=float, default=0.02)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--loss", default="fidelity", choices=["ce", "mse", "kl", "fidelity"]
    )

    p.add_argument("--max_population_size", type=int, default=30)
    p.add_argument("--number_genomes", type=int, default=2000)

    p.add_argument("--input_qubits", type=int, default=4)  # teacher input dim
    p.add_argument("--output_qubits", type=int, default=2)  # teacher output block size

    p.add_argument("--n_train_inputs", type=int, default=64)
    p.add_argument("--n_test_inputs", type=int, default=64)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=None)

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

    # Registers for student genomes
    qubits = {"input": args.input_qubits, "output": args.output_qubits}
    register_map = register_wire_map(qubits)
    logger.info(f"register map: {register_map}")

    # Teacher wiring consistent with your setup: input wires first then output wires
    n_wires = args.input_qubits + args.output_qubits
    input_wires = register_map["input"]
    output_wires = register_map["output"]

    seed = args.seed
    if seed is None:
        seed = int(datetime.now().timestamp())

    objective = TeacherObjective(
        teacher_name=args.teacher,
        input_mode=args.input_mode,
        n_train_inputs=args.n_train_inputs,
        n_test_inputs=args.n_test_inputs,
        input_wires=input_wires,
        output_wires=output_wires,
        seed=seed,
    )

    # specify hyperparameter options for genome evaluation
    hyperparameters = {
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "log_every": 15,
        "batch_size": args.batch_size,
    }

    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
        ),
        objective=objective,
        hyperparameters=hyperparameters,
        run_for=args.number_genomes,
        input_registers={"input": args.input_qubits},
        output_registers={"output": args.output_qubits},
        target="pennylane",
    )
