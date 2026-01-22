import argparse
import os
import torch
import pennylane as qml

from src.evolution.exaqc import EXAQC
from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.population.steady_state_population import SteadyStatePopulation
from src.objectives.genome_objectives import (
    train_genome_objective,
    genome_to_torch_params,
)
from src.datasets import HalfAdderDataset

# -----------------------
# Globals
# -----------------------

dataset = HalfAdderDataset()
best_fitness = float("inf")
best_genome: CircuitGenome | None = None

os.makedirs("plots", exist_ok=True)


# -----------------------
# Dataset-specific helpers
# -----------------------


def make_half_adder_y_extractor(dataset: HalfAdderDataset):
    """
    Convert a target basis-state vector into [sum, carry] bits.

    Must match bits_to_statevector() encoding:
        index |= bit << (n_qubits - i - 1)
    """
    n_qubits = dataset.n_qubits
    out_q = list(dataset.output_qubits)  # (sum, carry)

    def y_extractor(target_state: torch.Tensor) -> torch.Tensor:
        probs = target_state.real**2 + target_state.imag**2
        idx = int(torch.argmax(probs).item())

        def bit_at_qubit(i: int) -> int:
            return (idx >> (n_qubits - 1 - i)) & 1

        return torch.tensor(
            [bit_at_qubit(out_q[0]), bit_at_qubit(out_q[1])],
            dtype=torch.float32,
        )

    return y_extractor


# -----------------------
# EXAQC objective
# -----------------------


def half_adder_objective(
    genome: CircuitGenome, target: str = "pennylane", loss: str = "fidelity"
):
    """
    Objective used by EXAQC.

    PennyLane:
      - train once across full truth table
      - fitness = average fidelity loss

    Qiskit ML:
      - train once across full truth table
      - fitness = task loss (MSE on output bits)
    """
    global best_fitness, best_genome

    # ---------- PennyLane backend ----------
    if target == "pennylane":
        genome = train_genome_objective(
            genome,
            dataset=dataset,
            backend="pennylane",
            steps=500,
            lr=0.05,
            loss=loss,
        )

        avg_loss = genome.fitness[f"{loss}_loss"]

        if avg_loss < best_fitness:
            best_fitness = avg_loss
            best_genome = genome
            print(
                f"🎯 New best genome {genome.genome_number} "
                f"(PennyLane) loss={avg_loss:.6f}"
            )

            # Plot best PennyLane circuit
            try:
                params = genome_to_torch_params(genome)
                example_input = torch.zeros(dataset.n_qubits, dtype=torch.complex128)
                # dev, qnode = genome.generate_pennylane_circuit(measure_registers=False)
                qnode = genome.circuit
                fig, _ = qml.draw_mpl(qnode)(example_input, params)
                fig.savefig(
                    f"plots/best_half_adder_pl_{avg_loss:.6f}.png",
                    dpi=200,
                    bbox_inches="tight",
                )
            except Exception as e:
                print(f"⚠️ Could not plot PennyLane circuit: {e}")

        return genome

    # ---------- Qiskit ML backend ----------
    if target == "qiskit":
        y_extractor = make_half_adder_y_extractor(dataset)

        genome = train_genome_objective(
            genome,
            dataset=dataset,
            backend="qiskit",
            steps=300,
            lr=0.05,
            loss=loss,
            qiskit_config={
                "n_qubits": dataset.n_qubits,
                "input_qubits": list(dataset.input_qubits),
                "output_qubits": list(dataset.output_qubits),
                "y_extractor": y_extractor,
            },
        )

        avg_loss = genome.fitness[f"{loss}_loss"]

        if avg_loss < best_fitness:
            best_fitness = avg_loss
            best_genome = genome
            print(
                f"🎯 New best genome {genome.genome_number} "
                f"(Qiskit ML) loss={avg_loss:.6f}"
            )

        return genome

    raise ValueError(f"Unknown backend: {target}")


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evolve half-adder quantum circuits with PennyLane or Qiskit ML."
    )
    parser.add_argument("--max_population_size", "-ms", type=int, default=50)
    parser.add_argument("--number_genomes", "-ng", type=int, default=1000)
    parser.add_argument("--allowed_gates", "-g", nargs="+", default=None)
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="qiskit",
        choices=["qiskit", "pennylane"],
    )
    parser.add_argument(
        "--loss",
        "-l",
        required=False,
        type=str,
        default="fidelity",
        help=(
            "If specified, EXAQC will use the provided option between [fidelity, mse, kl, angle] ",
            "otherwise it will use qiskit.",
        ),
    )
    args = parser.parse_args()

    if args.backend == "pennylane":
        allowed_gates = pennylane_gate_specifications
    else:
        allowed_gates = qiskit_gate_specifications
    if args.allowed_gates is not None:
        allowed_gates = allowed_gates.use_only(args.allowed_gates)

    exaqc = EXAQC(
        gate_specifications=allowed_gates,
        population=SteadyStatePopulation(max_population_size=args.max_population_size),
        registers={"input": 2, "output": 2},
        objective_function=half_adder_objective,
        target=args.backend,
    )

    exaqc.run_for(args.number_genomes)
