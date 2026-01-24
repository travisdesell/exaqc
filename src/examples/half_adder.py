import argparse

# import matplotlib.pyplot as plt
import pennylane as qml
import torch
import sys

from loguru import logger

from src.evolution.exaqc import EXAQC
from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.population.steady_state_population import SteadyStatePopulation
from src.objectives.genome_objectives import (
    train_genome_objective,
    genome_to_torch_params,
)
from src.quantum_datasets import HalfAdderDataset

best_genome: CircuitGenome = None

dataset = HalfAdderDataset()


def half_adder_objective(genome: CircuitGenome):
    """
    Memetic objective:
      - For each input (a,b)
      - Train parameters via fidelity loss
      - Average final fidelity loss
    """
    global best_fitness, best_genome, dataset  # noqa: F824

    total_fidelity_loss = 0.0

    n_qubits = sum(genome.registers.values())
    example_input = torch.zeros(n_qubits, dtype=torch.int64)
    params = genome_to_torch_params(genome)

    for input_bits, target_state in dataset:
        genome = train_genome_objective(
            genome,
            input_bits=input_bits,
            target_state=target_state,
            target="pennylane",
            steps=500,
        )

        logger.info(f"Loss: {genome.fitness}")

        if genome.fitness is None:
            # No trainable parameters → evaluate directly
            qnode = genome.circuit
            with torch.no_grad():
                psi = qnode(input_bits, genome_to_torch_params(genome))
                psi = psi / torch.linalg.norm(psi)
                phi = target_state / torch.linalg.norm(target_state)

            fidelity_loss = 1.0 - torch.abs(torch.dot(psi.conj(), phi)) ** 2
            total_fidelity_loss += float(fidelity_loss.item())
        else:
            total_fidelity_loss += genome.fitness["fidelity_loss"]

    avg_loss = total_fidelity_loss / len(dataset)

    logger.info(f"genome fitness before setting to avg loss: {genome.fitness}")

    if genome.fitness is None:
        genome.fitness = {"fidelity_loss": avg_loss}
    else:
        genome.fitness["fidelity_loss"] = avg_loss

    logger.info(f"genome fitness after setting to avg loss: {genome.fitness}")

    if best_genome is None or genome.dominates(best_genome):
        logger.info(
            f"🎯 New best genome {genome.genome_number} "
            f"with fidelity loss: {avg_loss:.6f}"
        )

        best_genome = genome
        try:
            _, qnode = genome.generate_pennylane_circuit()
            fig, _ = qml.draw_mpl(qnode)(example_input, params)
            fig.savefig(
                f"plots/best_half_adder_{avg_loss:.6f}.png",
                dpi=200,
                bbox_inches="tight",
            )
        except Exception as e:
            logger.error(f"⚠️ Could not plot circuit (adjoint issue?): {e}")

    return genome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple CSV files into a single file.",
    )

    parser.add_argument(
        "--max_population_size",
        "-ms",
        type=int,
        default=50,
        help="The maximum population size for EXAQC's steady state population",
    )

    parser.add_argument(
        "--number_genomes",
        "-ng",
        type=int,
        default=1000,
        help="The maximum number of genomes to generate in this run of EXAQC",
    )

    parser.add_argument(
        "--allowed_gates",
        "-g",
        required=False,
        type=str,
        nargs="+",
        default=None,
        help=(
            "If specified, EXAQC will only add gates with the given method names, ",
            "otherwise it will use all available gates.",
        ),
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        required=False,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    # Parse arguments
    args = parser.parse_args()

    # remove the old logging handler.
    logger.remove()
    # create a new logging handler at the appropriate level
    logger.add(sys.stdout, level=args.logging_level)

    max_population_size = args.max_population_size
    number_genomes = args.number_genomes

    allowed_gates = pennylane_gate_specifications

    if args.allowed_gates is not None:
        allowed_gates = allowed_gates.use_only(args.allowed_gates)

    exaqc = EXAQC(
        gate_specifications=allowed_gates,
        population=SteadyStatePopulation(max_population_size=50),
        registers={"q": 4},
        objective_function=half_adder_objective,
        target="pennylane",
    )

    exaqc.run_for(number_genomes)

    _, qnode = best_genome.generate_pennylane_circuit()

    # for input_bits, target_state in dataset:
    #     print(input_bits)
    #     print(target_state)
    #     with torch.no_grad():
    #             psi = qnode(input_bits, genome_to_torch_params(best_genome))
    #             psi = psi / torch.linalg.norm(psi)
    #             phi = target_state / torch.linalg.norm(target_state)
    #             print(f"Expected: {phi}")
