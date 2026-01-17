import argparse
import matplotlib.pyplot as plt
import random
import pennylane as qml
import torch

from src.evolution.exaqc import EXAQC
from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.population.steady_state_population import SteadyStatePopulation

best_fitness = 1.0
count = 0


def random_objective_function(genome: CircuitGenome, target="pennylane"):
    """
    Computes a random fitness value for a given circuit. This will
    assign the genome's fitness attribute to the new fitness value. It
    also shows an example for modifying the genome's parameter values for
    parameterized gates.
    """
    global best_fitness, count

    for gate in genome.gates:
        for parameter, value in gate.parameters.items():
            # we can access the parameter values for each gate, and modify
            # them, e.g., if they are being trained
            gate.parameters[parameter] = value + random.uniform(-0.5, 0.5)

    torch_params = {
        f"{name}": torch.tensor(value, dtype=torch.float64)
        for gate in genome.gates
        for name, value in gate.parameters.items()
    }
    print(f"torch_params: {torch_params}")

    # just a test to make sure we can generate this circuit and not
    # break things
    _, circuit = genome.generate_pennylane_circuit()

    # Input qubits
    n_qubits = sum(genome.registers.values())
    input_bits = torch.zeros(n_qubits, dtype=torch.int64)

    # --- Forward pass (just to validate genome can run) ---
    try:
        _ = circuit(input_bits, torch_params)
    except Exception as e:
        print(f"Failed to run forward pass for genome {genome.genome_number}: {e}")
        state = None

    # make the fitnesses get progressively better
    genome.fitness = random.uniform(0.0, 1.0) - (count * 0.001)
    count += 1

    if genome.fitness < best_fitness:
        print(
            f"found new best genome number {genome.genome_number} with fitness: {genome.fitness}"
        )

        # plot each generated circuit to see how things are going
        try:
            fig, ax = qml.draw_mpl(circuit)(input_bits, torch_params)
            fig.savefig(f'plots/circuit_genome_{genome.fitness}.png', dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(
                f"Cannot plot adjoint operation: {e}"
            )
        best_fitness = genome.fitness


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

    args = parser.parse_args()

    max_population_size = args.max_population_size
    number_genomes = args.number_genomes

    allowed_gates = pennylane_gate_specifications

    if args.allowed_gates is not None:
        allowed_gates = allowed_gates.use_only(args.allowed_gates)

    exaqc = EXAQC(
        gate_specifications=allowed_gates,
        population=SteadyStatePopulation(max_population_size=50),
        registers={"a": 3, "b": 3},
        objective_function=random_objective_function,
    )

    exaqc.run_for(number_genomes)
