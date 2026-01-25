import argparse
import matplotlib.pyplot as plt
import random
import sys

from loguru import logger

from src.evolution.exaqc import EXAQC
from src.circuits.circuit import CircuitGenome
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.population.steady_state_population import SteadyStatePopulation

best_genome = None
count = 0


def random_objective_function(genome: CircuitGenome, target: str = "qiskit", loss: str = "fidelity_loss", batch_size: int = 0):
    """
    Computes a random fitness value for a given circuit. This will
    assign the genome's fitness attribute to the new fitness value. It
    also shows an example for modifying the genome's parameter values for
    parameterized gates.
    """
    global best_genome, count

    for gate in genome.gates:
        for parameter, value in gate.parameters.items():
            # we can access the parameter values for each gate, and modify
            # them, e.g., if they are being trained
            gate.parameters[parameter] = value + random.uniform(-0.5, 0.5)

    # just a test to make sure we can generate this circuit and not
    # break things
    circuit = genome.generate_qiskit_circuit()

    # make the fitnesses get progressively better
    genome.fitness = {
        "fidelity_loss": random.uniform(0.0, 1.0) - (count * 0.001),
    }
    count += 1

    if best_genome is None or genome.dominates(best_genome, loss=loss):
        logger.info(f"best genome is: {best_genome}")
        if best_genome is not None:
            logger.info(f"best genome fitness: {best_genome.fitness}")

        logger.info(
            f"found new best genome number {genome.genome_number} with fitness: {genome.fitness}"
        )

        # plot each generated circuit to see how things are going
        circuit.draw(output="mpl")
        plt.show()

        best_genome = genome


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

    allowed_gates = qiskit_gate_specifications

    if args.allowed_gates is not None:
        allowed_gates = allowed_gates.use_only(args.allowed_gates)

    exaqc = EXAQC(
        gate_specifications=allowed_gates,
        population=SteadyStatePopulation(max_population_size=50, loss="fidelity_loss"),
        input_registers={"a": 3, "b": 3},
        objective_function=random_objective_function,
        target="qiskit",
        loss="fidelity_loss",
    )

    exaqc.run_for(number_genomes)
