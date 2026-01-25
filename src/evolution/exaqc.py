import random

from collections.abc import Callable
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.gate_specifications import GateSpecifications
from src.circuits.registers import expand_registers

from src.evolution.crossover import (
    binary_crossover,
    exponential_crossover,
    n_ary_crossover,
)
from src.evolution.mutation import (
    add_gate,
    disable_gate,
    enable_gate,
    reorder_gate,
    qubit_swap,
)
from src.population.population_strategy import PopulationStrategy

class EXAQC:

    def __init__(
        self,
        gate_specifications: GateSpecifications,
        population: PopulationStrategy,
        objective_function: Callable[[CircuitGenome], None],
        input_qubits: list[tuple[str, int]] = None,
        input_registers: dict[str, int] = None,
        output_registers: dict[str, int] = None,
        output_qubits: list[tuple[str, int]] = None,
        target: str = "pennylane",
        loss: str = "fidelity",
        batch_size: int = None,
    ):
        """
        Creates an instance of Evolutionary Exploration of Augmenting Quantum Circuits given a
        particular population strategy, allowing the given gates (if specified).

        args:
            gate_specifications: is an object containing the allowed gates specifications for the search
                process, for either the pennylane or qiskit frameworks.
            population: is an instance of a subclass of the PopulationStrategy interface, utilized to get
                parents for mutation or crossover and insert children back into the population.
            objective_function: a method which takes a CircuitGenome, evaluates it and sets it's fitness
                value.
            input_registers: a dict of register names and sizes (the key is the qubit name, the value is its size). must
                be specified if input_qubits is not specified.
            input_qubits: a list of qubit tuples (name, register_index) which would be the expanded form of the
                input_registers. Must be specified if input_registers is not specified.
            output_registers: a dict of register names and sizes (the key is the qubit name, the value is its size). must
                be specified if output_qubits is not specified. If output_registers and output_qubits are None, they
                are set to the input registers/qubits.
            output_qubits: a list of qubit tuples (name, register_index) which would be the expanded form of the
                output_registers. Must be specified if output_registers is not specified. If output_registers
                and output_qubits are None, they are set to the input_registers/qubits.
            target: qiskit or pennylane
        """

        self.gate_specifications = gate_specifications
        self.population = population
        self.objective_function = objective_function
        self.target = target

        if input_registers is None and input_qubits is None:
            logger.critical("EXAQC requires *either* input_registers or input_qubits to be specified.")
            exit(1)

        if input_registers is not None and input_qubits is not None:
            logger.critical("EXAQC requires *either* input_registers or input_qubits to be specified, but not both.")
            exit(1)

        if output_registers is not None and output_qubits is not None:
            logger.critical("EXAQC requires *either* output_registers or output_qubits to be specified, but not both.")
            exit(1)

        self.input_qubits: list[tuple[str, int]] = input_qubits
        if self.input_qubits is None:
            self.input_qubits = expand_registers(input_registers)

        self.output_qubits: list[tuple[str, int]] = output_qubits
        if self.output_qubits is None:
            if output_registers is None:
                self.output_qubits = self.input_qubits.copy()
            else:
                self.output_qubits = expand_registers(output_registers)

        logger.info("Starting EXAQC with the following allowed gates:")
        for gate in sorted(
            self.gate_specifications.values(), key=lambda g: g.method_name
        ):
            logger.info(f"\t{gate}")

        # used to track how many genomes have been generated and set genome numbers
        self.genome_number = 0

        initial_genome = CircuitGenome(
            genome_number=self.next_genome_number(),
            target=self.target,
            input_qubits=self.input_qubits.copy(),
            output_qubits=self.output_qubits.copy(),
        )

        # generate the initial population
        for i in range(population.max_population_size):
            child = self.mutate(initial_genome)
            self.objective_function(
                child, target=self.target, loss=loss, batch_size=batch_size
            )

            self.population.insert_genome(child)

    def next_genome_number(self) -> int:
        """
        Increments and returns the next genome number.

        Returns:
            A new unique genome number for a new genome.
        """

        self.genome_number += 1
        return self.genome_number

    def mutate(self, parent: CircuitGenome) -> CircuitGenome:
        """
        Takes a given parent genome, makes a copy of it (with a new genome number) and
        then applies a random mutation to it.

        Args:
            parent: is the genome to mutate

        Returns:
            A mutated copy of the parent genome as a child.
        """

        child = parent.copy(genome_number=self.next_genome_number())

        # mutation_options = ["add_gate", "disable_gate", "enable_gate", "reorder_gate"]
        mutation_options = (
            ["add_gate"] * 5  # 50%
            + ["reorder_gate"] * 2  # 20%
            + ["qubit_swap"] * 2  # 20%
            + ["enable_gate"]  # 5%
            + ["disable_gate"]  # 5%
        )

        modified = False

        # only use the gates with which do not still require some validation from us to
        # ensure compatability
        allowed_gate_specifications = [
            v for v in self.gate_specifications.values() if v.needs_validation is False
        ]

        logger.info("starting mutation process")
        while not modified:
            # keep trying to mutate until successful

            mutation = random.choice(mutation_options)

            match mutation:
                case "add_gate":
                    gate_specification = random.choice(allowed_gate_specifications)
                    logger.info(f"\tattempting {mutation} with {gate_specification}")
                    modified = add_gate(gate_specification, child)

                case "disable_gate":
                    logger.info(f"\tattempting to mutate with {mutation}")
                    modified = disable_gate(child)

                case "enable_gate":
                    logger.info(f"\tattempting to mutate with {mutation}")
                    modified = enable_gate(child)

                case "reorder_gate":
                    logger.info(f"\tattempting to mutate with {mutation}")
                    modified = reorder_gate(child)

                case "qubit_swap":
                    logger.info(f"\tattempting to mutate with {mutation}")
                    modified = qubit_swap(child)

        return child

    def run_for(
        self,
        number_genomes: int,
        binary_crossover_rate: float = 0.10,
        n_ary_crossover_rate: float = 0.10,
        exponential_crossover_rate: float = 0.10,
        n_ary_parents: int = 4,
    ):
        """
        Runs EXAQC until it has generated and evaluated the given number of genomes.

        Args:
            number_genomes: how many genomes to generate with EXAQC
            binary_crossover_rate: what percentage of time to do binary crossover after
                the population has been initialized.
            n_ary_crossover_rate: what percentage of the time to do n-ary crossover
                after the population has been initialized.
            n_ary_parents: how many parents to use for n-ary crossover
        """

        # calculate this so we can use a single if set of statements based on
        # the random r value
        n_ary_cutoff = binary_crossover_rate + n_ary_crossover_rate
        exponential_cutoff = n_ary_cutoff + exponential_crossover_rate

        while self.genome_number < number_genomes:
            child = None

            r = random.uniform(0, 1.0)

            if (
                self.genome_number > self.population.max_population_size
                and r < binary_crossover_rate
            ):
                parents = self.population.get_parents(2)
                child = CircuitGenome(
                    genome_number=self.next_genome_number(),
                    target=self.target,
                    input_qubits=self.input_qubits.copy(),
                    output_qubits=self.output_qubits.copy(),
                )

                if binary_crossover(child, parents[0], parents[1]):
                    self.objective_function(child)
                else:
                    # two parents could not be used in exponential crossover, so try
                    # to generate a new child
                    continue

            elif (
                self.genome_number > self.population.max_population_size
                and r < n_ary_cutoff
            ):
                parents = self.population.get_parents(n_ary_parents)
                child = CircuitGenome(
                    genome_number=self.next_genome_number(),
                    target=self.target,
                    input_qubits=self.input_qubits.copy(),
                    output_qubits=self.output_qubits.copy(),
                )

                n_ary_crossover(child, parents)
                self.objective_function(child)

            elif (
                self.genome_number > self.population.max_population_size
                and r < exponential_cutoff
            ):
                parents = self.population.get_parents(2)
                child = CircuitGenome(
                    genome_number=self.next_genome_number(),
                    target=self.target,
                    input_qubits=self.input_qubits.copy(),
                    output_qubits=self.output_qubits.copy(),
                )

                exponential_crossover(child, parents[0], parents[1])
                self.objective_function(child)

            else:
                parent = self.population.get_parent()
                child = self.mutate(parent)
                self.objective_function(child)

            self.population.insert_genome(child)
