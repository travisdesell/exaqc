import random

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
    add_gate_with_selection,
    disable_gate,
    enable_gate,
    reorder_gate,
    qubit_swap,
)
from src.evolution.objective import Objective
from src.evolution.population_strategy import PopulationStrategy


class EXAQC:

    def __init__(
        self,
        gate_specifications: GateSpecifications,
        population: PopulationStrategy,
        objective: Objective,
        hyperparameters: dict[str, any],
        input_qubits: list[tuple[str, int]] = None,
        input_registers: dict[str, int] = None,
        output_registers: dict[str, int] = None,
        output_qubits: list[tuple[str, int]] = None,
        target: str = "pennylane",
    ):
        """
        Creates an instance of Evolutionary Exploration of Augmenting Quantum Circuits given a
        particular population strategy, allowing the given gates (if specified).

        args:
            gate_specifications: is an object containing the allowed gates specifications for the search
                process, for either the pennylane or qiskit frameworks.
            population: is an instance of a subclass of the PopulationStrategy interface, utilized to get
                parents for mutation or crossover and insert children back into the population.
            objective: a method which takes a CircuitGenome, evaluates it and sets it's fitness
                value.
            hyperparameters: a dict specifying which hyperparameters to use in the training process, and if
                this is an additional search space to search over.
            input_registers: a dict of register names and sizes (the key is the qubit name, the value is its size). must
                be specified if input_qubits is not specified.
            input_qubits: a list of qubit tuples (name, register_index) which would be the expanded form of the
                input_registers. Must be specified if input_registers is not specified.
            output_registers: a dict of register names and sizes (the key is the qubit name, the value is its
                size). must be specified if output_qubits is not specified. If output_registers and output_qubits
                are None, they are set to the input registers/qubits.
            output_qubits: a list of qubit tuples (name, register_index) which would be the expanded form of the
                output_registers. Must be specified if output_registers is not specified. If output_registers
                and output_qubits are None, they are set to the input_registers/qubits.
            target: qiskit or pennylane
        """

        self.gate_specifications = gate_specifications
        self.population = population
        self.objective = objective
        self.hyperparameters = hyperparameters
        self.target = target
        self.inserted_genomes = 0

        if input_registers is None and input_qubits is None:
            logger.critical(
                "EXAQC requires *either* input_registers or input_qubits to be specified."
            )
            exit(1)

        if input_registers is not None and input_qubits is not None:
            logger.critical(
                "EXAQC requires *either* input_registers or input_qubits to be specified, but not both."
            )
            exit(1)

        if output_registers is not None and output_qubits is not None:
            logger.critical(
                "EXAQC requires *either* output_registers or output_qubits to be specified, but not both."
            )
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

        # create a starting (empty) genome to initially generate genomes from
        self.initial_genome = CircuitGenome(
            genome_number=self.next_genome_number(),
            target=self.target,
            input_qubits=self.input_qubits.copy(),
            output_qubits=self.output_qubits.copy(),
        )

        self.saved_epochs = 5
        self.initial_genome.hyperparameters = self.get_hyperparameters()

    def get_hyperparameters(self):
        """
        Return:
            hyperparameters for a newly created child
        """

        if self.genome_number > 0 and self.genome_number % 100 == 0:
            # increase epochs every 100 genomes
            self.saved_epochs += 1

        # TODO: make an evolutionary strategy for handling hyperparameter options
        hyperparameters = self.hyperparameters.copy()

        """
        hyperparameters["learning_rate"] = random.choice(
            [0.001, 0.0005, 0.0001, 0.00005]
        )
        hyperparameters["epochs"] = random.choice([5, 10, 15, 20, 25, 30, 35, 40])
        """
        hyperparameters["learning_rate"] = 0.0005
        # hyperparameters["epochs"] = random.choice([5, 10])
        hyperparameters["epochs"] = self.saved_epochs

        return hyperparameters

    def next_genome_number(self) -> int:
        """
        Increments and returns the next genome number.

        Returns:
            A new unique genome number for a new genome.
        """

        self.genome_number += 1
        return self.genome_number

    def mutate(self, parent: CircuitGenome, metadata: dict[str, any]) -> CircuitGenome:
        """
        Takes a given parent genome, makes a copy of it (with a new genome number) and
        then applies a random mutation to it.

        Args:
            parent: is the genome to mutate

        Returns:
            A mutated copy of the parent genome as a child.
        """

        child = parent.copy(genome_number=None)
        # sets the childs metadata to new metadata (shouldn't directly copy from
        # the parent).
        child.metadata = metadata

        # mutation_options = ["add_gate", "disable_gate", "enable_gate", "reorder_gate"]
        mutation_options = (
            ["add_gate"] * 11  # 65%
            + ["reorder_gate"] * 2  # 10%
            + ["qubit_swap"] * 2  # 10%
            + ["enable_gate"]  # 5%
            + ["disable_gate"] * 2  # 10%
            + ["clone"] * 2  # 10%
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

            logger.info(f"\tattempting to mutate with {mutation}")
            match mutation:
                case "add_gate":
                    modified = add_gate_with_selection(
                        allowed_gate_specifications, child
                    )
                    # gate_specification = random.choice(allowed_gate_specifications)
                    # logger.info(f"\tattempting {mutation} with {gate_specification}")
                    # modified = add_gate(gate_specification, child)

                case "clone":
                    # don't need to do anything, just keep the copied child, but set
                    # modified to true so we count this as a valid mutation
                    modified = True

                case "disable_gate":
                    modified = disable_gate(child)

                case "enable_gate":
                    modified = enable_gate(child)

                case "reorder_gate":
                    modified = reorder_gate(child)

                case "qubit_swap":
                    modified = qubit_swap(child)

        return child

    def generate_genome(
        self,
        binary_crossover_rate: float = 0.10,
        n_ary_crossover_rate: float = 0.10,
        exponential_crossover_rate: float = 0.10,
        n_ary_parents: int = 5,
    ) -> CircuitGenome:
        """
        Generates a single genome for EXAQC.

        Args:
            binary_crossover_rate: what percentage of time to do binary crossover after
                the population has been initialized.
            n_ary_crossover_rate: what percentage of the time to do n-ary crossover
                after the population has been initialized.
            n_ary_parents: how many parents to use for n-ary crossover
        Returns:
            A new child to evaluate for EXAQC.
        """

        if self.population.is_initializing():
            # still need to populate the initial population
            # mutation_count = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            mutation_count = random.choice([0, 1, 2])
            # mutation_count = random.choice([0, 1, 2, 3, 4])
            valid = False

            logger.info(f"generating a child via {mutation_count + 1} mutations.")
            while not valid:
                # keep trying to create a child from the initial
                # genome until we get a valid one, then send it out
                # in this initialization phase there is no specific
                # metadata for children
                metadata = {}
                child = self.mutate(self.initial_genome, metadata)

                for i in range(mutation_count):
                    child = self.mutate(child, metadata)
                valid = child.is_valid()

            child.genome_number = self.next_genome_number()
            child.hyperparameters = self.get_hyperparameters()
            child.metadata = {}

            return child

        else:
            # generate from the population as usual

            # calculate this so we can use a single if set of statements based on
            # the random r value
            n_ary_cutoff = binary_crossover_rate + n_ary_crossover_rate
            exponential_cutoff = n_ary_cutoff + exponential_crossover_rate

            child = None
            while child is None:
                r = random.uniform(0, 1.0)

                if r < binary_crossover_rate:
                    parents, metadata = self.population.get_parents(2)

                    if parents is None:
                        continue

                    child = CircuitGenome(
                        genome_number=None,
                        target=self.target,
                        input_qubits=self.input_qubits.copy(),
                        output_qubits=self.output_qubits.copy(),
                        metadata=metadata,
                    )

                    if not binary_crossover(child, parents[0], parents[1]):
                        # two parents could not be used in exponential crossover, so try
                        # to generate a new child
                        continue

                elif r < n_ary_cutoff:
                    parents, metadata = self.population.get_parents(n_ary_parents)

                    if parents is None:
                        continue

                    child = CircuitGenome(
                        genome_number=None,
                        target=self.target,
                        input_qubits=self.input_qubits.copy(),
                        output_qubits=self.output_qubits.copy(),
                        metadata=metadata,
                    )

                    n_ary_crossover(child, parents)

                elif r < exponential_cutoff:
                    parents, metadata = self.population.get_parents(2)

                    if parents is None:
                        continue

                    child = CircuitGenome(
                        genome_number=None,
                        target=self.target,
                        input_qubits=self.input_qubits.copy(),
                        output_qubits=self.output_qubits.copy(),
                        metadata=metadata,
                    )

                    exponential_crossover(child, parents[0], parents[1])

                else:
                    parent, metadata = self.population.get_parent()

                    # mutation_count = random.choice([0, 1, 2, 3, 4])
                    mutation_count = random.choice([1, 2])
                    valid = False

                    logger.info(
                        f"generating a child via {mutation_count + 1} mutations."
                    )
                    while not valid:
                        # keep trying to create a child from the initial
                        # genome until we get a valid one, then send it out
                        child = self.mutate(parent, metadata)

                        for i in range(mutation_count):
                            child = self.mutate(child, metadata)
                        valid = child.is_valid()

                if not child.is_valid():
                    logger.warning(
                        "child was invalid (inputs did not connect to outputs), trying again."
                    )
                    child = None

            # successfully generated a child
            child.genome_number = self.next_genome_number()
            child.hyperparameters = self.get_hyperparameters()
            return child

    def insert_genome(self, genome: CircuitGenome):
        """
        Trys to insert an evaluated genome into the population strategy.

        Args:
            genome: is the evaluated genome to insert
        """
        self.population.insert_genome(genome, current_genome_number=self.genome_number)
        self.inserted_genomes += 1

    def run_for(
        self,
        number_genomes: int,
    ):
        """
        Runs EXAQC until it has generated and evaluated the given number of genomes.

        Args:
            number_genomes: how many genomes to generate with EXAQC
        """

        while self.genome_number < number_genomes:
            child = self.generate_genome()
            self.objective(child)
            self.population.insert_genome(child)
