import bisect
import random

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.population.population_strategy import PopulationStrategy


class SteadyStatePopulation(PopulationStrategy):

    def __init__(self, max_population_size: int, loss: str = None):
        """
        Creates a steady state population with the specified max population size.  The population
        will be sorted in order by genome fitness. The get parent methods can be called at any
        time to generate random parent selection.  Genomes will be inserted if the population size
        is below the max population size, or if they are better than the least fit genome in the
        population.  If adding a genome would cause the population size to be greater than the
        max population size, the least fit genome will be removed to keep it under the max size.

        Args:
            max_population_size: is the maximum number of genomes that the population will hold.
        """

        self.max_population_size = max_population_size
        self.loss = loss

        self.insertions = 0

        # used to store the population, should be kept in sorted order.
        self.population: list[CircuitGenome] = []

    def get_parent(self, **kwargs) -> CircuitGenome:
        """
        Used to get two or more parents to be used in mutation or
        other operations to generate children.

        Args:
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A single CircuitGenome from the population. If the population is empty
            it will return None.
        """

        if len(self.population) > 0:
            return random.choice(self.population)
        else:
            return None

    def get_parents(self, n_parents: int = 2, **kwargs) -> list[CircuitGenome]:
        """
        Used to get two or more parents to be used in crossover or
        other operations to generate children.

        Args:
            n_parents: specifies how many parents to return by the method.
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A list of unique (non-duplicate) CircuitGenomes. If the size of the population
            is less than n_parents, it will return None.
        """
        if len(self.population) >= n_parents:
            return random.sample(self.population, n_parents)
        else:
            return None

    def insert_genome(self, genome: CircuitGenome, **kwargs) -> bool:
        """
        Inserts a genome back into the population.

        Args:
            genome: is the genome to insert into the population.
            **kwargs: is used to pass additional options to the method for
                inserting the genome, such as an island or species it came from.

        Returns:
            True if it was inserted into the population, False otherwise.
        """

        # TODO: don't add duplicate genomes to the population
        # options:
        # 1. if gate innovation numbers are the same, keep the genome with better fitness
        # 2. if gate innovation numbers are the same but fitness different, keep both

        bisect.insort(
            self.population, genome, key=lambda genome: genome.fitness["loss"]
        )

        self.insertions += 1

        if genome.genome_number == self.population[0].genome_number:
            # this was a new global best genome
            logger.info(
                f"[insertion {self.insertions}] Population found new GLOBAL best genome with fitness: {genome.fitness}"
            )

        if len(self.population) > self.max_population_size:
            # remove the last genome from the population
            del self.population[-1]
