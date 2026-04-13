import bisect
import random

from functools import cmp_to_key
from typing import Callable, Optional

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.population_strategy import PopulationStrategy
from src.utils.profiler import EXAQCProfiler


class SteadyStatePopulation(PopulationStrategy):

    def __init__(
        self,
        max_population_size: int,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
        out_dir: str = "artifacts",
        profiler: Optional[EXAQCProfiler] = None,
    ):
        """
        Creates a steady state population with the specified max population size.  The population
        will be sorted in order by genome fitness. The get parent methods can be called at any
        time to generate random parent selection.  Genomes will be inserted if the population size
        is below the max population size, or if they are better than the least fit genome in the
        population.  If adding a genome would cause the population size to be greater than the
        max population size, the least fit genome will be removed to keep it under the max size.

        Args:
            max_population_size: is the maximum number of genomes that the population will hold.
            compare: a compare function used for sorting genomes. this should return 0 if both
                genomes should be ranked the same, a negative value if the first genome should
                come before the second genome, and a positive number otherwise
            out_dir: is the directory to write out the logs and best found genomes
            profiler: A profiler class for recording execution steps to plot later
        """

        self.max_population_size = max_population_size
        self.compare = compare
        self.out_dir = out_dir

        self.insertions = 0

        # used to store the population, should be kept in sorted order.
        self.population: list[CircuitGenome] = []
        self.accuracy_best_genome = None

        self.profiler = profiler
        if self.profiler is None:
            self.profiler = EXAQCProfiler(
                out_dir=out_dir,
            )

    def is_initializing(self) -> bool:
        """
        Returns:
            True if the population is at max size
        """

        return len(self.population) < self.max_population_size

    def get_best_genome(self) -> CircuitGenome:
        """
        Returns:
            The best genome in the population if it exists, None otherwise.
            Will only return none if no genomes have been inserted yet (i.e.,
            the very beginning of the search).
        """

        if len(self.population) > 0:
            return self.population[0]
        else:
            return None

    def get_parent(self, **kwargs) -> tuple[CircuitGenome, dict[str, any]]:
        """
        Used to get two or more parents to be used in mutation or
        other operations to generate children.

        Args:
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A single CircuitGenome from the population. If the population is empty
            it will return None. The second return value (if a genome is returned)
            is the metadata for the generated child, which for this strategy is
            just empty.
        """

        if len(self.population) > 0:
            metadata = {}
            return random.choice(self.population), metadata
        else:
            return None, None

    def get_parents(
        self, n_parents: int = 2, **kwargs
    ) -> tuple[list[CircuitGenome], dict[str, any]]:
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
            is less than n_parents, it will return None. The second return value (if a
            genome is returned) is the metadata for the generated child, which for this
            strategy is just empy.
        """
        if len(self.population) >= n_parents:
            # sort the parents so the most fit is the first parent
            parents = random.sample(self.population, n_parents)
            parents.sort(key=cmp_to_key(self.compare))
            metadata = {}
            return parents, metadata
        else:
            return None, None

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

        # don't add duplicate genomes to the population
        # if gate innovation numbers are the same, keep the genome with better fitness
        for i in range(len(self.population)):
            match_genome = self.population[i]
            if match_genome.has_same_gates(genome):
                # two genomes had the same enabled gates, keep the one with better fitness

                if self.compare(match_genome, genome) > 0:
                    # the new genome has a better fitness, so remove the old genome
                    # and then the below bisect.insort will add it
                    logger.info(
                        f"removing genome from population because fitness: {match_genome.fitness} is"
                        f"worse than the new genome fitness: {genome.fitness} where both have"
                        "the same enabled gates."
                    )
                    logger.info(
                        f"population genome gates: {match_genome.get_gate_innovations()}"
                    )
                    logger.info(
                        f"new genome gates:        {genome.get_gate_innovations()}"
                    )
                    del self.population[i]
                    break
                else:
                    # discard the new genome
                    self.insertions += 1
                    return

        bisect.insort(
            self.population,
            genome,
            key=cmp_to_key(self.compare),
        )
        genome.metadata["insert_type"] = "inserted"

        self.insertions += 1

        if self.profiler is not None:
            self.profiler.record(step=self.insertions, population=self.population)

        if self.accuracy_best_genome is None or (
            "test_acc" in genome.fitness
            and self.accuracy_best_genome.fitness["test_acc"]
            < genome.fitness["test_acc"]
        ) or (
            "eval_return_mean" in genome.fitness
            and self.accuracy_best_genome.fitness["eval_return_mean"]
            < genome.fitness["eval_return_mean"]
        ):
            self.accuracy_best_genome = genome

            # this was a new genome with a best accuracy
            logger.success(
                f"[global insertion {self.insertions}] Population found new ACCURACY best genome "
                f"with fitness: {genome.fitness}"
            )
            if self.out_dir is not None:
                genome.save_circuit(insert_type="best_accuracy", out_dir=self.out_dir)

            if genome.fitness.get("test_acc", -1.0) == 100.0:
                # found a perfect solution, can quit
                exit(1)

        if genome.genome_number == self.population[0].genome_number:
            # this was a new global best genome
            logger.success(
                f"🎯 New best genome {genome.genome_number} "
                f"[insertion {self.insertions}] Population found new GLOBAL best genome with fitness: {genome.fitness}"
            )
            genome.metadata["insert_type"] = "global_best"
            if self.out_dir is not None:
                genome.save_circuit(insert_type="best_fitness", out_dir=self.out_dir)

        if len(self.population) > self.max_population_size:
            # remove the last genome from the population
            if genome == self.population[-1]:
                # the genome that was inserted is going to be discarded, so flag its metadata
                genome.metadata["insert_type"] = "discarded"
            del self.population[-1]

        if self.out_dir is not None:
            genome.save_circuit(
                insert_type="genome", out_dir=self.out_dir + "/all_genomes/"
            )
            self.profiler.plot_single_run()
