"""A single steady-state island for the island-model search.

An :class:`Island` is one fitness-sorted sub-population in the island model
(see :mod:`src.evolution.steady_state_islands`). It holds its own genomes,
tracks the neighboring islands it may exchange genomes with for inter-island
crossover, and supports repopulation during extinction events.
"""

from __future__ import annotations

import bisect
import random
from functools import cmp_to_key
from typing import Callable

from loguru import logger

from src.circuits.circuit import CircuitGenome


class Island:

    def __init__(
        self,
        id: int,
        max_size: int,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
    ):
        """
        Creates an island which holds a single (sorted) set of genomes.

        Args:
            id: is the id for the island
            max_size: is the maximum number of genomes for the island.
            compare: a compare function used for sorting genomes. this should return 0 if both
                genomes should be ranked the same, a negative value if the first genome should
                come before the second genome, and a positive number otherwise
        """

        self.id = id
        self.max_size = max_size
        self.insertions = 0
        self.compare = compare

        self.population: list[CircuitGenome] = []
        self.status = "initializing"
        self.repopulation_genome_number = 0

        # track which other islands this island can perform inter-island crossover
        # with
        self.neighbors: list[Island] = []

    def is_initializing(self) -> bool:
        """
        Returns:
            True if the island is still initializing.
        """

        return self.status == "initializing"

    def neighbor_population_size(self) -> int:
        """
        Calculate how many genomes are in the populations of each
        neighboring island.  Used to determine if there are enough
        neighboring genomes to perform inter-island crossover.

        Returns:
            The sum of the number of genomes in each
            neighboring island's population.
        """

        size = 0
        for neighbor in self.neighbors:
            size += len(neighbor.population)

        return size

    def best_neighbor(self) -> Island | None:
        """
        Finds the neighboring island which has the genome
        with the best fitness.

        Returns:
            The neighboring island which has the genome with the best fitness,
            or None if there are no neighbors or all neighbor populations are
            empty (so there is no genome to compare).
        """

        best = None

        for neighbor in self.neighbors:
            if len(neighbor.population) > 0:
                if (
                    best is None
                    or self.compare(neighbor.population[0], best.population[0]) < 0
                ):
                    # this island has a better best genome
                    best = neighbor

        return best

    def repopulate(self, repopulation_genome_number: int) -> None:
        """
        Removes all genomes from this island and sets its status
        to repopulating. Also sets the repopulation genome number
        so any genomes generated from before repopoulation are discarded
        unless they are a new global best.

        Args:
            repopulation_genome_number: genomes with a genome number below this
                are discarded on insertion (unless they are the global best),
                so pre-extinction genomes do not refill the island.

        Returns:
            None. Clears ``population`` and sets ``status`` to ``"repopulating"``.
        """

        self.status = "repopulating"
        self.repopulation_genome_number = repopulation_genome_number
        self.population = []

    def get_parent(self, **kwargs) -> CircuitGenome | None:
        """
        Used to get a single parent to be used in mutation or
        other operations to generate children.

        Args:
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A single CircuitGenome chosen at random from the population, or
            None if the population is empty.
        """

        if len(self.population) > 0:
            return random.choice(self.population)
        else:
            return None

    def get_parents(self, n_parents: int = 2, **kwargs) -> list[CircuitGenome] | None:
        """
        Used to get two or more parents to be used in crossover or
        other operations to generate children.

        Args:
            n_parents: specifies how many parents to return by the method.
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A list of unique (non-duplicate) CircuitGenomes, sorted best-first.
            If the size of the population is less than n_parents, it returns
            None.
        """
        if len(self.population) >= n_parents:
            # sort the parents so the most fit is the first parent
            parents = random.sample(self.population, n_parents)
            parents.sort(key=cmp_to_key(self.compare))
            return parents
        else:
            return None

    def insert_genome(self, genome: CircuitGenome, **kwargs) -> None:
        """
        Inserts a genome into this island's fitness-sorted population.

        Duplicate genomes (same enabled gates) are resolved by keeping the more
        fit one, the population is kept sorted, and once it exceeds ``max_size``
        the least fit genome is dropped. The genome's ``insert_type`` metadata is
        set to one of ``"global_best"``, ``"local_best"``, ``"inserted"`` or
        ``"discarded"`` to record what happened.

        Args:
            genome: is the genome to insert into the population.
            **kwargs: is used to pass additional options to the method for
                inserting the genome, such as an island or species it came from.

        Returns:
            None. Mutates the island's ``population`` and the genome's
            ``insert_type`` metadata in place.
        """

        if (
            "insert_type" not in genome.metadata
            or genome.metadata["insert_type"] != "global_best"
        ):
            # temporarily assign the insert type, if it doesn't yet exist as global best.
            # set it to inserted which we can change later if it it a local best or gets discarded
            genome.metadata["insert_type"] = "inserted"

        if (
            genome.genome_number < self.repopulation_genome_number
            and genome.metadata["insert_type"] != "global_best"
        ):
            # discard genomes that were generated from before the island was repopulated unless they
            # were a new global best
            logger.info(
                f"discarding genome with number {genome.genome_number} as it was less than "
                f"the repopulation genome number: {self.repopulation_genome_number} and was "
                f"not global best, metadata: {genome.metadata}"
            )
            genome.metadata["insert_type"] = "discarded"
            return

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
                    genome.metadata["insert_type"] = "discarded"
                    return

        bisect.insort(
            self.population,
            genome,
            key=cmp_to_key(self.compare),
        )

        self.insertions += 1

        if genome == self.population[0]:
            # this was a new best genome for the island
            if genome.metadata["insert_type"] != "global_best":
                # if the genome was inserted at the front of the population it
                # is a new local best unless it was already the global best
                genome.metadata["insert_type"] = "local_best"

            # this was a new global best genome
            logger.success(
                f"[local insertion {self.insertions}] island {self.id} found new LOCAL best "
                f"genome with fitness: {genome.fitness}"
            )

        if len(self.population) >= self.max_size:
            self.status = "full"

        if len(self.population) > self.max_size:
            # remove the last genome from the population
            if genome == self.population[-1]:
                # if the genome was inserted at the bottom of the population
                # and we're going to remove it, set it to discarded
                genome.metadata["insert_type"] = "discarded"

            del self.population[-1]
