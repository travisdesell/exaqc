import random
import json

import bisect
from functools import cmp_to_key
from typing import Callable, Optional

from loguru import logger
import os
import torch
import pennylane as qml
import matplotlib.pyplot as plt

from src.circuits.circuit import CircuitGenome
from src.evolution.population_strategy import PopulationStrategy
from src.objectives.genome_objectives import (
    genome_to_torch_params,
)
from src.utils.profiler import EXAQCProfiler


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

    def is_initializing(self) -> bool:
        """
        Returns:
            True if the island is still initializing.
        """

        return self.status == "initializing"

    def repopulate(self, repopulation_genome_number: int):
        """
        Removes all genomes from this island and sets its status
        to repopulating. Also sets the repopulation genome number
        so any genomes generated from before repopoulation are discarded
        unless they are a new global best.
        """

        self.status = "repopulating"
        self.repopulation_genome_number = repopulation_genome_number
        self.population = []

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
            # sort the parents so the most fit is the first parent
            parents = random.sample(self.population, n_parents)
            parents.sort(key=cmp_to_key(self.compare))
            return parents
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

        if genome.genome_number < self.repopulation_genome_number and not ('global_best' in genome.metadata and genome.metadata['global_best'] is True):
            # discard genomes that were generated from before the island was repopulated unless they
            # were a new global best
            logger.info(f"discarding genome with number {genome.genome_number} as it was less than the repopulation genome number: {self.repopulation_genome_number} and was not global best, metadata: {genome.metadata}")
            return

        # don't add duplicate genomes to the population
        # if gate innovation numbers are the same, keep the genome with better fitness
        for i in range(len(self.population)):
            match_genome = self.population[i]
            if match_genome.has_same_gates(genome):
                #two genomes had the same enabled gates, keep the one with better fitness

                if self.compare(match_genome, genome) > 0:
                    # the new genome has a better fitness, so remove the old genome
                    # and then the below bisect.insort will add it
                    logger.info(
                        f"removing genome from population because fitness: {match_genome.fitness} is"
                        f"worse than the new genome fitness: {genome.fitness} where both have"
                        "the same enabled gates."
                    )
                    logger.info(f"population genome gates: {match_genome.get_gate_innovations()}")
                    logger.info(f"new genome gates:        {genome.get_gate_innovations()}")
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


        self.insertions += 1

        if genome == self.population[0]:
            # this was a new best genome for the island
            self.last_new_best = self.insertions

            # this was a new global best genome
            logger.success(
                f"[local insertion {self.insertions}] island {self.id} found new LOCAL best "
                f"genome with fitness: {genome.fitness}"
            )

        if len(self.population) >= self.max_size:
            self.status = "full"

        if len(self.population) > self.max_size:
            # remove the last genome from the population
            del self.population[-1]

def island_compare(island1: Island, island2: Island) -> int:
        """
        Used to sort genomes by fitness, even if there are multiple objectives, for population
        management and crossover methods.

        Args:
            island1: will compare the best (genome in slot 0 of the island) of this island to the other island
            island2: the second genome to comapre to

        Returns: 0 if the two best genomes in the islands have equivalent fitnesses, a negative value if 
            island1.population[0] should be sorted before island2.population[0], and a positive value if 
            island2.population[0] should be sorted before island1.population[0] 
        """

        return island1.compare(island1.population[0], island2.population[0])


class SteadyStateIslands(PopulationStrategy):

    def __init__(
        self,
        n_islands: int,
        max_island_size: int,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
        intra_island_crossover_rate: float = 0.5,
        genomes_before_extinction: int = 50,
        genomes_for_next_extinction: int = 200,
        islands_to_extinct: int = 2,
        out_dir: str = None,
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
            genomes_before_extinction: is how many genomes are inserted into islands before an
                extinction event happens, which clears out the worst islands and repopulates them
            island_to_extinct: is how many islands to clear out in an extinction event
            compare: a compare function used for sorting genomes. this should return 0 if both
                genomes should be ranked the same, a negative value if the first genome should
                come before the second genome, and a positive number otherwise
            out_dir: is the directory to write out the best found genomes and log files, if not
                specified log files will not be written.
        """

        self.n_islands = n_islands
        self.max_island_size = max_island_size
        self.compare = compare
        self.intra_island_crossover_rate = intra_island_crossover_rate
        self.genomes_before_extinction = genomes_before_extinction
        self.genomes_for_next_extinction = genomes_for_next_extinction
        self.islands_to_extinct = islands_to_extinct
        self.out_dir = out_dir

        self.insertions = 0

        # used to store the island populations, should be kept in sorted order.
        self.islands: list[Island] = [
            Island(max_size=max_island_size, id=i, compare=compare)
            for i in range(self.n_islands)
        ]
        self.current_island = 0

        self.global_best_genome = None

        self.profiler = profiler
        if self.profiler is None and out_dir:
            self.profiler = EXAQCProfiler(
                out_dir=self.out_dir,
                topk=5,
            )

    def is_initializing(self) -> bool:
        """
        Returns:
            True if all islands are not still initializing.
        """

        for island in self.islands:
            if island.is_initializing():
                return True

        return False

    def increment_current_island(self):
        """
        Increments the current island in a round robin fashion.
        """

        self.current_island += 1
        if self.current_island >= len(self.islands):
            self.current_island = 0

    def get_best_genome(self) -> CircuitGenome:
        """
        Returns:
            The best genome across all islands, if it exists. None otherwise.  It would
            only return none if no genomes have been inserted yet (i.e., the very beginning
            of the search).
        """

        return self.global_best_genome

    def get_parent(self, **kwargs) -> tuple[CircuitGenome, dict[str, any]]:
        """
        Used to get a parent to be used in mutation or other operations to generate
        children. This will be generated from an island in a round robin fashion.

        Steps:
        1. get target island
        2. if target island full, get from its population
	    3. if target island repopulating - use random from best island if the
            best island has any genomes, otherwise use global best genome. these
            should usually be the same but sometimes a genome comes in on a 
            repopulating island which is a new best but happened from before the
            repopulation trigger.
	    4. if initializing - shouldnt ever happen , stop with error

        Args:
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A single CircuitGenome from the population and a dictionary of its
            metadata. If the population is empty it will return None.
        """

        target_island = self.islands[self.current_island]
        self.increment_current_island()

        metadata = {"target_island_id": target_island.id}

        if target_island.status == "full":
            return random.choice(target_island.population), metadata

        if target_island.status == "repopulating":
            if len(self.best_island.population) > 0:
                # get parent from best island
                return random.choice(self.best_island.population), metadata
            else:
                # in case the global best island ended up being repopulated
                return self.global_best_genome, metadata

        else:
            logger.error(f"tried to get a parent from an initializing island. This should never happen.")
            exit(1)

    def get_parents(
        self, n_parents: int = 2, **kwargs
    ) -> tuple[list[CircuitGenome], dict[str, any]]:
        """
        Used to get two or more parents to be used in crossover or
        other operations to generate children.

        Steps:
        1. Get target island in round robin fashion. 
        2. if target island initializing - can’t do this yet, fail with error
        3. if target island repopulating - get global best and N-1 from other islands
        4. else - get 1 genome from this island, N-1 from other islands

        Args:
            n_parents: specifies how many parents to return by the method.
            **kwargs: is used to pass additional options to the method to get
                a parent, e.g., specifying if it is for inter or intra-island
                crossover, or to come from a particular island or species.

        Returns:
            A list of unique (non-duplicate) CircuitGenomes and a dictionary of metadata
            for the child they generate. If it is not possible to generate the specified
            number of parents, i.e., the target island is too small for intra-island
            crossover or there are not enough islands with genomes for inter-island
            crossover, then it will return None.
        """

        target_island = self.islands[self.current_island]
        self.increment_current_island()

        metadata = {"target_island_id": target_island.id}

        parents = None

        if random.uniform(0.0, 1.0) < self.intra_island_crossover_rate:
            # do intra island crossover
            logger.info(
                f"intra island crossover on {target_island.status} island: potential parent "
                f"length: {len(target_island.population)}, n_parents: {n_parents}"
            )

            if target_island.status == "full":
                parents = target_island.get_parents(n_parents)

            elif target_island.status == "repopulating":
                if len(target_island.population) < n_parents:
                    # try to get parents from best island if we dont have enough
                    # in this repopulating island
                    parents = self.best_island.get_parents(n_parents)
                else:
                    parents = target_island.get_parents(n_parents)

            else:
                logger.error("Doing intra-island crossover on an initializing island, this should never happen.")
                exit(1)

        else:
            # do inter island crossover

            # potential other parents can come from all other islands
            potential_parents = []

            for island in self.islands:
                if island != target_island:
                    potential_parents.extend(island.population)

            logger.info(
                f"inter island crossover: potential parent length: {len(potential_parents)}, "
                f"n_parents - 1: {n_parents - 1}"
            )

            if len(potential_parents) < (n_parents - 1):
                # there were not enough parents to select
                logger.warning(
                    "There were not enough potential parents across all other islands "
                    f"{len(potential_parents)} to get the requested number of parents {n_parents}"
                )
                return None, None

            # get the first parent from either the target island (if it is full) or the
            # best island if it is repopulating 
            if target_island.status == "full":
                parents = [random.choice(target_island.population)]

            elif target_island.status == "repopulating":
                if len(self.best_island.population) > 0:
                    parents = [random.choice(self.best_island.population)]
                else:
                    # in case the global best island ended up being repopulated
                    parents = [self.global_best_genome]

            else:
                logger.error("Doing inter-island crossover on an initializing island, this should never happen.")
                exit(1)

            # get all the remaining parents from other islands randomly
            parents.extend(random.sample(potential_parents, n_parents - 1))

        if parents is None:
            return None, None
        else:
            return parents, metadata

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

        target_island = None
        current_genome_number = kwargs['current_genome_number']

        if not hasattr(genome.metadata, "target_island_id"):
            # genome was generated without metadata for a target island which
            # means it was generated for initialization.

            # insert it into the island with the least number of genomes

            min_size = self.max_island_size
            target_islands = []

            for island in self.islands:
                logger.debug(f"min_size: {min_size}")
                if len(island.population) < min_size:
                    # found a new smallest island so use this
                    min_size = len(island.population)
                    target_islands = [island]

                elif len(island.population) == min_size:
                    # this would be another island that would
                    # be a potential target with the same minimal
                    # number of genomes
                    target_islands.append(island)

            target_island_ids = [target_island.id for target_island in target_islands]
            logger.debug(f"target island ids: {target_island_ids}")

            target_island = random.choice(target_islands)
        else:
            # select the target island as the island it was generated for
            # from the metadata
            target_island = self.islands[genome.metadata["target_island_id"]]

        if self.profiler is not None:
            merged_population = []
            for island in self.islands:
                merged_population.extend(island.population)

            self.profiler.record(
                step=self.insertions,
                population=merged_population,
            )

        if (
            self.global_best_genome is None
            or self.compare(self.global_best_genome, genome) > 0
        ):
            self.global_best_genome = genome
            # set its metadata as global best so we can use this during repopulation
            # on the chance it would be discarded due to being generated from before
            # the island was repopulated
            self.global_best_genome.metadata['global_best'] = True

            # update the best island to the island of this genome
            self.best_island = target_island

            # this was a new global best genome
            logger.success(
                f"[global insertion {self.insertions}] Population found new GLOBAL best genome "
                f"with fitness: {genome.fitness}"
            )

            test_metric = genome.fitness.get("test_acc", None)
            if test_metric is None:
                test_metric = genome.fitness.get("test_fidelity")

            tag = (
                f"trainloss_{genome.fitness['train_loss']:.4f}_testloss_"
                f"{genome.fitness['test_loss']:.4f}_testacc_{test_metric:.3f}"
            )

            if self.out_dir is not None:
                self._save_best_circuit(genome, out_dir=self.out_dir, tag=tag)
                self.profiler.plot_single_run()

        # check to see if the genome was a new global best
        logger.debug(f"target island id: {target_island.id}")
        target_island.insert_genome(genome)
        self.insertions += 1

        if self.insertions > 0 and (self.insertions % self.genomes_before_extinction) == 0:
            # perform island repopulation, but only repopulate full islands as well as
            # islands which have had enough genomes inserted to be repopulated again
            full_islands = [island for island in self.islands if island.status == "full" and (current_genome_number - island.repopulation_genome_number) > self.genomes_for_next_extinction]

            logger.info(f"REPOPULATING AT ITERATION {self.insertions}")
            logger.info(
                f"\tisland strategy has {len(full_islands)} full islands ready for repopulation, "
                f"repopulating {self.islands_to_extinct}"
            )

            # the worst islands should be sorted first
            full_islands.sort(key=cmp_to_key(island_compare), reverse=True)
            logger.info("\tsorted full islands!")

            removed = 0
            while removed < self.islands_to_extinct and removed < len(full_islands):
                # remove up to islands_to_extinct islands (less if we don't have enough
                # full populations)
                target_island = full_islands[removed]

                logger.info(f"\trepopulating island {target_island.id}, with genome[0] fitness: {target_island.population[0].fitness}")

                target_island.repopulate(repopulation_genome_number=current_genome_number)
                removed += 1

    def _save_best_circuit(
        self, genome: CircuitGenome, out_dir: str = "artifacts/", tag: str = ""
    ):
        if genome is None:
            logger.error("Genome cannot be None")
            raise ValueError

        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, f"genome_{genome.genome_number}.json")
        logger.info(f"writing NEW BEST gnome to {json_path}")
        with open(json_path, "w") as fp:
            json.dump(genome.to_dict(), fp, ensure_ascii=False, indent=4)

        # --- Text gate list ---
        txt_path = os.path.join(out_dir, f"genome_{genome.genome_number}.txt")
        with open(txt_path, "w") as f:
            genome.sort_gates()
            f.write(f"Genome {genome.genome_number}\n")
            f.write(f"Qubits: {genome.qubits}\n\n")
            for g in genome.gates:
                if getattr(g, "enabled", True):
                    f.write(
                        f"{g.depth:.3f}  {g.method_name}  {g.qubits}  {g.parameters}\n"
                    )

        # --- PennyLane draw ---
        genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")
        # logger.info(f"genome circuit: {genome.circuit}")

        try:
            params = genome_to_torch_params(genome)
            x0 = torch.zeros(len(genome.input_indexes))
            fig, ax = qml.draw_mpl(genome.circuit)(x0, params)
            ax.set_title(f"Genome {genome.genome_number}")
            path = os.path.join(
                out_dir, f"best_genome_{genome.genome_number}_{tag}.png"
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not draw circuit: {e}")
