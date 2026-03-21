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

    def is_initializing(self) -> bool:
        """
        Returns:
            True if the island is still initializing.
        """

        return self.status == "initializing"

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

        # TODO: don't add duplicate genomes to the population
        # options:
        # 1. if gate innovation numbers are the same, keep the genome with better fitness
        # 2. if gate innovation numbers are the same but fitness different, keep both

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


class SteadyStateIslands(PopulationStrategy):

    def __init__(
        self,
        n_islands: int,
        max_island_size: int,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
        intra_island_crossover_rate: float = 0.5,
        genomes_before_extinction: int = 250,
        islands_to_extinct: int = 1,
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
        if self.profiler is None:
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

    def get_parent(self, **kwargs) -> tuple[CircuitGenome, dict[str, any]]:
        """
        Used to get a parent to be used in mutation or other operations to generate
        children. This will be generated from an island in a round robin fashion.

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

        if target_island.status == "repopulating":
            # get parent from best island
            return random.choice(self.best_island.population), metadata

        elif len(target_island.population) > 0:
            return random.choice(target_island.population), metadata
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

        if target_island.status != "full":
            # try to get enough parents from the target island
            logger.info(
                f"non full intra island crossover:  potential parent length: {len(target_island.population)}, "
                f"n_parents: {n_parents}"
            )
            parents = target_island.get_parents(n_parents)
        else:
            # try to do inter or intra-island crossover as specified

            if random.uniform(0.0, 1.0) < self.intra_island_crossover_rate:
                logger.info(
                    f"intra island crossover: potential parent length: {len(target_island.population)}, "
                    f"n_parents: {n_parents}"
                )
                parents = target_island.get_parents(n_parents)
            else:
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
                    return None, None

                # have at least one parent from the current island
                parents = [random.choice(target_island.population)]
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

        logger.debug(f"target island id: {target_island.id}")
        target_island.insert_genome(genome)
        self.insertions += 1

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
