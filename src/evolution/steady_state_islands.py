"""Island-model steady-state population strategy for EXAQC.

The search is split across several independent steady-state populations
(:class:`Island`), each keeping its own fitness-sorted set of genomes. Children
are produced by intra-island crossover (parents from one island) or
inter-island crossover (parents drawn from an island's neighbors, as defined by
a connection topology). Periodically the worst full islands undergo an
extinction event and are repopulated, spreading strong genomes while preserving
diversity. :class:`SteadyStateIslands` is the :class:`PopulationStrategy` that
ties these together for the ``master_worker`` driver.
"""

from __future__ import annotations

import argparse
import random

from functools import cmp_to_key
from typing import Callable, Optional

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.island import Island
from src.evolution.topology import assign_topology
from src.evolution.population_strategy import PopulationStrategy
from src.utils.profiler import EXAQCProfiler


def island_compare(island1: Island, island2: Island) -> int:
    """
    Compares two islands by the fitness of their best genome (slot 0), used to
    rank islands (e.g. to find the worst full islands for extinction). Both
    islands must be non-empty, since it reads ``population[0]`` of each.

    Args:
        island1: the first island whose best genome is compared.
        island2: the second island whose best genome is compared.

    Returns:
        0 if the two islands' best genomes have equivalent fitness, a negative
        value if ``island1``'s best should sort before ``island2``'s, and a
        positive value otherwise.
    """

    return island1.compare(island1.population[0], island2.population[0])


class SteadyStateIslands(PopulationStrategy):

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds this population strategy's command-line arguments to a parser.

        Entry points call this on the ``islands`` sub-parser so every script
        exposes the same flags with the same defaults and help text, rather
        than repeating them. Each argument corresponds to the like-named
        :meth:`__init__` keyword.

        Args:
            parser: The (sub-)parser to add this strategy's arguments to.

        Returns:
            None. Mutates ``parser`` by adding ``--n_islands``,
            ``--max_island_size``, ``--genomes_before_extinction``,
            ``--genomes_for_next_extinction``, ``--islands_to_extinct``,
            ``--primary_parent`` and ``--intra_island_crossover_rate``.
        """

        parser.add_argument(
            "--n_islands",
            type=int,
            default=10,
            help="Number of steady-state populations (islands) evolved in parallel.",
        )

        parser.add_argument(
            "--max_island_size",
            type=int,
            default=10,
            help="Maximum number of genomes retained in each island.",
        )

        parser.add_argument(
            "--genomes_before_extinction",
            type=int,
            default=100,
            help="Number of genomes inserted before the first island extinction event.",
        )

        parser.add_argument(
            "--genomes_for_next_extinction",
            type=int,
            default=200,
            help="Number of genomes that need to be inserted into an island before it can be repopulated again.",
        )

        parser.add_argument(
            "--islands_to_extinct",
            type=int,
            default=1,
            help="Number of worst islands cleared and repopulated at each extinction event.",
        )

        parser.add_argument(
            "--primary_parent",
            type=str,
            default="best",
            help=(
                "How the primary crossover parent is chosen: 'best' (highest-fitness "
                "parent first) or 'island' (the target island's genome first)."
            ),
        )

        parser.add_argument(
            "--intra_island_crossover_rate",
            type=float,
            default=0.5,
            help="Fraction of an island's offspring produced by crossover within the same island.",
        )

        parser.add_argument(
            "--topology",
            type=str,
            nargs="+",
            default=["fully_connected"],
            help=(
                "How islands are connected to each other, which determines which other islands "
                "an island can select genomes from for inter-island crossover. Options are: "
                "'fully_connected' (default), 'ring', 'star' (all islands connected to one center), "
                "'2d_mesh <x_dim> <y_dim>' (requires x_dim * y_dim == n_islands), "
                "'tree <children per node>', 'random <min_edges> <max_edges>' (connects all islands "
                "in a line, then randomly adds (uniform between (min_edges - 1) to (max_edges - 1) other "
                "edges from each island to another randomly selected island)."
            ),
        )

    def __init__(
        self,
        n_islands: int,
        max_island_size: int,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
        intra_island_crossover_rate: float = 0.5,
        genomes_before_extinction: int = 50,
        genomes_for_next_extinction: int = 100,
        islands_to_extinct: int = 2,
        primary_parent: str = "best",
        topology: list[str] = ["fully_connected"],
        out_dir: str = None,
        profiler: Optional[EXAQCProfiler] = None,
        save_training_plot: bool = False,
    ):
        """
        Creates an island-model population of ``n_islands`` steady-state
        populations, each holding up to ``max_island_size`` genomes sorted by
        fitness. The islands are wired together with the requested ``topology``,
        and the get-parent methods draw parents for intra- or inter-island
        crossover. As genomes are inserted, the worst full islands are
        periodically cleared and repopulated (extinction events).

        Args:
            n_islands: how many islands (steady-state populations) to evolve in
                parallel.
            max_island_size: the maximum number of genomes retained per island.
            compare: a compare function used for sorting genomes. this should
                return 0 if both genomes should be ranked the same, a negative
                value if the first genome should come before the second genome,
                and a positive number otherwise.
            intra_island_crossover_rate: when both intra- and inter-island
                crossover are possible, the probability of choosing intra-island.
            genomes_before_extinction: how many genomes are inserted before an
                extinction event happens, which clears out the worst islands and
                repopulates them.
            genomes_for_next_extinction: how many genomes need to be added to an
                island before it can be repopulated again.
            islands_to_extinct: how many islands to clear out in an extinction
                event.
            primary_parent: can be `best` or `island`, and it determines how the
                primary parent is selected when get_parents is called. If `best`,
                the parent genomes are sorted such that the first (primary)
                parent has the best fitness; if `island` then the first genome is
                the one from the target island for the child.
            topology: how the islands are connected, which determines which other
                islands an island can select genomes from for inter-island
                crossover (see :func:`~src.evolution.topology.assign_topology`).
                Options are: 'fully_connected' (default), 'ring',
                'star' (all islands connected to one center),
                '2d_mesh <x_dim> <y_dim>' (requires x_dim * y_dim == n_islands),
                'tree <children per node>', and
                'random <min_edges> <max_edges>'.
            out_dir: the directory to write out the best found genomes and log
                files; if not specified, files are not written.
            profiler: an optional profiler to record per-insertion population
                snapshots; created automatically from ``out_dir`` when omitted.
            save_training_plot: when True, each saved genome also gets a
                training-history line plot written next to its diagram (see
                :meth:`CircuitGenome.save_circuit`).
        """

        self.n_islands = n_islands
        self.max_island_size = max_island_size
        self.compare = compare
        self.save_training_plot = save_training_plot
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

        assign_topology(self.islands, topology)

        self.global_best_genome = None
        self.metric_best_genome = None

        if primary_parent not in ("best", "island"):
            logger.error(
                f"Error initializing island strategy. Primary parent was {primary_parent} "
                "and possible options are either `best` or `island`."
            )
            exit(1)

        self.primary_parent = primary_parent

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

    def increment_current_island(self) -> None:
        """
        Increments the current island index in a round robin fashion, wrapping
        back to 0 after the last island.

        Returns:
            None. Advances ``self.current_island`` in place.
        """

        self.current_island += 1
        if self.current_island >= len(self.islands):
            self.current_island = 0

    def get_best_genome(self) -> CircuitGenome | None:
        """
        Returns:
            The best genome across all islands, if it exists. None otherwise.  It would
            only return none if no genomes have been inserted yet (i.e., the very beginning
            of the search).
        """

        return self.global_best_genome

    def get_parent(
        self, **kwargs
    ) -> tuple[CircuitGenome | None, dict[str, any] | None]:
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
            A tuple of a single CircuitGenome and a dictionary of its metadata
            (carrying the ``target_island_id``). Returns ``(None, None)`` when no
            parent can be selected -- i.e. the target island is repopulating and
            none of its neighbors hold any genomes.
        """

        target_island = self.islands[self.current_island]
        self.increment_current_island()

        metadata = {"target_island_id": target_island.id}

        if target_island.status == "full":
            return random.choice(target_island.population), metadata

        if target_island.status == "repopulating":
            best_neighbor = target_island.best_neighbor()

            if best_neighbor is not None:
                return random.choice(best_neighbor.population), metadata
            else:
                return None, None

        else:
            logger.error(
                "tried to get a parent from an initializing island. This should never happen."
            )
            exit(1)

    def get_parents(
        self, n_parents: int = 2, **kwargs
    ) -> tuple[list[CircuitGenome], dict[str, any]]:
        """
        Used to get two or more parents to be used in crossover or
        other operations to generate children, for a target island selected
        in a round robin manner.

        Will perform either intra- or inter-island crossover. It will perform
        inter-island crossover if there are not enough parents on the target
        island to perform intra-island crossover. Similarly, it will perform
        intra-island crossover if there not enough genomes in neighboring island
        populations to perform inter-island crossover.

        If both can be performed, it will select inter or intra island crossover
        randomly based on the intra_island_crossover_rate.

        If the target island is repopulating, parents will be selected from its
        best neighbor if it has enough genomes.

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

        # check to see if we have enough genomes on the target or best
        # island to do intra-island crossover.  if there is only one
        # island always do intra-island crossover.
        # if we are supposed to do inter-island crossover but there are
        # not enough neighbors, fall back to intra-island crossover

        do_intra_island = random.uniform(0.0, 1.0) < self.intra_island_crossover_rate

        if (
            n_parents > target_island.neighbor_population_size()
            or (  # not enough neighbors for inter-island
                do_intra_island and len(target_island.population) >= n_parents
            )
        ):
            # try to do intra island crossover
            logger.info(
                f"intra island crossover on {target_island.status} island: potential parent "
                f"length: {len(target_island.population)}, n_parents: {n_parents}"
            )

            metadata["crossover_type"] = "intra-island"

            if target_island.status == "full":
                parents = target_island.get_parents(n_parents)

            elif target_island.status == "repopulating":
                # get parents from our best neighboring island while we are
                # still repopulating
                best_neighbor = target_island.best_neighbor()

                if best_neighbor is not None:
                    parents = best_neighbor.get_parents(n_parents)

            else:
                logger.error(
                    "Doing intra-island crossover on an initializing island, this should never happen."
                )
                exit(1)

        # there weren't enough parents at the target (or best) island to get
        # intra-island parents so fall back to inter-island parents
        if parents is None:
            # try inter island crossover

            # potential other parents can come from all other islands
            potential_parents = []

            metadata["crossover_type"] = "inter-island"

            for island in target_island.neighbors:
                potential_parents.extend(island.population)

            logger.info(
                f"inter island crossover: n neighbors: {len(target_island.neighbors)}, potential parent "
                f"length: {len(potential_parents)}, n_parents - 1: {n_parents - 1}"
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
                best_neighbor = target_island.best_neighbor()

                if best_neighbor is not None:
                    parents = [random.choice(best_neighbor.population)]

            else:
                logger.error(
                    "Doing inter-island crossover on an initializing island, this should never happen."
                )
                exit(1)

            # get all the remaining parents from other islands randomly
            parents.extend(random.sample(potential_parents, n_parents - 1))

        if parents is None:
            return None, None
        else:
            if self.primary_parent == "best":
                # sort the parents so that the most fit is the primary parent, otherwise
                # it will be from the target island
                parents.sort(key=cmp_to_key(self.compare))
                logger.debug(
                    f"sorted parents for best primary parent strategy, primary parent fitness: {parents[0].fitness}, "
                    f"worst parent fitness: {parents[-1].fitness}"
                )

            return parents, metadata

    def insert_genome(self, genome: CircuitGenome, **kwargs) -> None:
        """
        Inserts a genome into the island it was generated for, updates the
        global/metric best genomes, and triggers extinction events periodically.

        A genome carrying a ``target_island_id`` in its metadata is routed to
        that island; one generated for initialization (no target) is routed to
        one of the islands with the fewest genomes.

        Args:
            genome: is the genome to insert into the population.
            **kwargs: additional options for inserting the genome; must include
                ``current_genome_number`` (used to gate extinction/repopulation).

        Returns:
            None. Inserts the genome into an island and updates the strategy's
            best-genome tracking and extinction state in place.
        """

        target_island = None
        current_genome_number = kwargs["current_genome_number"]

        if "target_island_id" not in genome.metadata:
            # genome was generated without metadata for a target island which
            # means it was generated for initialization: insert it into one of
            # the islands that currently hold the fewest genomes.
            min_size = min(len(island.population) for island in self.islands)
            target_islands = [
                island for island in self.islands if len(island.population) == min_size
            ]
            target_island = random.choice(target_islands)
        else:
            # select the target island as the island it was generated for
            # from the metadata
            target_island = self.islands[genome.metadata["target_island_id"]]

        if self.profiler is not None:
            # Sort the merged snapshot so profiler Best/top-k match global ranking.
            merged_population: list[CircuitGenome] = []
            for island in self.islands:
                merged_population.extend(island.population)
            merged_population.sort(key=cmp_to_key(self.compare))

            self.profiler.record(
                step=self.insertions,
                population=merged_population,
            )

        if self.metric_best_genome is None or (
            "target_metric" in genome.fitness
            and self.metric_best_genome.fitness["target_metric"]
            <= genome.fitness["target_metric"]
        ):
            self.metric_best_genome = genome

            # this was a new genome with a best accuracy
            logger.success(
                f"[global insertion {self.insertions}] Population found new best genome for target_metric"
                f"with fitness: {genome.fitness}"
            )

            if self.out_dir is not None:
                genome.save_circuit(
                    insert_type="best_accuracy",
                    out_dir=self.out_dir,
                    save_training_plot=self.save_training_plot,
                )
                if self.profiler is not None:
                    self.profiler.plot_single_run()

        if (
            self.global_best_genome is None
            or self.compare(self.global_best_genome, genome) > 0
        ):
            self.global_best_genome = genome
            # set its metadata as global best so we can use this during repopulation
            # on the chance it would be discarded due to being generated from before
            # the island was repopulated
            genome.metadata["insert_type"] = "global_best"

            # this was a new global best genome
            logger.success(
                f"[global insertion {self.insertions}] Population found new GLOBAL best genome "
                f"with fitness: {genome.fitness}"
            )

            if self.out_dir is not None:
                genome.save_circuit(
                    insert_type="best_fitness",
                    out_dir=self.out_dir,
                    save_training_plot=self.save_training_plot,
                )
                if self.profiler is not None:
                    self.profiler.plot_single_run()

        # check to see if the genome was a new global best
        logger.debug(f"target island id: {target_island.id}")
        target_island.insert_genome(genome)
        self.insertions += 1

        if self.out_dir is not None:
            genome.save_circuit(
                insert_type="genome",
                out_dir=self.out_dir + "/all_genomes/",
                save_training_plot=self.save_training_plot,
            )

        if (
            self.insertions > 0
            and (self.insertions % self.genomes_before_extinction) == 0
        ):
            # perform island repopulation, but only repopulate full islands as well as
            # islands which have had enough genomes inserted to be repopulated again
            full_islands = [
                island
                for island in self.islands
                if island.status == "full"
                and (current_genome_number - island.repopulation_genome_number)
                > self.genomes_for_next_extinction
            ]

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

                logger.info(
                    f"\trepopulating island {target_island.id}, with genome[0] "
                    f"fitness: {target_island.population[0].fitness}"
                )

                target_island.repopulate(
                    repopulation_genome_number=current_genome_number
                )
                removed += 1
