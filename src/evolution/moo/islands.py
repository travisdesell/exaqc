"""Multi-objective extension of the EXAQC steady-state island strategy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Type

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.base_population import MultiObjectivePopulationBase
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.moo.pareto import (
    genome_dominates,
    objective_vector,
)
from src.evolution.population_strategy import mark_discarded
from src.evolution.steady_state_islands import (
    Island,
    SteadyStateIslands,
)
from src.evolution.topology import assign_topology


class MultiObjectiveIsland(Island):
    """Island backed by a multi-objective population strategy.

    This adapter preserves the interface and state used by
    ``SteadyStateIslands`` while delegating local population selection to
    NSGA-II or NSGA-III.

    Args:
        id: Unique island identifier.
        max_size: Maximum number of genomes retained by the island.
        population_class: Multi-objective population class, such as
            ``NSGA2`` or ``NSGA3``.
        objectives: Objective specifications used by the population.
        tournament_size: Number of candidates used for tournament parent
            selection.
        population_kwargs: Additional arguments forwarded to the
            multi-objective population class.
        compare: Total order used by island bookkeeping (extinction and the
            best neighbor). Local survival stays Pareto-based.
    """

    def __init__(
        self,
        id: int,
        max_size: int,
        population_class: Type[MultiObjectivePopulationBase],
        objectives: Sequence[ObjectiveSpec],
        tournament_size: int = 2,
        population_kwargs: dict[str, Any] | None = None,
        compare: Callable[[CircuitGenome, CircuitGenome], int] | None = None,
    ) -> None:
        """Initialize a multi-objective island."""
        population_kwargs = population_kwargs or {}

        self.id = id
        self.max_size = max_size
        self.compare = compare

        self.insertions = 0
        self.status = "initializing"
        self.repopulation_genome_number = 0
        self.neighbors: list[Island] = []

        self.strategy = population_class(
            max_population_size=max_size,
            objectives=objectives,
            tournament_size=tournament_size,
            **population_kwargs,
        )

    @property
    def population(self) -> list[CircuitGenome]:
        """Return the underlying multi-objective population.

        Returns:
            Population maintained by the island's MOO strategy.
        """
        return self.strategy.population

    @population.setter
    def population(
        self,
        value: list[CircuitGenome],
    ) -> None:
        """Replace the underlying population.

        Args:
            value: New island population.
        """
        self.strategy.population = value

    def restore(
        self,
        genomes: list[CircuitGenome],
        next_genome_number: int,
        cleared: bool,
    ) -> None:
        """Takes back the genomes this island held when a run stopped.

        Status matches :meth:`Island.restore`: full at capacity, repopulating
        when an extinction event cleared it, and initializing otherwise. The
        genomes are not sorted by a scalar fitness, because survival on this
        island is Pareto-based.

        Args:
            genomes: The genomes this island held, in any order.
            next_genome_number: The number the restarted run's next genome takes.
            cleared: Whether an extinction event emptied this island during the
                run.

        Returns:
            None. Restores ``population``, ``status``, ``insertions`` and
            ``repopulation_genome_number``.
        """
        self.population = list(genomes)
        self.insertions = len(self.population)
        self.strategy.insertions = len(self.population)

        if len(self.population) >= self.max_size:
            self.status = "full"
        elif cleared:
            self.status = "repopulating"
            self.repopulation_genome_number = next_genome_number
        else:
            self.status = "initializing"

        self.strategy._refresh_selection_metadata()

    def best_neighbor(self) -> Island | None:
        """Finds the neighboring island with the best representative genome.

        The representative is the multi-objective strategy's best genome, not
        whichever genome happens to sit at ``population[0]``.

        Returns:
            The neighboring island with the preferred representative, or
            ``None`` if no neighbor holds a genome.

        Raises:
            RuntimeError: If this island was built without a compare function.
        """
        compare = self.compare
        if compare is None:
            raise RuntimeError(
                f"Island {self.id} has no compare function, so it cannot rank neighbors."
            )

        best: Island | None = None
        best_genome: CircuitGenome | None = None

        for neighbor in self.neighbors:
            candidate = self._representative(neighbor)
            if candidate is None:
                continue

            if best is None or best_genome is None:
                best = neighbor
                best_genome = candidate
            elif compare(candidate, best_genome) < 0:
                best = neighbor
                best_genome = candidate

        return best

    @staticmethod
    def _representative(island: Island) -> CircuitGenome | None:
        """Return the genome island bookkeeping should treat as best.

        Args:
            island: Island whose representative is needed.

        Returns:
            The strategy's representative genome when the island has one, the
            first population member otherwise, or ``None`` when the island is
            empty.
        """
        if not island.population:
            return None

        strategy = getattr(island, "strategy", None)
        if strategy is not None:
            return strategy.get_best_genome()

        return island.population[0]

    def is_initializing(self) -> bool:
        """Return whether the island is still initializing.

        Returns:
            ``True`` while the island has not reached its maximum size.
        """
        return self.status == "initializing"

    def repopulate(
        self,
        repopulation_genome_number: int,
    ) -> None:
        """Clear the island and mark it for repopulation.

        Args:
            repopulation_genome_number: Genome number at which repopulation
                occurred. Older outstanding genomes can then be discarded.
        """
        self.status = "repopulating"
        self.repopulation_genome_number = repopulation_genome_number

        self.population = []

    def get_parent(
        self,
        **kwargs,
    ) -> CircuitGenome | None:
        """Select one parent using the island's MOO strategy.

        Args:
            **kwargs: Additional parent-selection arguments.

        Returns:
            Selected genome, or ``None`` if the island is empty.
        """
        parent, _ = self.strategy.get_parent(**kwargs)

        return parent

    def get_parents(
        self,
        n_parents: int = 2,
        **kwargs,
    ) -> list[CircuitGenome] | None:
        """Select multiple parents using the island's MOO strategy.

        Args:
            n_parents: Number of parents to select.
            **kwargs: Additional parent-selection arguments.

        Returns:
            Unique selected parents, or ``None`` if there are insufficient
            genomes.
        """
        parents, _ = self.strategy.get_parents(
            n_parents=n_parents,
            **kwargs,
        )

        return parents

    def insert_genome(
        self,
        genome: CircuitGenome,
        **kwargs: Any,
    ) -> bool:
        """Insert a genome using local multi-objective selection.

        Stale genomes created before an island repopulation are handled in
        the same manner as the original ``Island`` implementation.

        Args:
            genome: Evaluated genome.
            **kwargs: Additional insertion arguments.

        Returns:
            ``True`` if the genome should be recorded as evaluated. ``False``
            if it predates this island's repopulation, or if the local
            strategy rejected it as a duplicate of a better genome.
        """
        if (
            "insert_type" not in genome.metadata
            or genome.metadata["insert_type"] != "global_best"
        ):
            genome.metadata["insert_type"] = "inserted"

        if (
            genome.genome_number < self.repopulation_genome_number
            and genome.metadata["insert_type"] != "global_best"
        ):
            logger.info(
                f"discarding genome {genome.genome_number} "
                "because it predates island "
                f"{self.id} repopulation genome "
                f"{self.repopulation_genome_number}"
            )

            mark_discarded(genome, "generated_before_repopulation")

            return False

        recorded = self.strategy.insert_genome(
            genome,
            **kwargs,
        )

        self.insertions += 1

        if len(self.population) >= self.max_size:
            self.status = "full"

        if genome.metadata.get("insert_type") not in ("discarded", "global_best"):
            genome.metadata["insert_type"] = "inserted"

        return recorded


class MultiObjectiveSteadyStateIslands(SteadyStateIslands):
    """Steady-state islands using NSGA-II or NSGA-III locally.

    The existing EXAQC island model manages target-island scheduling,
    crossover routing, extinction, and repopulation. Each individual island
    delegates local survival and parent selection to a multi-objective
    population strategy.

    Args:
        population_class: Multi-objective population class, such as
            ``NSGA2`` or ``NSGA3``.
        objectives: Objective specifications.
        n_islands: Number of islands.
        max_island_size: Maximum population size of each island.
        tournament_size: MOO tournament size.
        population_kwargs: Additional arguments passed to each local
            population.
        **kwargs: Additional arguments forwarded to
            ``SteadyStateIslands``.
    """

    def __init__(
        self,
        population_class: Type[MultiObjectivePopulationBase],
        objectives: Sequence[ObjectiveSpec],
        n_islands: int,
        max_island_size: int,
        tournament_size: int = 2,
        population_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the multi-objective island strategy."""
        self.objectives = list(objectives)

        self.population_class = population_class

        self.tournament_size = tournament_size

        self.population_kwargs = population_kwargs or {}

        super().__init__(
            n_islands=n_islands,
            max_island_size=max_island_size,
            compare=self._compare_genomes,
            **kwargs,
        )

        # SteadyStateIslands builds scalar-fitness islands and wires their
        # topology. Replace them with MOO islands, then apply that same
        # topology: the replacement would otherwise drop every neighbor.
        self.islands = [
            MultiObjectiveIsland(
                id=island_id,
                max_size=max_island_size,
                population_class=population_class,
                objectives=self.objectives,
                tournament_size=tournament_size,
                compare=self.compare,
                population_kwargs={
                    **self.population_kwargs,
                    "seed": (
                        self.population_kwargs.get(
                            "seed",
                            0,
                        )
                        + island_id
                    ),
                },
            )
            for island_id in range(n_islands)
        ]
        assign_topology(self.islands, self.topology)

    def _compare_genomes(
        self,
        genome1: CircuitGenome,
        genome2: CircuitGenome,
    ) -> int:
        """Compare two genomes for island-level bookkeeping.

        Local population survival remains fully Pareto based. This comparator
        is needed only because the existing ``SteadyStateIslands`` code
        expects a total ordering when selecting a primary crossover parent,
        tracking a representative global best, and ordering islands during
        extinction.

        Pareto dominance is considered first. If neither genome dominates
        the other, the first configured objective is used as a deterministic
        tie breaker.

        Args:
            genome1: First genome.
            genome2: Second genome.

        Returns:
            Negative value if ``genome1`` is preferred, positive value if
            ``genome2`` is preferred, or zero if they are equivalent.
        """
        if genome_dominates(
            genome1,
            genome2,
            self.objectives,
        ):
            return -1

        if genome_dominates(
            genome2,
            genome1,
            self.objectives,
        ):
            return 1

        values1 = objective_vector(
            genome1,
            self.objectives,
        )

        values2 = objective_vector(
            genome2,
            self.objectives,
        )

        if values1[0] < values2[0]:
            return -1

        if values1[0] > values2[0]:
            return 1

        return 0
