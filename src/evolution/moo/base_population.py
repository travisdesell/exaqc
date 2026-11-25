"""Base population implementation for multi-objective EXAQC strategies."""

from __future__ import annotations

import random

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.moo.pareto import (
    genome_dominates,
    validate_genome_fitness,
)
from src.evolution.population_strategy import PopulationStrategy, mark_discarded

if TYPE_CHECKING:
    from src.utils.restart import RestartState


class MultiObjectivePopulationBase(PopulationStrategy):
    """Base class for steady-state multi-objective populations.

    This class implements the EXAQC ``PopulationStrategy`` interface and
    provides behavior shared by algorithms such as NSGA-II and NSGA-III.

    Subclasses are responsible only for algorithm-specific environmental
    selection, diversity metadata, tournament comparison, and representative
    Pareto-front selection.

    Evaluated genomes are recorded by the search's archive. This class only
    decides which genomes stay in the population.

    Args:
        max_population_size: Maximum number of genomes retained.
        objectives: Objective specifications used during optimization.
        tournament_size: Number of candidates sampled during parent
            tournament selection.
        seed: Random seed used by parent selection.

    Raises:
        ValueError: If ``max_population_size`` is not positive.
        ValueError: If fewer than two objectives are supplied.
        ValueError: If ``tournament_size`` is smaller than two.
    """

    def __init__(
        self,
        max_population_size: int,
        objectives: Sequence[ObjectiveSpec],
        tournament_size: int = 2,
        seed: int = 0,
    ) -> None:
        """Initialize a multi-objective population."""
        if max_population_size <= 0:
            raise ValueError("max_population_size must be greater than zero.")

        if len(objectives) < 2:
            raise ValueError(
                "Multi-objective optimization requires at least two objectives."
            )

        if tournament_size < 2:
            raise ValueError("tournament_size must be at least two.")

        self.max_population_size = int(max_population_size)

        self.objectives = list(objectives)

        self.tournament_size = int(tournament_size)

        self.population: list[CircuitGenome] = []

        self.insertions = 0

        self.rng = random.Random(seed)

    def is_initializing(self) -> bool:
        """Determine whether the initial population is still being filled.

        Returns:
            ``True`` if the population has fewer genomes than its configured
            maximum size.
        """
        return len(self.population) < self.max_population_size

    def restore(self, state: "RestartState") -> None:
        """Takes back the population a stopped run held, so its search continues.

        Args:
            state: The stopped run's state (see :mod:`src.utils.restart`).

        Returns:
            None. Refills ``population`` with the run's genomes and restores
            the insertion count. Pareto ranks are recomputed.
        """
        self.population = list(state.population)
        self.insertions = state.inserted_genomes
        self._refresh_selection_metadata()

        logger.info(
            "restored a {} population of {} genomes after {} insertions",
            self.algorithm_name,
            len(self.population),
            self.insertions,
        )

    def get_population(self) -> list[CircuitGenome]:
        """Returns the genomes in the population, best first.

        Order is Pareto rank, then the algorithm's diversity key, then genome
        number. A multi-objective population has no single scalar best, so the
        first genome is the representative the same ordering would pick.

        Returns:
            A new list of the population's genomes. Modifying it does not
            affect the strategy.
        """
        self._refresh_selection_metadata()
        return sorted(self.population, key=self._parent_sort_key)

    def get_pareto_front(
        self,
    ) -> list[CircuitGenome]:
        """Return the current non-dominated Pareto front.

        Returns:
            Genomes with Pareto rank zero.
        """
        if not self.population:
            return []

        self._refresh_selection_metadata()

        return [
            genome
            for genome in self.population
            if genome.metadata.get("pareto_rank") == 0
        ]

    def get_best_genome(
        self,
    ) -> CircuitGenome | None:
        """Return a representative genome from the Pareto front.

        A multi-objective population does not have a mathematically unique
        best genome. The concrete algorithm therefore selects a
        representative Pareto-optimal genome for compatibility with EXAQC
        code that expects ``get_best_genome``.

        Returns:
            Representative Pareto-optimal genome, or ``None`` when the
            population is empty.
        """
        if not self.population:
            return None

        self._refresh_selection_metadata()

        return self._representative_genome()

    def get_parent(
        self,
        **kwargs,
    ) -> tuple[
        CircuitGenome | None,
        dict[str, object] | None,
    ]:
        """Select a single parent using tournament selection.

        Args:
            **kwargs: Additional population-strategy arguments. These are
                accepted for compatibility with ``PopulationStrategy``.

        Returns:
            Selected parent and metadata describing the selection. Returns
            ``(None, None)`` when the population is empty.
        """
        if not self.population:
            return None, None

        self._refresh_selection_metadata()

        parent = self._run_tournament(excluded_genome_numbers=set())

        metadata = {
            "selection_algorithm": (self.algorithm_name),
            "parent_pareto_rank": (parent.metadata.get("pareto_rank")),
        }

        return parent, metadata

    def get_parents(
        self,
        n_parents: int = 2,
        **kwargs,
    ) -> tuple[
        list[CircuitGenome] | None,
        dict[str, object] | None,
    ]:
        """Select multiple unique parents using tournament selection.

        Args:
            n_parents: Number of unique parents requested.
            **kwargs: Additional population-strategy arguments.

        Returns:
            Selected parents and selection metadata. Returns ``(None, None)``
            when there are insufficient genomes.

        Raises:
            ValueError: If ``n_parents`` is not positive.
        """
        if n_parents <= 0:
            raise ValueError("n_parents must be positive.")

        if len(self.population) < n_parents:
            return None, None

        self._refresh_selection_metadata()

        parents: list[CircuitGenome] = []

        selected_numbers: set[int] = set()

        while len(parents) < n_parents:
            parent = self._run_tournament(excluded_genome_numbers=(selected_numbers))

            parents.append(parent)

            selected_numbers.add(parent.genome_number)

        parents.sort(key=self._parent_sort_key)

        metadata = {
            "selection_algorithm": (self.algorithm_name),
            "parent_pareto_ranks": [
                parent.metadata.get("pareto_rank") for parent in parents
            ],
        }

        return parents, metadata

    def insert_genome(
        self,
        genome: CircuitGenome,
        **kwargs: Any,
    ) -> bool:
        """Insert an evaluated genome into the population.

        The genome is temporarily combined with the current population.
        The concrete multi-objective algorithm then performs environmental
        selection to retain at most ``max_population_size`` genomes.

        A genome that loses environmental selection is still recorded: the
        search archives it as discarded. A genome that repeats a topology a
        better genome already holds is not recorded.

        Args:
            genome: Evaluated genome to insert.
            **kwargs: Additional population metadata. Accepted for
                compatibility with :class:`PopulationStrategy`.

        Returns:
            ``True`` if the genome should be recorded as evaluated, whether it
            was kept or discarded from a full population. ``False`` if it is a
            duplicate of a better genome already held and should not be
            recorded.
        """
        validate_genome_fitness(
            genome,
            self.objectives,
        )

        self.insertions += 1

        duplicate_result = self._handle_duplicate(genome)

        if duplicate_result is False:
            mark_discarded(
                genome,
                "duplicate_of_better",
                self._dominating_duplicate(genome),
            )
            self._log_population_state(genome, survived=False)
            return False

        combined_population = self.population + [genome]

        survivors = self._environmental_selection(
            combined_population,
            self.max_population_size,
        )

        survived = any(
            candidate.genome_number == genome.genome_number for candidate in survivors
        )

        self.population = survivors

        self._refresh_selection_metadata()

        if survived:
            if genome.metadata.get("insert_type") != "global_best":
                genome.metadata["insert_type"] = "inserted"
        else:
            mark_discarded(
                genome,
                "worse_than_population",
                self._worst_kept_number(survivors),
            )

        self._log_population_state(
            genome,
            survived,
        )

        return True

    def _dominating_duplicate(self, genome: CircuitGenome) -> int | None:
        """Return the genome number of a better duplicate still in the population.

        Args:
            genome: Candidate that was rejected as a dominated duplicate.

        Returns:
            The number of the population member that Pareto-dominates
            ``genome`` and has the same enabled gates, or ``None`` when no
            such member remains.
        """
        for existing in self.population:
            if existing.has_same_gates(genome) and genome_dominates(
                existing,
                genome,
                self.objectives,
            ):
                return existing.genome_number
        return None

    def _worst_kept_number(self, survivors: Sequence[CircuitGenome]) -> int | None:
        """Return the genome number of the worst survivor.

        Args:
            survivors: Genomes environmental selection kept.

        Returns:
            The number of the survivor that sorts last by the parent key, or
            ``None`` when ``survivors`` is empty.
        """
        if not survivors:
            return None
        worst = max(survivors, key=self._parent_sort_key)
        return worst.genome_number

    def _handle_duplicate(
        self,
        genome: CircuitGenome,
    ) -> bool | None:
        """Handle a genome with an existing matching topology.

        Matching topology is determined using
        ``CircuitGenome.has_same_gates``.

        If the new genome dominates the existing duplicate, the existing
        genome is removed. If the existing genome dominates the new genome,
        the new genome is immediately discarded. If neither dominates the
        other, both are allowed to participate in environmental selection.

        Args:
            genome: Candidate genome.

        Returns:
            ``True`` if an existing duplicate was removed, ``False`` if the
            new genome should be discarded, or ``None`` if no decisive
            duplicate relationship was found.
        """
        for index, existing in enumerate(self.population):
            if not existing.has_same_gates(genome):
                continue

            if genome_dominates(
                genome,
                existing,
                self.objectives,
            ):
                logger.info(
                    "Removing genome "
                    f"{existing.genome_number} "
                    "because duplicate genome "
                    f"{genome.genome_number} "
                    "Pareto-dominates it."
                )

                del self.population[index]

                return True

            if genome_dominates(
                existing,
                genome,
                self.objectives,
            ):
                logger.info(
                    "Discarding genome "
                    f"{genome.genome_number} "
                    "because duplicate genome "
                    f"{existing.genome_number} "
                    "Pareto-dominates it."
                )

                return False

            return None

        return None

    def _run_tournament(
        self,
        excluded_genome_numbers: set[int],
    ) -> CircuitGenome:
        """Run one parent-selection tournament.

        Args:
            excluded_genome_numbers: Genome identifiers that must not be
                selected.

        Returns:
            Tournament-winning genome.

        Raises:
            RuntimeError: If no eligible genomes remain.
        """
        available = [
            genome
            for genome in self.population
            if genome.genome_number not in excluded_genome_numbers
        ]

        if not available:
            raise RuntimeError("No genomes are available for " "parent selection.")

        sample_size = min(
            self.tournament_size,
            len(available),
        )

        candidates = self.rng.sample(
            available,
            sample_size,
        )

        winner = candidates[0]

        for candidate in candidates[1:]:
            winner = self._tournament_winner(
                winner,
                candidate,
            )

        return winner

    def _log_population_state(
        self,
        genome: CircuitGenome,
        survived: bool,
    ) -> None:
        """Log information about a population insertion.

        Args:
            genome: Genome that was evaluated.
            survived: Whether the genome survived selection.
        """
        objective_text = ", ".join(
            (f"{objective.name}=" f"{genome.fitness[objective.name]}")
            for objective in self.objectives
        )

        logger.info(
            f"[{self.algorithm_name} insertion "
            f"{self.insertions}] "
            f"genome={genome.genome_number}, "
            f"survived={survived}, "
            f"population_size="
            f"{len(self.population)}, "
            f"pareto_rank="
            f"{genome.metadata.get('pareto_rank')}, "
            f"{objective_text}"
        )

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of the optimization algorithm.

        Returns:
            Algorithm name.
        """
        raise NotImplementedError

    @abstractmethod
    def _environmental_selection(
        self,
        population: Sequence[CircuitGenome],
        population_size: int,
    ) -> list[CircuitGenome]:
        """Select genomes that survive environmental selection.

        Args:
            population: Candidate population.
            population_size: Maximum number of survivors.

        Returns:
            Selected genomes.
        """
        raise NotImplementedError

    @abstractmethod
    def _refresh_selection_metadata(
        self,
    ) -> None:
        """Refresh algorithm-specific ranking and diversity metadata."""
        raise NotImplementedError

    @abstractmethod
    def _tournament_winner(
        self,
        left: CircuitGenome,
        right: CircuitGenome,
    ) -> CircuitGenome:
        """Select the preferred genome in a binary comparison.

        Args:
            left: First candidate.
            right: Second candidate.

        Returns:
            Preferred candidate.
        """
        raise NotImplementedError

    @abstractmethod
    def _representative_genome(
        self,
    ) -> CircuitGenome:
        """Return a representative Pareto-optimal genome.

        Returns:
            Representative genome.
        """
        raise NotImplementedError

    @abstractmethod
    def _parent_sort_key(
        self,
        genome: CircuitGenome,
    ) -> tuple:
        """Return the ordering key for crossover parents.

        Args:
            genome: Parent genome.

        Returns:
            Tuple used to sort parents.
        """
        raise NotImplementedError
