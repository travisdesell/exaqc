"""Base population implementation for multi-objective EXAQC strategies."""

from __future__ import annotations

import argparse
import math
import random
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.moo.pareto import (
    genome_dominates,
    objective_vector,
    validate_genome_fitness,
)
from src.evolution.population_strategy import PopulationStrategy, mark_discarded

if TYPE_CHECKING:
    from src.utils.restart import RestartState


class MultiObjectivePopulationBase(PopulationStrategy):
    """Base class for steady-state multi-objective populations.

    This class implements the EXAQC ``PopulationStrategy`` interface and
    provides behavior shared by algorithms such as NSGA-II and NSGA-III. Every
    inserted genome is first combined with the current population and the
    concrete algorithm's environmental selection then keeps at most
    ``max_population_size`` of them, so the population is an (N + 1) -> N
    steady-state version of the generational algorithm.

    A population strategy only selects and ranks genomes: everything written to
    disk goes through the run's :class:`~src.utils.genome_archive.GenomeArchive`.

    Subclasses are responsible only for algorithm-specific environmental
    selection, diversity metadata, and tournament comparison.

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

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds the flags every multi-objective population shares to a parser.

        Args:
            parser: The (sub-)parser to add the arguments to.

        Returns:
            None. Mutates ``parser`` by adding ``--max_population_size`` and
            ``--tournament_size``.
        """

        parser.add_argument(
            "--max_population_size",
            type=int,
            default=30,
            help="Maximum number of genomes retained in the multi-objective population.",
        )
        parser.add_argument(
            "--tournament_size",
            type=int,
            default=2,
            help="Number of genomes sampled for each parent-selection tournament.",
        )

    def __init__(
        self,
        max_population_size: int,
        objectives: Sequence[ObjectiveSpec],
        tournament_size: int = 2,
        seed: int = 0,
    ) -> None:
        """Initialize a multi-objective population.

        Args:
            max_population_size: Maximum number of genomes retained.
            objectives: Objective specifications used during optimization.
            tournament_size: Number of candidates sampled during parent
                tournament selection.
            seed: Random seed used by parent selection.

        Returns:
            None. Sets the population's configuration, an empty
            ``population`` and its random number generator.

        Raises:
            ValueError: If ``max_population_size`` is not positive.
            ValueError: If fewer than two objectives are supplied.
            ValueError: If ``tournament_size`` is smaller than two.
        """
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
        self._last_pareto_signature: tuple[int, ...] = ()

    def run_info(self) -> dict[str, Any]:
        """Describes the strategy's configuration, for the run's archive.

        Returns:
            The algorithm name and the objectives it optimizes (each with its
            fitness key and whether it is minimized or maximized), so tools
            reading the archive can draw the run's Pareto front.
        """

        return {
            "multi_objective": {
                "algorithm": self.algorithm_name,
                "objectives": [
                    {
                        "name": objective.name,
                        "sign": objective.sign,
                        "direction": "minimize" if objective.sign > 0 else "maximize",
                    }
                    for objective in self.objectives
                ],
            }
        }

    def restore(self, state: "RestartState") -> None:
        """Takes back the population a stopped run held, so its search continues.

        Pareto ranks and the algorithm's diversity metadata are recomputed from
        the restored genomes' fitness, so nothing beyond the genomes themselves
        needs to have been recorded.

        Args:
            state: The stopped run's state (see :mod:`src.utils.restart`).

        Returns:
            None. Refills ``population`` with the run's genomes, restores the
            insertion count and refreshes the selection metadata.
        """

        self.population = list(state.population)
        self.insertions = state.inserted_genomes
        self._refresh_selection_metadata()
        self._last_pareto_signature = self._pareto_signature()

        logger.info(
            "restored a {} population of {} genomes after {} insertions",
            self.algorithm_name,
            len(self.population),
            self.insertions,
        )

    def is_initializing(self) -> bool:
        """Determine whether the initial population is still being filled.

        Returns:
            ``True`` if the population has fewer genomes than its configured
            maximum size.
        """
        return len(self.population) < self.max_population_size

    def get_population(self) -> list[CircuitGenome]:
        """Returns the genomes in the population, best first.

        Genomes are ordered by Pareto rank, then by the algorithm's diversity
        preference (see :meth:`_parent_sort_key`).

        Returns:
            A new list of the population's genomes.
        """

        self._refresh_selection_metadata()
        return sorted(self.population, key=self._parent_sort_key)

    def get_pareto_front(self) -> list[CircuitGenome]:
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

    def get_best_genome(self) -> CircuitGenome | None:
        """Return a representative genome from the Pareto front.

        A multi-objective population has no unique best genome, so the Pareto
        front genome with the best value of the *first* objective is used (for
        example the lowest loss when the objectives are ``loss`` and
        ``n_enabled_gates``). Genome number breaks ties.

        Returns:
            The representative Pareto-optimal genome, or ``None`` when the
            population is empty.
        """
        front = self.get_pareto_front()

        if not front:
            return None

        return min(front, key=self._primary_key)

    def get_parent(
        self, **kwargs: Any
    ) -> tuple[CircuitGenome | None, dict[str, Any] | None]:
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
            "selection_algorithm": self.algorithm_name,
            "parent_pareto_rank": parent.metadata.get("pareto_rank"),
        }

        return parent, metadata

    def get_parents(
        self, n_parents: int = 2, **kwargs: Any
    ) -> tuple[list[CircuitGenome] | None, dict[str, Any] | None]:
        """Select multiple unique parents using tournament selection.

        Args:
            n_parents: Number of unique parents requested.
            **kwargs: Additional population-strategy arguments.

        Returns:
            Selected parents (best first) and selection metadata. Returns
            ``(None, None)`` when there are insufficient genomes.

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
            parent = self._run_tournament(excluded_genome_numbers=selected_numbers)
            parents.append(parent)
            selected_numbers.add(parent.genome_number)

        parents.sort(key=self._parent_sort_key)

        metadata = {
            "selection_algorithm": self.algorithm_name,
            "parent_pareto_ranks": [
                parent.metadata.get("pareto_rank") for parent in parents
            ],
        }

        return parents, metadata

    def insert_genome(self, genome: CircuitGenome, **kwargs: Any) -> bool:
        """Insert an evaluated genome into the population.

        The genome is temporarily combined with the current population and the
        concrete multi-objective algorithm then performs environmental
        selection to retain at most ``max_population_size`` genomes. The
        genome's ``insert_type`` metadata records the outcome: ``global_best``
        when it becomes the representative genome (see
        :meth:`get_best_genome`), ``inserted`` when it survives, or
        ``discarded`` with a ``discard_reason`` (see
        :func:`~src.evolution.population_strategy.mark_discarded`).

        Args:
            genome: Evaluated genome to insert.
            **kwargs: Additional population metadata.

        Returns:
            True: every genome is recorded as evaluated, whether it was kept or
            discarded.

        Raises:
            ValueError: If the genome has not been evaluated.
            KeyError: If an objective is missing from the genome's fitness.
            TypeError: If an objective value is not numeric.
        """
        validate_genome_fitness(genome, self.objectives)

        self.insertions += 1

        duplicate = self._dominating_duplicate(genome)
        if duplicate is not None:
            mark_discarded(genome, "duplicate_of_better", duplicate.genome_number)
            self._log_population_state(genome, survived=False)
            return True

        previous_best = self.get_best_genome()

        survivors = self._environmental_selection(
            self.population + [genome],
            self.max_population_size,
        )
        survived = any(candidate is genome for candidate in survivors)

        self.population = survivors
        self._refresh_selection_metadata()

        if not survived:
            # it lost to the population's least preferred survivor
            worst = max(self.population, key=self._parent_sort_key)
            mark_discarded(genome, "worse_than_population", worst.genome_number)
        else:
            genome.metadata["insert_type"] = "inserted"
            best = self.get_best_genome()
            if best is genome and best is not previous_best:
                genome.metadata["insert_type"] = "global_best"
                logger.success(
                    f"[{self.algorithm_name} insertion {self.insertions}] genome "
                    f"{genome.genome_number} is the new best Pareto front genome by "
                    f"{self.objectives[0].name}: {genome.fitness}"
                )

        self._log_front_if_changed()
        self._log_population_state(genome, survived)

        return True

    def _dominating_duplicate(self, genome: CircuitGenome) -> CircuitGenome | None:
        """Resolve a genome whose enabled gates match one already held.

        Matching topology is determined using
        ``CircuitGenome.has_same_gates``. If the new genome Pareto-dominates
        the existing duplicate, the existing genome is removed so the new one
        can take its place. If neither dominates the other, both take part in
        environmental selection.

        Args:
            genome: Candidate genome.

        Returns:
            The held duplicate that Pareto-dominates ``genome`` (so ``genome``
            should be discarded), or ``None`` when there is none.
        """
        for index, existing in enumerate(self.population):
            if not existing.has_same_gates(genome):
                continue

            if genome_dominates(genome, existing, self.objectives):
                logger.info(
                    f"removing genome {existing.genome_number} because duplicate "
                    f"genome {genome.genome_number} Pareto-dominates it."
                )
                del self.population[index]
                return None

            if genome_dominates(existing, genome, self.objectives):
                logger.info(
                    f"discarding genome {genome.genome_number} because duplicate "
                    f"genome {existing.genome_number} Pareto-dominates it."
                )
                return existing

            return None

        return None

    def _primary_key(self, genome: CircuitGenome) -> tuple[float, int]:
        """Orders genomes by their first objective, in minimization form.

        Args:
            genome: The genome to order.

        Returns:
            The genome's transformed first objective (non-finite values sort
            last) and its genome number as a tie breaker.
        """

        value = float(objective_vector(genome, self.objectives)[0])
        if not math.isfinite(value):
            value = math.inf
        return value, genome.genome_number

    def _pareto_signature(self) -> tuple[int, ...]:
        """Identifies the current Pareto front by its genome numbers.

        Returns:
            The sorted genome numbers of the Pareto front.
        """

        return tuple(sorted(genome.genome_number for genome in self.get_pareto_front()))

    def _run_tournament(self, excluded_genome_numbers: set[int]) -> CircuitGenome:
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
            raise RuntimeError("No genomes are available for parent selection.")

        sample_size = min(self.tournament_size, len(available))
        candidates = self.rng.sample(available, sample_size)

        winner = candidates[0]
        for candidate in candidates[1:]:
            winner = self._tournament_winner(winner, candidate)

        return winner

    def _log_front_if_changed(self) -> None:
        """Logs the Pareto front whenever its membership changes.

        Returns:
            None. Updates ``_last_pareto_signature``.
        """
        signature = self._pareto_signature()

        if signature == self._last_pareto_signature:
            return

        self._last_pareto_signature = signature

        logger.success(
            f"[{self.algorithm_name}] Pareto front updated: {len(signature)} genomes "
            f"{list(signature)}."
        )

    def _log_population_state(self, genome: CircuitGenome, survived: bool) -> None:
        """Log information about a population insertion.

        Args:
            genome: Genome that was evaluated.
            survived: Whether the genome survived selection.

        Returns:
            None.
        """
        objective_text = ", ".join(
            f"{objective.name}={genome.fitness[objective.name]}"
            for objective in self.objectives
        )

        logger.info(
            f"[{self.algorithm_name} insertion {self.insertions}] "
            f"genome={genome.genome_number}, survived={survived}, "
            f"population_size={len(self.population)}, "
            f"pareto_rank={genome.metadata.get('pareto_rank')}, {objective_text}"
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
    def _refresh_selection_metadata(self) -> None:
        """Refresh algorithm-specific ranking and diversity metadata.

        Returns:
            None. Updates the metadata of every genome in ``population``.
        """
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
    def _parent_sort_key(self, genome: CircuitGenome) -> tuple[float, float, int]:
        """Return the ordering key for crossover parents.

        Args:
            genome: Parent genome.

        Returns:
            Tuple used to sort parents, most preferred first.
        """
        raise NotImplementedError
