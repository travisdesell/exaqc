"""Base population implementation for multi-objective EXAQC strategies."""

from __future__ import annotations

import os
import random

from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.moo.pareto import (
    genome_dominates,
    validate_genome_fitness,
)
from src.evolution.population_strategy import PopulationStrategy
from src.utils.profiler import EXAQCProfiler


class MultiObjectivePopulationBase(PopulationStrategy):
    """Base class for steady-state multi-objective populations.

    This class implements the EXAQC ``PopulationStrategy`` interface and
    provides behavior shared by algorithms such as NSGA-II and NSGA-III.

    Subclasses are responsible only for algorithm-specific environmental
    selection, diversity metadata, tournament comparison, and representative
    Pareto-front selection.

    Args:
        max_population_size: Maximum number of genomes retained.
        objectives: Objective specifications used during optimization.
        tournament_size: Number of candidates sampled during parent
            tournament selection.
        out_dir: Directory used for saved artifacts.
        profiler: Optional EXAQC profiler. A profiler is created when this
            argument is ``None``.
        seed: Random seed used by parent selection.
        save_all_genomes: Whether every evaluated genome should be saved.
        save_pareto_front: Whether Pareto-front genomes should be saved when
            front membership changes.

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
        out_dir: str = "artifacts",
        profiler: Optional[EXAQCProfiler] = None,
        seed: int = 0,
        save_all_genomes: bool = True,
        save_pareto_front: bool = True,
    ) -> None:
        """Initialize a multi-objective population."""
        if max_population_size <= 0:
            raise ValueError("max_population_size must be greater than zero.")

        if len(objectives) < 2:
            raise ValueError(
                "Multi-objective optimization requires at least " "two objectives."
            )

        if tournament_size < 2:
            raise ValueError("tournament_size must be at least two.")

        self.max_population_size = int(max_population_size)

        self.objectives = list(objectives)

        self.tournament_size = int(tournament_size)

        self.out_dir = out_dir

        self.save_all_genomes = bool(save_all_genomes)

        self.save_pareto_front_enabled = bool(save_pareto_front)

        self.population: list[CircuitGenome] = []

        self.insertions = 0

        self.rng = random.Random(seed)

        self.profiler = profiler

        if self.profiler is None:
            self.profiler = EXAQCProfiler(
                out_dir=out_dir,
            )

        self._last_pareto_signature: tuple[int, ...] = ()

    def is_initializing(self) -> bool:
        """Determine whether the initial population is still being filled.

        Returns:
            ``True`` if the population has fewer genomes than its configured
            maximum size.
        """
        return len(self.population) < self.max_population_size

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
    ) -> Optional[CircuitGenome]:
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
        Optional[CircuitGenome],
        Optional[dict[str, object]],
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
        Optional[list[CircuitGenome]],
        Optional[dict[str, object]],
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
        **kwargs,
    ) -> bool:
        """Insert an evaluated genome into the population.

        The genome is temporarily combined with the current population.
        The concrete multi-objective algorithm then performs environmental
        selection to retain at most ``max_population_size`` genomes.

        Args:
            genome: Evaluated genome to insert.
            **kwargs: Additional population metadata.

        Returns:
            ``True`` if the genome survives environmental selection.
        """
        validate_genome_fitness(
            genome,
            self.objectives,
        )

        self.insertions += 1

        duplicate_result = self._handle_duplicate(genome)

        if duplicate_result is False:
            genome.metadata["insert_type"] = "discarded_duplicate"

            self._record_and_save(
                genome,
                survived=False,
            )

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

        genome.metadata["insert_type"] = "inserted" if survived else "discarded"

        self._record_and_save(
            genome,
            survived,
        )

        self._log_population_state(
            genome,
            survived,
        )

        return survived

    def _handle_duplicate(
        self,
        genome: CircuitGenome,
    ) -> Optional[bool]:
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

    def _record_and_save(
        self,
        genome: CircuitGenome,
        survived: bool,
    ) -> None:
        """Record profiling data and save population artifacts.

        Args:
            genome: Genome that was just evaluated.
            survived: Whether the genome survived environmental selection.
        """
        if self.profiler is not None:
            self.profiler.record(
                step=self.insertions,
                population=self.population,
            )

        if self.out_dir is None:
            return

        if self.save_all_genomes:
            genome.save_circuit(
                insert_type=("genome_inserted" if survived else "genome_discarded"),
                out_dir=os.path.join(
                    self.out_dir,
                    "all_genomes",
                ),
            )

        if self.save_pareto_front_enabled:
            self._save_pareto_front_if_changed()

        if self.profiler is not None:
            self.profiler.plot_single_run()

    def _save_pareto_front_if_changed(
        self,
    ) -> None:
        """Save the Pareto front when its membership changes."""
        pareto_front = self.get_pareto_front()

        signature = tuple(sorted(genome.genome_number for genome in pareto_front))

        if signature == self._last_pareto_signature:
            return

        self._last_pareto_signature = signature

        pareto_directory = os.path.join(
            self.out_dir,
            "pareto_front",
        )

        os.makedirs(
            pareto_directory,
            exist_ok=True,
        )

        logger.success(
            f"[{self.algorithm_name}] "
            "Pareto front updated with "
            f"{len(pareto_front)} genomes."
        )

        for index, genome in enumerate(pareto_front):
            genome.save_circuit(
                insert_type=(f"pareto_{index}"),
                out_dir=pareto_directory,
            )

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
