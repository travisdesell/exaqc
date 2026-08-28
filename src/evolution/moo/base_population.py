# src/evolution/moo/base_population.py

from __future__ import annotations

import os
import random

from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.objective_spec import (
    ObjectiveSpec,
)
from src.evolution.moo.pareto import (
    assign_pareto_ranks,
    genome_dominates,
    validate_genome_fitness,
)
from src.evolution.population_strategy import PopulationStrategy
from src.utils.profiler import EXAQCProfiler


class MultiObjectivePopulationBase(PopulationStrategy):
    """Base class for steady-state multi-objective populations.

    Args:
        max_population_size: Maximum number of retained genomes.
        objectives: Multi-objective fitness specifications.
        tournament_size: Number of candidates used in parent selection.
        out_dir: Directory used for saved artifacts.
        profiler: Optional EXAQC profiler.
        seed: Random seed used by population selection.
        save_all_genomes: Whether every evaluated genome should be saved.
        save_pareto_front: Whether the current Pareto front should be saved.
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
    ):
        if max_population_size <= 0:
            raise ValueError(
                "max_population_size must be greater than zero."
            )

        if len(objectives) < 2:
            raise ValueError(
                "Multi-objective optimization requires at least "
                "two objectives."
            )

        if tournament_size < 2:
            raise ValueError(
                "tournament_size must be at least two."
            )

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

        self.best_accuracy_genome: Optional[CircuitGenome] = None
        self.best_return_genome: Optional[CircuitGenome] = None

        self._last_pareto_signature: tuple[int, ...] = ()

    def is_initializing(self) -> bool:
        """Return whether the initial population is still being filled."""
        return len(self.population) < self.max_population_size

    def get_pareto_front(self) -> list[CircuitGenome]:
        """Return the current non-dominated front.

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

    def get_best_genome(self) -> Optional[CircuitGenome]:
        """Return one representative genome from the Pareto front.

        Multi-objective optimization has no unique best genome. Subclasses
        choose a representative solution for compatibility with existing
        EXAQC execution code.

        Returns:
            Representative genome, or ``None`` for an empty population.
        """
        if not self.population:
            return None

        self._refresh_selection_metadata()
        return self._representative_genome()

    def get_parent(
        self,
        **kwargs,
    ) -> tuple[Optional[CircuitGenome], Optional[dict[str, object]]]:
        """Select one parent using multi-objective tournament selection.

        Returns:
            Selected genome and child metadata.
        """
        if not self.population:
            return None, None

        self._refresh_selection_metadata()

        winner = self._run_tournament(
            excluded_genome_numbers=set(),
        )

        metadata = {
            "selection_algorithm": self.algorithm_name,
            "parent_pareto_rank": winner.metadata.get(
                "pareto_rank"
            ),
        }

        return winner, metadata

    def get_parents(
        self,
        n_parents: int = 2,
        **kwargs,
    ) -> tuple[
        Optional[list[CircuitGenome]],
        Optional[dict[str, object]],
    ]:
        """Select unique parents using tournament selection.

        Args:
            n_parents: Number of unique parents requested.

        Returns:
            Selected parents and child metadata.
        """
        if n_parents <= 0:
            raise ValueError("n_parents must be positive.")

        if len(self.population) < n_parents:
            return None, None

        self._refresh_selection_metadata()

        parents: list[CircuitGenome] = []
        selected_numbers: set[int] = set()

        while len(parents) < n_parents:
            parent = self._run_tournament(
                excluded_genome_numbers=selected_numbers,
            )

            parents.append(parent)
            selected_numbers.add(parent.genome_number)

        parents.sort(
            key=self._parent_sort_key,
        )

        metadata = {
            "selection_algorithm": self.algorithm_name,
            "parent_pareto_ranks": [
                parent.metadata.get("pareto_rank")
                for parent in parents
            ],
        }

        return parents, metadata

    def insert_genome(
        self,
        genome: CircuitGenome,
        **kwargs,
    ) -> bool:
        """Insert an evaluated genome using steady-state selection.

        The new genome is added to the current population and environmental
        selection retains at most ``max_population_size`` genomes.

        Args:
            genome: Evaluated genome.
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
            self._record_and_save(genome, survived=False)
            return False

        combined_population = self.population + [genome]

        survivors = self._environmental_selection(
            combined_population,
            self.max_population_size,
        )

        survived = any(
            survivor.genome_number == genome.genome_number
            for survivor in survivors
        )

        self.population = survivors
        self._refresh_selection_metadata()

        if survived:
            genome.metadata["insert_type"] = "inserted"
        else:
            genome.metadata["insert_type"] = "discarded"

        self._update_scalar_best_trackers(genome)
        self._record_and_save(genome, survived=survived)
        self._log_population_state(genome, survived)

        return survived

    def _handle_duplicate(
        self,
        genome: CircuitGenome,
    ) -> Optional[bool]:
        """Handle genomes with identical enabled gate innovations.

        Args:
            genome: Candidate genome.

        Returns:
            ``False`` if the new genome should immediately be discarded.
            ``True`` if an existing duplicate was removed.
            ``None`` if no duplicate was found or both should be retained.
        """
        for index, existing in enumerate(self.population):
            if not existing.has_same_gates(genome):
                continue

            new_dominates = genome_dominates(
                genome,
                existing,
                self.objectives,
            )
            existing_dominates = genome_dominates(
                existing,
                genome,
                self.objectives,
            )

            if new_dominates:
                logger.info(
                    f"Removing duplicate genome "
                    f"{existing.genome_number}; new genome "
                    f"{genome.genome_number} dominates it."
                )
                del self.population[index]
                return True

            if existing_dominates:
                logger.info(
                    f"Discarding duplicate genome "
                    f"{genome.genome_number}; existing genome "
                    f"{existing.genome_number} dominates it."
                )
                return False

            # Neither dominates the other. They may have the same structure
            # but different trained parameters and different trade-offs.
            return None

        return None

    def _run_tournament(
        self,
        excluded_genome_numbers: set[int],
    ) -> CircuitGenome:
        """Run one tournament while excluding selected parents."""
        available = [
            genome
            for genome in self.population
            if genome.genome_number not in excluded_genome_numbers
        ]

        if not available:
            raise RuntimeError(
                "No available genomes remain for parent selection."
            )

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

    def _update_scalar_best_trackers(
        self,
        genome: CircuitGenome,
    ) -> None:
        """Track best accuracy and best return for convenience."""
        if "test_acc" in genome.fitness:
            if (
                self.best_accuracy_genome is None
                or float(genome.fitness["test_acc"])
                > float(
                    self.best_accuracy_genome.fitness["test_acc"]
                )
            ):
                self.best_accuracy_genome = genome

                logger.success(
                    f"[multi-objective insertion {self.insertions}] "
                    f"New best accuracy genome "
                    f"{genome.genome_number}: "
                    f"{genome.fitness['test_acc']}"
                )

                if self.out_dir is not None:
                    genome.save_circuit(
                        insert_type="best_accuracy",
                        out_dir=self.out_dir,
                    )

        if "eval_return_mean" in genome.fitness:
            if (
                self.best_return_genome is None
                or float(genome.fitness["eval_return_mean"])
                > float(
                    self.best_return_genome.fitness[
                        "eval_return_mean"
                    ]
                )
            ):
                self.best_return_genome = genome

                logger.success(
                    f"[multi-objective insertion {self.insertions}] "
                    f"New best return genome "
                    f"{genome.genome_number}: "
                    f"{genome.fitness['eval_return_mean']}"
                )

                if self.out_dir is not None:
                    genome.save_circuit(
                        insert_type="best_return",
                        out_dir=self.out_dir,
                    )

    def _record_and_save(
        self,
        genome: CircuitGenome,
        survived: bool,
    ) -> None:
        """Record profiler information and save artifacts."""
        if self.profiler is not None:
            self.profiler.record(
                step=self.insertions,
                population=self.population,
            )

        if self.out_dir is None:
            return

        if self.save_all_genomes:
            genome.save_circuit(
                insert_type=(
                    "genome_inserted"
                    if survived
                    else "genome_discarded"
                ),
                out_dir=os.path.join(
                    self.out_dir,
                    "all_genomes",
                ),
            )

        if self.save_pareto_front_enabled:
            self._save_pareto_front_if_changed()

        if self.profiler is not None:
            self.profiler.plot_single_run()

    def _save_pareto_front_if_changed(self) -> None:
        """Save the Pareto front only when its membership changes."""
        pareto_front = self.get_pareto_front()

        signature = tuple(
            sorted(
                genome.genome_number
                for genome in pareto_front
            )
        )

        if signature == self._last_pareto_signature:
            return

        self._last_pareto_signature = signature

        pareto_dir = os.path.join(
            self.out_dir,
            "pareto_front",
        )

        os.makedirs(
            pareto_dir,
            exist_ok=True,
        )

        logger.success(
            f"Pareto front updated: "
            f"{len(pareto_front)} genomes."
        )

        for index, pareto_genome in enumerate(pareto_front):
            pareto_genome.save_circuit(
                insert_type=f"pareto_{index}",
                out_dir=pareto_dir,
            )

    def _log_population_state(
        self,
        genome: CircuitGenome,
        survived: bool,
    ) -> None:
        """Log multi-objective insertion information."""
        objective_text = ", ".join(
            f"{objective.name}="
            f"{genome.fitness[objective.name]}"
            for objective in self.objectives
        )

        logger.info(
            f"[{self.algorithm_name} insertion {self.insertions}] "
            f"genome={genome.genome_number} "
            f"survived={survived} "
            f"population_size={len(self.population)} "
            f"pareto_rank="
            f"{genome.metadata.get('pareto_rank')} "
            f"{objective_text}"
        )

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the population algorithm name."""
        raise NotImplementedError

    @abstractmethod
    def _environmental_selection(
        self,
        population: Sequence[CircuitGenome],
        population_size: int,
    ) -> list[CircuitGenome]:
        """Select surviving genomes."""
        raise NotImplementedError

    @abstractmethod
    def _refresh_selection_metadata(self) -> None:
        """Refresh rank and diversity metadata."""
        raise NotImplementedError

    @abstractmethod
    def _tournament_winner(
        self,
        left: CircuitGenome,
        right: CircuitGenome,
    ) -> CircuitGenome:
        """Choose a tournament winner."""
        raise NotImplementedError

    @abstractmethod
    def _representative_genome(self) -> CircuitGenome:
        """Return a representative Pareto-optimal genome."""
        raise NotImplementedError

    @abstractmethod
    def _parent_sort_key(
        self,
        genome: CircuitGenome,
    ) -> tuple:
        """Return a key used to order selected parents."""
        raise NotImplementedError