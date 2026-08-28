# src/evolution/population/multi_objective_population.py

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.multi_objective.nsga2 import (
    nsga2_better,
    nsga2_environmental_selection,
    rank_and_crowding,
)
from src.evolution.multi_objective.nsga3 import (
    choose_reference_divisions,
    das_dennis_reference_directions,
    nsga3_environmental_selection,
    nsga3_tournament,
)
from src.evolution.multi_objective.objective_spec import ObjectiveSpec
from src.evolution.multi_objective.pareto import (
    assign_pareto_metadata,
)
from src.evolution.population_strategy import PopulationStrategy


MultiObjectiveAlgorithm = Literal["nsga2", "nsga3"]


class MultiObjectivePopulation(PopulationStrategy):
    """Steady-state multi-objective EXAQC population.

    Args:
        max_population_size: Maximum number of retained genomes.
        objectives: Ordered objective definitions.
        algorithm: Multi-objective survival algorithm.
        tournament_size: Number of candidates used for parent selection.
        reference_divisions: NSGA-III simplex divisions. When omitted, an
            appropriate value is selected automatically.
        seed: Random seed.
        out_dir: Artifact output directory.
        profiler: Optional EXAQC profiler.
    """

    def __init__(
        self,
        *,
        max_population_size: int,
        objectives: Sequence[ObjectiveSpec],
        algorithm: MultiObjectiveAlgorithm = "nsga2",
        tournament_size: int = 2,
        reference_divisions: Optional[int] = None,
        seed: int = 0,
        out_dir: str = "artifacts",
        profiler=None,
    ):
        super().__init__()

        if max_population_size <= 0:
            raise ValueError(
                "max_population_size must be positive."
            )

        if len(objectives) < 2:
            raise ValueError(
                "Multi-objective optimization requires at least "
                "two objectives."
            )

        if algorithm not in {"nsga2", "nsga3"}:
            raise ValueError(
                f"Unknown multi-objective algorithm '{algorithm}'."
            )

        self.max_population_size = int(max_population_size)
        self.objectives = list(objectives)
        self.algorithm = algorithm
        self.tournament_size = max(2, int(tournament_size))
        self.out_dir = out_dir
        self.profiler = profiler
        self.population: list[CircuitGenome] = []
        self.rng = random.Random(seed)

        self.reference_directions: np.ndarray | None = None

        if algorithm == "nsga3":
            divisions = reference_divisions

            if divisions is None:
                divisions = choose_reference_divisions(
                    n_objectives=len(self.objectives),
                    target_population_size=self.max_population_size,
                )

            self.reference_directions = (
                das_dennis_reference_directions(
                    n_objectives=len(self.objectives),
                    divisions=divisions,
                )
            )

            logger.info(
                f"NSGA-III generated "
                f"{len(self.reference_directions)} reference directions "
                f"with divisions={divisions}."
            )

    def __len__(self) -> int:
        """Return the current population size."""
        return len(self.population)

    def is_initializing(self) -> bool:
        """Return whether the initial population is still being filled."""
        return len(self.population) < self.max_population_size

    def insert(self, genome: CircuitGenome) -> bool:
        """Insert an evaluated genome and perform environmental selection.

        Args:
            genome: Evaluated offspring genome.

        Returns:
            ``True`` when the inserted genome survives selection.
        """
        if genome.fitness is None:
            raise ValueError(
                f"Genome {genome.genome_number} has no fitness."
            )

        combined = self.population + [genome]

        if self.algorithm == "nsga2":
            survivors = nsga2_environmental_selection(
                combined,
                population_size=self.max_population_size,
                objectives=self.objectives,
            )
        else:
            survivors = nsga3_environmental_selection(
                combined,
                population_size=self.max_population_size,
                objectives=self.objectives,
                reference_directions=self.reference_directions,
                rng=self.rng,
            )

        survived = any(
            candidate.genome_number == genome.genome_number
            for candidate in survivors
        )

        self.population = survivors

        logger.info(
            f"Inserted genome={genome.genome_number} "
            f"survived={survived} "
            f"population_size={len(self.population)} "
            f"pareto_rank={genome.metadata.get('pareto_rank')}"
        )

        return survived

    def add(self, genome: CircuitGenome) -> bool:
        """Alias for insertion used by some population APIs."""
        return self.insert(genome)

    def _refresh_selection_metadata(self) -> None:
        """Refresh Pareto and diversity metadata."""
        if self.algorithm == "nsga2":
            rank_and_crowding(
                self.population,
                self.objectives,
            )
        else:
            assign_pareto_metadata(
                self.population,
                self.objectives,
            )

            if len(self.population) > 0:
                refreshed = nsga3_environmental_selection(
                    self.population,
                    population_size=len(self.population),
                    objectives=self.objectives,
                    reference_directions=self.reference_directions,
                    rng=self.rng,
                )
                self.population = refreshed

    def get_parent(self, **kwargs) -> CircuitGenome:
        """Select a parent using multi-objective tournament selection."""
        if not self.population:
            raise RuntimeError(
                "Cannot select a parent from an empty population."
            )

        self._refresh_selection_metadata()

        candidates = [
            self.rng.choice(self.population)
            for _ in range(self.tournament_size)
        ]

        winner = candidates[0]

        for candidate in candidates[1:]:
            if self.algorithm == "nsga2":
                winner = nsga2_better(winner, candidate)
            else:
                winner = nsga3_tournament(winner, candidate)

        return winner

    def get_best_genome(self) -> CircuitGenome:
        """Return a representative genome from the first Pareto front.

        Multi-objective optimization has no unique best genome. This method
        returns a deterministic representative for compatibility with older
        EXAQC code.
        """
        front = self.get_pareto_front()

        if not front:
            raise RuntimeError("Population is empty.")

        if self.algorithm == "nsga2":
            return max(
                front,
                key=lambda genome: (
                    genome.metadata.get(
                        "crowding_distance",
                        0.0,
                    ),
                    -genome.genome_number,
                ),
            )

        return min(
            front,
            key=lambda genome: (
                genome.metadata.get(
                    "reference_distance",
                    np.inf,
                ),
                genome.genome_number,
            ),
        )

    def get_pareto_front(self) -> list[CircuitGenome]:
        """Return the current non-dominated front."""
        if not self.population:
            return []

        self._refresh_selection_metadata()

        return [
            genome
            for genome in self.population
            if genome.metadata.get("pareto_rank") == 0
        ]