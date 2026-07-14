# src/evolution/moo/nsga2.py

from __future__ import annotations

import math

from collections.abc import Sequence
from typing import Optional

import numpy as np

from src.circuits.circuit import CircuitGenome
from src.evolution.multi_objective.base_population import (
    MultiObjectivePopulationBase,
)
from src.evolution.multi_objective.objective_spec import (
    ObjectiveSpec,
)
from src.evolution.multi_objective.pareto import (
    assign_pareto_ranks,
    objective_matrix,
)
from src.utils.profiler import EXAQCProfiler


def calculate_crowding_distance(
    population: Sequence[CircuitGenome],
    front: Sequence[int],
    objectives: Sequence[ObjectiveSpec],
) -> dict[int, float]:
    """Calculate NSGA-II crowding distances for one Pareto front.

    Args:
        population: Complete population.
        front: Population indices belonging to the front.
        objectives: Objective specifications.

    Returns:
        Mapping from population index to crowding distance.
    """
    distances = {
        population_index: 0.0
        for population_index in front
    }

    if not front:
        return distances

    if len(front) <= 2:
        return {
            population_index: float("inf")
            for population_index in front
        }

    values = objective_matrix(
        population,
        objectives,
    )

    front_array = np.asarray(
        front,
        dtype=np.int64,
    )
    front_values = values[front_array]

    for objective_index in range(front_values.shape[1]):
        local_order = np.argsort(
            front_values[:, objective_index],
            kind="stable",
        )

        first_population_index = front[
            int(local_order[0])
        ]
        last_population_index = front[
            int(local_order[-1])
        ]

        distances[first_population_index] = float("inf")
        distances[last_population_index] = float("inf")

        minimum = float(
            front_values[
                local_order[0],
                objective_index,
            ]
        )
        maximum = float(
            front_values[
                local_order[-1],
                objective_index,
            ]
        )

        objective_range = maximum - minimum

        if (
            not np.isfinite(objective_range)
            or objective_range <= 1e-12
        ):
            continue

        for position in range(1, len(local_order) - 1):
            local_index = int(local_order[position])
            population_index = front[local_index]

            if math.isinf(distances[population_index]):
                continue

            previous_value = float(
                front_values[
                    local_order[position - 1],
                    objective_index,
                ]
            )
            next_value = float(
                front_values[
                    local_order[position + 1],
                    objective_index,
                ]
            )

            distances[population_index] += (
                next_value - previous_value
            ) / objective_range

    return distances


class NSGA2Population(MultiObjectivePopulationBase):
    """Steady-state NSGA-II population for EXAQC."""

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
        super().__init__(
            max_population_size=max_population_size,
            objectives=objectives,
            tournament_size=tournament_size,
            out_dir=out_dir,
            profiler=profiler,
            seed=seed,
            save_all_genomes=save_all_genomes,
            save_pareto_front=save_pareto_front,
        )

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "nsga2"

    def _assign_rank_and_crowding(
        self,
        population: Sequence[CircuitGenome],
    ) -> list[list[int]]:
        """Assign Pareto rank and crowding distance."""
        fronts = assign_pareto_ranks(
            population,
            self.objectives,
        )

        for genome in population:
            genome.metadata.pop(
                "crowding_distance",
                None,
            )

        for front in fronts:
            distances = calculate_crowding_distance(
                population,
                front,
                self.objectives,
            )

            for population_index in front:
                population[population_index].metadata[
                    "crowding_distance"
                ] = float(
                    distances[population_index]
                )

        return fronts

    def _environmental_selection(
        self,
        population: Sequence[CircuitGenome],
        population_size: int,
    ) -> list[CircuitGenome]:
        """Select NSGA-II survivors."""
        population = list(population)
        fronts = self._assign_rank_and_crowding(population)

        if len(population) <= population_size:
            return population

        survivors: list[CircuitGenome] = []

        for front in fronts:
            remaining = population_size - len(survivors)

            if remaining <= 0:
                break

            if len(front) <= remaining:
                survivors.extend(
                    population[index]
                    for index in front
                )
                continue

            ordered_front = sorted(
                (
                    population[index]
                    for index in front
                ),
                key=lambda genome: (
                    float(
                        genome.metadata.get(
                            "crowding_distance",
                            0.0,
                        )
                    ),
                    -genome.genome_number,
                ),
                reverse=True,
            )

            survivors.extend(
                ordered_front[:remaining]
            )
            break

        self._assign_rank_and_crowding(survivors)
        return survivors

    def _refresh_selection_metadata(self) -> None:
        """Refresh Pareto-rank and crowding metadata."""
        if self.population:
            self._assign_rank_and_crowding(
                self.population
            )

    def _tournament_winner(
        self,
        left: CircuitGenome,
        right: CircuitGenome,
    ) -> CircuitGenome:
        """Choose a winner using NSGA-II crowded comparison."""
        left_rank = int(
            left.metadata.get(
                "pareto_rank",
                np.iinfo(np.int64).max,
            )
        )
        right_rank = int(
            right.metadata.get(
                "pareto_rank",
                np.iinfo(np.int64).max,
            )
        )

        if left_rank < right_rank:
            return left

        if right_rank < left_rank:
            return right

        left_distance = float(
            left.metadata.get(
                "crowding_distance",
                0.0,
            )
        )
        right_distance = float(
            right.metadata.get(
                "crowding_distance",
                0.0,
            )
        )

        if left_distance > right_distance:
            return left

        if right_distance > left_distance:
            return right

        return self.rng.choice([left, right])

    def _representative_genome(self) -> CircuitGenome:
        """Return a diverse member of the first Pareto front."""
        pareto_front = [
            genome
            for genome in self.population
            if genome.metadata.get("pareto_rank") == 0
        ]

        return max(
            pareto_front,
            key=lambda genome: (
                float(
                    genome.metadata.get(
                        "crowding_distance",
                        0.0,
                    )
                ),
                -genome.genome_number,
            ),
        )

    def _parent_sort_key(
        self,
        genome: CircuitGenome,
    ) -> tuple:
        """Sort selected parents by rank and crowding."""
        return (
            int(
                genome.metadata.get(
                    "pareto_rank",
                    np.iinfo(np.int64).max,
                )
            ),
            -float(
                genome.metadata.get(
                    "crowding_distance",
                    0.0,
                )
            ),
            genome.genome_number,
        )