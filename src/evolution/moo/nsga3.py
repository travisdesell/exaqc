# src/evolution/moo/nsga3.py

from __future__ import annotations

import math

from collections.abc import Sequence
from typing import Optional

import numpy as np
from loguru import logger

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


def _integer_compositions(
    total: int,
    parts: int,
) -> list[list[int]]:
    """Generate ordered integer compositions.

    Args:
        total: Integer total.
        parts: Number of composition entries.

    Returns:
        All non-negative ordered compositions.
    """
    if parts == 1:
        return [[total]]

    compositions = []

    for value in range(total + 1):
        for remainder in _integer_compositions(
            total - value,
            parts - 1,
        ):
            compositions.append(
                [value] + remainder
            )

    return compositions


def generate_reference_directions(
    n_objectives: int,
    divisions: int,
) -> np.ndarray:
    """Generate Das-Dennis reference directions.

    Args:
        n_objectives: Number of objectives.
        divisions: Number of simplex divisions.

    Returns:
        Reference-direction matrix.
    """
    if n_objectives < 2:
        raise ValueError(
            "NSGA-III requires at least two objectives."
        )

    if divisions < 1:
        raise ValueError(
            "reference divisions must be at least one."
        )

    compositions = _integer_compositions(
        divisions,
        n_objectives,
    )

    return np.asarray(
        compositions,
        dtype=np.float64,
    ) / float(divisions)


def choose_reference_divisions(
    n_objectives: int,
    population_size: int,
) -> int:
    """Choose a reference division count automatically."""
    divisions = 1

    while (
        math.comb(
            divisions + n_objectives - 1,
            n_objectives - 1,
        )
        < population_size
    ):
        divisions += 1

    return divisions


def sanitize_objective_matrix(
    values: np.ndarray,
) -> np.ndarray:
    """Replace non-finite objective values with finite penalties."""
    values = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    for objective_index in range(values.shape[1]):
        column = values[:, objective_index]
        finite_mask = np.isfinite(column)

        if not np.any(finite_mask):
            values[:, objective_index] = 1.0
            continue

        finite_values = column[finite_mask]
        penalty = float(np.max(finite_values)) + max(
            1.0,
            float(np.ptp(finite_values)),
        )

        column[~finite_mask] = penalty
        values[:, objective_index] = column

    return values


def normalize_objectives(
    values: np.ndarray,
) -> np.ndarray:
    """Normalize objective values for reference association."""
    values = sanitize_objective_matrix(values)

    ideal_point = np.min(
        values,
        axis=0,
    )

    translated = values - ideal_point

    objective_ranges = np.max(
        translated,
        axis=0,
    )

    objective_ranges = np.where(
        objective_ranges > 1e-12,
        objective_ranges,
        1.0,
    )

    return translated / objective_ranges


def associate_reference_directions(
    normalized_values: np.ndarray,
    reference_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate each solution with a reference direction.

    Returns:
        Reference-direction indices and perpendicular distances.
    """
    direction_norms = np.linalg.norm(
        reference_directions,
        axis=1,
        keepdims=True,
    )

    unit_directions = reference_directions / np.maximum(
        direction_norms,
        1e-12,
    )

    projection_lengths = (
        normalized_values
        @ unit_directions.T
    )

    projections = (
        projection_lengths[:, :, None]
        * unit_directions[None, :, :]
    )

    residuals = (
        normalized_values[:, None, :]
        - projections
    )

    distances = np.linalg.norm(
        residuals,
        axis=2,
    )

    nearest_direction = np.argmin(
        distances,
        axis=1,
    )

    nearest_distance = distances[
        np.arange(len(normalized_values)),
        nearest_direction,
    ]

    return nearest_direction, nearest_distance


class NSGA3Population(MultiObjectivePopulationBase):
    """Steady-state NSGA-III population for EXAQC."""

    def __init__(
        self,
        max_population_size: int,
        objectives: Sequence[ObjectiveSpec],
        tournament_size: int = 2,
        reference_divisions: Optional[int] = None,
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

        if reference_divisions is None:
            reference_divisions = choose_reference_divisions(
                len(self.objectives),
                max_population_size,
            )

        self.reference_divisions = int(reference_divisions)

        self.reference_directions = (
            generate_reference_directions(
                n_objectives=len(self.objectives),
                divisions=self.reference_divisions,
            )
        )

        logger.info(
            f"Created NSGA-III population with "
            f"{len(self.reference_directions)} reference directions "
            f"using divisions={self.reference_divisions}."
        )

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return "nsga3"

    def _assign_reference_metadata(
        self,
        population: Sequence[CircuitGenome],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Associate the population with reference directions."""
        if not population:
            return (
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=np.float64),
            )

        values = objective_matrix(
            population,
            self.objectives,
        )

        normalized = normalize_objectives(values)

        reference_indices, reference_distances = (
            associate_reference_directions(
                normalized,
                self.reference_directions,
            )
        )

        for index, genome in enumerate(population):
            genome.metadata["reference_direction"] = int(
                reference_indices[index]
            )
            genome.metadata["reference_distance"] = float(
                reference_distances[index]
            )

        return reference_indices, reference_distances

    def _environmental_selection(
        self,
        population: Sequence[CircuitGenome],
        population_size: int,
    ) -> list[CircuitGenome]:
        """Select survivors using NSGA-III niching."""
        population = list(population)

        fronts = assign_pareto_ranks(
            population,
            self.objectives,
        )

        reference_indices, reference_distances = (
            self._assign_reference_metadata(population)
        )

        if len(population) <= population_size:
            return population

        selected_indices: list[int] = []
        splitting_front: Optional[list[int]] = None

        for front in fronts:
            if (
                len(selected_indices) + len(front)
                <= population_size
            ):
                selected_indices.extend(front)
            else:
                splitting_front = list(front)
                break

        if splitting_front is None:
            survivors = [
                population[index]
                for index in selected_indices[:population_size]
            ]
            assign_pareto_ranks(
                survivors,
                self.objectives,
            )
            self._assign_reference_metadata(survivors)
            return survivors

        niche_counts = np.zeros(
            len(self.reference_directions),
            dtype=np.int64,
        )

        for population_index in selected_indices:
            niche_counts[
                reference_indices[population_index]
            ] += 1

        candidates_by_reference: dict[int, list[int]] = {
            reference_index: []
            for reference_index in range(
                len(self.reference_directions)
            )
        }

        for population_index in splitting_front:
            reference_index = int(
                reference_indices[population_index]
            )

            candidates_by_reference[
                reference_index
            ].append(population_index)

        remaining = population_size - len(selected_indices)

        while remaining > 0:
            available_references = [
                reference_index
                for reference_index, candidates
                in candidates_by_reference.items()
                if candidates
            ]

            if not available_references:
                break

            minimum_count = min(
                niche_counts[reference_index]
                for reference_index in available_references
            )

            least_occupied_references = [
                reference_index
                for reference_index in available_references
                if niche_counts[reference_index]
                == minimum_count
            ]

            chosen_reference = self.rng.choice(
                least_occupied_references
            )

            candidates = candidates_by_reference[
                chosen_reference
            ]

            if niche_counts[chosen_reference] == 0:
                chosen_index = min(
                    candidates,
                    key=lambda population_index: (
                        reference_distances[
                            population_index
                        ],
                        population[
                            population_index
                        ].genome_number,
                    ),
                )
            else:
                chosen_index = self.rng.choice(
                    candidates
                )

            selected_indices.append(chosen_index)
            candidates.remove(chosen_index)
            niche_counts[chosen_reference] += 1
            remaining -= 1

        survivors = [
            population[index]
            for index in selected_indices
        ]

        assign_pareto_ranks(
            survivors,
            self.objectives,
        )
        self._assign_reference_metadata(survivors)

        return survivors

    def _refresh_selection_metadata(self) -> None:
        """Refresh Pareto and reference metadata."""
        if not self.population:
            return

        assign_pareto_ranks(
            self.population,
            self.objectives,
        )
        self._assign_reference_metadata(
            self.population
        )

    def _tournament_winner(
        self,
        left: CircuitGenome,
        right: CircuitGenome,
    ) -> CircuitGenome:
        """Choose an NSGA-III tournament winner."""
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
                "reference_distance",
                np.inf,
            )
        )
        right_distance = float(
            right.metadata.get(
                "reference_distance",
                np.inf,
            )
        )

        if left_distance < right_distance:
            return left

        if right_distance < left_distance:
            return right

        return self.rng.choice([left, right])

    def _representative_genome(self) -> CircuitGenome:
        """Return a representative first-front genome."""
        pareto_front = [
            genome
            for genome in self.population
            if genome.metadata.get("pareto_rank") == 0
        ]

        return min(
            pareto_front,
            key=lambda genome: (
                float(
                    genome.metadata.get(
                        "reference_distance",
                        np.inf,
                    )
                ),
                genome.genome_number,
            ),
        )

    def _parent_sort_key(
        self,
        genome: CircuitGenome,
    ) -> tuple:
        """Sort parents by rank and reference distance."""
        return (
            int(
                genome.metadata.get(
                    "pareto_rank",
                    np.iinfo(np.int64).max,
                )
            ),
            float(
                genome.metadata.get(
                    "reference_distance",
                    np.inf,
                )
            ),
            genome.genome_number,
        )