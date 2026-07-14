# src/evolution/moo/pareto.py

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.evolution.multi_objective.objective_spec import ObjectiveSpec

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome


def validate_genome_fitness(
    genome: CircuitGenome,
    objectives: Sequence[ObjectiveSpec],
) -> None:
    """Validate that a genome contains all requested objectives.

    Args:
        genome: Genome to validate.
        objectives: Objective specifications.

    Raises:
        ValueError: If the genome has no fitness dictionary.
        KeyError: If an objective is missing.
        TypeError: If an objective cannot be converted to a number.
    """
    if genome.fitness is None:
        raise ValueError(
            f"Genome {genome.genome_number} has not been evaluated."
        )

    for objective in objectives:
        if objective.name not in genome.fitness:
            raise KeyError(
                f"Genome {genome.genome_number} is missing objective "
                f"{objective.name!r}. Available fitness values are "
                f"{sorted(genome.fitness.keys())}."
            )

        try:
            float(genome.fitness[objective.name])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Fitness value {objective.name!r} for genome "
                f"{genome.genome_number} is not numeric: "
                f"{genome.fitness[objective.name]!r}."
            ) from exc


def objective_vector(
    genome: CircuitGenome,
    objectives: Sequence[ObjectiveSpec],
) -> np.ndarray:
    """Extract a minimization-form objective vector.

    Args:
        genome: Evaluated genome.
        objectives: Ordered objective specifications.

    Returns:
        One-dimensional objective vector.
    """
    validate_genome_fitness(genome, objectives)

    values = []

    for objective in objectives:
        value = objective.minimization_value(
            genome.fitness[objective.name]
        )

        if not np.isfinite(value):
            value = np.inf

        values.append(value)

    return np.asarray(values, dtype=np.float64)


def objective_matrix(
    population: Sequence[CircuitGenome],
    objectives: Sequence[ObjectiveSpec],
) -> np.ndarray:
    """Build an objective matrix for a population.

    Args:
        population: Genome population.
        objectives: Ordered objective specifications.

    Returns:
        Array with shape ``[population_size, n_objectives]``.
    """
    if not population:
        return np.empty((0, len(objectives)), dtype=np.float64)

    return np.stack(
        [
            objective_vector(genome, objectives)
            for genome in population
        ],
        axis=0,
    )


def vector_dominates(
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    """Check whether one minimization vector dominates another.

    Args:
        left: First minimization objective vector.
        right: Second minimization objective vector.

    Returns:
        ``True`` if ``left`` is no worse in every objective and strictly
        better in at least one.
    """
    return bool(
        np.all(left <= right)
        and np.any(left < right)
    )


def genome_dominates(
    left: CircuitGenome,
    right: CircuitGenome,
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    """Check whether one genome dominates another.

    Args:
        left: First genome.
        right: Second genome.
        objectives: Objective specifications.

    Returns:
        ``True`` if ``left`` Pareto-dominates ``right``.
    """
    return vector_dominates(
        objective_vector(left, objectives),
        objective_vector(right, objectives),
    )


def fast_non_dominated_sort(
    population: Sequence[CircuitGenome],
    objectives: Sequence[ObjectiveSpec],
) -> list[list[int]]:
    """Perform fast non-dominated sorting.

    Args:
        population: Evaluated population.
        objectives: Objective specifications.

    Returns:
        Pareto fronts represented by population indices.
    """
    population_size = len(population)

    if population_size == 0:
        return []

    values = objective_matrix(population, objectives)

    domination_sets: list[list[int]] = [
        [] for _ in range(population_size)
    ]
    domination_counts = np.zeros(
        population_size,
        dtype=np.int64,
    )

    first_front: list[int] = []

    for left_index in range(population_size):
        for right_index in range(left_index + 1, population_size):
            left = values[left_index]
            right = values[right_index]

            if vector_dominates(left, right):
                domination_sets[left_index].append(right_index)
                domination_counts[right_index] += 1

            elif vector_dominates(right, left):
                domination_sets[right_index].append(left_index)
                domination_counts[left_index] += 1

    for index in range(population_size):
        if domination_counts[index] == 0:
            first_front.append(index)

    fronts: list[list[int]] = []

    if first_front:
        fronts.append(first_front)

    current_front_index = 0

    while current_front_index < len(fronts):
        next_front: list[int] = []

        for left_index in fronts[current_front_index]:
            for right_index in domination_sets[left_index]:
                domination_counts[right_index] -= 1

                if domination_counts[right_index] == 0:
                    next_front.append(right_index)

        if next_front:
            fronts.append(next_front)

        current_front_index += 1

    return fronts


def assign_pareto_ranks(
    population: Sequence[CircuitGenome],
    objectives: Sequence[ObjectiveSpec],
) -> list[list[int]]:
    """Assign Pareto-rank metadata to a population.

    Args:
        population: Genome population.
        objectives: Objective specifications.

    Returns:
        Computed Pareto fronts.
    """
    fronts = fast_non_dominated_sort(
        population,
        objectives,
    )

    for genome in population:
        genome.metadata.pop("pareto_rank", None)

    for rank, front in enumerate(fronts):
        for population_index in front:
            population[population_index].metadata[
                "pareto_rank"
            ] = int(rank)

    return fronts