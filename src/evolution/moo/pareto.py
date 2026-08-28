"""Pareto utilities shared by multi-objective population strategies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.objective_spec import ObjectiveSpec


def validate_genome_fitness(
    genome: CircuitGenome,
    objectives: Sequence[ObjectiveSpec],
) -> None:
    """Validate that a genome contains all required objective values.

    Args:
        genome: Genome whose fitness dictionary should be validated.
        objectives: Objective specifications required by the population.

    Raises:
        ValueError: If the genome has not been evaluated.
        KeyError: If an objective is missing from the fitness dictionary.
        TypeError: If an objective cannot be converted to a floating-point
            value.
    """
    if genome.fitness is None:
        raise ValueError(f"Genome {genome.genome_number} has not been evaluated.")

    for objective in objectives:
        if objective.name not in genome.fitness:
            raise KeyError(
                f"Genome {genome.genome_number} does not contain "
                f"objective '{objective.name}'. Available fitness values: "
                f"{sorted(genome.fitness.keys())}."
            )

        try:
            float(genome.fitness[objective.name])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Objective '{objective.name}' for genome "
                f"{genome.genome_number} must be numeric."
            ) from exc


def objective_vector(
    genome: CircuitGenome,
    objectives: Sequence[ObjectiveSpec],
) -> np.ndarray:
    """Return the transformed minimization objective vector for a genome.

    Raw values in ``genome.fitness`` are transformed according to each
    objective's sign and weight. Non-finite objective values are converted
    to positive infinity so that they behave as poor minimization values.

    Args:
        genome: Genome whose objectives should be extracted.
        objectives: Objective specifications defining the transformation.

    Returns:
        One-dimensional NumPy array containing transformed objective values.
    """
    validate_genome_fitness(
        genome,
        objectives,
    )

    values: list[float] = []

    for objective in objectives:
        value = objective.transform(genome.fitness[objective.name])

        if not np.isfinite(value):
            value = np.inf

        values.append(value)

    return np.asarray(
        values,
        dtype=np.float64,
    )


def genome_dominates(
    left: CircuitGenome,
    right: CircuitGenome,
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    """Determine whether one genome Pareto-dominates another.

    All objectives are assumed to have already been transformed into
    minimization form.

    Genome ``left`` dominates genome ``right`` when ``left`` is no worse
    for every objective and strictly better for at least one objective.

    Args:
        left: First genome.
        right: Genome being compared against.
        objectives: Objective specifications.

    Returns:
        ``True`` if ``left`` Pareto-dominates ``right``.
    """
    left_values = objective_vector(
        left,
        objectives,
    )

    right_values = objective_vector(
        right,
        objectives,
    )

    no_worse = np.all(left_values <= right_values)

    strictly_better = np.any(left_values < right_values)

    return bool(no_worse and strictly_better)


def non_dominated_sort(
    population: Sequence[CircuitGenome],
    objectives: Sequence[ObjectiveSpec],
) -> list[list[int]]:
    """Partition a population into Pareto fronts.

    The first returned front contains all non-dominated genomes. The second
    front contains genomes dominated only by members of the first front,
    and so on.

    Args:
        population: Population to rank.
        objectives: Objective specifications.

    Returns:
        List of Pareto fronts. Each front contains indexes into
        ``population``.
    """
    population_size = len(population)

    if population_size == 0:
        return []

    domination_sets: list[list[int]] = [[] for _ in range(population_size)]

    domination_counts = [0 for _ in range(population_size)]

    first_front: list[int] = []

    for left_index in range(population_size):
        for right_index in range(
            left_index + 1,
            population_size,
        ):
            left = population[left_index]
            right = population[right_index]

            if genome_dominates(
                left,
                right,
                objectives,
            ):
                domination_sets[left_index].append(right_index)

                domination_counts[right_index] += 1

            elif genome_dominates(
                right,
                left,
                objectives,
            ):
                domination_sets[right_index].append(left_index)

                domination_counts[left_index] += 1

    for index, count in enumerate(domination_counts):
        if count == 0:
            first_front.append(index)

    fronts: list[list[int]] = []

    if first_front:
        fronts.append(first_front)

    front_index = 0

    while front_index < len(fronts):
        next_front: list[int] = []

        for genome_index in fronts[front_index]:
            for dominated_index in domination_sets[genome_index]:
                domination_counts[dominated_index] -= 1

                if domination_counts[dominated_index] == 0:
                    next_front.append(dominated_index)

        if next_front:
            fronts.append(next_front)

        front_index += 1

    return fronts


def assign_pareto_ranks(
    population: Sequence[CircuitGenome],
    objectives: Sequence[ObjectiveSpec],
) -> list[list[int]]:
    """Calculate and store Pareto ranks for a population.

    Pareto rank zero corresponds to the non-dominated front.

    Args:
        population: Population whose ranks should be updated.
        objectives: Objective specifications.

    Returns:
        Pareto fronts represented as indexes into ``population``.
    """
    fronts = non_dominated_sort(
        population,
        objectives,
    )

    for genome in population:
        genome.metadata.pop(
            "pareto_rank",
            None,
        )

    for rank, front in enumerate(fronts):
        for index in front:
            population[index].metadata["pareto_rank"] = rank

    return fronts
