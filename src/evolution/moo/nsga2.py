"""NSGA-II population strategy for EXAQC."""

from __future__ import annotations

import math

from collections.abc import Sequence

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.base_population import (
    MultiObjectivePopulationBase,
)
from src.evolution.moo.pareto import (
    assign_pareto_ranks,
    objective_vector,
)


class NSGA2(MultiObjectivePopulationBase):
    """Steady-state NSGA-II population strategy.

    NSGA-II ranks candidate solutions using Pareto dominance and maintains
    diversity within each Pareto front using crowding distance.

    All objective values are internally treated as minimization objectives.
    Maximization objectives should therefore use ``sign=-1.0`` in their
    ``ObjectiveSpec``.
    """

    @property
    def algorithm_name(self) -> str:
        """Return the optimization algorithm name.

        Returns:
            ``"nsga2"``.
        """
        return "nsga2"

    def _refresh_selection_metadata(
        self,
    ) -> None:
        """Refresh Pareto ranks and crowding distances."""
        if not self.population:
            return

        fronts = assign_pareto_ranks(
            self.population,
            self.objectives,
        )

        for front in fronts:
            self._assign_crowding_distance(
                self.population,
                front,
            )

    def _assign_crowding_distance(
        self,
        population: Sequence[
            CircuitGenome
        ],
        front: list[int],
    ) -> None:
        """Calculate NSGA-II crowding distance for one Pareto front.

        Args:
            population: Population containing the front.
            front: Genome indexes belonging to the Pareto front.
        """
        if not front:
            return

        for index in front:
            population[index].metadata[
                "crowding_distance"
            ] = 0.0

        if len(front) <= 2:
            for index in front:
                population[index].metadata[
                    "crowding_distance"
                ] = math.inf

            return

        for objective_index in range(
            len(self.objectives)
        ):
            sorted_front = sorted(
                front,
                key=lambda index: (
                    objective_vector(
                        population[index],
                        self.objectives,
                    )[objective_index]
                ),
            )

            first_index = sorted_front[0]
            last_index = sorted_front[-1]

            population[first_index].metadata[
                "crowding_distance"
            ] = math.inf

            population[last_index].metadata[
                "crowding_distance"
            ] = math.inf

            minimum = objective_vector(
                population[first_index],
                self.objectives,
            )[objective_index]

            maximum = objective_vector(
                population[last_index],
                self.objectives,
            )[objective_index]

            objective_range = (
                maximum - minimum
            )

            if math.isclose(
                objective_range,
                0.0,
            ):
                continue

            for position in range(
                1,
                len(sorted_front) - 1,
            ):
                index = sorted_front[position]

                genome = population[index]

                if math.isinf(
                    genome.metadata[
                        "crowding_distance"
                    ]
                ):
                    continue

                previous_value = (
                    objective_vector(
                        population[
                            sorted_front[
                                position - 1
                            ]
                        ],
                        self.objectives,
                    )[objective_index]
                )

                next_value = (
                    objective_vector(
                        population[
                            sorted_front[
                                position + 1
                            ]
                        ],
                        self.objectives,
                    )[objective_index]
                )

                normalized_distance = (
                    next_value
                    - previous_value
                ) / objective_range

                genome.metadata[
                    "crowding_distance"
                ] += normalized_distance

    def _environmental_selection(
        self,
        population: Sequence[
            CircuitGenome
        ],
        population_size: int,
    ) -> list[CircuitGenome]:
        """Perform NSGA-II environmental selection.

        Complete Pareto fronts are inserted while space remains. When only
        part of a front can fit, genomes with the greatest crowding distance
        are selected.

        Args:
            population: Candidate population.
            population_size: Maximum number of survivors.

        Returns:
            Selected survivor population.
        """
        population = list(population)

        fronts = assign_pareto_ranks(
            population,
            self.objectives,
        )

        for front in fronts:
            self._assign_crowding_distance(
                population,
                front,
            )

        survivors: list[
            CircuitGenome
        ] = []

        for front in fronts:
            if (
                len(survivors)
                + len(front)
                <= population_size
            ):
                survivors.extend(
                    population[index]
                    for index in front
                )

                continue

            remaining = (
                population_size
                - len(survivors)
            )

            if remaining <= 0:
                break

            candidates = [
                population[index]
                for index in front
            ]

            candidates.sort(
                key=lambda genome: (
                    genome.metadata.get(
                        "crowding_distance",
                        0.0,
                    )
                ),
                reverse=True,
            )

            survivors.extend(
                candidates[:remaining]
            )

            break

        survivor_fronts = (
            assign_pareto_ranks(
                survivors,
                self.objectives,
            )
        )

        for front in survivor_fronts:
            self._assign_crowding_distance(
                survivors,
                front,
            )

        return survivors

    def _tournament_winner(
        self,
        left: CircuitGenome,
        right: CircuitGenome,
    ) -> CircuitGenome:
        """Choose the preferred NSGA-II tournament candidate.

        Lower Pareto rank is preferred. If both genomes have the same rank,
        the genome with greater crowding distance is preferred.

        Args:
            left: First tournament candidate.
            right: Second tournament candidate.

        Returns:
            Preferred candidate.
        """
        left_rank = left.metadata.get(
            "pareto_rank",
            math.inf,
        )

        right_rank = right.metadata.get(
            "pareto_rank",
            math.inf,
        )

        if left_rank < right_rank:
            return left

        if right_rank < left_rank:
            return right

        left_distance = left.metadata.get(
            "crowding_distance",
            0.0,
        )

        right_distance = right.metadata.get(
            "crowding_distance",
            0.0,
        )

        if left_distance > right_distance:
            return left

        if right_distance > left_distance:
            return right

        return (
            left
            if left.genome_number
            <= right.genome_number
            else right
        )

    def _representative_genome(
        self,
    ) -> CircuitGenome:
        """Return one representative Pareto-optimal genome.

        The Pareto-front genome with the largest crowding distance is used.
        Genome number provides a deterministic tie breaker.

        Returns:
            Representative Pareto-optimal genome.

        Raises:
            RuntimeError: If the Pareto front is unexpectedly empty.
        """
        front = self.get_pareto_front()

        if not front:
            raise RuntimeError(
                "Cannot choose a representative "
                "from an empty Pareto front."
            )

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

    def _parent_sort_key(
        self,
        genome: CircuitGenome,
    ) -> tuple:
        """Return NSGA-II ordering for crossover parents.

        Args:
            genome: Selected parent.

        Returns:
            Tuple ordered by Pareto rank, decreasing crowding distance,
            and genome number.
        """
        return (
            genome.metadata.get(
                "pareto_rank",
                math.inf,
            ),
            -genome.metadata.get(
                "crowding_distance",
                0.0,
            ),
            genome.genome_number,
        )