"""NSGA-III population strategy for EXAQC."""

from __future__ import annotations

import argparse
import math

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.base_population import (
    MultiObjectivePopulationBase,
)
from src.evolution.moo.pareto import (
    assign_pareto_ranks,
    objective_vector,
)

#: Where a non-finite objective lands after normalization: far beyond every
#: finite genome (which normalize to roughly [0, 1]), so it is never mistaken
#: for one near the ideal point.
NON_FINITE_NORMALIZED = 1.0e6


class NSGA3(MultiObjectivePopulationBase):
    """Steady-state NSGA-III population strategy.

    NSGA-III combines non-dominated sorting with reference-direction-based
    niching. It is primarily intended for optimization problems containing
    three or more objectives.

    All objectives are internally represented as minimization objectives.

    Args:
        reference_divisions: Number of divisions used to construct the
            Das-Dennis reference-direction simplex, or ``None`` to use the
            most divisions whose reference directions do not outnumber the
            population (see :meth:`default_reference_divisions`).
        *args: Positional arguments passed to
            ``MultiObjectivePopulationBase``.
        **kwargs: Keyword arguments passed to
            ``MultiObjectivePopulationBase``.

    Raises:
        ValueError: If ``reference_divisions`` is not positive.
        ValueError: If no reference directions can be generated.
    """

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds NSGA-III's command-line arguments to a parser.

        Args:
            parser: The (sub-)parser to add the arguments to.

        Returns:
            None. Mutates ``parser`` by adding the shared multi-objective
            population flags and ``--reference_divisions``.
        """

        MultiObjectivePopulationBase.initialize_parser(parser)
        parser.add_argument(
            "--reference_divisions",
            type=int,
            default=None,
            help=(
                "Divisions of the Das-Dennis reference-direction simplex. Defaults to the "
                "most divisions whose reference directions do not outnumber "
                "--max_population_size (max_population_size - 1 for two objectives)."
            ),
        )

    @staticmethod
    def default_reference_divisions(n_objectives: int, population_size: int) -> int:
        """Chooses how finely to divide the reference simplex for a population.

        NSGA-III works best with roughly one reference direction per
        population member. A simplex with ``H`` divisions over ``M`` objectives
        has ``C(H + M - 1, M - 1)`` directions, so this returns the largest
        ``H`` whose direction count does not exceed the population size.

        Args:
            n_objectives: Number of optimization objectives.
            population_size: Maximum number of genomes the population holds.

        Returns:
            The number of divisions, at least one.
        """

        divisions = 1
        while (
            math.comb(divisions + 1 + n_objectives - 1, n_objectives - 1)
            <= population_size
        ):
            divisions += 1
        return divisions

    def __init__(
        self,
        *args: Any,
        reference_divisions: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an NSGA-III population.

        Args:
            *args: Positional arguments passed to
                ``MultiObjectivePopulationBase``.
            reference_divisions: Number of simplex divisions, or ``None`` for
                :meth:`default_reference_divisions`.
            **kwargs: Keyword arguments passed to
                ``MultiObjectivePopulationBase``.

        Returns:
            None. Sets ``reference_divisions`` and ``reference_directions``.

        Raises:
            ValueError: If ``reference_divisions`` is not positive.
            ValueError: If no reference directions can be generated.
        """
        super().__init__(
            *args,
            **kwargs,
        )

        if reference_divisions is None:
            reference_divisions = self.default_reference_divisions(
                len(self.objectives), self.max_population_size
            )

        if reference_divisions <= 0:
            raise ValueError("reference_divisions must be greater than zero.")

        self.reference_divisions = int(reference_divisions)

        self.reference_directions = self._generate_reference_directions(
            n_objectives=len(self.objectives),
            divisions=(self.reference_divisions),
        )

        if len(self.reference_directions) == 0:
            raise ValueError("No NSGA-III reference directions " "were generated.")

    @property
    def algorithm_name(self) -> str:
        """Return the optimization algorithm name.

        Returns:
            ``"nsga3"``.
        """
        return "nsga3"

    def run_info(self) -> dict[str, Any]:
        """Describes the strategy's configuration, for the run's archive.

        Returns:
            The shared multi-objective description (see
            :meth:`MultiObjectivePopulationBase.run_info`) plus the number of
            reference divisions and directions used.
        """

        info = super().run_info()
        info["multi_objective"]["reference_divisions"] = self.reference_divisions
        info["multi_objective"]["reference_directions"] = len(self.reference_directions)
        return info

    @staticmethod
    def _generate_reference_directions(
        n_objectives: int,
        divisions: int,
    ) -> np.ndarray:
        """Generate Das-Dennis simplex reference directions.

        Args:
            n_objectives: Number of optimization objectives.
            divisions: Number of simplex divisions.

        Returns:
            Two-dimensional array whose rows are reference directions.
        """
        directions: list[list[float]] = []

        def generate(
            remaining: int,
            objective_index: int,
            current: list[int],
        ) -> None:
            """Recursively generate integer simplex coordinates.

            Args:
                remaining: Remaining divisions.
                objective_index: Current objective dimension.
                current: Partial integer direction.

            Returns:
                None. Appends each completed direction to ``directions``.
            """
            if objective_index == n_objectives - 1:
                complete = current + [remaining]

                directions.append([value / divisions for value in complete])

                return

            for value in range(remaining + 1):
                generate(
                    remaining - value,
                    objective_index + 1,
                    current + [value],
                )

        generate(
            remaining=divisions,
            objective_index=0,
            current=[],
        )

        return np.asarray(
            directions,
            dtype=np.float64,
        )

    def _objective_matrix(
        self,
        population: Sequence[CircuitGenome],
    ) -> np.ndarray:
        """Construct the transformed objective matrix.

        Args:
            population: Population to convert.

        Returns:
            Matrix with one row per genome and one column per objective.
        """
        return np.asarray(
            [
                objective_vector(
                    genome,
                    self.objectives,
                )
                for genome in population
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _find_extreme_points(
        shifted_values: np.ndarray,
    ) -> np.ndarray:
        """Find NSGA-III extreme points using ASF values.

        Args:
            shifted_values: Objective values translated by the ideal point.

        Returns:
            Array containing one extreme point per objective.
        """
        n_objectives = shifted_values.shape[1]

        extreme_points: list[np.ndarray] = []

        for objective_index in range(n_objectives):
            weights = np.full(
                n_objectives,
                1e-6,
                dtype=np.float64,
            )

            weights[objective_index] = 1.0

            asf_values = np.max(
                shifted_values / weights,
                axis=1,
            )

            extreme_index = int(np.argmin(asf_values))

            extreme_points.append(shifted_values[extreme_index])

        return np.asarray(
            extreme_points,
            dtype=np.float64,
        )

    @staticmethod
    def _calculate_intercepts(
        shifted_values: np.ndarray,
        extreme_points: np.ndarray,
    ) -> np.ndarray:
        """Calculate objective-axis intercepts for NSGA-III normalization.

        A linear system using the extreme points is solved when possible.
        If the system is singular or produces invalid intercepts, per-objective
        maxima are used as a safe fallback.

        Args:
            shifted_values: Objective values translated by the ideal point.
            extreme_points: Extreme points calculated with achievement
                scalarizing functions.

        Returns:
            Positive normalization intercepts.
        """
        n_objectives = shifted_values.shape[1]

        fallback = np.max(
            shifted_values,
            axis=0,
        )

        fallback[
            np.isclose(
                fallback,
                0.0,
            )
        ] = 1.0

        try:
            coefficients = np.linalg.solve(
                extreme_points,
                np.ones(
                    n_objectives,
                    dtype=np.float64,
                ),
            )

            intercepts = 1.0 / coefficients

        except np.linalg.LinAlgError:
            return fallback

        invalid = ~np.isfinite(intercepts) | (intercepts <= 1e-12)

        if np.any(invalid):
            return fallback

        return intercepts

    def _normalize_objectives(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Normalize transformed objectives for reference association.

        The ideal point, extreme points and intercepts are found from the
        genomes whose objectives are all finite. A non-finite objective (a
        genome whose training failed) is then placed far beyond every finite
        genome, so it associates with a reference direction at a large
        distance rather than looking ideal.

        Args:
            values: Transformed minimization objective matrix.

        Returns:
            Normalized objective matrix, with every entry finite.
        """
        if values.size == 0:
            return values

        finite_rows = np.all(np.isfinite(values), axis=1)

        if not np.any(finite_rows):
            return np.full_like(values, NON_FINITE_NORMALIZED)

        finite_values = values[finite_rows]

        ideal_point = np.min(
            finite_values,
            axis=0,
        )

        shifted_finite = finite_values - ideal_point

        extreme_points = self._find_extreme_points(shifted_finite)

        intercepts = self._calculate_intercepts(
            shifted_finite,
            extreme_points,
        )

        with np.errstate(invalid="ignore", over="ignore"):
            normalized = (values - ideal_point) / intercepts

        normalized[~np.isfinite(normalized)] = NON_FINITE_NORMALIZED

        return normalized

    def _associate_reference_directions(
        self,
        population: Sequence[CircuitGenome],
    ) -> None:
        """Associate each genome with its nearest reference direction.

        Perpendicular Euclidean distance from a normalized objective vector
        to each reference direction is used.

        Args:
            population: Population whose associations should be updated.

        Returns:
            None. Sets ``reference_index`` and ``reference_distance`` in each
            genome's metadata.
        """
        if not population:
            return

        values = self._objective_matrix(population)

        finite_rows = np.all(np.isfinite(values), axis=1)

        normalized_values = self._normalize_objectives(values)

        directions = self.reference_directions

        norms = np.linalg.norm(
            directions,
            axis=1,
        )

        norms[np.isclose(norms, 0.0)] = 1.0

        unit_directions = directions / norms[:, np.newaxis]

        for genome_index, genome in enumerate(population):
            point = normalized_values[genome_index]

            projection_lengths = unit_directions @ point

            projected_points = projection_lengths[:, np.newaxis] * unit_directions

            distances = np.linalg.norm(
                point - projected_points,
                axis=1,
            )

            reference_index = int(np.argmin(distances))

            genome.metadata["reference_index"] = reference_index

            # a genome with a non-finite objective is never close to a reference
            genome.metadata["reference_distance"] = (
                float(distances[reference_index])
                if finite_rows[genome_index]
                else math.inf
            )

    def _refresh_selection_metadata(
        self,
    ) -> None:
        """Refresh Pareto ranks and NSGA-III reference associations.

        Returns:
            None. Sets ``pareto_rank``, ``reference_index`` and
            ``reference_distance`` in the metadata of every genome in
            ``population``.
        """
        if not self.population:
            return

        assign_pareto_ranks(
            self.population,
            self.objectives,
        )

        self._associate_reference_directions(self.population)

    def _environmental_selection(
        self,
        population: Sequence[CircuitGenome],
        population_size: int,
    ) -> list[CircuitGenome]:
        """Perform NSGA-III environmental selection.

        Complete Pareto fronts are accepted until the next complete front
        would exceed the requested population size. Reference-direction
        niching is then used to select the remaining genomes.

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

        self._associate_reference_directions(population)

        survivors: list[CircuitGenome] = []

        splitting_front: list[CircuitGenome] = []

        for front in fronts:
            candidates = [population[index] for index in front]

            if len(survivors) + len(candidates) <= population_size:
                survivors.extend(candidates)

                continue

            splitting_front = candidates
            break

        remaining = population_size - len(survivors)

        if remaining > 0 and splitting_front:
            selected = self._niching_selection(
                survivors=survivors,
                candidates=(splitting_front),
                n_select=remaining,
            )

            survivors.extend(selected)

        assign_pareto_ranks(
            survivors,
            self.objectives,
        )

        self._associate_reference_directions(survivors)

        return survivors

    def _niching_selection(
        self,
        survivors: Sequence[CircuitGenome],
        candidates: Sequence[CircuitGenome],
        n_select: int,
    ) -> list[CircuitGenome]:
        """Select members of a partial front using NSGA-III niching.

        Args:
            survivors: Genomes already selected from better Pareto fronts.
            candidates: Genomes belonging to the splitting front.
            n_select: Number of additional genomes required.

        Returns:
            Genomes selected from the splitting front.
        """
        remaining_candidates = list(candidates)

        selected: list[CircuitGenome] = []

        niche_counts = np.zeros(
            len(self.reference_directions),
            dtype=np.int64,
        )

        for genome in survivors:
            reference_index = int(genome.metadata["reference_index"])

            niche_counts[reference_index] += 1

        while len(selected) < n_select and remaining_candidates:
            available_references = sorted(
                {
                    int(genome.metadata["reference_index"])
                    for genome in remaining_candidates
                }
            )

            minimum_niche_count = min(
                niche_counts[reference_index]
                for reference_index in available_references
            )

            least_occupied = [
                reference_index
                for reference_index in available_references
                if niche_counts[reference_index] == minimum_niche_count
            ]

            reference_index = self.rng.choice(least_occupied)

            niche_candidates = [
                genome
                for genome in remaining_candidates
                if int(genome.metadata["reference_index"]) == reference_index
            ]

            if niche_counts[reference_index] == 0:
                winner = min(
                    niche_candidates,
                    key=lambda genome: (
                        genome.metadata.get(
                            "reference_distance",
                            math.inf,
                        ),
                        genome.genome_number,
                    ),
                )

            else:
                winner = self.rng.choice(niche_candidates)

            selected.append(winner)

            remaining_candidates.remove(winner)

            niche_counts[reference_index] += 1

        return selected

    def _tournament_winner(
        self,
        left: CircuitGenome,
        right: CircuitGenome,
    ) -> CircuitGenome:
        """Choose the preferred NSGA-III tournament candidate.

        Pareto rank is compared first. Reference-direction distance provides
        the secondary preference.

        Args:
            left: First candidate.
            right: Second candidate.

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
            "reference_distance",
            math.inf,
        )

        right_distance = right.metadata.get(
            "reference_distance",
            math.inf,
        )

        if left_distance < right_distance:
            return left

        if right_distance < left_distance:
            return right

        return left if left.genome_number <= right.genome_number else right

    def _parent_sort_key(
        self,
        genome: CircuitGenome,
    ) -> tuple[float, float, int]:
        """Return NSGA-III ordering for crossover parents.

        Args:
            genome: Selected parent genome.

        Returns:
            Tuple ordered by Pareto rank, reference distance, and genome
            number.
        """
        return (
            genome.metadata.get(
                "pareto_rank",
                math.inf,
            ),
            genome.metadata.get(
                "reference_distance",
                math.inf,
            ),
            genome.genome_number,
        )
