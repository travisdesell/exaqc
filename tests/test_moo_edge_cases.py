"""Important edge cases for ``src.evolution.moo`` (excluding NSGA-III).

These tests target subtle failure modes rather than happy-path coverage:
non-finite objectives, maximization signs in Pareto dominance, NSGA-II
crowding when objective ranges collapse, duplicate-topology handling, and
stale genomes after multi-objective island repopulation.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

from src.evolution.moo.islands import MultiObjectiveIsland
from src.evolution.moo.nsga2 import NSGA2
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.moo.pareto import (
    genome_dominates,
    non_dominated_sort,
    objective_vector,
)

LOSS = ObjectiveSpec(name="loss", sign=1.0)
ACCURACY = ObjectiveSpec(name="accuracy", sign=-1.0)
OBJECTIVES = [LOSS, ACCURACY]


class _MockGenome:
    """Minimal genome stand-in for multi-objective unit tests.

    Args:
        genome_number: Unique genome identifier.
        fitness: Fitness dictionary, or ``None`` if unevaluated.
        topology_id: Shared identifier used by ``has_same_gates``.
        metadata: Mutable metadata dictionary.
    """

    def __init__(
        self,
        genome_number: int,
        fitness: dict[str, float] | None,
        topology_id: str = "unique",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the mock genome."""
        self.genome_number = genome_number
        self.fitness = fitness
        self.topology_id = topology_id
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    def has_same_gates(self, other: _MockGenome) -> bool:
        """Return whether two mocks share the same topology id.

        Args:
            other: Genome being compared.

        Returns:
            ``True`` when both genomes use the same ``topology_id``.
        """
        return self.topology_id == other.topology_id


def _make_nsga2(max_population_size: int = 4) -> NSGA2:
    """Create an NSGA-II population with artifact saving disabled.

    Args:
        max_population_size: Maximum retained population size.

    Returns:
        Configured ``NSGA2`` instance.
    """
    return NSGA2(
        max_population_size=max_population_size,
        objectives=OBJECTIVES,
        tournament_size=2,
        out_dir="unused",
        profiler=MagicMock(),
        save_all_genomes=False,
        save_pareto_front=False,
        seed=0,
    )


def test_objective_vector_maps_nonfinite_to_positive_inf() -> None:
    """Non-finite raw fitness must become +inf after minimization transform."""
    genome = _MockGenome(
        genome_number=1,
        fitness={"loss": float("nan"), "accuracy": float("inf")},
    )

    values = objective_vector(genome, OBJECTIVES)

    assert values.shape == (2,)
    assert math.isinf(float(values[0])) and float(values[0]) > 0
    assert math.isinf(float(values[1])) and float(values[1]) > 0


def test_dominance_respects_maximization_sign() -> None:
    """Higher accuracy must dominate when that objective uses ``sign=-1``."""
    better = _MockGenome(
        genome_number=1,
        fitness={"loss": 0.5, "accuracy": 0.9},
    )
    worse = _MockGenome(
        genome_number=2,
        fitness={"loss": 0.5, "accuracy": 0.7},
    )

    assert genome_dominates(better, worse, OBJECTIVES)
    assert not genome_dominates(worse, better, OBJECTIVES)

    # Identical objective vectors never dominate each other.
    twin = _MockGenome(
        genome_number=3,
        fitness={"loss": 0.5, "accuracy": 0.9},
    )
    assert not genome_dominates(better, twin, OBJECTIVES)
    assert non_dominated_sort([better, twin], OBJECTIVES) == [[0, 1]]


def test_nsga2_crowding_skips_nonfinite_objective_range() -> None:
    """Crowding must stay finite when every objective on a front is +/-inf.

    Mapping non-finite fitness to +inf makes ``maximum - minimum`` become
    NaN (``inf - inf``). Without an explicit guard that NaN contaminates
    crowding distance and breaks tournament / truncation ordering.
    """
    population = [
        _MockGenome(
            genome_number=index,
            fitness={"loss": float("nan"), "accuracy": float("nan")},
        )
        for index in range(3)
    ]
    nsga2 = _make_nsga2()

    nsga2._assign_crowding_distance(population, [0, 1, 2])

    for genome in population:
        distance = genome.metadata["crowding_distance"]
        assert math.isfinite(distance) or math.isinf(distance)
        assert not math.isnan(distance)


def test_nsga2_discards_dominated_duplicate_topology() -> None:
    """A worse duplicate topology is rejected before environmental selection."""
    nsga2 = _make_nsga2()
    existing = _MockGenome(
        genome_number=1,
        fitness={"loss": 0.2, "accuracy": 0.8},
        topology_id="shared",
    )
    dominated = _MockGenome(
        genome_number=2,
        fitness={"loss": 0.4, "accuracy": 0.6},
        topology_id="shared",
    )
    nsga2.population = [existing]

    assert nsga2._handle_duplicate(dominated) is False
    assert len(nsga2.population) == 1
    assert nsga2.population[0].genome_number == 1

    # A dominating duplicate replaces the existing genome.
    improved = _MockGenome(
        genome_number=3,
        fitness={"loss": 0.1, "accuracy": 0.9},
        topology_id="shared",
    )
    assert nsga2._handle_duplicate(improved) is True
    assert nsga2.population == []


def test_moo_island_discards_stale_genomes_after_repopulation() -> None:
    """Genomes older than a repopulation event are discarded unless global best."""
    island = MultiObjectiveIsland(
        id=0,
        max_size=3,
        population_class=NSGA2,
        objectives=OBJECTIVES,
        tournament_size=2,
        population_kwargs={
            "out_dir": "unused",
            "profiler": MagicMock(),
            "save_all_genomes": False,
            "save_pareto_front": False,
            "seed": 0,
        },
    )
    island.repopulation_genome_number = 100
    island.strategy.insert_genome = MagicMock(return_value=True)

    stale = _MockGenome(genome_number=10, fitness={"loss": 0.1, "accuracy": 0.9})
    assert island.insert_genome(stale) is False
    assert stale.metadata["insert_type"] == "discarded"
    island.strategy.insert_genome.assert_not_called()

    global_best = _MockGenome(
        genome_number=10,
        fitness={"loss": 0.1, "accuracy": 0.9},
        metadata={"insert_type": "global_best"},
    )
    assert island.insert_genome(global_best) is True
    island.strategy.insert_genome.assert_called_once_with(global_best)
    assert global_best.metadata["insert_type"] == "inserted"
