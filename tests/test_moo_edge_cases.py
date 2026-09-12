"""Important edge cases for ``src.evolution.moo`` (excluding NSGA-III).

These tests target subtle failure modes rather than happy-path coverage:
objective-spec validation, non-finite / weighted objectives, Pareto front
structure, NSGA-II crowding and truncation, duplicate-topology handling,
parent selection, and multi-objective island topology / bookkeeping.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.evolution.moo.islands import (
    MultiObjectiveIsland,
    MultiObjectiveSteadyStateIslands,
)
from src.evolution.moo.nsga2 import NSGA2
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.moo.pareto import (
    assign_pareto_ranks,
    genome_dominates,
    non_dominated_sort,
    objective_vector,
    validate_genome_fitness,
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


def _make_nsga2(max_population_size: int = 4, seed: int = 0) -> NSGA2:
    """Create an NSGA-II population with artifact saving disabled.

    Args:
        max_population_size: Maximum retained population size.
        seed: RNG seed for tournament selection.

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
        seed=seed,
    )


def _island_kwargs() -> dict[str, Any]:
    """Return default kwargs for a disposable multi-objective island.

    Returns:
        Keyword arguments for ``MultiObjectiveIsland`` / island strategy
        population construction with saving disabled.
    """
    return {
        "out_dir": "unused",
        "profiler": MagicMock(),
        "save_all_genomes": False,
        "save_pareto_front": False,
        "seed": 0,
    }


# ---------------------------------------------------------------------------
# ObjectiveSpec
# ---------------------------------------------------------------------------


def test_objective_spec_rejects_invalid_configuration() -> None:
    """Empty names, bad signs, and non-positive weights must raise."""
    with pytest.raises(ValueError, match="empty"):
        ObjectiveSpec(name="", sign=1.0)

    with pytest.raises(ValueError, match="sign"):
        ObjectiveSpec(name="loss", sign=0.5)

    with pytest.raises(ValueError, match="weight"):
        ObjectiveSpec(name="loss", sign=1.0, weight=0.0)

    with pytest.raises(ValueError, match="weight"):
        ObjectiveSpec(name="loss", sign=1.0, weight=-2.0)


def test_objective_spec_transform_applies_sign_and_weight() -> None:
    """Maximization and weighting must both affect the minimization form."""
    maximize = ObjectiveSpec(name="accuracy", sign=-1.0, weight=2.0)
    minimize = ObjectiveSpec(name="loss", sign=1.0, weight=3.0)

    assert maximize.transform(0.8) == pytest.approx(-1.6)
    assert minimize.transform(0.5) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


def test_validate_genome_fitness_rejects_missing_and_non_numeric() -> None:
    """Unevaluated, incomplete, or non-numeric fitness must fail validation."""
    unevaluated = _MockGenome(genome_number=1, fitness=None)
    with pytest.raises(ValueError, match="has not been evaluated"):
        validate_genome_fitness(unevaluated, OBJECTIVES)

    missing = _MockGenome(genome_number=2, fitness={"loss": 0.1})
    with pytest.raises(KeyError, match="accuracy"):
        validate_genome_fitness(missing, OBJECTIVES)

    bad_type = _MockGenome(
        genome_number=3,
        fitness={"loss": 0.1, "accuracy": "good"},
    )
    with pytest.raises(TypeError, match="must be numeric"):
        validate_genome_fitness(bad_type, OBJECTIVES)


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


def test_objective_vector_maps_negative_inf_accuracy_through_sign() -> None:
    """``-inf`` accuracy (maximize) must become a poor minimization value.

    Transform is ``sign * value`` with ``sign=-1``, so ``-inf`` becomes
    ``+inf`` before the non-finite remap; either way the genome is treated
    as dominated / worst on that axis.
    """
    genome = _MockGenome(
        genome_number=1,
        fitness={"loss": 0.1, "accuracy": float("-inf")},
    )

    values = objective_vector(genome, OBJECTIVES)

    assert values[0] == pytest.approx(0.1)
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

    twin = _MockGenome(
        genome_number=3,
        fitness={"loss": 0.5, "accuracy": 0.9},
    )
    assert not genome_dominates(better, twin, OBJECTIVES)
    assert non_dominated_sort([better, twin], OBJECTIVES) == [[0, 1]]


def test_weighted_objectives_scale_vectors_but_not_incomparability() -> None:
    """Weights scale objective vectors; they do not invent dominance.

    Two trade-off genomes stay mutually non-dominating even under large
    weights, because dominance still requires no-worse on every axis.
    """
    unweighted = [
        ObjectiveSpec(name="loss", sign=1.0),
        ObjectiveSpec(name="accuracy", sign=-1.0),
    ]
    genome_a = _MockGenome(
        genome_number=1,
        fitness={"loss": 0.1, "accuracy": 0.5},
    )
    genome_b = _MockGenome(
        genome_number=2,
        fitness={"loss": 0.5, "accuracy": 0.9},
    )
    assert not genome_dominates(genome_a, genome_b, unweighted)
    assert not genome_dominates(genome_b, genome_a, unweighted)

    weighted = [
        ObjectiveSpec(name="loss", sign=1.0, weight=100.0),
        ObjectiveSpec(name="accuracy", sign=-1.0, weight=1.0),
    ]
    assert not genome_dominates(genome_a, genome_b, weighted)
    assert not genome_dominates(genome_b, genome_a, weighted)
    assert objective_vector(genome_a, weighted)[0] == pytest.approx(10.0)
    assert objective_vector(genome_b, weighted)[0] == pytest.approx(50.0)


def test_non_dominated_sort_empty_and_dominance_chain() -> None:
    """Empty populations and total order chains must produce expected fronts."""
    assert non_dominated_sort([], OBJECTIVES) == []

    best = _MockGenome(1, {"loss": 0.1, "accuracy": 0.9})
    mid = _MockGenome(2, {"loss": 0.2, "accuracy": 0.8})
    worst = _MockGenome(3, {"loss": 0.3, "accuracy": 0.7})

    fronts = non_dominated_sort([best, mid, worst], OBJECTIVES)
    assert fronts == [[0], [1], [2]]

    ranks = assign_pareto_ranks([best, mid, worst], OBJECTIVES)
    assert ranks == [[0], [1], [2]]
    assert best.metadata["pareto_rank"] == 0
    assert mid.metadata["pareto_rank"] == 1
    assert worst.metadata["pareto_rank"] == 2


def test_non_dominated_sort_tradeoff_front() -> None:
    """Incomparable trade-off genomes must share the first front."""
    low_loss = _MockGenome(1, {"loss": 0.1, "accuracy": 0.5})
    high_acc = _MockGenome(2, {"loss": 0.5, "accuracy": 0.9})
    dominated = _MockGenome(3, {"loss": 0.6, "accuracy": 0.4})

    fronts = non_dominated_sort([low_loss, high_acc, dominated], OBJECTIVES)

    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]


# ---------------------------------------------------------------------------
# NSGA-II crowding / selection
# ---------------------------------------------------------------------------


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


def test_nsga2_crowding_skips_zero_range_objective() -> None:
    """A constant objective on a front must not divide by zero."""
    population = [
        _MockGenome(1, {"loss": 0.1, "accuracy": 0.5}),
        _MockGenome(2, {"loss": 0.1, "accuracy": 0.7}),
        _MockGenome(3, {"loss": 0.1, "accuracy": 0.9}),
    ]
    nsga2 = _make_nsga2()

    nsga2._assign_crowding_distance(population, [0, 1, 2])

    # Boundary genomes on the accuracy axis keep infinite crowding.
    assert math.isinf(population[0].metadata["crowding_distance"])
    assert math.isinf(population[2].metadata["crowding_distance"])
    # Interior genome gets a finite positive contribution from accuracy only.
    interior = population[1].metadata["crowding_distance"]
    assert math.isfinite(interior)
    assert interior > 0.0
    assert not math.isnan(interior)


def test_nsga2_environmental_selection_truncates_by_crowding() -> None:
    """When a front overflows capacity, higher crowding distance survives."""
    # Three mutually non-dominated genomes; keep only two.
    low_loss = _MockGenome(1, {"loss": 0.0, "accuracy": 0.0})
    mid = _MockGenome(2, {"loss": 0.5, "accuracy": 0.5})
    high_acc = _MockGenome(3, {"loss": 1.0, "accuracy": 1.0})
    nsga2 = _make_nsga2(max_population_size=2)

    survivors = nsga2._environmental_selection(
        [low_loss, mid, high_acc],
        population_size=2,
    )

    survivor_numbers = {genome.genome_number for genome in survivors}
    assert len(survivors) == 2
    # Extreme points have infinite crowding; the interior midpoint is dropped.
    assert survivor_numbers == {1, 3}


def test_nsga2_tournament_prefers_lower_rank_then_crowding() -> None:
    """Tournament must prefer better Pareto rank, then larger crowding."""
    nsga2 = _make_nsga2()
    better_rank = _MockGenome(1, {"loss": 0.1, "accuracy": 0.9})
    worse_rank = _MockGenome(2, {"loss": 0.5, "accuracy": 0.5})
    better_rank.metadata["pareto_rank"] = 0
    better_rank.metadata["crowding_distance"] = 0.0
    worse_rank.metadata["pareto_rank"] = 1
    worse_rank.metadata["crowding_distance"] = math.inf

    assert nsga2._tournament_winner(better_rank, worse_rank) is better_rank

    crowded = _MockGenome(3, {"loss": 0.2, "accuracy": 0.8})
    sparse = _MockGenome(4, {"loss": 0.3, "accuracy": 0.7})
    crowded.metadata["pareto_rank"] = 0
    crowded.metadata["crowding_distance"] = 2.0
    sparse.metadata["pareto_rank"] = 0
    sparse.metadata["crowding_distance"] = 0.1

    assert nsga2._tournament_winner(crowded, sparse) is crowded


def test_nsga2_get_parents_rejects_invalid_and_insufficient() -> None:
    """Parent selection must guard invalid counts and undersized populations."""
    nsga2 = _make_nsga2(max_population_size=3)
    nsga2.population = [
        _MockGenome(1, {"loss": 0.1, "accuracy": 0.9}),
        _MockGenome(2, {"loss": 0.2, "accuracy": 0.8}),
    ]

    with pytest.raises(ValueError, match="n_parents"):
        nsga2.get_parents(n_parents=0)

    parents, metadata = nsga2.get_parents(n_parents=3)
    assert parents is None
    assert metadata is None

    parents, metadata = nsga2.get_parents(n_parents=2)
    assert parents is not None
    assert len(parents) == 2
    assert {parent.genome_number for parent in parents} == {1, 2}
    assert metadata is not None
    assert metadata["selection_algorithm"] == "nsga2"


def test_nsga2_insert_rejects_unevaluated_genome() -> None:
    """Inserting a genome without fitness must fail before selection."""
    nsga2 = _make_nsga2()
    with pytest.raises(ValueError, match="has not been evaluated"):
        nsga2.insert_genome(_MockGenome(1, fitness=None))


def test_nsga2_constructor_rejects_invalid_sizes() -> None:
    """Population construction must reject illegal size / objective counts."""
    with pytest.raises(ValueError, match="max_population_size"):
        NSGA2(
            max_population_size=0,
            objectives=OBJECTIVES,
            out_dir="unused",
            profiler=MagicMock(),
            save_all_genomes=False,
            save_pareto_front=False,
        )

    with pytest.raises(ValueError, match="at least two objectives"):
        NSGA2(
            max_population_size=2,
            objectives=[LOSS],
            out_dir="unused",
            profiler=MagicMock(),
            save_all_genomes=False,
            save_pareto_front=False,
        )

    with pytest.raises(ValueError, match="tournament_size"):
        NSGA2(
            max_population_size=2,
            objectives=OBJECTIVES,
            tournament_size=1,
            out_dir="unused",
            profiler=MagicMock(),
            save_all_genomes=False,
            save_pareto_front=False,
        )


# ---------------------------------------------------------------------------
# Duplicate topologies
# ---------------------------------------------------------------------------


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

    improved = _MockGenome(
        genome_number=3,
        fitness={"loss": 0.1, "accuracy": 0.9},
        topology_id="shared",
    )
    assert nsga2._handle_duplicate(improved) is True
    assert nsga2.population == []


def test_nsga2_keeps_incomparable_duplicate_for_selection() -> None:
    """Incomparable same-topology genomes must both enter selection."""
    nsga2 = _make_nsga2(max_population_size=2)
    existing = _MockGenome(
        genome_number=1,
        fitness={"loss": 0.1, "accuracy": 0.5},
        topology_id="shared",
    )
    rival = _MockGenome(
        genome_number=2,
        fitness={"loss": 0.5, "accuracy": 0.9},
        topology_id="shared",
    )
    nsga2.population = [existing]

    assert nsga2._handle_duplicate(rival) is None
    assert nsga2.insert_genome(rival) is True
    assert {genome.genome_number for genome in nsga2.population} == {1, 2}


# ---------------------------------------------------------------------------
# Islands
# ---------------------------------------------------------------------------


def test_moo_island_discards_stale_genomes_after_repopulation() -> None:
    """Genomes older than a repopulation event are discarded unless global best."""
    island = MultiObjectiveIsland(
        id=0,
        max_size=3,
        population_class=NSGA2,
        objectives=OBJECTIVES,
        tournament_size=2,
        population_kwargs=_island_kwargs(),
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


def test_moo_island_repopulate_clears_population_and_marks_status() -> None:
    """Repopulation must empty the island and record the cutoff genome number."""
    island = MultiObjectiveIsland(
        id=0,
        max_size=3,
        population_class=NSGA2,
        objectives=OBJECTIVES,
        tournament_size=2,
        population_kwargs=_island_kwargs(),
    )
    island.population = [
        _MockGenome(1, {"loss": 0.1, "accuracy": 0.9}),
    ]

    island.repopulate(repopulation_genome_number=50)

    assert island.status == "repopulating"
    assert island.repopulation_genome_number == 50
    assert island.population == []


def test_moo_islands_preserve_topology_neighbors_after_replacement() -> None:
    """Replacing SO islands with MOO islands must re-apply the topology.

    ``SteadyStateIslands.__init__`` assigns neighbors, then
    ``MultiObjectiveSteadyStateIslands`` replaces ``self.islands``. Without
    re-running ``assign_topology``, every island has no neighbors and
    inter-island crossover cannot run.
    """
    strategy = MultiObjectiveSteadyStateIslands(
        population_class=NSGA2,
        objectives=OBJECTIVES,
        n_islands=4,
        max_island_size=3,
        tournament_size=2,
        topology=["ring"],
        out_dir="unused",
        profiler=MagicMock(),
        population_kwargs=_island_kwargs(),
    )

    for island in strategy.islands:
        assert isinstance(island, MultiObjectiveIsland)
        assert hasattr(island, "neighbors")
        assert len(island.neighbors) == 2

    # Ring: island i connects to i-1 and i+1 (mod n).
    assert {neighbor.id for neighbor in strategy.islands[0].neighbors} == {1, 3}
    assert {neighbor.id for neighbor in strategy.islands[2].neighbors} == {1, 3}


def test_moo_island_compare_uses_dominance_then_first_objective() -> None:
    """Island bookkeeping compare must prefer dominance, then objective 0."""
    strategy = MultiObjectiveSteadyStateIslands(
        population_class=NSGA2,
        objectives=OBJECTIVES,
        n_islands=2,
        max_island_size=2,
        tournament_size=2,
        topology=["fully_connected"],
        out_dir="unused",
        profiler=MagicMock(),
        population_kwargs=_island_kwargs(),
    )

    better = _MockGenome(1, {"loss": 0.1, "accuracy": 0.9})
    worse = _MockGenome(2, {"loss": 0.5, "accuracy": 0.5})
    assert strategy._compare_genomes(better, worse) < 0
    assert strategy._compare_genomes(worse, better) > 0

    # Incomparable on the Pareto front: lower loss (objective 0) wins.
    low_loss = _MockGenome(3, {"loss": 0.1, "accuracy": 0.5})
    high_acc = _MockGenome(4, {"loss": 0.5, "accuracy": 0.9})
    assert strategy._compare_genomes(low_loss, high_acc) < 0
    assert strategy._compare_genomes(high_acc, low_loss) > 0

    twin_a = _MockGenome(5, {"loss": 0.2, "accuracy": 0.8})
    twin_b = _MockGenome(6, {"loss": 0.2, "accuracy": 0.8})
    assert strategy._compare_genomes(twin_a, twin_b) == 0


def test_moo_island_marks_full_after_reaching_capacity() -> None:
    """Island status must become ``full`` once local capacity is reached."""
    island = MultiObjectiveIsland(
        id=0,
        max_size=2,
        population_class=NSGA2,
        objectives=OBJECTIVES,
        tournament_size=2,
        population_kwargs=_island_kwargs(),
    )

    assert island.is_initializing()
    assert island.insert_genome(
        _MockGenome(1, {"loss": 0.1, "accuracy": 0.9}, topology_id="a")
    )
    assert island.status == "initializing"

    assert island.insert_genome(
        _MockGenome(2, {"loss": 0.2, "accuracy": 0.8}, topology_id="b")
    )
    assert island.status == "full"
    assert not island.is_initializing()
