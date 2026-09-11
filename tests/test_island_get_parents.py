"""Unit tests for :meth:`SteadyStateIslands.get_parents` selection logic.

``get_parents`` picks a target island (round-robin) and returns ``n_parents``
parents for a crossover, choosing between **intra-island** crossover (all parents
from the target island) and **inter-island** crossover (one parent from the
target island, the rest sampled from neighboring islands). The behavior under
test:

* fall back to **intra**-island crossover when inter-island is impossible
  (the neighboring islands hold too few genomes for ``n_parents``),
* fall back to **inter**-island crossover when intra-island is impossible
  (the target island's own population is too small for ``n_parents``),
* return ``(None, None)`` when neither is possible,
* otherwise return the parents plus metadata carrying the target island id and
  the crossover type.

The tests construct each situation deterministically by populating the islands
directly and, where both crossovers are possible, pinning
``random.uniform`` so the intra/inter coin flip is controlled.
"""

from __future__ import annotations

from functools import cmp_to_key

import pytest

import src.evolution.steady_state_islands as steady_state_islands
from src.evolution.island import Island
from src.evolution.steady_state_islands import SteadyStateIslands


class MockGenome:
    """A minimal stand-in genome ordered by a single ``loss`` value.

    Args:
        loss: The genome's fitness loss; smaller is better under
            :func:`compare`.
        genome_number: A unique identifier for the genome.
    """

    def __init__(self, loss: float, genome_number: int = 0) -> None:
        self.fitness = {"loss": loss, "target_metric": -loss}
        self.genome_number = genome_number
        self.metadata: dict[str, object] = {}

    def has_same_gates(self, other: "MockGenome") -> bool:
        """Treats every mock genome as structurally distinct.

        Args:
            other: The genome being compared against during insertion.

        Returns:
            False always, so insertion never treats two mocks as duplicates.
        """

        return False


def compare(genome1: MockGenome, genome2: MockGenome) -> int:
    """Orders genomes by ``loss``, smaller-is-better.

    Args:
        genome1: The first genome to compare.
        genome2: The second genome to compare.

    Returns:
        A negative number if ``genome1`` is better (smaller loss), a positive
        number if ``genome2`` is better, and 0 if they are equal.
    """

    loss1 = genome1.fitness["loss"]
    loss2 = genome2.fitness["loss"]
    return (loss1 > loss2) - (loss1 < loss2)


#: Monotonic counter so every created genome gets a distinct number/loss.
_next_number = 0


def make_population(size: int) -> list[MockGenome]:
    """Builds a population of ``size`` genomes with distinct losses.

    Args:
        size: How many genomes to create.

    Returns:
        A list of ``size`` :class:`MockGenome` objects with unique, increasing
        losses and genome numbers.
    """

    global _next_number
    population = []
    for _ in range(size):
        population.append(
            MockGenome(loss=0.01 * (_next_number + 1), genome_number=_next_number)
        )
        _next_number += 1
    return population


def make_strategy(
    n_islands: int,
    intra_island_crossover_rate: float = 0.5,
    primary_parent: str = "best",
) -> SteadyStateIslands:
    """Builds a fully connected island strategy for testing ``get_parents``.

    Every island neighbors every other (``fully_connected`` topology), islands
    start empty, and ``current_island`` is 0.

    Args:
        n_islands: Number of islands.
        intra_island_crossover_rate: The intra/inter crossover coin-flip rate.
        primary_parent: ``"best"`` or ``"island"`` primary-parent policy.

    Returns:
        A configured :class:`SteadyStateIslands` with empty islands.
    """

    return SteadyStateIslands(
        n_islands=n_islands,
        max_island_size=32,
        compare=compare,
        intra_island_crossover_rate=intra_island_crossover_rate,
        primary_parent=primary_parent,
        topology=["fully_connected"],
        out_dir=None,
    )


def fill_island(island: Island, size: int, status: str = "full") -> None:
    """Populates an island with ``size`` genomes and sets its status.

    Args:
        island: The island to populate.
        size: How many genomes to give it.
        status: The island status to set (``"full"`` for these tests).

    Returns:
        None. Sets ``island.population`` and ``island.status`` in place.
    """

    island.population = make_population(size)
    island.status = status


# ---------------------------------------------------------------------------
# Fall back to intra-island crossover when inter-island is impossible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parents", [2, 3, 4])
def test_falls_back_to_intra_when_inter_impossible(n_parents: int) -> None:
    """With too few neighbor genomes, crossover falls back to intra-island.

    The target island has enough genomes for ``n_parents`` but its neighbors
    are empty, so inter-island crossover is impossible and all parents must come
    from the target island.
    """

    strategy = make_strategy(n_islands=3)
    fill_island(strategy.islands[0], n_parents + 1)  # target, enough for intra
    # neighbors (islands 1 and 2) stay empty -> inter-island impossible

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "intra-island"
    assert metadata["target_island_id"] == 0

    # every parent came from the target island's population
    target_ids = {id(genome) for genome in strategy.islands[0].population}
    assert all(id(parent) in target_ids for parent in parents)
    # parents are unique and, under "best", ordered best-first
    assert len({id(parent) for parent in parents}) == n_parents
    assert parents == sorted(parents, key=cmp_to_key(compare))


# ---------------------------------------------------------------------------
# Fall back to inter-island crossover when intra-island is impossible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parents", [2, 3, 4])
def test_falls_back_to_inter_when_intra_impossible(n_parents: int) -> None:
    """With too small a target population, crossover falls back to inter-island.

    The target island holds a single genome (too few for ``n_parents``) while
    its neighbors hold plenty, so intra-island crossover is impossible and the
    remaining parents are drawn from neighboring islands.
    """

    strategy = make_strategy(n_islands=3)
    fill_island(strategy.islands[0], 1)  # target too small for intra
    fill_island(strategy.islands[1], n_parents)  # neighbors: ample genomes
    fill_island(strategy.islands[2], n_parents)

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "inter-island"
    assert metadata["target_island_id"] == 0

    # exactly one parent from the target island, the rest from neighbors
    target_ids = {id(genome) for genome in strategy.islands[0].population}
    neighbor_ids = {
        id(genome) for island in strategy.islands[1:] for genome in island.population
    }
    from_target = [parent for parent in parents if id(parent) in target_ids]
    from_neighbors = [parent for parent in parents if id(parent) in neighbor_ids]
    assert len(from_target) == 1
    assert len(from_neighbors) == n_parents - 1

    assert len({id(parent) for parent in parents}) == n_parents
    assert parents == sorted(parents, key=cmp_to_key(compare))


# ---------------------------------------------------------------------------
# Neither crossover possible -> (None, None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parents", [2, 3, 4])
def test_returns_none_when_neither_possible(n_parents: int) -> None:
    """Too few genomes for both intra- and inter-island crossover yields None.

    The target island has ``n_parents - 1`` genomes (too few for intra), and the
    single neighbor has ``n_parents - 2`` (too few to complete inter), so neither
    crossover can produce ``n_parents`` parents.
    """

    strategy = make_strategy(n_islands=2)
    fill_island(strategy.islands[0], n_parents - 1)  # too few for intra
    fill_island(strategy.islands[1], max(0, n_parents - 2))  # too few for inter

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is None
    assert metadata is None


# ---------------------------------------------------------------------------
# Both possible -> the intra/inter coin flip decides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parents", [2, 3])
def test_uses_intra_when_both_possible_and_flip_selects_intra(
    n_parents: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both are possible and the coin flip favors intra, intra is used."""

    strategy = make_strategy(n_islands=3, intra_island_crossover_rate=0.5)
    fill_island(strategy.islands[0], n_parents)  # intra possible
    fill_island(strategy.islands[1], n_parents)  # inter also possible
    fill_island(strategy.islands[2], n_parents)

    # uniform() < 0.5 -> do_intra_island is True
    monkeypatch.setattr(steady_state_islands.random, "uniform", lambda low, high: 0.0)

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "intra-island"
    assert metadata["target_island_id"] == 0
    target_ids = {id(genome) for genome in strategy.islands[0].population}
    assert all(id(parent) in target_ids for parent in parents)


@pytest.mark.parametrize("n_parents", [2, 3])
def test_uses_inter_when_both_possible_and_flip_selects_inter(
    n_parents: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both are possible and the coin flip favors inter, inter is used."""

    strategy = make_strategy(n_islands=3, intra_island_crossover_rate=0.5)
    fill_island(strategy.islands[0], n_parents)  # intra possible
    fill_island(strategy.islands[1], n_parents)  # inter also possible
    fill_island(strategy.islands[2], n_parents)

    # uniform() >= 0.5 -> do_intra_island is False
    monkeypatch.setattr(steady_state_islands.random, "uniform", lambda low, high: 0.9)

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "inter-island"
    assert metadata["target_island_id"] == 0
    # exactly one parent from the target island
    target_ids = {id(genome) for genome in strategy.islands[0].population}
    assert len([parent for parent in parents if id(parent) in target_ids]) == 1


# ---------------------------------------------------------------------------
# Metadata / round-robin target selection
# ---------------------------------------------------------------------------


def test_target_island_id_advances_round_robin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successive calls target islands in round-robin order via the metadata."""

    n_parents = 2
    strategy = make_strategy(n_islands=3)
    for island in strategy.islands:
        fill_island(island, n_parents)

    # force intra-island crossover so each call succeeds deterministically
    monkeypatch.setattr(steady_state_islands.random, "uniform", lambda low, high: 0.0)

    _, first = strategy.get_parents(n_parents)
    _, second = strategy.get_parents(n_parents)
    _, third = strategy.get_parents(n_parents)
    _, fourth = strategy.get_parents(n_parents)

    assert first["target_island_id"] == 0
    assert second["target_island_id"] == 1
    assert third["target_island_id"] == 2
    # wraps back around to the first island
    assert fourth["target_island_id"] == 0


# ---------------------------------------------------------------------------
# Empty neighbor populations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parents", [2, 3, 4])
def test_inter_island_draws_only_from_non_empty_neighbors(n_parents: int) -> None:
    """Inter-island crossover ignores empty neighbors and uses the rest.

    The target island is too small for intra-island crossover, and only one of
    its neighbors holds genomes (the others are empty), so all non-target
    parents must be sampled from that single non-empty neighbor.
    """

    strategy = make_strategy(n_islands=4)
    fill_island(strategy.islands[0], 1)  # target too small for intra
    # island 1 and island 3 are left empty; island 2 supplies the parents
    fill_island(strategy.islands[2], n_parents)

    assert strategy.islands[1].population == []
    assert strategy.islands[3].population == []

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "inter-island"
    assert metadata["target_island_id"] == 0

    target_ids = {id(genome) for genome in strategy.islands[0].population}
    non_empty_neighbor_ids = {id(genome) for genome in strategy.islands[2].population}
    from_target = [parent for parent in parents if id(parent) in target_ids]
    from_neighbor = [
        parent for parent in parents if id(parent) in non_empty_neighbor_ids
    ]
    assert len(from_target) == 1
    assert len(from_neighbor) == n_parents - 1
    # nothing was invented from the empty neighbors
    assert len({id(parent) for parent in parents}) == n_parents


@pytest.mark.parametrize("n_parents", [3, 4])
def test_inter_island_impossible_when_empty_neighbors_starve_the_pool(
    n_parents: int,
) -> None:
    """Empty neighbors can drop the pool below what inter-island needs.

    The target is too small for intra-island crossover and, with most neighbors
    empty, the remaining neighbor genomes are fewer than the ``n_parents - 1``
    inter-island crossover requires, so ``get_parents`` returns ``(None, None)``.
    """

    strategy = make_strategy(n_islands=4)
    fill_island(strategy.islands[0], n_parents - 1)  # too small for intra
    # only one neighbor has genomes, and too few to complete inter-island
    fill_island(strategy.islands[1], n_parents - 2)
    # islands 2 and 3 remain empty

    assert strategy.islands[2].population == []
    assert strategy.islands[3].population == []

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is None
    assert metadata is None


@pytest.mark.parametrize("n_parents", [2, 3, 4])
def test_intra_fallback_when_partial_empty_neighbors_are_too_small(
    n_parents: int,
) -> None:
    """Intra-island is used when partially-empty neighbors are still too small.

    Some neighbors are empty and the rest together hold fewer than ``n_parents``
    genomes, so inter-island crossover is impossible and all parents come from
    the (sufficiently large) target island.
    """

    strategy = make_strategy(n_islands=3)
    fill_island(strategy.islands[0], n_parents + 1)  # target: enough for intra
    fill_island(strategy.islands[1], n_parents - 1)  # neighbor total < n_parents
    # island 2 is left empty

    assert strategy.islands[2].population == []

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "intra-island"
    assert metadata["target_island_id"] == 0

    target_ids = {id(genome) for genome in strategy.islands[0].population}
    assert all(id(parent) in target_ids for parent in parents)


@pytest.mark.parametrize("n_parents", [2, 3])
def test_inter_island_with_empty_neighbor_when_both_possible(
    n_parents: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced inter-island crossover skips an empty neighbor without error.

    Both crossovers are possible (the target has enough genomes and the
    non-empty neighbors hold plenty), and the coin flip is pinned to inter; an
    empty neighbor in the mix must simply be skipped.
    """

    strategy = make_strategy(n_islands=4, intra_island_crossover_rate=0.5)
    fill_island(strategy.islands[0], n_parents)  # intra possible
    # island 1 empty; islands 2 and 3 supply inter-island parents
    fill_island(strategy.islands[2], n_parents)
    fill_island(strategy.islands[3], n_parents)

    assert strategy.islands[1].population == []

    # uniform() >= 0.5 -> do_intra_island is False -> inter-island
    monkeypatch.setattr(steady_state_islands.random, "uniform", lambda low, high: 0.9)

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "inter-island"

    empty_neighbor_ids = {id(genome) for genome in strategy.islands[1].population}
    assert empty_neighbor_ids == set()
    assert len({id(parent) for parent in parents}) == n_parents


# ---------------------------------------------------------------------------
# Repopulating target island (parents are sourced from the best neighbor)
# ---------------------------------------------------------------------------


def make_repopulating_target(strategy: SteadyStateIslands) -> None:
    """Marks island 0 as repopulating with an empty population.

    Args:
        strategy: The strategy whose island 0 is put into the repopulating
            state (empty population), as after an extinction event.

    Returns:
        None. Mutates ``strategy.islands[0]`` in place.
    """

    strategy.islands[0].status = "repopulating"
    strategy.islands[0].population = []


def test_repopulating_target_returns_none_when_all_neighbors_empty() -> None:
    """A repopulating target with only empty neighbors yields ``(None, None)``.

    ``best_neighbor`` returns ``None`` (no populated neighbor) and there are no
    genomes anywhere for inter-island crossover, so ``get_parents`` returns
    ``(None, None)`` rather than raising.
    """

    strategy = make_strategy(n_islands=3)
    make_repopulating_target(strategy)
    # neighbors (islands 1 and 2) are left empty

    parents, metadata = strategy.get_parents(2)

    assert parents is None
    assert metadata is None


@pytest.mark.parametrize("n_parents", [2, 3])
def test_repopulating_target_draws_parents_from_populated_neighbors(
    n_parents: int,
) -> None:
    """A repopulating target sources all its parents from populated neighbors.

    The target island is empty (repopulating), so the first parent comes from
    its best populated neighbor and the rest from neighboring islands via
    inter-island crossover.
    """

    strategy = make_strategy(n_islands=3)
    make_repopulating_target(strategy)
    fill_island(strategy.islands[1], n_parents)
    fill_island(strategy.islands[2], n_parents)

    parents, metadata = strategy.get_parents(n_parents)

    assert parents is not None
    assert len(parents) == n_parents
    assert metadata["crossover_type"] == "inter-island"
    assert metadata["target_island_id"] == 0

    # the empty target contributes nothing; every parent comes from a neighbor
    neighbor_ids = {
        id(genome) for island in strategy.islands[1:] for genome in island.population
    }
    assert all(id(parent) in neighbor_ids for parent in parents)


# ---------------------------------------------------------------------------
# insert_genome routes to the target island recorded in metadata
# ---------------------------------------------------------------------------


def test_insert_genome_routes_to_target_island() -> None:
    """A genome with a ``target_island_id`` is inserted into that island.

    Genomes record the island they were bred for (via ``get_parents``), and
    insertion must honor that rather than falling back to the smallest island.
    Island 0 is made the largest here so the smallest-island fallback would send
    the genome elsewhere if the target id were ignored.
    """

    strategy = make_strategy(n_islands=3)
    for _ in range(3):
        strategy.islands[0].insert_genome(MockGenome(0.5))
    assert len(strategy.islands[0].population) == 3

    targeted = MockGenome(0.4)
    targeted.metadata["target_island_id"] = 0
    strategy.insert_genome(targeted, current_genome_number=100)

    assert targeted in strategy.islands[0].population
    assert targeted not in strategy.islands[1].population
    assert targeted not in strategy.islands[2].population
