"""Unit tests for :meth:`src.evolution.island.Island.best_neighbor`.

``best_neighbor`` scans an island's ``neighbors`` and returns the neighboring
island whose best genome (``population[0]``, since each island keeps its
population sorted with the best genome first) is best under the island's
``compare`` function. These tests exercise the normal selection behavior, tie
handling, and the degenerate edge cases (no neighbors, and a neighbor with an
empty population).
"""

from __future__ import annotations

from src.evolution.island import Island


class MockGenome:
    """A minimal stand-in genome ordered by a single scalar value.

    Args:
        value: The genome's fitness value; smaller is treated as better by
            :func:`compare`.
    """

    def __init__(self, value: float) -> None:
        self.value = value


def compare(genome1: MockGenome, genome2: MockGenome) -> int:
    """Orders genomes by ``value``, smaller-is-better.

    Args:
        genome1: The first genome to compare.
        genome2: The second genome to compare.

    Returns:
        A negative number if ``genome1`` is better (smaller value), a positive
        number if ``genome2`` is better, and 0 if they are equal.
    """

    return (genome1.value > genome2.value) - (genome1.value < genome2.value)


def make_island(island_id: int, population: list[MockGenome]) -> Island:
    """Builds an island with a given (already best-first) population.

    Args:
        island_id: The id assigned to the island.
        population: The island's population, expected to be ordered best-first
            (as the strategy maintains it); ``best_neighbor`` only reads
            ``population[0]``.

    Returns:
        A configured :class:`Island` with no neighbors set.
    """

    island = Island(id=island_id, max_size=10, compare=compare)
    island.population = population
    return island


def make_home_with_neighbors(best_values: list[float | None]) -> Island:
    """Builds a home island wired to one neighbor per entry in ``best_values``.

    Args:
        best_values: One entry per neighbor. A float becomes that neighbor's
            single-genome (``population[0]``) value; ``None`` gives the neighbor
            an empty population.

    Returns:
        A home :class:`Island` whose ``neighbors`` are the constructed islands,
        in order.
    """

    home = Island(id=0, max_size=10, compare=compare)
    neighbors: list[Island] = []
    for offset, value in enumerate(best_values):
        population = [] if value is None else [MockGenome(value)]
        neighbors.append(make_island(offset + 1, population))
    home.neighbors = neighbors
    return home


def test_best_neighbor_selects_minimum_value() -> None:
    """The neighbor with the best (smallest-value) genome is returned."""

    home = make_home_with_neighbors([5.0, 2.0, 8.0])
    assert home.best_neighbor() is home.neighbors[1]


def test_best_neighbor_when_first_is_best() -> None:
    """The first neighbor is returned when it holds the best genome."""

    home = make_home_with_neighbors([1.0, 5.0, 8.0])
    assert home.best_neighbor() is home.neighbors[0]


def test_best_neighbor_when_last_is_best() -> None:
    """The last neighbor is returned when it holds the best genome."""

    home = make_home_with_neighbors([8.0, 5.0, 1.0])
    assert home.best_neighbor() is home.neighbors[2]


def test_best_neighbor_single_neighbor() -> None:
    """With exactly one neighbor, that neighbor is returned."""

    home = make_home_with_neighbors([3.0])
    assert home.best_neighbor() is home.neighbors[0]


def test_best_neighbor_tie_returns_first_best() -> None:
    """On a tie for best, the earliest such neighbor is returned.

    ``best_neighbor`` only replaces its running best on a strictly better
    comparison, so the first neighbor achieving the best value wins.
    """

    home = make_home_with_neighbors([3.0, 1.0, 1.0])
    assert home.best_neighbor() is home.neighbors[1]


def test_best_neighbor_negative_values() -> None:
    """Selection works when fitness values are negative (e.g. RL losses)."""

    home = make_home_with_neighbors([-1.0, -7.0, -3.0])
    assert home.best_neighbor() is home.neighbors[1]


def test_best_neighbor_returns_none_when_no_populated_neighbors() -> None:
    """best_neighbor returns None when there is no populated neighbor to pick.

    With no neighbors at all, or with neighbors that all have empty populations,
    there is no best neighbor to return, so it yields None rather than raising.
    Callers guard against this None return.
    """

    # no neighbors at all
    home = Island(id=0, max_size=10, compare=compare)
    assert home.neighbors == []
    assert home.best_neighbor() is None

    # neighbors exist but all of them have empty populations
    home = make_home_with_neighbors([None, None, None])
    assert home.best_neighbor() is None


def test_best_neighbor_skips_empty_neighbor_populations() -> None:
    """Neighbors with empty populations are skipped during selection.

    ``best_neighbor`` ignores any neighbor whose population is empty and returns
    the best among the neighbors that actually hold genomes.
    """

    # a trailing empty neighbor is skipped, leaving the only populated one
    home = make_home_with_neighbors([5.0, None])
    assert home.best_neighbor() is home.neighbors[0]

    # an empty neighbor in the middle is skipped in favor of a better populated
    # neighbor that follows it
    home = make_home_with_neighbors([5.0, None, 2.0])
    assert home.best_neighbor() is home.neighbors[2]


def test_best_neighbor_does_not_mutate_neighbors() -> None:
    """Selecting a best neighbor leaves the neighbor list unchanged."""

    home = make_home_with_neighbors([5.0, 2.0, 8.0])
    before = list(home.neighbors)
    home.best_neighbor()
    assert home.neighbors == before
