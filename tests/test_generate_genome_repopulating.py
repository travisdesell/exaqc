"""Regression tests for choosing parents while islands repopulate.

A Walker2d run with a ``tree 3`` topology crashed its MPI master: island 12, a
leaf still repopulating after one extinction, had only island 3 as a neighbor,
and island 3 was emptied by the next extinction. ``get_parent`` then returned
``(None, None)`` and ``generate_genome``'s mutation branch tried to copy ``None``.
The crash was not reported, so every worker waited on the dead master until
the job hit its time limit. These tests rebuild that situation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.evolution.exaqc import EXAQC
from src.evolution.steady_state_islands import SteadyStateIslands
from tests.test_exaqc_insert_archive import build_search, compare


def build_tree_search() -> tuple[EXAQC, SteadyStateIslands]:
    """Builds a search whose leaf island 12 has only empty neighbors.

    The 13 islands form a tree with 3 children per island, so island 12 is a
    leaf whose only neighbor is island 3. Islands 3 and 12 are repopulating and
    empty, as they were after the two extinctions in the crashed run. The other
    islands are full of copies of the search's initial genome, and the next
    island asked for a parent is 12.

    Returns:
        The search and its island population strategy.
    """

    population = SteadyStateIslands(
        n_islands=13,
        max_island_size=2,
        compare=compare,
        topology=["tree", "3"],
    )
    search = build_search(population, MagicMock())

    for island in population.islands:
        if island.id in (3, 12):
            island.repopulate(repopulation_genome_number=10)
            continue

        island.population = []
        for number in range(2):
            genome = search.initial_genome.copy(genome_number=number)
            genome.encoder = search.initial_encoder.copy()
            genome.decoder = search.initial_decoder.copy()
            genome.fitness = {"loss": float(number), "target_metric": -float(number)}
            island.population.append(genome)
        island.status = "full"

    population.current_island = 12
    return search, population


def test_a_repopulating_island_with_empty_neighbors_gives_no_parent() -> None:
    """get_parent returns ``(None, None)`` when no neighbor has a genome."""

    _, population = build_tree_search()

    assert [neighbor.id for neighbor in population.islands[12].neighbors] == [3]
    assert population.get_parent() == (None, None)


def test_mutation_moves_on_when_an_island_has_no_parent() -> None:
    """generate_genome skips a parentless island instead of mutating ``None``.

    Every draw picks mutation (not crossover), so the first attempt lands on
    island 12, finds no parent, and the next one uses island 0.
    """

    search, population = build_tree_search()
    parent_calls: list[Any] = []
    get_parent = population.get_parent

    def recording_get_parent(**kwargs: Any) -> Any:
        """Records each parent draw and defers to the real ``get_parent``."""
        result = get_parent(**kwargs)
        parent_calls.append(result)
        return result

    with (
        patch("src.evolution.exaqc.random.uniform", return_value=0.99),
        patch.object(population, "get_parent", side_effect=recording_get_parent),
    ):
        child = search.generate_genome()

    assert parent_calls[0] == (None, None)
    assert len(parent_calls) == 2
    assert child.metadata["target_island_id"] == 0
    assert child.genome_number is not None


def test_a_parent_from_an_initializing_island_raises() -> None:
    """Asking an initializing island for a parent raises instead of exiting.

    ``exit`` on the MPI master would leave the workers waiting forever, while
    an exception aborts the job when run under ``python -m mpi4py``.
    """

    population = SteadyStateIslands(
        n_islands=2, max_island_size=2, compare=compare, topology=["ring"]
    )

    with pytest.raises(RuntimeError, match="initializing island"):
        population.get_parent()
