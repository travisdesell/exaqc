"""Tests for :meth:`src.evolution.population_strategy.PopulationStrategy.from_args`.

Every entry point builds its population strategy the same way, through
``PopulationStrategy.from_args``. These tests pin that shared behavior -- the
sub-command chosen selects the concrete strategy -- so the entry-point tests do
not have to. (The run's output directory is owned by ``GenomeArchive``; see
``tests/test_genome_archive.py``.)
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from src.circuits.circuit import CircuitGenome
from src.evolution.population_strategy import PopulationStrategy
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """Orders genomes by a ``loss`` fitness (lower first).

    Args:
        genome1: The first genome.
        genome2: The second genome.

    Returns:
        Negative when ``genome1`` is better, positive when ``genome2`` is.
    """

    return genome1.fitness["loss"] - genome2.fitness["loss"]


def steady_state_args(**overrides: Any) -> Namespace:
    """Builds a parsed-args namespace selecting the steady-state strategy.

    Args:
        **overrides: Values replacing the defaults below.

    Returns:
        The namespace ``from_args`` reads.
    """

    values = {
        "population_strategy": "steady_state",
        "max_population_size": 5,
    }
    values.update(overrides)
    return Namespace(**values)


def islands_args(**overrides: Any) -> Namespace:
    """Builds a parsed-args namespace selecting the islands strategy.

    Args:
        **overrides: Values replacing the defaults below.

    Returns:
        The namespace ``from_args`` reads.
    """

    values = {
        "population_strategy": "islands",
        "n_islands": 2,
        "max_island_size": 3,
        "genomes_before_extinction": 100,
        "genomes_for_next_extinction": 100,
        "islands_to_extinct": 1,
        "primary_parent": "best",
        "intra_island_crossover_rate": 0.5,
        "topology": ["fully_connected"],
    }
    values.update(overrides)
    return Namespace(**values)


def test_from_args_builds_steady_state_population() -> None:
    """A ``steady_state`` sub-command yields a SteadyStatePopulation."""

    population = PopulationStrategy.from_args(steady_state_args(), compare)

    assert isinstance(population, SteadyStatePopulation)
    assert not isinstance(population, SteadyStateIslands)
    assert population.max_population_size == 5


def test_from_args_builds_islands_population() -> None:
    """An ``islands`` sub-command yields a SteadyStateIslands strategy."""

    population = PopulationStrategy.from_args(islands_args(), compare)

    assert isinstance(population, SteadyStateIslands)
    assert len(population.islands) == 2


def test_from_args_writes_nothing_to_disk(tmp_path, monkeypatch) -> None:
    """Building a strategy no longer creates an output directory.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)

    PopulationStrategy.from_args(steady_state_args(), compare)
    PopulationStrategy.from_args(islands_args(), compare)

    assert list(tmp_path.iterdir()) == []
