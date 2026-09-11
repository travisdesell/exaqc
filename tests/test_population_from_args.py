"""Tests for :meth:`src.evolution.population_strategy.PopulationStrategy.from_args`.

Every entry point builds its population strategy the same way, through
``PopulationStrategy.from_args``. These tests pin that shared behavior -- the
sub-command chosen selects the concrete strategy, and the strategy's output
directory is created -- so the entry-point tests do not have to.
"""

from __future__ import annotations

from argparse import Namespace

from src.circuits.circuit import CircuitGenome
from src.evolution.population_strategy import PopulationStrategy
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """Orders genomes by a ``loss`` fitness (lower first)."""

    return genome1.fitness["loss"] - genome2.fitness["loss"]


def steady_state_args(out_dir: str, **overrides) -> Namespace:
    """Builds a parsed-args namespace selecting the steady-state strategy."""

    values = {
        "population_strategy": "steady_state",
        "max_population_size": 5,
        "out_dir": out_dir,
        "save_training_plot": False,
    }
    values.update(overrides)
    return Namespace(**values)


def islands_args(out_dir: str, **overrides) -> Namespace:
    """Builds a parsed-args namespace selecting the islands strategy."""

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
        "out_dir": out_dir,
        "save_training_plot": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_from_args_builds_steady_state_population(tmp_path) -> None:
    """A ``steady_state`` sub-command yields a SteadyStatePopulation."""

    population = PopulationStrategy.from_args(steady_state_args(str(tmp_path)), compare)

    assert isinstance(population, SteadyStatePopulation)
    assert not isinstance(population, SteadyStateIslands)


def test_from_args_builds_islands_population(tmp_path) -> None:
    """An ``islands`` sub-command yields a SteadyStateIslands strategy."""

    population = PopulationStrategy.from_args(islands_args(str(tmp_path)), compare)

    assert isinstance(population, SteadyStateIslands)


def test_from_args_creates_the_output_directory(tmp_path) -> None:
    """The strategy's output directory is created if it does not exist."""

    out_dir = tmp_path / "fresh" / "nested" / "out"
    assert not out_dir.exists()

    PopulationStrategy.from_args(steady_state_args(str(out_dir)), compare)

    assert out_dir.is_dir()
