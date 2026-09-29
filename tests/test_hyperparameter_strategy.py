"""Tests for the hyperparameter strategies EXAQC chooses children's training settings with."""

from __future__ import annotations

import argparse
import math
import random

from functools import partial
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.evolution.exaqc import EXAQC
from src.evolution.hyperparameter_strategy import (
    FixedHyperparameters,
    HyperparameterStrategy,
    SimplexHyperparameters,
    TunedHyperparameter,
)
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.examples import classification
from tests.test_exaqc_insert_archive import build_search, compare

LEARNING_RATE = TunedHyperparameter.parse("learning_rate=log:1e-3:5e-2:1e-5:0.3")

#: The arguments ``classification`` requires.
REQUIRED_ARGUMENTS = [
    "--dataset",
    "iris",
    "--input_qubits",
    "4",
    "--output_qubits",
    "2",
    "-ms",
    "uniform",
    "1",
    "3",
    "-ps",
    "uniform",
    "2",
    "3",
    "--out_dir",
    "out",
]


def genome_with(number: int, **hyperparameters: Any) -> MagicMock:
    """Builds a stand-in genome carrying only a number and hyperparameters.

    Args:
        number: The genome number.
        **hyperparameters: The genome's hyperparameters.

    Returns:
        The stand-in genome.
    """

    genome = MagicMock()
    genome.genome_number = number
    genome.hyperparameters = dict(hyperparameters)
    return genome


def test_parse_reads_both_ranges() -> None:
    """A five-field specification sets the burn-in and full ranges."""

    assert LEARNING_RATE == TunedHyperparameter(
        name="learning_rate",
        scale="log",
        initial_min=1e-3,
        initial_max=5e-2,
        min=1e-5,
        max=0.3,
    )


def test_parse_defaults_the_full_range_to_the_burn_in_range() -> None:
    """A three-field specification clamps values to the burn-in range."""

    parsed = TunedHyperparameter.parse("epochs=int:5:20")
    assert (parsed.min, parsed.max) == (5.0, 20.0)


@pytest.mark.parametrize(
    "spec",
    [
        "learning_rate",
        "learning_rate=log:1e-3",
        "learning_rate=cubic:1:2",
        "learning_rate=log:a:b",
        "learning_rate=log:1e-2:1e-3",
        "learning_rate=log:1e-3:1e-2:1e-2:1e-1",
        "weight_decay=log:0:1e-3",
    ],
)
def test_parse_rejects_bad_specifications(spec: str) -> None:
    """Malformed, unordered or non-positive log specifications are rejected.

    Args:
        spec: The bad specification.
    """

    with pytest.raises(ValueError):
        TunedHyperparameter.parse(spec)


def test_search_space_round_trips_and_clamps() -> None:
    """Log values step in log10 space and come back clamped to the full range."""

    assert LEARNING_RATE.to_search_space(1e-2) == pytest.approx(-2.0)
    assert LEARNING_RATE.from_search_space(-2.0) == pytest.approx(1e-2)
    assert LEARNING_RATE.from_search_space(5.0) == 0.3
    assert LEARNING_RATE.from_search_space(-9.0) == 1e-5

    epochs = TunedHyperparameter.parse("epochs=int:5:20:1:50")
    assert epochs.from_search_space(7.6) == 8
    assert isinstance(epochs.from_search_space(7.6), int)


def test_fixed_returns_a_copy_of_the_base() -> None:
    """The fixed strategy never changes, or shares, the configured values."""

    base = {"learning_rate": 0.01}
    generated = FixedHyperparameters().generate(base, [], initializing=False)
    assert generated == base and generated is not base
    assert FixedHyperparameters().run_info() == {}


def test_burn_in_draws_from_the_initial_range() -> None:
    """While initializing, tuned values fall in the burn-in range, others are copied."""

    strategy = SimplexHyperparameters([LEARNING_RATE], rng=random.Random(1))
    metadata: dict[str, Any] = {}
    for _ in range(200):
        generated = strategy.generate(
            {"learning_rate": 0.005, "epochs": 3}, [], True, metadata
        )
        assert 1e-3 <= generated["learning_rate"] <= 5e-2
        assert generated["epochs"] == 3
    assert metadata["hyperparameter_generation"]["phase"] == "burn_in"


def test_simplex_step_follows_the_paper() -> None:
    """``h_avg + r * (h_best - h_avg)`` in log space, best being the lowest index."""

    population = [
        genome_with(10, learning_rate=1e-2),
        genome_with(11, learning_rate=1e-3),
        genome_with(12, learning_rate=1e-4),
    ]
    rng = MagicMock()
    rng.sample.return_value = [2, 0, 1]
    rng.random.return_value = 0.75  # r = 0.75 * 2.0 - 0.5 = 1.0

    strategy = SimplexHyperparameters([LEARNING_RATE], n_genomes=3, rng=rng)
    metadata: dict[str, Any] = {}
    generated = strategy.generate({"learning_rate": 5e-3}, population, False, metadata)

    # the others average -3.5 in log10; r = 1 lands on the best genome's -2
    assert generated["learning_rate"] == pytest.approx(1e-2)
    assert metadata["hyperparameter_generation"] == {
        "strategy": "simplex",
        "phase": "simplex",
        "best_genome": 10,
        "other_genomes": [11, 12],
        "r": pytest.approx(1.0),
    }

    rng.random.return_value = 0.25  # r = 0: the average of the others
    generated = strategy.generate({"learning_rate": 5e-3}, population, False)
    assert math.log10(generated["learning_rate"]) == pytest.approx(-3.5)


def test_simplex_falls_back_to_burn_in_with_too_few_genomes() -> None:
    """With fewer genomes than a step needs, values are drawn as in the burn-in."""

    strategy = SimplexHyperparameters([LEARNING_RATE], n_genomes=4)
    metadata: dict[str, Any] = {}
    strategy.generate({"learning_rate": 5e-3}, [genome_with(1)], False, metadata)
    assert metadata["hyperparameter_generation"]["phase"] == "burn_in"


def test_simplex_uses_the_base_value_when_a_genome_has_none() -> None:
    """A genome without the tuned key counts as holding the configured value."""

    population = [genome_with(1), genome_with(2)]
    rng = MagicMock()
    rng.sample.return_value = [0, 1]
    rng.random.return_value = 0.3
    strategy = SimplexHyperparameters([LEARNING_RATE], n_genomes=2, rng=rng)
    generated = strategy.generate({"learning_rate": 5e-3}, population, False)
    assert generated["learning_rate"] == pytest.approx(5e-3)


def test_simplex_rejects_bad_settings() -> None:
    """Fewer than two genomes, or tuning a key twice, is rejected."""

    with pytest.raises(ValueError):
        SimplexHyperparameters([LEARNING_RATE], n_genomes=1)
    with pytest.raises(ValueError):
        SimplexHyperparameters([LEARNING_RATE, LEARNING_RATE])


def test_island_children_learn_from_their_island() -> None:
    """An island search hands SHO the child's target island, not every genome."""

    population = SteadyStateIslands(n_islands=2, max_island_size=2, compare=compare)
    population.islands[0].population = ["a0", "a1"]
    population.islands[1].population = ["b0"]
    assert population.get_population_for_child({"target_island_id": 0}) == [
        "a0",
        "a1",
    ]
    assert population.get_population_for_child({"target_island_id": 1}) == ["b0"]


def test_exaqc_generates_children_with_the_strategy() -> None:
    """EXAQC records SHO's run_info and gives burn-in children drawn values."""

    archive = MagicMock()
    population = SteadyStatePopulation(max_population_size=3, compare=compare)
    strategy = SimplexHyperparameters([LEARNING_RATE], n_genomes=2)
    with patch(
        "tests.test_exaqc_insert_archive.EXAQC",
        partial(EXAQC, hyperparameter_strategy=strategy),
    ):
        search = build_search(population, archive)

    assert search.hyperparameter_strategy is strategy
    run_info = archive.set_run_info.call_args.kwargs
    assert run_info["hyperparameter_strategy"]["tuned"] == [LEARNING_RATE.as_dict()]

    child = search.generate_genome()
    assert 1e-3 <= child.hyperparameters["learning_rate"] <= 5e-2
    assert child.metadata["hyperparameter_generation"]["phase"] == "burn_in"
    assert child.hyperparameters["epochs"] == 1


def test_exaqc_defaults_to_fixed_hyperparameters() -> None:
    """Without a strategy every child gets the configured hyperparameters."""

    search = build_search(
        SteadyStatePopulation(max_population_size=3, compare=compare), None
    )
    assert isinstance(search.hyperparameter_strategy, FixedHyperparameters)
    child = search.generate_genome()
    assert child.hyperparameters == search.hyperparameters
    assert "hyperparameter_generation" not in child.metadata


def test_classification_parser_defaults_to_fixed() -> None:
    """Without the new flags a classification run keeps fixed hyperparameters."""

    args = classification.build_parser().parse_args(
        [*REQUIRED_ARGUMENTS, "steady_state"]
    )
    assert args.hyperparameter_strategy == "fixed"
    assert isinstance(HyperparameterStrategy.from_args(args), FixedHyperparameters)
    assert isinstance(
        HyperparameterStrategy.from_args(argparse.Namespace()), FixedHyperparameters
    )


def test_classification_parser_builds_sho() -> None:
    """``--hyperparameter_strategy simplex`` tunes the learning rate by default."""

    parser = classification.build_parser()
    args = parser.parse_args(
        [*REQUIRED_ARGUMENTS, "--hyperparameter_strategy", "simplex", "steady_state"]
    )
    assert args.sho_tune == classification.DEFAULT_SHO_TUNE
    assert all(isinstance(spec, str) for spec in args.sho_tune)
    assert (args.sho_genomes, args.sho_l1, args.sho_l2) == (4, 2.0, 0.5)
    strategy = HyperparameterStrategy.from_args(args)
    assert isinstance(strategy, SimplexHyperparameters)
    assert strategy.tuned == [LEARNING_RATE]

    args = parser.parse_args(
        [
            "--hyperparameter_strategy",
            "simplex",
            "--sho_tune",
            "learning_rate=log:1e-3:5e-2",
            "epochs=int:5:20",
            *REQUIRED_ARGUMENTS,
            "steady_state",
        ]
    )
    assert len(HyperparameterStrategy.from_args(args).tuned) == 2


@pytest.mark.parametrize(
    "bad",
    [
        ["--sho_tune", "batch_size=int:2:8"],
        ["--sho_tune", "learning_rate=log:0:1"],
        ["--sho_genomes", "1"],
        ["--hyperparameter_strategy", "bayesian"],
    ],
)
def test_classification_parser_rejects_bad_sho_arguments(bad: list[str]) -> None:
    """Untunable names, bad specifications and bad counts fail at parse time.

    Args:
        bad: The offending arguments.
    """

    with pytest.raises(SystemExit):
        classification.build_parser().parse_args(
            [*REQUIRED_ARGUMENTS, *bad, "steady_state"]
        )
