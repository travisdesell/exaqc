"""Tests for the multi-objective population strategies (NSGA-II and NSGA-III).

The strategies only read a genome's ``fitness``, ``metadata``, ``genome_number``
and ``has_same_gates``, so a small stand-in genome keeps these tests fast and
independent of circuit construction. ``src.examples.classification_moo``'s
objective parsing and population factory are covered at the end.
"""

from __future__ import annotations

import itertools
import math
import random
from argparse import Namespace
from typing import Any

import pytest

from src.evolution.moo.nsga2 import NSGA2
from src.evolution.moo.nsga3 import NSGA3
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.evolution.moo.pareto import non_dominated_sort
from src.examples.classification_moo import (
    build_population,
    parse_objectives,
    structural_fitness,
)

OBJECTIVES = [ObjectiveSpec("loss"), ObjectiveSpec("n_enabled_gates")]


class FakeGenome:
    """A stand-in genome carrying only what the population strategies read.

    Args:
        genome_number: The genome's number.
        loss: Its ``loss`` fitness.
        gates: Its ``n_enabled_gates`` fitness.
        topology: Genomes with equal ``topology`` have the same enabled gates.
    """

    def __init__(
        self, genome_number: int, loss: float, gates: float, topology: int | None = None
    ) -> None:
        """Creates the stand-in genome.

        Args:
            genome_number: The genome's number.
            loss: Its ``loss`` fitness.
            gates: Its ``n_enabled_gates`` fitness.
            topology: Genomes with equal ``topology`` have the same enabled
                gates; ``None`` makes the genome unique.

        Returns:
            None. Sets ``genome_number``, ``fitness``, ``metadata`` and
            ``topology``.
        """

        self.genome_number = genome_number
        self.fitness: dict[str, float] = {"loss": loss, "n_enabled_gates": gates}
        self.metadata: dict[str, Any] = {}
        self.topology = topology

    def has_same_gates(self, other: "FakeGenome") -> bool:
        """Reports whether two stand-in genomes have the same enabled gates.

        Args:
            other: The genome to compare with.

        Returns:
            True when both share a non-``None`` ``topology``.
        """

        return self.topology is not None and self.topology == other.topology


def dominates(left: FakeGenome, right: FakeGenome) -> bool:
    """Brute-force Pareto dominance for two minimized objectives.

    Args:
        left: The first genome.
        right: The second genome.

    Returns:
        True when ``left`` is no worse in both objectives and better in one.
    """

    a = (left.fitness["loss"], left.fitness["n_enabled_gates"])
    b = (right.fitness["loss"], right.fitness["n_enabled_gates"])
    return all(x <= y for x, y in zip(a, b)) and a != b


def test_non_dominated_sort_matches_brute_force() -> None:
    """Every genome is ranked below all genomes that dominate it, and front 0 is exact."""

    rng = random.Random(1)
    for _ in range(30):
        population = [
            FakeGenome(i, rng.randint(0, 5), rng.randint(0, 5)) for i in range(25)
        ]
        fronts = non_dominated_sort(population, OBJECTIVES)
        rank = {index: r for r, front in enumerate(fronts) for index in front}

        assert sorted(rank) == list(range(25))
        for i, j in itertools.permutations(range(25), 2):
            if dominates(population[i], population[j]):
                assert rank[i] < rank[j]
        assert set(fronts[0]) == {
            i
            for i in range(25)
            if not any(dominates(population[j], population[i]) for j in range(25))
        }


@pytest.mark.parametrize("population_class", [NSGA2, NSGA3])
def test_population_keeps_size_and_records_every_genome(
    population_class: type[NSGA2] | type[NSGA3],
) -> None:
    """Inserting past the maximum size keeps the size fixed and records every genome."""

    rng = random.Random(2)
    population = population_class(max_population_size=10, objectives=OBJECTIVES)
    for number in range(200):
        genome = FakeGenome(number, rng.random(), rng.randint(1, 30))
        assert population.insert_genome(genome) is True
        assert genome.metadata["insert_type"] in (
            "inserted",
            "global_best",
            "discarded",
        )

    assert len(population.population) == 10
    front = population.get_pareto_front()
    assert front
    for genome in front:
        assert not any(dominates(other, genome) for other in population.population)

    ranks = [genome.metadata["pareto_rank"] for genome in population.get_population()]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize("population_class", [NSGA2, NSGA3])
def test_best_genome_is_lowest_first_objective_on_front(
    population_class: type[NSGA2] | type[NSGA3],
) -> None:
    """The representative genome is the front member with the best first objective."""

    population = population_class(max_population_size=5, objectives=OBJECTIVES)
    for number, (loss, gates) in enumerate([(0.5, 1), (0.1, 9), (0.3, 4), (0.9, 9)]):
        population.insert_genome(FakeGenome(number, loss, gates))

    assert population.get_best_genome().genome_number == 1


def test_dominated_duplicate_is_discarded() -> None:
    """A genome dominated by a held genome with the same gates is discarded."""

    population = NSGA2(max_population_size=5, objectives=OBJECTIVES)
    population.insert_genome(FakeGenome(1, 0.2, 3, topology=7))
    duplicate = FakeGenome(2, 0.4, 3, topology=7)
    population.insert_genome(duplicate)

    assert duplicate.metadata["insert_type"] == "discarded"
    assert duplicate.metadata["discard_reason"] == "duplicate_of_better"
    assert duplicate.metadata["lost_to"] == 1
    assert [genome.genome_number for genome in population.population] == [1]


def test_crowding_distance_is_finite_with_a_failed_genome() -> None:
    """A non-finite loss does not turn other genomes' crowding distances into NaN."""

    population = NSGA2(max_population_size=10, objectives=OBJECTIVES)
    for number, (loss, gates) in enumerate(
        [(0.1, 9), (0.2, 6), (0.3, 4), (0.4, 2), (math.nan, 1)]
    ):
        population.insert_genome(FakeGenome(number, loss, gates))

    for genome in population.population:
        assert not math.isnan(genome.metadata["crowding_distance"])


@pytest.mark.parametrize("population_class", [NSGA2, NSGA3])
def test_failed_genome_is_dominated(
    population_class: type[NSGA2] | type[NSGA3],
) -> None:
    """A genome whose loss is NaN is off the front even with the fewest gates.

    Args:
        population_class: The strategy under test.
    """

    population = population_class(max_population_size=6, objectives=OBJECTIVES)
    for number, (loss, gates) in enumerate(
        [(0.1, 10), (0.2, 8), (0.3, 6), (0.4, 4), (0.5, 2)]
    ):
        population.insert_genome(FakeGenome(number, loss, gates))
    failed = FakeGenome(9, math.nan, 1)
    population.insert_genome(failed)

    assert failed.metadata["pareto_rank"] > 0
    assert failed not in population.get_pareto_front()
    assert population.get_best_genome().genome_number == 0
    if population_class is NSGA3:
        assert failed.metadata["reference_distance"] == math.inf


@pytest.mark.parametrize(
    "n_objectives, population_size, expected",
    [(2, 30, 29), (2, 8, 7), (3, 30, 6)],
)
def test_default_reference_divisions(
    n_objectives: int, population_size: int, expected: int
) -> None:
    """The default divisions give the most directions that do not outnumber the population.

    Args:
        n_objectives: Number of objectives.
        population_size: Maximum population size.
        expected: The divisions expected.
    """

    divisions = NSGA3.default_reference_divisions(n_objectives, population_size)
    assert divisions == expected
    assert math.comb(divisions + n_objectives - 1, n_objectives - 1) <= population_size


def test_run_info_records_objectives() -> None:
    """The archive's run_info names the algorithm and each objective's direction."""

    objectives = [ObjectiveSpec("target_metric", sign=-1.0), ObjectiveSpec("loss")]
    info = NSGA3(max_population_size=8, objectives=objectives).run_info()

    assert info["multi_objective"]["algorithm"] == "nsga3"
    assert info["multi_objective"]["objectives"][0] == {
        "name": "target_metric",
        "sign": -1.0,
        "direction": "maximize",
    }
    assert info["multi_objective"]["reference_directions"] == 8


def test_parse_objectives() -> None:
    """``--objectives`` values parse into signed specifications, and bad ones are rejected."""

    parsed = parse_objectives(["loss", "n_enabled_gates:1", "target_metric:-1"])
    assert [(o.name, o.sign) for o in parsed] == [
        ("loss", 1.0),
        ("n_enabled_gates", 1.0),
        ("target_metric", -1.0),
    ]

    for bad in (
        ["loss"],
        ["loss", "loss"],
        ["loss", "accuracy"],
        ["loss", "n_enabled_gates:2"],
    ):
        with pytest.raises(ValueError):
            parse_objectives(bad)


def test_build_population_selects_algorithm() -> None:
    """The sub-command picks the class and its flags reach the constructor."""

    args = Namespace(
        population_strategy="nsga3",
        max_population_size=12,
        tournament_size=3,
        reference_divisions=None,
        objectives=["loss", "n_enabled_gates"],
        seed=0,
    )
    population = build_population(args, None)

    assert isinstance(population, NSGA3)
    assert population.max_population_size == 12
    assert population.tournament_size == 3
    assert population.reference_divisions == 11


def test_structural_fitness_counts_enabled_gates_only() -> None:
    """Disabled gates and their parameters are not counted."""

    class Gate:
        """A stand-in gate with an enabled flag and parameters."""

        def __init__(self, enabled: bool, parameters: dict[str, float]) -> None:
            """Creates the stand-in gate.

            Args:
                enabled: Whether the gate is enabled.
                parameters: The gate's parameters.

            Returns:
                None. Sets ``enabled`` and ``parameters``.
            """

            self.enabled = enabled
            self.parameters = parameters

    genome = Namespace(
        gates=[
            Gate(True, {"theta": 0.1}),
            Gate(True, {}),
            Gate(False, {"a": 1, "b": 2}),
        ]
    )

    assert structural_fitness(genome) == {"n_enabled_gates": 2.0, "n_parameters": 1.0}


@pytest.mark.parametrize("noisy", [True, False])
def test_noise_gap_compares_a_trained_genome_with_its_noiseless_self(
    noisy: bool,
) -> None:
    """A noisy genome records its noise gap; a noiseless one records none.

    The noiseless re-score uses the same best weights, so the gap is the
    difference between the recorded (noisy) validation metrics and that
    re-score.

    Args:
        noisy: Whether the genome runs with depolarizing noise.
    """

    import torch

    from src.examples.classification_moo import MultiObjectiveClassificationObjective
    from src.metrics.mean_class_accuracy import MeanClassAccuracy
    from tests.supervised_trainer_test_utils import (
        build_classification_genome,
        cross_entropy_on_logits,
        make_balanced_binary_dataloaders,
    )

    torch.manual_seed(0)
    genome, n_features = build_classification_genome(
        genome_number=1,
        target="pennylane",
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        epochs=1,
    )
    if noisy:
        genome.hyperparameters.update(noise_type="depolarizing", noise_p=0.2)
    train, validation = make_balanced_binary_dataloaders(n_features=n_features)
    objective = MultiObjectiveClassificationObjective(
        training_dataloader=train,
        validation_dataloader=validation,
        training_loss_function=cross_entropy_on_logits,
        validation_loss_function=cross_entropy_on_logits,
        metrics={"mean_class_accuracy": MeanClassAccuracy(n_labels=2)},
    )

    objective(genome)

    if not noisy:
        assert "noise_gap" not in genome.fitness
        return
    assert {"noise_gap", "noise_gap_accuracy"} <= set(genome.fitness)
    assert genome.fitness["noise_gap"] != 0.0  # 20% depolarizing changes the loss
    assert "noise_gap" not in genome.metadata.get("initial_fitness", {})
    parse_objectives(["loss", "noise_gap"])  # selectable


def test_noise_gap_needs_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Choosing noise_gap for a noiseless run stops before any work.

    Args:
        monkeypatch: Used to set the command line.
    """

    import sys

    from src.examples import classification_moo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification_moo",
            "-ms",
            "uniform",
            "1",
            "3",
            "-ps",
            "uniform",
            "5",
            "5",
            "--dataset",
            "iris",
            "--input_qubits",
            "4",
            "--output_qubits",
            "2",
            "--objectives",
            "loss",
            "noise_gap",
            "--out_dir",
            "unused",
            "nsga2",
        ],
    )
    with pytest.raises(SystemExit):
        classification_moo.main()
