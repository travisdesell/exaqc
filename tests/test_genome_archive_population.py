"""Tests for the population record an archive keeps and the series built from it.

A run stores only how its population *changed* at each insertion, rather than a
fixed set of statistics computed while it ran. These tests pin the two halves of
that: that membership is reconstructed exactly from the recorded changes, and
that any metric a genome recorded can be summarized over the genomes that were
alive at each step.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.utils.genome_archive import GenomeArchive


class FakeGenome:
    """Minimal genome exposing what :meth:`GenomeArchive.add_genome` reads."""

    def __init__(
        self,
        genome_number: int,
        loss: float,
        gates: list[str],
        returns: list[float] | None = None,
    ) -> None:
        """Builds a genome that serializes like a real one.

        Args:
            genome_number: The genome's number.
            loss: Its ``fitness["loss"]``; ``target_metric`` mirrors it.
            gates: The method names of its enabled gates.
            returns: The per-episode returns it recorded while training, if any.
        """

        self.genome_number = genome_number
        self.serialized: dict[str, Any] = {
            "genome_number": genome_number,
            "target": "pennylane",
            "fitness": {"loss": loss, "target_metric": 1.0 - loss},
            "gates": [
                {"method_name": name, "enabled": True, "parameters": {}}
                for name in gates
            ],
            "metadata": {
                "insert_type": "inserted",
                "generated_by": ["add_gate"],
                "parent_genomes": [],
            },
        }
        if returns is not None:
            self.serialized["metadata"]["training_episode_metrics"] = [
                {"episode": episode, "return": value}
                for episode, value in enumerate(returns)
            ]

    def to_dict(self) -> dict[str, Any]:
        """Returns the genome's serialized form.

        Returns:
            The dict :meth:`GenomeArchive.add_genome` stores.
        """

        return self.serialized


def build_archive(directory, genomes: list[FakeGenome]) -> GenomeArchive:
    """Creates an archive holding the given genomes.

    Args:
        directory: The run directory to create.
        genomes: The genomes to store, in insertion order.

    Returns:
        The archive, still open for writing.
    """

    archive = GenomeArchive.create(str(directory))
    for insertion, genome in enumerate(genomes, start=1):
        archive.add_genome(genome, insertion=insertion)
    return archive


def test_membership_is_reconstructed_at_every_step(tmp_path) -> None:
    """Genomes entering and leaving the population are both recovered.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = [
        FakeGenome(number, loss=0.1 * number, gates=["h"]) for number in (1, 2, 3, 4)
    ]
    archive = build_archive(tmp_path / "run", genomes)

    populations = {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 4]}
    for step, members in populations.items():
        archive.record_population(
            step=step, population=[genomes[number - 1] for number in members]
        )

    for step, members in populations.items():
        assert archive.population_at(step) == members
    # genome 3 was evicted at step 4, so it is absent from the latest membership
    assert archive.population_at() == [1, 2, 4]
    archive.close()


def test_an_unchanged_population_records_no_step(tmp_path) -> None:
    """A step that changed nothing costs no row.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = [FakeGenome(1, loss=0.5, gates=["h"])]
    archive = build_archive(tmp_path / "run", genomes)

    for step in (1, 2, 3):
        archive.record_population(step=step, population=genomes)

    recorded = archive.connection.execute(
        "SELECT step FROM population_events ORDER BY step"
    ).fetchall()
    assert [row[0] for row in recorded] == [1]
    archive.close()


def test_reopening_continues_from_the_recorded_membership(tmp_path) -> None:
    """A resumed run records the change, not the whole population again.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = [
        FakeGenome(number, loss=0.1 * number, gates=["h"]) for number in (1, 2, 3)
    ]
    archive = build_archive(tmp_path / "run", genomes)
    archive.record_population(step=1, population=genomes[:2])
    archive.close()

    reopened = GenomeArchive.create(str(tmp_path / "run"))
    reopened.record_population(step=2, population=genomes[1:])

    added, removed = reopened.connection.execute(
        "SELECT added, removed FROM population_events WHERE step = 2"
    ).fetchone()
    assert added == "[3]"
    assert removed == "[1]"
    assert reopened.population_at() == [2, 3]
    reopened.close()


def test_population_series_summarizes_the_living_genomes(tmp_path) -> None:
    """Statistics cover the genomes alive at each step, not every genome stored.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = [
        FakeGenome(1, loss=0.9, gates=["h"]),
        FakeGenome(2, loss=0.5, gates=["h"]),
        FakeGenome(3, loss=0.1, gates=["h"]),
    ]
    archive = build_archive(tmp_path / "run", genomes)
    archive.record_population(step=1, population=genomes[:1])
    archive.record_population(step=2, population=genomes[:2])
    # genome 1 is evicted, so it stops counting towards the population's mean
    archive.record_population(step=3, population=genomes[1:])

    series = archive.population_series("loss")
    assert series["step"] == [1, 2, 3]
    assert series["population_size"] == [1, 2, 2]
    assert series["best"] == pytest.approx([0.9, 0.5, 0.1])
    assert series["worst"] == pytest.approx([0.9, 0.9, 0.5])
    assert series["mean"] == pytest.approx([0.9, 0.7, 0.3])
    archive.close()


def test_population_series_follows_the_metric_direction(tmp_path) -> None:
    """For a metric where larger is better, best and worst swap ends.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = [
        FakeGenome(1, loss=0.9, gates=["h"], returns=[-5.0]),
        FakeGenome(2, loss=0.5, gates=["h"], returns=[4.0]),
    ]
    archive = build_archive(tmp_path / "run", genomes)
    archive.record_population(step=1, population=genomes)

    metric = "training_episode_metrics.return"
    ascending = archive.population_series(metric, higher_is_better=False)
    descending = archive.population_series(metric, higher_is_better=True)

    assert ascending["best"] == pytest.approx([-5.0])
    assert descending["best"] == pytest.approx([4.0])
    assert descending["worst"] == pytest.approx([-5.0])
    archive.close()


def test_series_metrics_span_columns_fitness_and_training_metrics(tmp_path) -> None:
    """Anything a genome recorded can be charted, whatever its task.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    archive = build_archive(
        tmp_path / "run", [FakeGenome(1, loss=0.5, gates=["h"], returns=[1.0, 2.0])]
    )

    metrics = archive.series_metrics()
    assert "n_enabled_gates" in metrics
    assert "loss" in metrics and "target_metric" in metrics
    assert "training_episode_metrics.return" in metrics
    # the series' own counter numbers its records rather than measuring anything
    assert "training_episode_metrics.episode" not in metrics
    archive.close()


def test_circuit_complexity_is_recorded_for_each_genome(tmp_path) -> None:
    """Decomposition costs are summed from the gate specifications when archiving.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    archive = build_archive(
        tmp_path / "run",
        [
            FakeGenome(1, loss=0.5, gates=["h", "h"]),
            FakeGenome(2, loss=0.4, gates=["h", "cx", "cx"]),
        ],
    )

    recorded = archive.connection.execute(
        "SELECT genome_number, n_cnot FROM genomes ORDER BY genome_number"
    ).fetchall()
    assert recorded == [(1, 0), (2, 2)]
    archive.close()


def test_population_series_rejects_a_metric_no_genome_recorded(tmp_path) -> None:
    """An unknown metric is refused rather than charted as nothing.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    archive = build_archive(tmp_path / "run", [FakeGenome(1, loss=0.5, gates=["h"])])
    archive.record_population(step=1, population=[FakeGenome(1, loss=0.5, gates=["h"])])

    with pytest.raises(ValueError, match="was not recorded"):
        archive.population_series("no_such_metric")
    archive.close()
