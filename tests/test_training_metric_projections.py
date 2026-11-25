"""Tests for recording a genome's pre-training scores and its best-step metrics.

A genome is scored three ways while it trains: its inherited weights at epoch or
episode 0, the weights it keeps at its best step, and the last weights trained
(which it discards when training stopped improving). These tests check that:

* both trainers keep the inherited weights' metrics (``initial_*_metrics``),
  and the reinforcement-learning trainer takes its training returns at the best
  episode rather than at the last one;
* the objectives score the inherited weights with the same formula as fitness
  (``metadata["initial_fitness"]``);
* the archive projects each series' initial, best and last records, holds the
  inherited weights' fitness and the gain training made as columns, reads older
  archives without them as unrecorded, and adds them when such an archive is
  reopened to continue its run;
* ``training_gain_summary`` summarizes that gain over MCP.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.objectives.classification_objective import ClassificationObjective
from src.utils.artifact_viewer.mcp_tools import DashboardTools
from src.utils.artifact_viewer.server import RunRegistry
from src.utils.genome_archive import (
    ARCHIVE_FILENAME,
    GenomeArchive,
    _metric_projections,
)

from tests.test_exaqc_dashboard import FakeGenome, build_run, gate
from tests.test_pretraining_evaluation import _classification_setup, _rl_setup

#: The ``genomes`` table as archive format 6 declared it, before the metric
#: projections and the inherited weights' fitness.
_FORMAT_6_GENOMES = """
CREATE TABLE genomes(
    genome_number INTEGER PRIMARY KEY,
    insertion INTEGER,
    saved_at REAL,
    insert_type TEXT,
    generated_by TEXT,
    crossover_type TEXT,
    island INTEGER,
    n_gates INTEGER,
    n_enabled_gates INTEGER,
    n_parameters INTEGER,
    n_cnot INTEGER,
    n_rot INTEGER,
    max_innovation_number INTEGER,
    generated_at_insertion INTEGER,
    evaluation_seconds REAL,
    evaluated_host TEXT,
    evaluated_rank INTEGER,
    discard_reason TEXT,
    final_metrics TEXT,
    fitness TEXT,
    request_wait_seconds REAL,
    generation_seconds REAL,
    turnaround_seconds REAL,
    loss REAL GENERATED ALWAYS AS (json_extract(fitness, '$.loss')) VIRTUAL,
    target_metric REAL GENERATED ALWAYS AS
        (json_extract(fitness, '$.target_metric')) VIRTUAL
);
CREATE TABLE sqlar(name TEXT PRIMARY KEY, mode INT, mtime INT, sz INT, data BLOB);
CREATE TABLE population_events(
    step INTEGER PRIMARY KEY, recorded_at REAL, added TEXT, removed TEXT
);
CREATE TABLE genome_parents(
    child INTEGER NOT NULL, parent INTEGER NOT NULL, PRIMARY KEY(child, parent)
);
CREATE TABLE run_info(key TEXT PRIMARY KEY, value TEXT);
"""


def _write_untouched_format_6_archive(directory: Path) -> Path:
    """Writes a format-6 archive holding one genome, never reopened since.

    Args:
        directory: The run directory to create.

    Returns:
        The run directory.
    """

    directory.mkdir(parents=True)
    connection = sqlite3.connect(directory / ARCHIVE_FILENAME)
    connection.executescript(_FORMAT_6_GENOMES)
    connection.execute(
        "INSERT INTO genomes(genome_number, insertion, insert_type, generated_by, "
        "final_metrics, fitness) VALUES (1, 1, 'inserted', '[\"add_gate\"]', ?, ?)",
        (
            json.dumps({"validation_epoch_metrics.loss": 0.4}),
            json.dumps({"loss": 0.4, "target_metric": 0.6}),
        ),
    )
    connection.commit()
    connection.close()
    return directory


def _genome_with_initial_fitness(
    genome_number: int,
    target_metric: float,
    initial_target_metric: float,
    generated_by: list[str],
) -> FakeGenome:
    """Builds a fake genome whose inherited weights scored ``initial_target_metric``.

    Args:
        genome_number: The genome's number.
        target_metric: Its fitness ``target_metric`` after training.
        initial_target_metric: The ``target_metric`` its inherited weights had.
        generated_by: The operators that generated it.

    Returns:
        The genome, with a two-epoch validation history whose best epoch is 1.
    """

    genome = FakeGenome(
        genome_number,
        loss=1.0 - target_metric,
        target_metric=target_metric,
        parents=[0],
        generated_by=generated_by,
        gates=[gate(1)],
    )
    metadata = genome.serialized["metadata"]
    metadata["initial_fitness"] = {
        "loss": 1.0 - initial_target_metric,
        "target_metric": initial_target_metric,
    }
    metadata["validation_epoch_metrics"] = [
        {"epoch": 0, "loss": 1.0 - initial_target_metric},
        {"epoch": 1, "loss": 1.0 - target_metric},
        {"epoch": 2, "loss": 0.9},
    ]
    metadata["best_epoch"] = 1
    return genome


def _write_format_6_run(directory: Path) -> Path:
    """Writes a format-6 run, then reopens it the way continuing the run does.

    Args:
        directory: The run directory to create.

    Returns:
        The run directory.
    """

    _write_untouched_format_6_archive(directory)
    GenomeArchive.create(str(directory), expect_existing=True).close()
    return directory


def test_supervised_records_the_inherited_weights_and_their_fitness() -> None:
    """Epoch 0's evaluation is kept apart and scored like fitness is."""

    genome, objective = _classification_setup(epochs=2)
    objective(genome)

    metadata = genome.metadata
    assert (
        metadata["initial_validation_metrics"]
        is metadata["validation_epoch_metrics"][0]
    )
    assert "loss" in metadata["initial_training_metrics"]
    assert metadata["initial_fitness"] == ClassificationObjective.fitness_from_metrics(
        metadata["initial_training_metrics"], metadata["initial_validation_metrics"]
    )


@pytest.mark.parametrize("task", ["classification", "reinforcement_learning"])
def test_initial_fitness_is_the_fitness_when_nothing_trains(task: str) -> None:
    """With no training, the inherited weights' fitness is the genome's fitness.

    Args:
        task: Which task's trainer and objective to check.
    """

    if task == "classification":
        genome, objective = _classification_setup(epochs=0)
    else:
        genome, objective, _ = _rl_setup(episodes=0)
    objective(genome)

    assert genome.metadata["initial_fitness"] == genome.fitness


def _scripted_rl_training(
    monkeypatch: pytest.MonkeyPatch,
    returns: list[float],
    evaluations: list[float],
) -> Any:
    """Trains an RL genome with scripted training and evaluation returns.

    Args:
        monkeypatch: Used to script ``run_update`` and ``evaluate``.
        returns: Each training episode's return, in order.
        evaluations: Each evaluation's ``return_mean``, episode 0 first.

    Returns:
        The trained genome; every episode is evaluated and training never stops
        early.
    """

    genome, objective, _ = _rl_setup(episodes=len(returns))
    genome.hyperparameters["log_every"] = 1
    genome.hyperparameters["improvement_cutoff"] = 0
    genome.hyperparameters["ema_alpha"] = 0.5
    trainer = objective.trainer
    real_evaluate = trainer.evaluate
    remaining_returns = list(returns)
    remaining_evaluations = list(evaluations)

    def scripted_update(*args: Any, **kwargs: Any) -> tuple[float, dict[str, Any]]:
        """Returns the next scripted training return.

        Args:
            *args: Ignored update arguments.
            **kwargs: Ignored update arguments.

        Returns:
            The return and an empty info dict.
        """

        return remaining_returns.pop(0), {}

    def scripted_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Runs the real evaluation with the next scripted return mean.

        Args:
            *args: Forwarded to the real evaluation.
            **kwargs: Forwarded to the real evaluation.

        Returns:
            The real results with ``return_mean`` replaced.
        """

        results = real_evaluate(*args, **kwargs)
        results["return_mean"] = remaining_evaluations.pop(0)
        return results

    monkeypatch.setattr(trainer, "run_update", scripted_update)
    monkeypatch.setattr(trainer, "evaluate", scripted_evaluate)
    objective(genome)
    return genome


def test_rl_training_returns_are_taken_at_the_best_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training term describes the kept weights, not the discarded last ones.

    Args:
        monkeypatch: Used to script training and evaluation returns.
    """

    genome = _scripted_rl_training(
        monkeypatch,
        returns=[10.0, 0.0, 4.0, 8.0],
        evaluations=[0.0, 5.0, 1.0, 1.0, 1.0],
    )

    assert genome.metadata["best_episode"] == 1
    best_training = genome.metadata["best_training_metrics"]
    # as of episode 1: the moving average is seeded with its return
    assert best_training == {"return_mean": 10.0, "best_episode_return": 10.0}
    assert genome.fitness["train_return_mean"] == 10.0
    assert genome.fitness["eval_return_mean"] == 5.0


def test_rl_keeping_the_inherited_weights_keeps_their_training_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When training never improves, the training term is the inherited weights'.

    Args:
        monkeypatch: Used to script training and evaluation returns.
    """

    genome = _scripted_rl_training(
        monkeypatch, returns=[50.0, 50.0], evaluations=[7.0, 1.0, 1.0]
    )

    assert genome.metadata["best_episode"] == 0
    assert (
        genome.metadata["best_training_metrics"]
        == genome.metadata["initial_training_metrics"]
    )
    assert genome.fitness == genome.metadata["initial_fitness"]


def test_metric_projections_pick_the_initial_best_and_last_records() -> None:
    """Each series contributes its step-0, best-step and last records."""

    metadata = {
        "validation_epoch_metrics": [
            {"epoch": 0, "loss": 0.9},
            {"epoch": 1, "loss": 0.5},
            {"epoch": 2, "loss": 0.7},
        ],
        "training_epoch_metrics": [
            {"epoch": 1, "loss": 0.4},
            {"epoch": 2, "loss": 0.3},
        ],
        "best_epoch": 1,
    }

    initial, best, last = _metric_projections(metadata)

    # a training series has no epoch 0: nothing was trained before epoch 1
    assert initial == {"validation_epoch_metrics.loss": 0.9}
    assert best == {
        "validation_epoch_metrics.loss": 0.5,
        "training_epoch_metrics.loss": 0.4,
    }
    assert last == {
        "validation_epoch_metrics.loss": 0.7,
        "training_epoch_metrics.loss": 0.3,
    }


def test_the_archive_holds_the_projections_and_the_training_gain(
    tmp_path: Path,
) -> None:
    """The summary row carries every projection, and the gain is a column.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run = build_run(
        tmp_path / "run",
        [_genome_with_initial_fitness(1, 0.8, 0.6, ["add_gate"])],
    )

    with GenomeArchive.open_readonly(run) as archive:
        (summary,) = archive.list_genomes(sort_key="genome_number")
        assert summary["initial_metrics"] == {"validation_epoch_metrics.loss": 0.4}
        assert summary["best_metrics"] == {
            "validation_epoch_metrics.loss": pytest.approx(0.2)
        }
        assert summary["last_metrics"] == {"validation_epoch_metrics.loss": 0.9}
        assert summary["initial_fitness"]["target_metric"] == 0.6

        gain, initial = archive.connection.execute(
            "SELECT target_gain, initial_target_metric FROM genomes"
        ).fetchone()
        assert gain == pytest.approx(0.2)
        assert initial == 0.6
        assert "target_gain" in archive.series_metrics()
        assert archive.metric_keys() == ["validation_epoch_metrics.loss"]


def test_an_older_archive_reads_and_is_brought_up_to_date(tmp_path: Path) -> None:
    """A format-6 archive opens without its new columns, and gains them on restart.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run = _write_format_6_run(tmp_path / "older_run")

    # GenomeArchive.create above stands in for reopening the run to continue it
    with GenomeArchive.open_readonly(str(run)) as archive:
        assert "target_gain" in archive.genome_columns
        (summary,) = archive.list_genomes(sort_key="genome_number")
        assert summary["best_metrics"] == {}
        assert summary["initial_fitness"] is None

    with GenomeArchive.create(str(run), expect_existing=True) as archive:
        archive.add_genome(
            _genome_with_initial_fitness(2, 0.9, 0.5, ["add_gate"]), insertion=2
        )

    with GenomeArchive.open_readonly(str(run)) as archive:
        gains = dict(
            archive.connection.execute("SELECT genome_number, target_gain FROM genomes")
        )
        assert gains[1] is None
        assert gains[2] == pytest.approx(0.4)


def test_an_unopened_older_archive_reads_its_new_columns_as_unrecorded(
    tmp_path: Path,
) -> None:
    """Reading a format-6 archive that was never reopened does not fail.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    directory = _write_untouched_format_6_archive(tmp_path / "untouched")

    with GenomeArchive.open_readonly(str(directory)) as archive:
        assert "best_metrics" not in archive.genome_columns
        (summary,) = archive.list_genomes(sort_key="genome_number")
        assert summary["best_metrics"] == {}
        assert summary["initial_fitness"] is None
        assert archive.metric_keys() == []
        assert "target_gain" not in archive.series_metrics()


def test_training_gain_summary_reports_what_training_added(tmp_path: Path) -> None:
    """The MCP tool summarizes gains overall, by starting score and by operator.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = [
        _genome_with_initial_fitness(1, 0.8, 0.6, ["add_gate"]),
        _genome_with_initial_fitness(2, 0.7, 0.7, ["clone"]),
        _genome_with_initial_fitness(3, 0.9, 0.5, ["add_gate"]),
        _genome_with_initial_fitness(4, 0.6, 0.6, ["clone"]),
    ]
    build_run(tmp_path / "runs" / "gains", genomes)
    tools = DashboardTools(RunRegistry(watch_directory=str(tmp_path / "runs")))

    summary = tools.training_gain_summary("gains")
    assert summary["overall"]["n"] == 4
    assert summary["overall"]["improved"] == 0.5
    assert summary["overall"]["unchanged"] == 0.5
    assert len(summary["by_starting_score"]) == 4
    assert sum(row["n"] for row in summary["by_starting_score"]) == 4

    grouped = tools.training_gain_summary("gains", group_by="generated_by")
    by_operator = {row["group"]: row for row in grouped["rows"]}
    assert by_operator["add_gate"]["improved"] == 1.0
    assert by_operator["clone"]["unchanged"] == 1.0

    with pytest.raises(ValueError, match="group_by"):
        tools.training_gain_summary("gains", group_by="loss")


def test_training_gain_summary_explains_an_archive_too_old_for_it(
    tmp_path: Path,
) -> None:
    """An archive without the gain column is refused with what to use instead.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    _write_untouched_format_6_archive(tmp_path / "runs" / "untouched")
    tools = DashboardTools(RunRegistry(watch_directory=str(tmp_path / "runs")))

    with pytest.raises(ValueError, match="format 7"):
        tools.training_gain_summary("untouched")
