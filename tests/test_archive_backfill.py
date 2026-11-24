"""Tests for bringing archives written before format 7 up to it.

An archive written before format 7 holds every genome's JSON but none of the
summary columns format 7 added. These tests check that the backfill:

* re-derives each genome's initial, best and last metric projections;
* takes ``initial_fitness`` from a genome's own metadata, scores it from a
  reinforcement-learning genome's episode-0 evaluation with the search's own
  formula, and leaves it ``NULL`` where it cannot be known;
* rewrites nothing else, records itself in ``run_info``, and changes nothing
  when run again;
* leaves the original untouched under ``--output_dir`` and refuses an archive a
  search may still be writing to.

It also checks that moving the reinforcement-learning fitness formula into a
module-level function left the objective's fitness unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.examples import backfill_archive as backfill_entry
from src.objectives.reinforcement_learning_objective import (
    ReinforcementLearningObjective,
    reinforcement_learning_fitness,
)
from src.utils.archive_backfill import (
    BACKFILL_RUN_INFO_KEY,
    ArchiveInUseError,
    backfill_archive,
)
from src.utils.artifact_viewer.mcp_tools import DashboardTools
from src.utils.artifact_viewer.server import RunRegistry
from src.utils.genome_archive import ARCHIVE_FILENAME, GenomeArchive

from tests.test_exaqc_dashboard import FakeGenome, build_run, gate

#: The ``genomes`` columns archive format 7 added, generated ones first (a
#: column a generated column reads cannot be dropped before it).
_FORMAT_7_COLUMNS = (
    "target_gain",
    "initial_target_metric",
    "initial_loss",
    "initial_fitness",
    "initial_metrics",
    "best_metrics",
    "last_metrics",
)


def _rl_genome(genome_number: int, with_episode_0: bool = True) -> FakeGenome:
    """Builds a reinforcement-learning genome as a format-6 search stored it.

    Args:
        genome_number: The genome's number.
        with_episode_0: Whether its evaluations start with episode 0's
            evaluation of the inherited weights (runs from 2026-09-30 on).

    Returns:
        The genome: evaluation returns 10 (episode 0), 50 (episode 1, its best)
        and 30 (episode 2), and no ``initial_fitness``.
    """

    genome = FakeGenome(
        genome_number,
        loss=-45.0,
        target_metric=50.0,
        parents=[0],
        generated_by=["add_gate"],
        gates=[gate(1)],
        task="reinforcement_learning",
        task_target="walker2d",
    )
    genome.serialized["fitness"].update(
        {
            "eval_return_mean": 50.0,
            "eval_return_std": 2.0,
            "train_return_mean": 5.0,
            "best_episode_return": 6.0,
            "env_id": "Walker2d-v5",
            "eval_policy": "greedy",
        }
    )
    evaluations = [
        {
            "return_mean": 10.0,
            "return_std": 1.0,
            "best_episode_return": 12.0,
            "episode": 0,
        },
        {
            "return_mean": 50.0,
            "return_std": 2.0,
            "best_episode_return": 55.0,
            "episode": 1,
        },
        {
            "return_mean": 30.0,
            "return_std": 3.0,
            "best_episode_return": 40.0,
            "episode": 2,
        },
    ]
    genome.serialized["metadata"].update(
        {
            "evaluation_episode_metrics": (
                evaluations if with_episode_0 else evaluations[1:]
            ),
            "training_episode_metrics": [
                {"episode": 1, "return": 5.0, "loss": 0.3},
                {"episode": 2, "return": 7.0, "loss": 0.2},
            ],
            "best_episode": 1,
            "eval_policy": "greedy",
        }
    )
    return genome


def _classification_genome(
    genome_number: int, initial_fitness: dict[str, float] | None = None
) -> FakeGenome:
    """Builds a classification genome with a two-epoch training history.

    Args:
        genome_number: The genome's number.
        initial_fitness: The ``metadata["initial_fitness"]`` it recorded, as a
            format-7 search does; ``None`` for one stored before that.

    Returns:
        The genome, whose best epoch is 1.
    """

    genome = FakeGenome(
        genome_number, loss=0.3, target_metric=0.8, parents=[0], generated_by=["clone"]
    )
    metadata = genome.serialized["metadata"]
    metadata["validation_epoch_metrics"] = [
        {"epoch": 0, "loss": 0.9},
        {"epoch": 1, "loss": 0.3},
        {"epoch": 2, "loss": 0.5},
    ]
    metadata["training_epoch_metrics"] = [
        {"epoch": 1, "loss": 0.4},
        {"epoch": 2, "loss": 0.2},
    ]
    metadata["best_epoch"] = 1
    if initial_fitness is not None:
        metadata["initial_fitness"] = initial_fitness
    return genome


def _write_format_6_run(
    directory: Path, genomes: list[FakeGenome], run_info: dict[str, Any]
) -> Path:
    """Writes a run, then strips it back to what archive format 6 stored.

    Args:
        directory: The run directory to create.
        genomes: The genomes to store.
        run_info: ``run_info`` values to record (replacing ``build_run``'s).

    Returns:
        The archive's path. Its genomes were saved an hour ago, so the run reads
        as finished.
    """

    build_run(directory, genomes)
    path = directory / ARCHIVE_FILENAME
    connection = sqlite3.connect(path)
    for column in _FORMAT_7_COLUMNS:
        connection.execute(f"ALTER TABLE genomes DROP COLUMN {column}")
    connection.execute("ALTER TABLE genomes ADD COLUMN final_metrics TEXT")
    connection.execute(
        "UPDATE genomes SET final_metrics = ?, saved_at = saved_at - 3600",
        (json.dumps({"recorded": 1.0}),),
    )
    connection.executemany(
        "INSERT OR REPLACE INTO run_info(key, value) VALUES (?, ?)",
        [
            (key, json.dumps(value))
            for key, value in {"format_version": 6, **run_info}.items()
        ],
    )
    connection.commit()
    connection.close()
    return path


def _write_rl_run(directory: Path) -> Path:
    """Writes a format-6 reinforcement-learning run with a training-return weight of 0.25.

    Args:
        directory: The run directory to create.

    Returns:
        The archive's path: genome 1 has an episode-0 evaluation, genome 2 does
        not.
    """

    return _write_format_6_run(
        directory,
        [_rl_genome(1), _rl_genome(2, with_episode_0=False)],
        {
            "task": "reinforcement_learning",
            "task_target": "walker2d",
            "arguments": {"env": "walker2d", "train_vs_validation_bias": 0.25},
        },
    )


def _rows(path: Path) -> list[tuple[Any, ...]]:
    """Reads every column of every genome's summary row.

    Args:
        path: The archive.

    Returns:
        The rows, in genome-number order.
    """

    connection = sqlite3.connect(path)
    rows = connection.execute("SELECT * FROM genomes ORDER BY genome_number").fetchall()
    connection.close()
    return rows


def _sha256(path: Path) -> str:
    """Hashes a file's bytes.

    Args:
        path: The file.

    Returns:
        Its SHA-256 hex digest.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rl_archive_gains_projections_and_derived_initial_fitness(
    tmp_path: Path,
) -> None:
    """Episode 0 is scored with the run's bias, and the gain becomes queryable.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = _write_rl_run(tmp_path / "rl")

    result = backfill_archive(str(path))

    assert (result.genomes, result.updated) == (2, 2)
    assert result.initial_fitness_derived == 1
    assert result.initial_fitness_unavailable == 1
    with GenomeArchive.open_readonly(str(path)) as archive:
        first, second = archive.list_genomes(sort_key="genome_number")
        assert first["initial_metrics"] == {
            "evaluation_episode_metrics.return_mean": 10.0,
            "evaluation_episode_metrics.return_std": 1.0,
            "evaluation_episode_metrics.best_episode_return": 12.0,
        }
        assert first["best_metrics"]["evaluation_episode_metrics.return_mean"] == 50.0
        assert first["best_metrics"]["training_episode_metrics.loss"] == 0.3
        assert first["last_metrics"]["evaluation_episode_metrics.return_mean"] == 30.0
        assert first["initial_fitness"] == {
            "loss": -(0.25 * 10.0 + 0.75 * 10.0),
            "target_metric": 10.0,
            "eval_return_mean": 10.0,
            "eval_return_std": 1.0,
            "train_return_mean": 10.0,
            "best_episode_return": 12.0,
            "env_id": "Walker2d-v5",
            "eval_policy": "greedy",
        }
        # a genome stored before episode 0 was recorded gets its projections only
        assert second["initial_metrics"] == {}
        assert second["best_metrics"]["evaluation_episode_metrics.return_mean"] == 50.0
        assert second["initial_fitness"] is None

        gains = dict(
            archive.connection.execute("SELECT genome_number, target_gain FROM genomes")
        )
        assert gains == {1: 40.0, 2: None}
        assert "target_gain" in archive.series_metrics()
        run_info = archive.run_info()
        assert run_info["format_version"] == 6

    tools = DashboardTools(RunRegistry(run_directories=[str(path.parent)]))
    summary = tools.training_gain_summary(path.parent.name)
    assert summary["overall"]["n"] == 1
    assert summary["overall"]["improved"] == 1.0


def test_backfill_rewrites_nothing_a_run_recorded(tmp_path: Path) -> None:
    """Fitness and the old final_metrics column are kept as the run wrote them.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = _write_rl_run(tmp_path / "rl")
    connection = sqlite3.connect(path)
    before = connection.execute(
        "SELECT genome_number, fitness, final_metrics, saved_at FROM genomes"
    ).fetchall()
    connection.close()

    backfill_archive(str(path))

    connection = sqlite3.connect(path)
    after = connection.execute(
        "SELECT genome_number, fitness, final_metrics, saved_at FROM genomes"
    ).fetchall()
    connection.close()
    assert after == before


def test_classification_genomes_get_projections_but_no_derived_initial_fitness(
    tmp_path: Path,
) -> None:
    """Without stored epoch-0 training metrics, only a recorded initial fitness is used.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    recorded = {"loss": 0.8, "target_metric": 0.4}
    path = _write_format_6_run(
        tmp_path / "iris",
        [_classification_genome(1), _classification_genome(2, recorded)],
        {"task": "classification", "task_target": "iris"},
    )

    result = backfill_archive(str(path))

    assert result.initial_fitness_recorded == 1
    assert result.initial_fitness_unavailable == 1
    with GenomeArchive.open_readonly(str(path)) as archive:
        first, second = archive.list_genomes(sort_key="genome_number")
        assert first["initial_metrics"] == {"validation_epoch_metrics.loss": 0.9}
        assert first["best_metrics"] == {
            "validation_epoch_metrics.loss": 0.3,
            "training_epoch_metrics.loss": 0.4,
        }
        assert first["last_metrics"] == {
            "validation_epoch_metrics.loss": 0.5,
            "training_epoch_metrics.loss": 0.2,
        }
        assert first["initial_fitness"] is None
        assert second["initial_fitness"] == recorded
        assert archive.metric_keys() == [
            "training_epoch_metrics.loss",
            "validation_epoch_metrics.loss",
        ]


def test_a_second_backfill_changes_nothing(tmp_path: Path) -> None:
    """Running the backfill again writes no row and records nothing new.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = _write_rl_run(tmp_path / "rl")
    backfill_archive(str(path))
    rows = _rows(path)
    with GenomeArchive.open_readonly(str(path)) as archive:
        run_info = archive.run_info()

    result = backfill_archive(str(path))

    assert (result.genomes, result.updated) == (2, 0)
    assert _rows(path) == rows
    with GenomeArchive.open_readonly(str(path)) as archive:
        assert archive.run_info() == run_info


def test_an_archive_written_at_format_7_needs_no_backfill(tmp_path: Path) -> None:
    """A current archive's rows already hold what the backfill would write.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run = build_run(
        tmp_path / "current",
        [_classification_genome(1, {"loss": 0.8, "target_metric": 0.4})],
    )
    # a current archive's genomes are fresh; skip the idle check
    result = backfill_archive(run, idle_minutes=0)

    assert result.updated == 0
    with GenomeArchive.open_readonly(run) as archive:
        assert BACKFILL_RUN_INFO_KEY not in archive.run_info()


def test_the_backfill_is_recorded_in_run_info(tmp_path: Path) -> None:
    """``run_info`` says when, with what code and how much was backfilled.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = _write_rl_run(tmp_path / "rl")
    backfill_archive(str(path))

    with GenomeArchive.open_readonly(str(path)) as archive:
        record = archive.run_info()[BACKFILL_RUN_INFO_KEY]
    assert record["genomes"] == 2
    assert record["updated"] == 2
    assert record["initial_fitness"] == {"recorded": 0, "derived": 1, "unavailable": 1}
    assert record["backfilled_at"] > 0
    assert "git_commit" in record


def test_output_dir_backfills_a_copy_and_leaves_the_source_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command copies the run and never writes the original.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to set the command line.
    """

    source = _write_rl_run(tmp_path / "runs" / "rl_1")
    digest = _sha256(source)
    output = tmp_path / "backfilled"

    monkeypatch.setattr(
        sys,
        "argv",
        ["backfill_archive", str(source.parent), "--output_dir", str(output)],
    )
    backfill_entry.main()

    assert _sha256(source) == digest
    with GenomeArchive.open_readonly(str(source)) as archive:
        assert "best_metrics" not in archive.genome_columns
    copy = output / "rl_1" / ARCHIVE_FILENAME
    with GenomeArchive.open_readonly(str(copy)) as archive:
        assert archive.run_info()[BACKFILL_RUN_INFO_KEY]["updated"] == 2

    # an existing copy is never overwritten
    with pytest.raises(SystemExit) as exit_info:
        backfill_entry.main()
    assert exit_info.value.code == 1
    assert _sha256(source) == digest


def test_an_archive_still_being_written_is_refused(tmp_path: Path) -> None:
    """A genome saved within --idle_minutes refuses the archive, and drops its copy.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run = build_run(tmp_path / "runs" / "live", [_rl_genome(1)])

    with pytest.raises(ArchiveInUseError):
        backfill_archive(run)
    with pytest.raises(ArchiveInUseError):
        backfill_entry.backfill_one(run, str(tmp_path / "out"), idle_minutes=10.0)
    assert not (tmp_path / "out" / "live").exists()


def test_the_command_requires_an_explicit_destination() -> None:
    """Neither or both of --output_dir and --in_place is rejected."""

    parser = backfill_entry.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--in_place", "--output_dir", "out"])
    args = parser.parse_args(["run", "--in_place"])
    assert args.in_place and args.output_dir is None


@pytest.mark.parametrize("eval_policy", ["greedy", "both"])
def test_rl_fitness_function_matches_the_objective_formula(eval_policy: str) -> None:
    """The module-level formula gives the fitness the objective always computed.

    Args:
        eval_policy: The evaluation regime the genome was scored under.
    """

    training = {"return_mean": 100.0, "best_episode_return": 120.0}
    validation = {"return_mean": 300.0, "return_std": 5.0}
    if eval_policy == "both":
        validation.update(
            {"greedy_return_mean": 300.0, "stochastic_return_mean": 250.0}
        )
    # the fitness as the objective computed it before the formula was moved
    expected = {
        "loss": -(0.25 * 100.0 + 0.75 * 300.0),
        "target_metric": 300.0,
        "eval_return_mean": 300.0,
        "eval_return_std": 5.0,
        "train_return_mean": 100.0,
        "best_episode_return": 120.0,
        "env_id": "Fake-v0",
        "eval_policy": eval_policy,
    }
    if eval_policy == "both":
        expected.update(
            {"eval_greedy_return_mean": 300.0, "eval_stochastic_return_mean": 250.0}
        )

    objective = ReinforcementLearningObjective(
        environment=SimpleNamespace(env_id="Fake-v0"),
        trainer=SimpleNamespace(),
        train_vs_validation_bias=0.25,
    )
    genome = SimpleNamespace(metadata={"eval_policy": eval_policy})

    assert objective.fitness_from_metrics(genome, training, validation) == expected
    assert (
        reinforcement_learning_fitness(
            training,
            validation,
            train_vs_validation_bias=0.25,
            env_id="Fake-v0",
            eval_policy=eval_policy,
        )
        == expected
    )
