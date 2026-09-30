"""Tests for recording run timing in the archive and serving it to the dashboard.

Archive format 6 records three per-genome timing columns (how long a worker
waited for the genome, how long the master took to generate it, and its
turnaround from generation to insertion) and a ``master_timing`` table the MPI
master fills every :data:`~src.evolution.master_worker.MASTER_TIMING_RECORD_EVERY`
insertions. The dashboard's timing endpoint, its runs list and the
``get_run_timing`` MCP tool read them; archives written before format 6 must
still open, showing "no timing recorded".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.evolution import master_worker
from src.evolution.steady_state_population import SteadyStatePopulation
from src.utils import restart
from src.utils.artifact_viewer.mcp_tools import DashboardTools
from src.utils.artifact_viewer.server import (
    ArtifactViewer,
    RenderService,
    RunRegistry,
    master_busy_percent,
    timing_payload_from,
)
from src.utils.genome_archive import ARCHIVE_FILENAME, TIMING_COLUMNS, GenomeArchive
from src.utils.phase_timer import PhaseTimer

from tests.test_exaqc_dashboard import build_run, standard_genomes
from tests.test_exaqc_insert_archive import build_search, compare
from tests.test_master_timing import ScriptedWorkerComm


def timed_genomes() -> list[Any]:
    """Builds the standard four genomes, each with recorded timing.

    Genome ``n`` waited ``n`` ms for its worker, took ``2n`` ms to generate,
    was evaluated for ``3n`` ms and was inserted ``4n`` ms after it was
    generated.

    Returns:
        The genomes, in insertion order.
    """

    genomes = standard_genomes()
    for genome in genomes:
        n = genome.genome_number
        genome.serialized["metadata"]["timing"] = {
            "request_wait_seconds": n / 1000,
            "generation_seconds": 2 * n / 1000,
            "evaluation_seconds": 3 * n / 1000,
            "generated_at": 1000.0,
            "inserted_at": 1000.0 + 4 * n / 1000,
        }
    return genomes


def timer(**phases: tuple[float, int]) -> PhaseTimer:
    """Builds a phase timer holding the given totals.

    Args:
        **phases: Each phase's ``(seconds, count)``.

    Returns:
        The timer.
    """

    built = PhaseTimer()
    for phase, (seconds, count) in phases.items():
        built.seconds[phase] = seconds
        built.counts[phase] = count
    return built


def build_timed_run(directory: Path) -> str:
    """Writes a run whose genomes and master both recorded timing.

    The master recorded two intervals: insertions 1-2, busy 1 s of 4 s, and
    insertions 3-4, busy 3 s of 4 s.

    Args:
        directory: The run directory to create.

    Returns:
        The run directory, as a string.
    """

    run_directory = build_run(directory, timed_genomes())
    with GenomeArchive.create(run_directory, expect_existing=True) as archive:
        archive.record_master_timing(
            step=2,
            wall_seconds=4.0,
            genomes=2,
            phases=timer(idle=(3.0, 4), generate=(0.5, 2), insert=(0.5, 2)),
            insert_parts=timer(archive=(0.2, 2), best_files=(0.3, 1)),
        )
        archive.record_master_timing(
            step=4,
            wall_seconds=4.0,
            genomes=2,
            phases=timer(idle=(1.0, 4), generate=(1.0, 2), insert=(2.0, 2)),
            insert_parts=timer(archive=(0.4, 2), best_files=(1.6, 2)),
        )
    return run_directory


def make_old_format(run_directory: str) -> None:
    """Rewrites a run's archive as one written before format 6.

    Args:
        run_directory: The run directory holding the archive.

    Returns:
        None. Drops the timing columns and the ``master_timing`` table.
    """

    connection = sqlite3.connect(str(Path(run_directory) / ARCHIVE_FILENAME))
    try:
        for column in TIMING_COLUMNS:
            connection.execute(f"ALTER TABLE genomes DROP COLUMN {column}")
        connection.execute("DROP TABLE master_timing")
        connection.commit()
    finally:
        connection.close()


def test_genomes_record_their_timing_columns(tmp_path: Path) -> None:
    """Wait and generation are copied from the genome; turnaround is computed."""

    run_directory = build_run(tmp_path / "run", timed_genomes())

    with GenomeArchive.open_readonly(run_directory) as reader:
        assert reader.has_timing_columns()
        summary = reader.get_summary(3)
        assert summary["request_wait_seconds"] == pytest.approx(0.003)
        assert summary["generation_seconds"] == pytest.approx(0.006)
        assert summary["turnaround_seconds"] == pytest.approx(0.012)
        # the timing columns chart, sort and list like any other numeric column
        assert set(TIMING_COLUMNS) <= set(reader.series_metrics())
        numbers = [
            row["genome_number"]
            for row in reader.list_genomes(
                sort_key="turnaround_seconds", descending=True
            )
        ]
        assert numbers == [4, 3, 2, 1]


def test_genomes_without_timing_record_nulls(tmp_path: Path) -> None:
    """A genome that recorded no timing (e.g. a serial run's wait) stores NULLs."""

    run_directory = build_run(tmp_path / "run", standard_genomes())

    with GenomeArchive.open_readonly(run_directory) as reader:
        summary = reader.get_summary(1)
        assert all(summary[column] is None for column in TIMING_COLUMNS)


def test_master_timing_rows_round_trip_and_summarize(tmp_path: Path) -> None:
    """Master rows decode, and worker intervals line up with them."""

    run_directory = build_timed_run(tmp_path / "run")

    with GenomeArchive.open_readonly(run_directory) as reader:
        rows = reader.master_timing()
        assert [row["step"] for row in rows] == [2, 4]
        assert rows[0]["phases"]["idle"] == {"seconds": 3.0, "count": 4}
        assert rows[1]["insert_parts"]["best_files"] == {"seconds": 1.6, "count": 2}

        first, second = reader.worker_timing([2, 4])
        assert first["genomes"] == 2 and second["genomes"] == 2
        # genomes 1-2 waited 1 and 2 ms and were evaluated for 3 and 6 ms
        assert first["request_wait_seconds"] == pytest.approx(0.0015)
        assert first["wait_fraction"] == pytest.approx(3 / 12)
        assert second["turnaround_seconds"] == pytest.approx(0.014)

        # busy is 1 s + 3 s of 8 s recorded
        assert master_busy_percent(rows) == pytest.approx(50.0)
        payload = timing_payload_from(reader)

    assert payload["recorded"] is True
    assert payload["master"]["step"] == [2, 4]
    assert payload["master"]["busy_percent"] == pytest.approx([25.0, 75.0])
    assert payload["master"]["idle_percent"] == pytest.approx([75.0, 25.0])
    assert payload["master"]["genomes_per_second"] == pytest.approx([0.5, 0.5])
    assert payload["master"]["phase_ms"]["generate"] == pytest.approx([250.0, 500.0])
    assert payload["master"]["insert_part_ms"]["best_files"] == pytest.approx(
        [300.0, 800.0]
    )
    assert payload["workers"]["step"] == [2, 4]
    assert payload["workers"]["request_wait_ms"] == pytest.approx([1.5, 3.5])
    summary = payload["summary"]
    assert summary["busy_percent"] == pytest.approx(50.0)
    assert summary["genomes_per_second"] == pytest.approx(0.5)
    assert summary["request_wait_ms"] == pytest.approx(2.5)
    assert summary["wait_percent"] == pytest.approx(100 * 10 / 40)


def test_a_run_without_master_timing_charts_genome_timing_by_insertion(
    tmp_path: Path,
) -> None:
    """A serial run has worker series (over fixed intervals) but no master series."""

    run_directory = build_run(tmp_path / "run", timed_genomes())

    with GenomeArchive.open_readonly(run_directory) as reader:
        payload = timing_payload_from(reader)

    assert payload["recorded"] is True
    assert payload["master"] == {}
    # four genomes fit in one interval of 100 insertions
    assert payload["workers"]["step"] == [4]
    assert "busy_percent" not in payload["summary"]
    assert payload["summary"]["evaluation_ms"] == pytest.approx(7.5)


def test_an_archive_from_before_timing_opens_with_no_timing_recorded(
    tmp_path: Path,
) -> None:
    """Old archives keep working everywhere; timing just reads as not recorded."""

    run_directory = build_timed_run(tmp_path / "run")
    make_old_format(run_directory)

    with GenomeArchive.open_readonly(run_directory) as reader:
        assert not reader.has_timing_columns()
        assert reader.master_timing() == []
        assert not set(TIMING_COLUMNS) & set(reader.series_metrics())
        summary = reader.get_summary(2)
        assert all(summary[column] is None for column in TIMING_COLUMNS)
        # sorting by a column the archive lacks falls back rather than failing
        assert len(reader.list_genomes(sort_key="request_wait_seconds")) == 4
        assert reader.points("turnaround_seconds")["y"] == [None] * 4
        assert timing_payload_from(reader) == {
            "recorded": False,
            "master": {},
            "workers": {},
            "summary": {},
        }

    viewer = ArtifactViewer(
        RunRegistry(run_directories=[run_directory]), RenderService(processes=0)
    )
    assert viewer.runs_payload()["runs"][0]["master_busy_percent"] is None
    assert viewer.timing_payload(0)["recorded"] is False

    tools = DashboardTools(
        RunRegistry(run_directories=[run_directory]),
        RenderService(processes=0),
        "http://dash.test:8000",
    )
    timing = tools.get_run_timing(0)
    assert timing["recorded"] is False
    assert "no timing was recorded" in timing["note"]
    # the roll-up reads the missing columns as NULL rather than refusing the run
    rows = tools.query_sql(
        "SELECT COUNT(*), COUNT(request_wait_seconds) FROM genomes", runs=[0]
    )["rows"]
    assert rows == [[4, 0]]


def test_an_archive_from_before_timing_cannot_be_restarted(tmp_path: Path) -> None:
    """Adding genomes to an old-format archive is refused with a clear reason."""

    run_directory = build_run(
        tmp_path / "run", standard_genomes(), run_info={"arguments": {}}
    )
    make_old_format(run_directory)

    with pytest.raises(ValueError, match="per-genome timing"):
        restart.load(run_directory)


def test_the_master_records_timing_rows_in_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows land every ``MASTER_TIMING_RECORD_EVERY`` insertions plus the remainder.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Shrinks the recording interval so a short run records
            several rows.
    """

    monkeypatch.setattr(master_worker, "MASTER_TIMING_RECORD_EVERY", 2)
    archive = GenomeArchive.create(str(tmp_path / "run"))
    # rendering the best genome's images is not what is under test
    monkeypatch.setattr(archive, "write_current_best", lambda genome, kind: None)
    search = build_search(
        SteadyStatePopulation(max_population_size=3, compare=compare), archive
    )

    try:
        master_worker.master(
            comm=ScriptedWorkerComm(),
            rank=0,
            exaqc=search,
            run_for=5,
            timing_report_every=0,
        )
    finally:
        search.close()

    with GenomeArchive.open_readonly(str(tmp_path / "run")) as reader:
        rows = reader.master_timing()
        assert [row["step"] for row in rows] == [2, 4, 5]
        assert [row["genomes"] for row in rows] == [2, 2, 1]
        assert {"idle", "generate", "send", "deserialize", "insert"} <= set(
            rows[0]["phases"]
        )
        assert "population" in rows[0]["insert_parts"]
        summaries = reader.list_genomes(sort_key="genome_number")
        assert all(summary["generation_seconds"] is not None for summary in summaries)
        assert all(summary["turnaround_seconds"] is not None for summary in summaries)


def test_get_run_timing_summarizes_and_downsamples(tmp_path: Path) -> None:
    """The MCP tool returns the summary and capped series of a timed run."""

    run_directory = build_timed_run(tmp_path / "run")
    tools = DashboardTools(
        RunRegistry(run_directories=[run_directory]),
        RenderService(processes=0),
        "http://dash.test:8000",
    )

    timing = tools.get_run_timing(0, max_points=1)

    assert timing["recorded"] is True
    assert timing["summary"]["busy_percent"] == pytest.approx(50.0)
    assert len(timing["master"]["step"]) == 1
    assert len(timing["master"]["phase_ms"]["generate"]) == 1
    assert len(timing["workers"]["step"]) == 1
    assert timing["recorded_points"] == 2
    assert timing["dashboard_url"].endswith("/run/0")
