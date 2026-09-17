"""Tests for restarting a run that is distributed across MPI ranks.

Everything a restart restores is rebuilt on the master rank, so these tests run a
real search across processes with ``mpiexec`` -- rather than the serial path the
rest of the restart tests take -- and check that continuing it across a fresh set
of processes carries on the run: genome numbers and insertions continue, nothing
is duplicated, and the population comes back. They also check that requeueing a
run that has already evaluated everything asked for stops cleanly instead of
hanging on the workers or evaluating more genomes.

The search itself is :mod:`tests.mpi_restart_driver`, which scores a genome by
its number instead of training it, so a run takes seconds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.utils.genome_archive import GenomeArchive

#: Ranks to run with: one master and two workers, so genomes really are
#: evaluated away from the rank that owns the archive.
RANKS = 3

#: How long a single run may take before the test gives up on it. A run that
#: hangs (workers never told to stop) fails here rather than stalling the suite.
TIMEOUT_SECONDS = 180

pytestmark = pytest.mark.skipif(
    shutil.which("mpiexec") is None,
    reason="mpiexec is not available, so the MPI paths cannot be exercised",
)


def run_search(out_dir: Path, number_genomes: int) -> subprocess.CompletedProcess[str]:
    """Runs the driver under ``mpiexec``, continuing whatever is in ``out_dir``.

    Args:
        out_dir: The run's output directory.
        number_genomes: The total the run should end up having evaluated.

    Returns:
        The finished process, with its output captured.
    """

    return subprocess.run(
        [
            shutil.which("mpiexec"),
            "-n",
            str(RANKS),
            sys.executable,
            "-m",
            "tests.mpi_restart_driver",
            "--mutation_strategy",
            "uniform",
            "1",
            "2",
            "--parent_strategy",
            "uniform",
            "2",
            "3",
            # the strategy flags above take any number of values, so a flag that
            # takes exactly one has to end the list before the sub-command
            "--out_dir",
            str(out_dir),
            "--number_genomes",
            str(number_genomes),
            "steady_state",
            "--max_population_size",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        cwd=Path(__file__).resolve().parent.parent,
    )


def archive_state(out_dir: Path) -> dict[str, Any]:
    """Reads what a run's archive holds.

    Args:
        out_dir: The run's output directory.

    Returns:
        The genome numbers stored, the highest insertion, the population's
        members, and how many restarts were recorded.
    """

    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        numbers = [number for number, _ in reader.iter_genome_dicts()]
        insertions = reader.connection.execute(
            "SELECT MAX(insertion) FROM genomes"
        ).fetchone()[0]
        return {
            "numbers": numbers,
            "insertions": insertions,
            "population": reader.population_at(),
            "restarts": len(reader.run_info().get("restarts") or []),
        }


def test_a_distributed_run_continues_across_a_restart(tmp_path) -> None:
    """Continuing an MPI run carries on its numbering, insertions and population.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    out_dir = tmp_path / "run"

    first = run_search(out_dir, number_genomes=6)
    assert first.returncode == 0, first.stderr

    stopped = archive_state(out_dir)
    # the master drains the genomes still being evaluated when the budget is
    # reached, so a run holds at least what was asked for
    assert len(stopped["numbers"]) >= 6

    second = run_search(out_dir, number_genomes=stopped["insertions"] + 5)
    assert second.returncode == 0, second.stderr

    continued = archive_state(out_dir)

    assert len(continued["numbers"]) > len(stopped["numbers"])
    assert continued["insertions"] > stopped["insertions"]
    # nothing was renumbered or evaluated twice
    assert len(continued["numbers"]) == len(set(continued["numbers"]))
    assert set(stopped["numbers"]) <= set(continued["numbers"])
    # the genomes the restart added are numbered past everything already stored
    added = set(continued["numbers"]) - set(stopped["numbers"])
    assert min(added) > max(stopped["numbers"])
    # the restart is recorded, from where the stopped run had reached
    assert continued["restarts"] == 1
    assert continued["population"]


def test_requeueing_a_finished_distributed_run_stops_cleanly(tmp_path) -> None:
    """A finished MPI run asked for no more genomes exits without doing anything.

    A restart that has nothing to do must still shut its workers down, and must
    not record itself: a requeue script firing after a run finishes would
    otherwise fill the run's record with restarts that evaluated nothing.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    out_dir = tmp_path / "run"

    assert run_search(out_dir, number_genomes=6).returncode == 0
    finished = archive_state(out_dir)

    for _ in range(2):
        again = run_search(out_dir, number_genomes=6)
        assert again.returncode == 0, again.stderr

    unchanged = archive_state(out_dir)

    assert unchanged["numbers"] == finished["numbers"]
    assert unchanged["insertions"] == finished["insertions"]
    assert unchanged["restarts"] == 0


def test_the_driver_records_what_a_restart_needs(tmp_path) -> None:
    """The distributed run records the arguments and innovation numbers to restart.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    out_dir = tmp_path / "run"
    assert run_search(out_dir, number_genomes=4).returncode == 0

    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        run_info = reader.run_info()
        innovations = reader.connection.execute(
            "SELECT MAX(max_innovation_number) FROM genomes"
        ).fetchone()[0]

    arguments = run_info["arguments"]
    assert arguments["population_strategy"] == "steady_state"
    assert arguments["max_population_size"] == 4
    assert innovations is not None
    # the run's own record is JSON, so a restart reads it back as it was written
    assert json.loads(json.dumps(arguments)) == arguments
