"""Tests for restarting a run that is distributed across MPI ranks.

Everything a restart restores is rebuilt on the master rank, so these tests run a
real search across processes with ``mpiexec`` -- rather than the serial path the
rest of the restart tests take -- and check that continuing it across a fresh set
of processes carries on the run: genome numbers and insertions continue, nothing
is duplicated, and the population comes back. They also check that requeueing a
run that has already evaluated everything asked for stops cleanly instead of
hanging on the workers or evaluating more genomes.

Whether MPI can run at all is established by launching a trivial program rather
than by looking for ``mpiexec`` on the path: a machine with fewer cores than
ranks may refuse to start them, and a launcher built against a different MPI than
``mpi4py`` may fail to start Python. Where that cannot be made to work these
tests skip, saying what was tried.

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

#: Ranks to run with: a master and a worker, so genomes really are evaluated
#: away from the rank that owns the archive. Kept as low as that leaves room for
#: machines with few cores, which an MPI launcher may refuse to oversubscribe.
RANKS = 2

#: How long a single run may take before the test gives up on it. A run that
#: hangs (workers never told to stop) fails here rather than stalling the suite.
TIMEOUT_SECONDS = 180

#: The launcher arguments that were found to work, or why none did; filled in by
#: :func:`mpiexec_command` the first time it is asked.
_LAUNCHER: list[str] | str | None = None


def mpiexec_command() -> list[str]:
    """Returns a launcher that can actually start :data:`RANKS` ranks here.

    Having ``mpiexec`` on the path does not mean it can run: a machine with
    fewer cores than ranks may refuse to start them unless oversubscription is
    allowed, and a launcher built against a different MPI than ``mpi4py`` may
    fail to start Python at all. Rather than assume, this launches a trivial
    MPI program and keeps the first form that reports the expected number of
    ranks. The answer is worked out once and reused.

    Returns:
        The launcher and its arguments, ready to have a command appended.

    Raises:
        Skipped: (via :func:`pytest.skip`) When no form could run, carrying what
            the attempts printed so a failure elsewhere can be diagnosed.
    """

    global _LAUNCHER

    if _LAUNCHER is None:
        _LAUNCHER = _find_launcher()

    if isinstance(_LAUNCHER, str):
        pytest.skip(_LAUNCHER)

    return list(_LAUNCHER)


def _find_launcher() -> list[str] | str:
    """Finds a working ``mpiexec`` invocation, or explains why there is none.

    Returns:
        The launcher arguments that started :data:`RANKS` ranks, or a message
        saying what was tried and what happened.
    """

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        return "mpiexec is not available, so the MPI paths cannot be exercised"

    probe = "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
    attempts: list[str] = []

    # a launcher that will not oversubscribe refuses more ranks than cores, so
    # the plain form is tried first and oversubscription only as a fallback
    for arguments in ([mpiexec], [mpiexec, "--oversubscribe"]):
        command = [*arguments, "-n", str(RANKS), sys.executable, "-c", probe]
        try:
            finished = subprocess.run(
                command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            attempts.append(f"{' '.join(arguments)}: timed out")
            continue

        if finished.returncode == 0 and finished.stdout.split() == [str(RANKS)] * RANKS:
            return arguments

        attempts.append(
            f"{' '.join(arguments)}: exit {finished.returncode}, "
            f"stdout {finished.stdout.strip()!r}, stderr {finished.stderr.strip()!r}"
        )

    return (
        f"no mpiexec invocation could start {RANKS} ranks here, so the MPI paths "
        f"cannot be exercised ({'; '.join(attempts)})"
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
            *mpiexec_command(),
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


def assert_ran(finished: subprocess.CompletedProcess[str]) -> None:
    """Asserts a launched run succeeded, reporting everything it printed if not.

    An MPI launcher can fail with nothing on either stream, so a bare exit code
    says very little; this puts whatever the run did print into the failure.

    Args:
        finished: The finished run.

    Returns:
        None. Fails the test when the run did not exit cleanly.
    """

    assert finished.returncode == 0, (
        f"the run exited {finished.returncode}\n"
        f"stdout: {finished.stdout.strip()!r}\n"
        f"stderr: {finished.stderr.strip()!r}"
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
    assert_ran(first)

    stopped = archive_state(out_dir)
    # the master drains the genomes still being evaluated when the budget is
    # reached, so a run holds at least what was asked for
    assert len(stopped["numbers"]) >= 6

    second = run_search(out_dir, number_genomes=stopped["insertions"] + 5)
    assert_ran(second)

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

    assert_ran(run_search(out_dir, number_genomes=6))
    finished = archive_state(out_dir)

    for _ in range(2):
        assert_ran(run_search(out_dir, number_genomes=6))

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
    assert_ran(run_search(out_dir, number_genomes=4))

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
