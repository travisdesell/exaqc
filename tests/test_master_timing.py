"""Tests for the MPI master's and workers' timing instrumentation.

The master logs, every ``--timing_report_every`` insertions and once for the
whole run, how its time split between waiting on workers and generating,
sending, deserializing and inserting genomes (with insertion broken into its
parts by :attr:`EXAQC.insert_timer`). Each worker records how long it waited
for every genome as ``timing["request_wait_seconds"]`` and logs its own split
when the search ends. The loops are driven here through a scripted stand-in
for the MPI communicator, so no MPI launch is needed.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from loguru import logger
from mpi4py import MPI

from src.circuits.circuit import CircuitGenome
from src.evolution import master_worker
from src.evolution.exaqc import EXAQC
from src.evolution.steady_state_population import SteadyStatePopulation
from src.utils.phase_timer import PhaseTimer
from src.utils.restart import OVERRIDABLE_ARGUMENTS

from tests.test_exaqc_insert_archive import FakeGenome, build_search, compare


class ScriptedWorkerComm:
    """Stands in for ``MPI.COMM_WORLD`` as seen by the master, with one worker.

    The simulated worker (rank 1) starts by requesting a genome; whenever the
    master sends it one, it "evaluates" it (assigning a fitness) and queues the
    evaluated genome followed by its next request.
    """

    def __init__(self) -> None:
        """Queues the worker's first request.

        Returns:
            None. Sets ``inbox`` (messages the master will receive) and
            ``sent`` (every message the master sent).
        """

        self.inbox: list[tuple[Any, int]] = [
            (None, master_worker.tag_ids["genome_request"])
        ]
        self.sent: list[tuple[Any, int, int]] = []

    def Get_size(self) -> int:
        """Returns the communicator size: the master plus one worker.

        Returns:
            2.
        """

        return 2

    def recv(self, source: int, status: MPI.Status) -> Any:
        """Delivers the next queued message from the worker.

        Args:
            source: Ignored (every message comes from rank 1).
            status: Filled in with the message's tag and source.

        Returns:
            The message's data.
        """

        data, tag = self.inbox.pop(0)
        status.Set_tag(tag)
        status.Set_source(1)
        return data

    def send(self, data: Any, dest: int, tag: int) -> None:
        """Records a message to the worker, simulating its evaluation.

        Args:
            data: The serialized genome, or ``None`` to stop the worker.
            dest: The worker's rank.
            tag: The message tag.

        Returns:
            None. Appends to ``sent`` and, for a genome, queues its evaluated
            copy and the worker's next request.
        """

        self.sent.append((data, dest, tag))
        if data is None:
            return
        genome = CircuitGenome.from_dict(data)
        genome.fitness = {"loss": float(genome.genome_number), "target_metric": 0.0}
        self.inbox.append((genome.to_dict(), master_worker.tag_ids["genome_response"]))
        self.inbox.append((None, master_worker.tag_ids["genome_request"]))


class ScriptedMasterComm:
    """Stands in for ``MPI.COMM_WORLD`` as seen by a worker.

    The master hands out the given serialized genomes in order, then ``None``.
    """

    def __init__(self, genomes: list[dict[str, Any]]) -> None:
        """Queues the genomes the master will hand out.

        Args:
            genomes: Serialized genomes for the worker to evaluate.

        Returns:
            None. Sets ``to_hand_out`` and ``sent``.
        """

        self.to_hand_out: list[dict[str, Any] | None] = [*genomes, None]
        self.sent: list[tuple[Any, int, int]] = []

    def send(self, data: Any, dest: int, tag: int) -> None:
        """Records a message the worker sent to the master.

        Args:
            data: The message data.
            dest: The master's rank.
            tag: The message tag.

        Returns:
            None. Appends to ``sent``.
        """

        self.sent.append((data, dest, tag))

    def recv(self, source: int, tag: int, status: MPI.Status) -> Any:
        """Delivers the master's next genome, or ``None`` once all are handed out.

        Args:
            source: Ignored (always the master).
            tag: Ignored.
            status: Filled in with the message's tag.

        Returns:
            The next serialized genome, or ``None``.
        """

        data = self.to_hand_out.pop(0)
        status.Set_tag(
            master_worker.tag_ids["genome"]
            if data is not None
            else master_worker.tag_ids["genome_response"]
        )
        return data


@pytest.fixture
def info_messages() -> Iterator[list[str]]:
    """Captures the INFO-and-above log messages written during a test.

    Yields:
        The list the messages are appended to.
    """

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="INFO"
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def test_phase_timer_accumulates_and_differences() -> None:
    """Phases add up, and ``since`` keeps only what was added after a copy."""

    timer = PhaseTimer()
    timer.add("generate", 0.002)
    timer.add("generate", 0.004)
    earlier = timer.copy()
    timer.add("generate", 0.010)
    timer.add("insert", 0.020)
    with timer.time("send"):
        pass

    assert earlier.counts == {"generate": 2}
    assert timer.counts == {"generate": 3, "insert": 1, "send": 1}
    assert timer.total(("generate", "insert")) == pytest.approx(0.036)

    interval = timer.since(earlier)
    assert interval.counts == {"generate": 1, "insert": 1, "send": 1}
    assert interval.seconds["generate"] == pytest.approx(0.010)
    assert interval.describe(("generate", "insert", "never")) == (
        "generate 10.00 ms (x1), insert 20.00 ms (x1)"
    )
    assert PhaseTimer().describe() == "none"


def test_phase_timer_records_a_block_that_raises() -> None:
    """A timed block that raises is still counted."""

    timer = PhaseTimer()
    with pytest.raises(ValueError):
        with timer.time("insert"):
            raise ValueError("insert failed")

    assert timer.counts == {"insert": 1}


def test_insert_genome_times_each_part_of_an_insertion() -> None:
    """Every part of an insertion is timed once per insertion."""

    search = build_search(
        SteadyStatePopulation(max_population_size=4, compare=compare), MagicMock()
    )

    search.insert_genome(FakeGenome(1, loss=0.5, target_metric=0.5))
    search.insert_genome(FakeGenome(2, loss=0.6, target_metric=0.4))

    assert search.insert_timer.counts == {
        "population": 2,
        "archive": 2,
        "population_events": 2,
    }


def test_master_reports_timing_every_interval_and_for_the_whole_run(
    info_messages: list[str],
) -> None:
    """Reports land every ``timing_report_every`` insertions, plus one at the end.

    Args:
        info_messages: Captured INFO log messages.
    """

    search = build_search(
        SteadyStatePopulation(max_population_size=3, compare=compare), MagicMock()
    )
    comm = ScriptedWorkerComm()

    master_worker.master(
        comm=comm, rank=0, exaqc=search, run_for=4, timing_report_every=2
    )

    reports = [
        message for message in info_messages if message.startswith("master timing")
    ]
    assert [report.split("]")[0] for report in reports] == [
        "master timing [insertions 1-2",
        "master timing [insertions 3-4",
        "master timing [whole run",
    ]
    assert "2 genomes inserted" in reports[0]
    assert "4 genomes inserted" in reports[-1]
    for report in reports:
        assert "busy" in report and "idle" in report
        assert "generate" in report and "insert parts: population" in report

    # four genomes handed out, then the worker was told to stop
    assert [data is None for data, _, _ in comm.sent] == [False] * 4 + [True]
    assert search.inserted_genomes == 4


def test_master_with_reporting_off_still_reports_the_whole_run(
    info_messages: list[str],
) -> None:
    """``timing_report_every=0`` leaves only the final report.

    Args:
        info_messages: Captured INFO log messages.
    """

    search = build_search(
        SteadyStatePopulation(max_population_size=3, compare=compare), MagicMock()
    )

    master_worker.master(
        comm=ScriptedWorkerComm(),
        rank=0,
        exaqc=search,
        run_for=3,
        timing_report_every=0,
    )

    reports = [
        message for message in info_messages if message.startswith("master timing")
    ]
    assert len(reports) == 1
    assert reports[0].startswith("master timing [whole run]: 3 genomes inserted")


def test_worker_records_how_long_it_waited_for_each_genome(
    info_messages: list[str],
) -> None:
    """Each returned genome carries its request wait; the worker logs its split.

    Args:
        info_messages: Captured INFO log messages.
    """

    search = build_search(
        SteadyStatePopulation(max_population_size=3, compare=compare), MagicMock()
    )
    genomes = [search.generate_genome().to_dict() for _ in range(2)]
    comm = ScriptedMasterComm(genomes)

    def objective(genome: CircuitGenome) -> None:
        """Assigns a fitness without training."""
        genome.fitness = {"loss": 1.0, "target_metric": 0.0}

    master_worker.worker(comm=comm, rank=3, objective=objective)

    responses = [
        data
        for data, _, tag in comm.sent
        if tag == master_worker.tag_ids["genome_response"]
    ]
    assert len(responses) == 2
    for response in responses:
        timing = response["metadata"]["timing"]
        assert timing["request_wait_seconds"] >= 0.0
        assert "evaluation_seconds" in timing

    summaries = [
        message for message in info_messages if message.startswith("worker 3 timing")
    ]
    assert len(summaries) == 1
    assert summaries[0].startswith("worker 3 timing: 2 genomes in")


def test_timing_report_every_defaults_to_1000_and_is_overridable_on_restart() -> None:
    """The flag defaults to 1000 and, being about logging only, may change on restart."""

    parser = argparse.ArgumentParser()
    EXAQC.initialize_parser(parser)
    defaults = {action.dest: action.default for action in parser._actions}

    assert defaults["timing_report_every"] == 1000
    assert "timing_report_every" in OVERRIDABLE_ARGUMENTS
