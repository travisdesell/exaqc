from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from mpi4py import MPI
from mpi4py.MPI import Intracomm

from src.circuits.circuit import CircuitGenome
from src.evolution.exaqc import EXAQC
from src.evolution.objective import Objective, evaluate_genome
from src.utils.phase_timer import PhaseTimer

tag_ids = {
    "genome": 1,
    "genome_request": 2,
    "genome_response": 3,
    "search_over": 4,
}

# get the reverse dict of tags
tags = {}
for key, value in tag_ids.items():
    tags[value] = key

#: The master loop's phases, in the order its timing reports list them:
#: ``idle`` is time spent waiting for a worker's message, the rest is work.
MASTER_IDLE_PHASE: str = "idle"
MASTER_BUSY_PHASES: tuple[str, ...] = ("generate", "send", "deserialize", "insert")

#: How many insertions each ``master_timing`` row in the run's archive covers,
#: which the dashboard charts. Finer than the default log interval
#: (``--timing_report_every``) so short runs still chart several points.
MASTER_TIMING_RECORD_EVERY: int = 100

#: The parts of an insertion (see :meth:`EXAQC.insert_genome`), in report order.
INSERT_PHASES: tuple[str, ...] = (
    "population",
    "archive",
    "best_files",
    "population_events",
)


@dataclass
class ReportWindow:
    """Where the interval a master timing report covers began.

    Attributes:
        started: ``time.perf_counter()`` when the interval began.
        timer: A copy of the master loop's timer at that point.
        insert_timer: A copy of the search's insertion timer at that point.
        inserted: How many genomes the master had inserted by then.
    """

    started: float
    timer: PhaseTimer
    insert_timer: PhaseTimer
    inserted: int


def start_window(timer: PhaseTimer, exaqc: EXAQC, inserted: int) -> ReportWindow:
    """Starts a new timing-report interval at the current moment.

    Args:
        timer: The master loop's timer.
        exaqc: The search, whose ``insert_timer`` is copied.
        inserted: How many genomes the master has inserted so far.

    Returns:
        The window marking where the interval began.
    """

    return ReportWindow(
        started=time.perf_counter(),
        timer=timer.copy(),
        insert_timer=exaqc.insert_timer.copy(),
        inserted=inserted,
    )


def report_master_timing(
    label: str,
    window: ReportWindow,
    timer: PhaseTimer,
    exaqc: EXAQC,
    inserted: int,
) -> None:
    """Logs where the master's time went since a window began.

    The busy fraction is the time spent generating, sending, deserializing and
    inserting genomes; the idle fraction is time spent waiting for a worker's
    message. A busy fraction near 100% means workers are queueing on the master.

    Args:
        label: What the interval is, e.g. ``"insertions 1-1000"``.
        window: Where the interval began.
        timer: The master loop's timer.
        exaqc: The search, whose ``insert_timer`` breaks down insertion time.
        inserted: How many genomes the master has inserted so far.

    Returns:
        None. Writes one INFO log line.
    """

    wall = time.perf_counter() - window.started
    interval = timer.since(window.timer)
    insert_interval = exaqc.insert_timer.since(window.insert_timer)
    count = inserted - window.inserted
    busy = interval.total(MASTER_BUSY_PHASES)
    idle = interval.total((MASTER_IDLE_PHASE,))

    def percent(seconds: float) -> float:
        """Returns ``seconds`` as a percentage of the interval's wall time."""

        return 100.0 * seconds / wall if wall > 0 else 0.0

    logger.info(
        "master timing [{}]: {} genomes inserted in {:.1f} s ({:.2f} genomes/s) | "
        "busy {:.1f}%, idle {:.1f}% | {} | insert parts: {}",
        label,
        count,
        wall,
        count / wall if wall > 0 else 0.0,
        percent(busy),
        percent(idle),
        interval.describe((MASTER_IDLE_PHASE, *MASTER_BUSY_PHASES)),
        insert_interval.describe(INSERT_PHASES),
    )


def receive(comm: Intracomm, timer: PhaseTimer) -> tuple[Any, int, int]:
    """Waits for the next message from any worker, timing the wait as idle.

    Args:
        comm: The MPI communicator.
        timer: The master loop's timer.

    Returns:
        The message's data, its tag id and the rank that sent it.
    """

    status = MPI.Status()
    with timer.time(MASTER_IDLE_PHASE):
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)

    tag_id = status.Get_tag()
    source = status.Get_source()

    # logged lazily: formatting the genome itself on every message is costly
    logger.debug("master process received tag {} from source {}", tags[tag_id], source)
    return data, tag_id, source


def generate_genome(exaqc: EXAQC, timer: PhaseTimer) -> CircuitGenome:
    """Generates a genome for a worker, timing it as the ``generate`` phase.

    Args:
        exaqc: The search generating the genome.
        timer: The master loop's timer.

    Returns:
        The new genome, with how long it took to generate recorded as its
        ``timing["generation_seconds"]``.
    """

    started = time.perf_counter()
    genome = exaqc.generate_genome()
    seconds = time.perf_counter() - started
    timer.add("generate", seconds)
    genome.metadata.setdefault("timing", {})["generation_seconds"] = seconds
    return genome


def record_master_timing(
    window: ReportWindow, timer: PhaseTimer, exaqc: EXAQC, inserted: int
) -> None:
    """Stores the master's timing since a window began in the run's archive.

    Args:
        window: Where the interval began.
        timer: The master loop's timer.
        exaqc: The search, whose ``archive`` receives the row (nothing is
            written when it has none) and whose ``insert_timer`` breaks down
            insertion time.
        inserted: How many genomes the master has inserted so far.

    Returns:
        None. Writes one ``master_timing`` row, keyed by the search's total
        insertion count so rows continue across a restart.
    """

    if exaqc.archive is None:
        return
    exaqc.archive.record_master_timing(
        step=exaqc.inserted_genomes,
        wall_seconds=time.perf_counter() - window.started,
        genomes=inserted - window.inserted,
        phases=timer.since(window.timer),
        insert_parts=exaqc.insert_timer.since(window.insert_timer),
    )


def insert_response(exaqc: EXAQC, data: dict[str, Any], timer: PhaseTimer) -> None:
    """Deserializes an evaluated genome from a worker and inserts it.

    Args:
        exaqc: The search to insert into.
        data: The serialized genome the worker sent back.
        timer: The master loop's timer.

    Returns:
        None. Inserts the genome into ``exaqc``.
    """

    with timer.time("deserialize"):
        genome = CircuitGenome.from_dict(data)
    with timer.time("insert"):
        exaqc.insert_genome(genome)


def master(
    comm: Intracomm,
    rank: int,
    exaqc: EXAQC,
    run_for: int,
    timing_report_every: int = 1000,
) -> None:
    """
    The master process which will generate genomes and receive results from workers
    to perform the EXAQC search.

    Every ``timing_report_every`` insertions, and once for the whole run at the
    end, it logs where its time went (see :func:`report_master_timing`), so a
    run shows whether workers are waiting on the master. Independently, every
    :data:`MASTER_TIMING_RECORD_EVERY` insertions (and for the final partial
    interval) it stores the same breakdown in the run's archive for the
    dashboard (see :func:`record_master_timing`).

    Args:
        comm: is the MPI COMM WORLD
        rank: is the rank of the master process (should be 0)
        exaqc: is an initialized EXAQC search object
        run_for: is how many genomes to generate
        timing_report_every: how many insertions each periodic timing report
            covers; 0 keeps only the final report.

    Returns:
        None. Runs the search to completion, inserting every evaluated genome
        into ``exaqc`` and recording ``master_timing`` rows in its archive, and
        tells every worker to stop.
    """
    n_workers = comm.Get_size() - 1

    evaluated_genomes = 0
    # counts every insertion, including the ones drained after run_for is reached
    inserted = 0

    timer = PhaseTimer()
    run_window = start_window(timer, exaqc, inserted)
    # the interval the next log report covers
    window = run_window
    # the interval the next master_timing row in the archive covers
    archive_window = run_window

    while evaluated_genomes < run_for:
        data, tag_id, source = receive(comm, timer)

        if tag_id == tag_ids["genome_request"]:
            genome = generate_genome(exaqc, timer)
            with timer.time("send"):
                comm.send(genome.to_dict(), dest=source, tag=tag_ids["genome"])

        elif tag_id == tag_ids["genome_response"]:
            insert_response(exaqc, data, timer)
            inserted += 1

            evaluated_genomes += 1
            logger.info(f"evaluated {evaluated_genomes} of max {run_for} genomes")

            if inserted % MASTER_TIMING_RECORD_EVERY == 0:
                record_master_timing(archive_window, timer, exaqc, inserted)
                archive_window = start_window(timer, exaqc, inserted)

            if timing_report_every > 0 and inserted % timing_report_every == 0:
                report_master_timing(
                    f"insertions {window.inserted + 1}-{inserted}",
                    window,
                    timer,
                    exaqc,
                    inserted,
                )
                window = start_window(timer, exaqc, inserted)

    # receive last genomes and cleanup
    finished_workers = 0
    while finished_workers < n_workers:
        data, tag_id, source = receive(comm, timer)

        if tag_id == tag_ids["genome_request"]:
            comm.send(None, dest=source, tag=tag_ids["genome_response"])
            finished_workers += 1

        elif tag_id == tag_ids["genome_response"]:
            insert_response(exaqc, data, timer)
            inserted += 1

    # the last, partial interval, when it inserted anything
    if inserted > archive_window.inserted:
        record_master_timing(archive_window, timer, exaqc, inserted)
    report_master_timing("whole run", run_window, timer, exaqc, inserted)


def worker(
    comm: Intracomm,
    rank: int,
    objective: Objective,
) -> None:
    """
    This is a worker process which will repeatedly request new genomes
    from the master process, evaluate them with the objective function and
    send the genome back to the master process to be inserted into the search.

    How long the worker waited for each genome -- from sending its request to
    receiving the genome, on this worker's own clock -- is recorded in the
    genome's ``timing["request_wait_seconds"]``, and when the search ends the
    worker logs how its time split between waiting and evaluating.

    Args:
        comm: is the MPI COMM WORLD
        rank: is the rank of the master process (should be 0)
        objective: is the objective function used to evaluate genomes

    Returns:
        None. Returns once the master signals that the search is over.
    """

    started = time.perf_counter()
    waiting_seconds = 0.0
    evaluating_seconds = 0.0
    requests = 0
    evaluated = 0

    while True:
        # request a genome from the main process, timing how long it takes to
        # arrive (the master's queue, generation and the transfer)
        requested = time.perf_counter()
        comm.send(None, dest=0, tag=tag_ids["genome_request"])

        status = MPI.Status()
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        request_wait = time.perf_counter() - requested
        waiting_seconds += request_wait
        requests += 1
        tag_id = status.Get_tag()

        # logged lazily: formatting the genome itself on every message is costly
        logger.debug("worker process {} received tag: {}", rank, tags[tag_id])
        if data is None:
            # the search is over, the worker can quit
            break

        genome = CircuitGenome.from_dict(data)

        # records how long the evaluation took and which rank and host ran it
        evaluate_genome(objective, genome, rank=rank)

        # written after evaluation, so an objective replacing the metadata
        # cannot lose it
        timing = genome.metadata.setdefault("timing", {})
        timing["request_wait_seconds"] = request_wait
        evaluating_seconds += float(timing.get("evaluation_seconds", 0.0))
        evaluated += 1

        comm.send(genome.to_dict(), dest=0, tag=tag_ids["genome_response"])

    wall = time.perf_counter() - started
    logger.info(
        "worker {} timing: {} genomes in {:.1f} s | waiting on the master {:.1f} s "
        "({:.1f}%, {:.2f} ms per request) | evaluating {:.1f} s ({:.1f}%)",
        rank,
        evaluated,
        wall,
        waiting_seconds,
        100.0 * waiting_seconds / wall if wall > 0 else 0.0,
        1000.0 * waiting_seconds / requests if requests else 0.0,
        evaluating_seconds,
        100.0 * evaluating_seconds / wall if wall > 0 else 0.0,
    )


def run_evolution(
    objective: Objective,
    build_exaqc: Callable[[], EXAQC],
    run_for: int,
    timing_report_every: int = 1000,
) -> None:
    """Runs an EXAQC search serially or across MPI ranks, as available.

    The execution mode is chosen from ``MPI.COMM_WORLD``'s size so the same
    entry point works with or without ``mpiexec``:

    * **1 process** (run without ``mpiexec``, or a single rank): builds the
      search and runs it in-process via :meth:`~src.evolution.exaqc.EXAQC.run_for`,
      since there are no worker ranks to distribute genomes to.
    * **more than 1 process**: rank 0 is the master that generates genomes and
      owns the population; every other rank is a worker that evaluates genomes
      with ``objective``.

    Only the serial run and the master (rank 0) build the ``EXAQC`` object, so
    worker ranks never construct the search machinery (gate set, encoder/decoder,
    population). ``build_exaqc`` is therefore a deferred factory that is invoked
    only on those ranks; the ``EXAQC`` it returns must wrap the same
    ``objective`` passed here.

    Args:
        objective: The objective used to evaluate genomes; needed by every worker
            rank (and by the ``EXAQC`` built for the serial/master run).
        build_exaqc: A zero-argument factory returning the fully-configured
            :class:`~src.evolution.exaqc.EXAQC` search. Called only on the serial
            run and the MPI master, never on workers.
        run_for: How many genomes to generate and evaluate before stopping.
        timing_report_every: How many insertions each of the MPI master's
            periodic timing reports covers (0 keeps only its final report).
            Unused by a serial run, which has no workers to wait on it.

    Returns:
        None. Runs the search to completion on this rank; the serial run and the
        master close the search's output archive when it ends, even on error.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1 and rank != 0:
        # worker rank: only the objective is needed to evaluate genomes
        worker(comm=comm, rank=rank, objective=objective)
        return

    # serial run or MPI master: build the search machinery here (and only here)
    exaqc = build_exaqc()

    try:
        if size == 1:
            # no worker ranks to distribute to, so run the search in-process
            exaqc.run_for(run_for)
        else:
            master(
                comm=comm,
                rank=rank,
                exaqc=exaqc,
                run_for=run_for,
                timing_report_every=timing_report_every,
            )
    finally:
        exaqc.close()
