from typing import Callable

from loguru import logger

from mpi4py import MPI
from mpi4py.MPI import Intracomm

from src.circuits.circuit import CircuitGenome
from src.evolution.exaqc import EXAQC
from src.evolution.objective import Objective

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


def master(comm: Intracomm, rank: int, exaqc: EXAQC, run_for: int):
    """
    The master process which will generate genomes and receive results from workers
    to perform the EXAQC search.

    Args:
        comm: is the MPI COMM WORLD
        rank: is the rank of the master process (should be 0)
        exaqc: is an initialized EXAQC search object
        run_for: is how many genomes to generate
    """
    n_workers = comm.Get_size() - 1

    evaluated_genomes = 0

    while evaluated_genomes < run_for:
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)

        tag_id = status.Get_tag()
        source = status.Get_source()

        logger.debug(
            f"master process received tag {tags[tag_id]} from source {source} and data: {data}"
        )

        if tag_id == tag_ids["genome_request"]:
            genome = exaqc.generate_genome()
            comm.send(genome.to_dict(), dest=source, tag=tag_ids["genome"])

        elif tag_id == tag_ids["genome_response"]:
            genome = CircuitGenome.from_dict(data)
            exaqc.insert_genome(genome)

            evaluated_genomes += 1
            logger.info(f"evaluated {evaluated_genomes} of max {run_for} genomes")

    # receive last genomes and cleanup
    finished_workers = 0
    while finished_workers < n_workers:
        status = MPI.Status()
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)

        tag_id = status.Get_tag()
        source = status.Get_source()

        logger.debug(
            f"master process received tag {tags[tag_id]} from source {source} and data: {data}"
        )

        if tag_id == tag_ids["genome_request"]:
            comm.send(None, dest=source, tag=tag_ids["genome_response"])
            finished_workers += 1

        elif tag_id == tag_ids["genome_response"]:
            genome = CircuitGenome.from_dict(data)
            exaqc.insert_genome(genome)


def worker(
    comm: Intracomm,
    rank: int,
    objective: Objective,
):
    """
    This is a worker process which will repeatedly request new genomes
    from the master process, evaluate them with the objective function and
    send the genome back to the master process to be inserted into the search.

    Args:
        comm: is the MPI COMM WORLD
        rank: is the rank of the master process (should be 0)
        objective: is the objective function used to evaluate genomes
    """

    while True:
        # request a genome from the main process
        comm.send(None, dest=0, tag=tag_ids["genome_request"])

        status = MPI.Status()
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag_id = status.Get_tag()

        logger.debug(
            f"worker process {rank} received tag: {tags[tag_id]}, received data: {data}"
        )
        if data is None:
            # the search is over, the worker can quit
            break

        genome = CircuitGenome.from_dict(data)

        objective(genome)

        comm.send(genome.to_dict(), dest=0, tag=tag_ids["genome_response"])


def run_evolution(
    objective: Objective,
    build_exaqc: Callable[[], EXAQC],
    run_for: int,
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

    Returns:
        None. Runs the search to completion on this rank.
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

    if size == 1:
        # no worker ranks to distribute to, so run the search in-process
        exaqc.run_for(run_for)
    else:
        master(comm=comm, rank=rank, exaqc=exaqc, run_for=run_for)
