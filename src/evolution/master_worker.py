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


def master_worker(exaqc: EXAQC, run_for: int) -> None:
    """Runs an MPI master/worker EXAQC search over a pre-built ``EXAQC``.

    Rank 0 acts as the master: it uses ``exaqc`` to generate genomes and owns
    the population. Every other rank is a worker that repeatedly requests a
    genome, evaluates it with ``exaqc.objective``, and returns it. The search
    stops once ``run_for`` genomes have been evaluated.

    All search configuration -- the allowed gate set, initial encoder/decoder,
    population strategy, mutation/parent strategies and crossover rates,
    registers, backend target and task metadata -- lives on the passed-in
    ``exaqc``; see :class:`~src.evolution.exaqc.EXAQC` for those parameters.

    Args:
        exaqc: A fully-constructed :class:`~src.evolution.exaqc.EXAQC` search.
            It is built identically on every rank; only rank 0 drives it as the
            master, while workers use its ``objective`` to evaluate genomes.
        run_for: How many genomes to generate and evaluate before stopping.

    Returns:
        None. Runs the search to completion (all ranks return when it ends).
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master(comm=comm, rank=rank, exaqc=exaqc, run_for=run_for)

    else:
        worker(comm=comm, rank=rank, objective=exaqc.objective)
