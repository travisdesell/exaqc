from loguru import logger

from mpi4py import MPI
from mpi4py.MPI import Intracomm

from src.circuits.circuit import CircuitGenome
from src.circuits.gate_specifications import GateSpecifications
from src.evolution.exaqc import EXAQC
from src.evolution.objective import Objective
from src.evolution.population_strategy import PopulationStrategy

from typing import Callable

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
    comm: Intracomm, rank: int, objective: Objective,
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


def master_worker(
    gate_specifications: GateSpecifications,
    population: PopulationStrategy,
    objective: Objective,
    hyperparameters: dict[str, any],
    run_for: int,
    input_qubits: list[tuple[str, int]] = None,
    input_registers: dict[str, int] = None,
    output_registers: dict[str, int] = None,
    output_qubits: list[tuple[str, int]] = None,
    target: str = "pennylane",
):
    """
    Creates an instance of Evolutionary Exploration of Augmenting Quantum Circuits given a
    particular population strategy, allowing the given gates (if specified), and uses the main process
    as the master in the master work strategy. Workers will asynchronously get new tasks (genomes)
    to evaluate and send the results back to the master process.

    args:
        gate_specifications: is an object containing the allowed gates specifications for the search
            process, for either the pennylane or qiskit frameworks.
        population: is an instance of a subclass of the PopulationStrategy interface, utilized to get
            parents for mutation or crossover and insert children back into the population.
        objective: an instantiated Objective which can be called with a CircuitGenome as an argument
            to be trained and have its fitness evaluated.
        hyperparameters: a dict specifying which hyperparameters to use in the training process, and if
            this is an additional search space to search over.
        run_for: how many genomes to generate in the search process.
        input_registers: a dict of register names and sizes (the key is the qubit name, the value is its size). must
            be specified if input_qubits is not specified.
        input_qubits: a list of qubit tuples (name, register_index) which would be the expanded form of the
            input_registers. Must be specified if input_registers is not specified.
        output_registers: a dict of register names and sizes (the key is the qubit name, the value is its
            size). must be specified if output_qubits is not specified. If output_registers and output_qubits
            are None, they are set to the input registers/qubits.
        output_qubits: a list of qubit tuples (name, register_index) which would be the expanded form of the
            output_registers. Must be specified if output_registers is not specified. If output_registers
            and output_qubits are None, they are set to the input_registers/qubits.
        target: qiskit or pennylane
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        exaqc = EXAQC(
            gate_specifications=gate_specifications,
            population=population,
            objective=objective,
            hyperparameters=hyperparameters,
            input_registers=input_registers,
            input_qubits=input_qubits,
            output_registers=output_registers,
            output_qubits=output_qubits,
            target=target,
        )

        master(comm=comm, rank=rank, exaqc=exaqc, run_for=run_for)

    else:
        worker(comm=comm, rank=rank, objective=objective)
