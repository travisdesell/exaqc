"""A minimal search driver, run under ``mpiexec`` by the MPI restart test.

This mirrors what a search entry point does -- decide whether to continue the run
in ``--out_dir``, build (or restore) the population, run the search across the
master and worker ranks -- with an objective that scores a genome by its number
instead of training it, so a restart can be exercised across processes in
seconds.

It is deliberately not named ``test_*``: pytest must not collect it, because it
is a program the test runs, not a test itself.
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import initialize_decoder
from src.circuits.encoder import initialize_encoder
from src.circuits.gate_specifications import GateSpecifications
from src.utils import restart
from src.evolution.exaqc import EXAQC
from src.evolution.master_worker import run_evolution
from src.evolution.population_strategy import PopulationStrategy
from src.utils.genome_archive import GenomeArchive


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> float:
    """Orders genomes by ``fitness["loss"]``, lower first.

    Args:
        genome1: The first genome.
        genome2: The second genome.

    Returns:
        Negative when ``genome1`` is better, positive when ``genome2`` is.
    """

    return genome1.fitness["loss"] - genome2.fitness["loss"]


def objective(genome: CircuitGenome) -> None:
    """Scores a genome by its number, so the search runs without training.

    Args:
        genome: The genome to score.

    Returns:
        None. Sets the genome's fitness.
    """

    genome.fitness = {
        "loss": float(genome.genome_number),
        "target_metric": -float(genome.genome_number),
    }


def build_parser() -> argparse.ArgumentParser:
    """Builds the driver's parser, with the same run-output flags a search has.

    Returns:
        The parser, carrying ``--out_dir``, ``--number_genomes``, the restart
        flags and the population sub-command.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    EXAQC.initialize_parser(parser)
    GateSpecifications.initialize_parser(parser)
    GenomeArchive.initialize_parser(parser)
    PopulationStrategy.initialize_parser(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Runs (or continues) a small search across whatever ranks are available.

    Args:
        argv: The arguments to parse; the command line when omitted.

    Returns:
        None. Writes the run's archive under ``--out_dir``.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    # every rank writes to the same captured pipe, and the search logs a line per
    # gate, so only warnings are kept
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    def fail(message: str) -> NoReturn:
        """Reports why the run cannot start, the way a parser does.

        Args:
            message: The explanation.

        Raises:
            SystemExit: Always.
        """

        parser.error(message)

    args, restart_state, run_for = restart.prepare(args, fail)

    def build_exaqc() -> EXAQC:
        """Builds the search on the serial run or the MPI master.

        Returns:
            The search, restored when it continues an existing run.
        """

        population = (
            PopulationStrategy.from_args(args, compare)
            if restart_state is None
            else restart.restored_strategy(restart_state, args, compare)
        )

        search = EXAQC(
            gate_specifications=GateSpecifications.from_args(args),
            population=population,
            archive=GenomeArchive.from_args(args, restarting=restart_state is not None),
            objective=objective,
            initial_encoder=initialize_encoder(
                target=args.target,
                encoding_str="linear",
                n_inputs=4,
                n_outputs=2,
                quantum_input_mode="ry",
                n_input_qubits=2,
            ),
            initial_decoder=initialize_decoder(
                target=args.target, decoding_str="linear", n_inputs=4, n_outputs=2
            ),
            hyperparameters={
                "quantum_input_mode": "ry",
                "quantum_output_mode": "probs",
                "epochs": 1,
                "learning_rate": 0.01,
            },
            mutation_strategy=args.mutation_strategy,
            parent_strategy=args.parent_strategy,
            input_registers={"input": 2},
            output_registers={"input": 2},
            task="classification",
            task_target="iris",
            restarting=restart_state is not None,
        )

        if restart_state is not None:
            restart.resume(search, restart_state, args)

        return search

    run_evolution(objective=objective, build_exaqc=build_exaqc, run_for=run_for)


if __name__ == "__main__":
    main(sys.argv[1:])
