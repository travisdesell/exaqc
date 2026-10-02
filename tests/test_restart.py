"""Tests for restarting a stopped search from its archive.

``src.utils.restart`` reads back what a stopped, canceled or crashed run
left in its ``genomes.sqlar``: the arguments it was started with, the genomes its
population held, the counters it stopped at, and the highest gate innovation
number it handed out. These tests run small real searches, stop them, restart
them from the archive, and check that the restarted search continues the run
rather than colliding with it -- genome numbers and innovation numbers carry on,
the population comes back, and islands keep the connections the run actually
used (which matters for a ``random`` topology, drawn afresh every time the
islands are built).
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend before anything imports pyplot, so
# writing a best-genome image stays headless.
import matplotlib

matplotlib.use("Agg")

from argparse import Namespace  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.circuits.decoder import initialize_decoder  # noqa: E402
from src.circuits.encoder import initialize_encoder  # noqa: E402
from src.circuits.pennylane_gate_specifications import (  # noqa: E402
    pennylane_gate_specifications,
)
from src.utils import restart  # noqa: E402
from src.evolution.exaqc import EXAQC  # noqa: E402
from src.evolution.innovation import innovation_number_generator  # noqa: E402
from src.evolution.population_strategy import PopulationStrategy  # noqa: E402
from src.evolution.steady_state_islands import SteadyStateIslands  # noqa: E402
from src.evolution.steady_state_population import SteadyStatePopulation  # noqa: E402
from src.utils.genome_archive import GenomeArchive  # noqa: E402

#: The arguments a steady-state run records, as an entry point would.
STEADY_STATE_ARGUMENTS: dict[str, Any] = {
    "population_strategy": "steady_state",
    "max_population_size": 4,
    "number_genomes": 6,
    "out_dir": "recorded/by/the/run",
}

#: The arguments an island run records; the random topology is drawn when the
#: islands are built, so rebuilding from these alone gives different neighbors.
ISLAND_ARGUMENTS: dict[str, Any] = {
    "population_strategy": "islands",
    "n_islands": 4,
    "max_island_size": 2,
    "genomes_before_extinction": 100,
    "genomes_for_next_extinction": 200,
    "islands_to_extinct": 1,
    "primary_parent": "best",
    "intra_island_crossover_rate": 0.5,
    "topology": ["random", "1", "2"],
    "number_genomes": 8,
}


def compare(genome1: Any, genome2: Any) -> float:
    """Orders genomes by ``fitness["loss"]``, lower first.

    Args:
        genome1: The first genome.
        genome2: The second genome.

    Returns:
        Negative when ``genome1`` is better, positive when ``genome2`` is.
    """

    return genome1.fitness["loss"] - genome2.fitness["loss"]


def objective(genome: CircuitGenome) -> None:
    """Scores a genome by its number, so a search runs without training anything.

    Args:
        genome: The genome to score.

    Returns:
        None. Sets the genome's fitness.
    """

    genome.fitness = {
        "loss": float(genome.genome_number),
        "target_metric": -float(genome.genome_number),
    }


def build_search(
    population: PopulationStrategy,
    archive: GenomeArchive | None,
    restarting: bool = False,
) -> EXAQC:
    """Builds a small classification-shaped search around a population strategy.

    Args:
        population: The population strategy to search with.
        archive: The archive to record genomes in.
        restarting: Whether this search continues a run the archive holds.

    Returns:
        The configured search, scoring genomes with :func:`objective`.
    """

    search = EXAQC(
        gate_specifications=pennylane_gate_specifications,
        population=population,
        objective=objective,
        initial_encoder=initialize_encoder(
            target="pennylane",
            encoding_str="linear",
            n_inputs=4,
            n_outputs=2,
            quantum_input_mode="ry",
            n_input_qubits=2,
        ),
        initial_decoder=initialize_decoder(
            target="pennylane", decoding_str="linear", n_inputs=4, n_outputs=2
        ),
        hyperparameters={
            "quantum_input_mode": "ry",
            "quantum_output_mode": "probs",
            "epochs": 1,
            "learning_rate": 0.01,
        },
        mutation_strategy=["uniform", "1", "2"],
        parent_strategy=["uniform", "2", "3"],
        input_registers={"input": 2},
        output_registers={"input": 2},
        task="classification",
        task_target="iris",
        archive=archive,
        restarting=restarting,
    )
    search.objective = objective
    return search


def run_a_search(
    run_dir: Any,
    population: PopulationStrategy,
    arguments: dict[str, Any],
    genomes: int,
) -> None:
    """Runs a search into a new archive and closes it, as a stopped run leaves it.

    Args:
        run_dir: The run's output directory.
        population: The population strategy to search with.
        arguments: The arguments to record with the run.
        genomes: How many genomes to evaluate before stopping.

    Returns:
        None. Leaves a complete archive in ``run_dir``.
    """

    archive = GenomeArchive.create(str(run_dir), arguments=arguments)
    try:
        build_search(population, archive).run_for(genomes)
    finally:
        archive.close()


def restored_population(state: restart.RestartState) -> PopulationStrategy:
    """Rebuilds a population strategy from a run's arguments and restores it.

    Args:
        state: The stopped run's state.

    Returns:
        The strategy, holding what the run held when it stopped.
    """

    population = PopulationStrategy.from_args(
        restart.restart_arguments(state, Namespace()), compare
    )
    population.restore(state)
    return population


def test_a_stopped_steady_state_run_continues_where_it_left_off(tmp_path) -> None:
    """A restarted search keeps the population and carries on numbering genomes.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "run"
    run_a_search(
        run_dir, SteadyStatePopulation(4, compare), STEADY_STATE_ARGUMENTS, genomes=6
    )

    with GenomeArchive.open_readonly(str(run_dir)) as reader:
        stopped_at = reader.population_at()
        highest = reader.max_genome_number()

    state = restart.load(str(run_dir))
    assert state.inserted_genomes == 6
    assert state.next_genome_number == highest + 1
    assert state.arguments == STEADY_STATE_ARGUMENTS

    population = restored_population(state)
    assert sorted(genome.genome_number for genome in population.get_population()) == (
        stopped_at
    )

    archive = GenomeArchive.create(str(run_dir), arguments=STEADY_STATE_ARGUMENTS)
    try:
        search = build_search(population, archive, restarting=True)
        restart.resume(search, state)

        # the generator resumes past every number the stopped run handed out, so
        # no gate the restarted run creates can reuse one for a different gate
        assert (
            innovation_number_generator.current_innovation_number
            >= state.max_innovation_number
        )

        child = search.generate_genome()
        assert child.genome_number == state.next_genome_number
        inherited = {
            gate.innovation_number
            for parent in state.population
            for gate in parent.gates
        }
        assert all(
            gate.innovation_number > state.max_innovation_number
            for gate in child.gates
            if gate.innovation_number not in inherited
        )

        search.run_for(2)
        assert search.inserted_genomes == 8
    finally:
        archive.close()

    with GenomeArchive.open_readonly(str(run_dir)) as reader:
        # the restarted run appends: its insertions continue, and the population
        # it records is the one the restored search holds
        assert (
            reader.connection.execute("SELECT MAX(insertion) FROM genomes").fetchone()[
                0
            ]
            == 8
        )
        assert sorted(reader.population_at()) == sorted(
            genome.genome_number for genome in population.get_population()
        )


def test_restarted_islands_keep_the_connections_the_run_used(tmp_path) -> None:
    """Islands come back with their genomes, statuses and recorded neighbors.

    A ``random`` topology is drawn when the islands are built, so rebuilding the
    strategy from the run's arguments alone would give a different graph; the
    restart re-applies what the run recorded.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "islands"
    population = SteadyStateIslands(
        n_islands=4,
        max_island_size=2,
        compare=compare,
        topology=["random", "1", "2"],
    )
    run_a_search(run_dir, population, ISLAND_ARGUMENTS, genomes=8)

    state = restart.load(str(run_dir))
    recorded = state.island_neighbors
    assert recorded is not None

    restored = restored_population(state)
    assert [
        [neighbor.id for neighbor in island.neighbors] for island in restored.islands
    ] == recorded

    with GenomeArchive.open_readonly(str(run_dir)) as reader:
        islands_of = {
            number: island
            for number, island in reader.connection.execute(
                "SELECT genome_number, island FROM genomes"
            )
        }

    for island in restored.islands:
        held = [genome.genome_number for genome in island.population]
        assert all(islands_of[number] == island.id for number in held)
        if not held:
            assert island.status == "initializing"
        elif len(held) >= island.max_size:
            assert island.status == "full"
        else:
            # an island that is not full fills from its best neighbor, and must
            # not discard the genomes the restarted run generates
            assert island.status == "repopulating"
            assert island.repopulation_genome_number == state.next_genome_number

    assert restored.insertions == state.inserted_genomes
    assert restored.global_best_genome.genome_number == state.best_genome.genome_number


def test_an_archive_that_predates_restarts_says_so(tmp_path) -> None:
    """A run that recorded no arguments cannot be restarted, and explains why.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "older"
    archive = GenomeArchive.create(str(run_dir))
    try:
        build_search(SteadyStatePopulation(4, compare), archive).run_for(2)
    finally:
        archive.close()

    with pytest.raises(ValueError, match="arguments"):
        restart.load(str(run_dir))


def test_restarting_needs_an_archive_holding_genomes(tmp_path) -> None:
    """Restarting reports a missing archive, and an archive with nothing in it.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    assert restart.archive_exists(str(tmp_path / "nothing")) is False
    with pytest.raises(FileNotFoundError):
        restart.load(str(tmp_path / "nothing"))

    empty = tmp_path / "empty"
    GenomeArchive.create(str(empty), arguments=STEADY_STATE_ARGUMENTS).close()

    assert restart.archive_exists(str(empty)) is True
    with pytest.raises(ValueError, match="no evaluated genomes"):
        restart.load(str(empty))


def test_a_restart_takes_its_configuration_from_the_run() -> None:
    """Only the overridable arguments come from the restart's own command line."""

    state = restart.RestartState(
        archive_path="run/genomes.sqlar",
        arguments={"dataset": "iris", "epochs": 3, "number_genomes": 10},
        run_info={},
    )
    given = Namespace(number_genomes=50, epochs=9, device="cuda")

    arguments = restart.restart_arguments(state, given)
    # the run's own configuration wins; how many genomes to evaluate does not
    assert arguments.epochs == 3
    assert arguments.dataset == "iris"
    assert arguments.number_genomes == 50
    # an argument the run never recorded keeps the value it was given
    assert arguments.device == "cuda"

    assert restart.differing_arguments(state, given) == {"epochs": (3, 9)}
    assert restart.differing_arguments(state, Namespace(number_genomes=50)) == {}


def test_a_restart_records_itself_without_disturbing_the_run(tmp_path) -> None:
    """Restarts are appended, leaving the original run's own record intact.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "run"
    run_a_search(
        run_dir, SteadyStatePopulation(4, compare), STEADY_STATE_ARGUMENTS, genomes=4
    )

    with GenomeArchive.open_readonly(str(run_dir)) as reader:
        original = reader.run_info()

    state = restart.load(str(run_dir))
    archive = GenomeArchive.create(str(run_dir), arguments=STEADY_STATE_ARGUMENTS)
    try:
        restart.record_restart(archive, state, Namespace(number_genomes=99))
        restart.record_restart(archive, state, Namespace(number_genomes=150))
        recorded = archive.run_info()
    finally:
        archive.close()

    assert recorded["command_line"] == original["command_line"]
    assert recorded["start_time"] == original["start_time"]
    assert recorded["arguments"] == original["arguments"]

    restarts = recorded["restarts"]
    assert [entry["number_genomes"] for entry in restarts] == [99, 150]
    assert all(entry["from_insertion"] == state.inserted_genomes for entry in restarts)
