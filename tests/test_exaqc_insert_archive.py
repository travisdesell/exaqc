"""Tests for how an EXAQC search records the genomes it inserts.

``EXAQC.insert_genome`` is the single insertion path for serial and MPI runs,
and the one place a search writes to disk: every recorded genome goes into the
run's :class:`~src.utils.genome_archive.GenomeArchive`, the current-best genome
files are rewritten only when a best changes, and the search history is
recorded. These tests pin that behavior with a mocked archive, and check a small
real search end to end.
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend before anything imports pyplot, so
# the end-to-end search can render its best-genome images headless.
import matplotlib

matplotlib.use("Agg")

import os  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import MagicMock, call  # noqa: E402

import pytest  # noqa: E402

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.circuits.decoder import initialize_decoder  # noqa: E402
from src.circuits.encoder import initialize_encoder  # noqa: E402
from src.circuits.pennylane_gate_specifications import (  # noqa: E402
    pennylane_gate_specifications,
)
from src.evolution import master_worker  # noqa: E402
from src.evolution.exaqc import EXAQC  # noqa: E402
from src.evolution.population_strategy import PopulationStrategy  # noqa: E402
from src.evolution.steady_state_islands import SteadyStateIslands  # noqa: E402
from src.evolution.steady_state_population import SteadyStatePopulation  # noqa: E402
from src.utils.genome_archive import ARCHIVE_FILENAME, GenomeArchive  # noqa: E402


class FakeGenome:
    """Stand-in genome carrying just what the population strategies read."""

    def __init__(
        self,
        genome_number: int,
        loss: float,
        target_metric: float,
        gates: tuple[int, ...] | None = None,
    ) -> None:
        """Creates the fake genome.

        Args:
            genome_number: The genome's number.
            loss: Its ``fitness["loss"]`` (lower is better).
            target_metric: Its ``fitness["target_metric"]`` (higher is better).
            gates: Innovation numbers of its enabled gates; genomes with equal
                gates are duplicates. Defaults to a set unique to this genome.
        """

        self.genome_number = genome_number
        self.fitness = {"loss": loss, "target_metric": target_metric}
        self.metadata: dict[str, Any] = {}
        self.gates = gates if gates is not None else (genome_number,)

    def has_same_gates(self, other: FakeGenome) -> bool:
        """Reports whether two fake genomes have the same enabled gates.

        Args:
            other: The genome to compare against.

        Returns:
            True if their gate innovation numbers match.
        """

        return self.gates == other.gates

    def get_gate_innovations(self) -> list[int]:
        """Returns the fake genome's gate innovation numbers.

        Returns:
            The innovation numbers.
        """

        return list(self.gates)


def compare(genome1: Any, genome2: Any) -> float:
    """Orders genomes by ``fitness["loss"]``, lower first.

    Args:
        genome1: The first genome.
        genome2: The second genome.

    Returns:
        Negative when ``genome1`` is better, positive when ``genome2`` is.
    """

    return genome1.fitness["loss"] - genome2.fitness["loss"]


def build_search(population: PopulationStrategy, archive: Any) -> EXAQC:
    """Builds a small classification-shaped EXAQC search.

    Args:
        population: The population strategy to search with.
        archive: The archive (or mock, or ``None``) to record genomes in.

    Returns:
        The configured search.
    """

    return EXAQC(
        gate_specifications=pennylane_gate_specifications,
        population=population,
        objective=lambda genome: None,
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
    )


def best_writes(archive: MagicMock) -> list[tuple[int, str]]:
    """Lists the current-best files a mocked archive was asked to write.

    Args:
        archive: The mocked archive.

    Returns:
        ``(genome_number, kind)`` for each ``write_current_best`` call, in order.
    """

    return [
        (recorded.args[0].genome_number, recorded.args[1])
        for recorded in archive.write_current_best.call_args_list
    ]


def test_run_info_is_recorded_when_the_search_starts() -> None:
    """The archive learns the task, target and strategy of the run."""

    archive = MagicMock()
    build_search(SteadyStatePopulation(max_population_size=4, compare=compare), archive)

    archive.set_run_info.assert_called_once()
    info = archive.set_run_info.call_args.kwargs
    assert info["task"] == "classification"
    assert info["task_target"] == "iris"
    assert info["target"] == "pennylane"
    assert info["population_strategy"] == "SteadyStatePopulation"
    assert "command_line" in info and "start_time" in info
    assert info["seed_genome_number"] == 1


def test_inserted_genomes_are_archived_in_insertion_order() -> None:
    """Every recorded genome is archived, with a history row after each."""

    archive = MagicMock()
    search = build_search(
        SteadyStatePopulation(max_population_size=4, compare=compare), archive
    )

    genomes = [
        FakeGenome(1, loss=0.5, target_metric=0.5),
        FakeGenome(2, loss=0.4, target_metric=0.6),
    ]
    for genome in genomes:
        search.insert_genome(genome)

    assert archive.add_genome.call_args_list == [
        call(genomes[0], insertion=1, island=None),
        call(genomes[1], insertion=2, island=None),
    ]
    assert [
        recorded.kwargs["step"] for recorded in archive.record_history.call_args_list
    ] == [1, 2]
    snapshot = archive.record_history.call_args_list[-1].kwargs["population"]
    assert [genome.genome_number for genome in snapshot] == [2, 1]


def test_best_files_are_rewritten_only_when_a_best_changes() -> None:
    """Fitness and target_metric bests are tracked independently."""

    archive = MagicMock()
    search = build_search(
        SteadyStatePopulation(max_population_size=2, compare=compare), archive
    )

    search.insert_genome(FakeGenome(1, loss=1.0, target_metric=0.5))
    assert best_writes(archive) == [(1, "fitness"), (1, "target_metric")]
    assert archive.plot_history.call_count == 1

    archive.reset_mock()
    search.insert_genome(FakeGenome(2, loss=2.0, target_metric=0.4))
    assert best_writes(archive) == []
    archive.plot_history.assert_not_called()

    archive.reset_mock()
    search.insert_genome(FakeGenome(3, loss=0.5, target_metric=0.3))
    assert best_writes(archive) == [(3, "fitness")]

    archive.reset_mock()
    # worse than the whole (full) population, so it is discarded at once, yet it
    # is still the best genome by target_metric and is still archived
    genome = FakeGenome(4, loss=3.0, target_metric=0.9)
    search.insert_genome(genome)
    assert genome.metadata["insert_type"] == "discarded"
    assert best_writes(archive) == [(4, "target_metric")]
    archive.add_genome.assert_called_once()
    assert search.target_metric_best_genome is genome


def test_rejected_duplicates_are_not_archived() -> None:
    """A duplicate of a better genome is counted but never recorded."""

    archive = MagicMock()
    population = SteadyStatePopulation(max_population_size=4, compare=compare)
    search = build_search(population, archive)

    search.insert_genome(FakeGenome(1, loss=0.5, target_metric=0.5, gates=(7,)))
    archive.reset_mock()

    search.insert_genome(FakeGenome(2, loss=0.9, target_metric=0.9, gates=(7,)))

    archive.add_genome.assert_not_called()
    archive.write_current_best.assert_not_called()
    archive.record_history.assert_not_called()
    assert search.inserted_genomes == 2
    assert search.target_metric_best_genome.genome_number == 1
    assert [genome.genome_number for genome in population.get_population()] == [1]


def test_island_genomes_are_archived_with_their_island() -> None:
    """Island strategies record which island each genome went into."""

    archive = MagicMock()
    population = SteadyStateIslands(n_islands=2, max_island_size=3, compare=compare)
    search = build_search(population, archive)

    genomes = [
        FakeGenome(number, loss=float(number), target_metric=0.0)
        for number in range(1, 5)
    ]
    for genome in genomes:
        search.insert_genome(genome)

    islands = [
        recorded.kwargs["island"] for recorded in archive.add_genome.call_args_list
    ]
    assert sorted(islands) == [0, 0, 1, 1]
    assert [genome.metadata["island_id"] for genome in genomes] == islands

    # the history snapshot merges the islands into one ranking, best first
    snapshot = archive.record_history.call_args_list[-1].kwargs["population"]
    assert [genome.genome_number for genome in snapshot] == [1, 2, 3, 4]


def test_get_population_returns_a_sorted_copy() -> None:
    """Callers can't disturb a population through its snapshot."""

    population = SteadyStatePopulation(max_population_size=4, compare=compare)
    for number, loss in ((1, 0.3), (2, 0.1), (3, 0.2)):
        population.insert_genome(FakeGenome(number, loss=loss, target_metric=0.0))

    snapshot = population.get_population()
    assert [genome.genome_number for genome in snapshot] == [2, 3, 1]

    snapshot.clear()
    assert len(population.population) == 3


def test_a_search_without_an_archive_writes_nothing(tmp_path, monkeypatch) -> None:
    """Without an archive, insertion still works and nothing reaches disk.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to ``chdir`` into ``tmp_path``.
    """

    monkeypatch.chdir(tmp_path)

    search = build_search(
        SteadyStatePopulation(max_population_size=4, compare=compare), archive=None
    )
    search.insert_genome(FakeGenome(1, loss=0.5, target_metric=0.5))
    search.close()

    assert list(tmp_path.iterdir()) == []
    assert search.target_metric_best_genome.genome_number == 1


def test_close_closes_the_archive() -> None:
    """Closing the search closes its archive."""

    archive = MagicMock()
    build_search(
        SteadyStatePopulation(max_population_size=4, compare=compare), archive
    ).close()

    archive.close.assert_called_once()


def test_run_evolution_closes_the_search_even_when_it_fails() -> None:
    """The archive is closed even if the search raises."""

    search = MagicMock()
    search.run_for.side_effect = RuntimeError("search failed")

    with pytest.raises(RuntimeError):
        master_worker.run_evolution(
            objective=MagicMock(), build_exaqc=lambda: search, run_for=3
        )

    search.close.assert_called_once()


def test_run_for_evaluates_exactly_the_requested_number_of_genomes() -> None:
    """A serial search evaluates ``number_genomes`` genomes; the seed doesn't count."""

    evaluated: list[int] = []

    def objective(genome: CircuitGenome) -> None:
        """Records which genome was evaluated and gives it a fitness."""
        evaluated.append(genome.genome_number)
        genome.fitness = {"loss": float(genome.genome_number), "target_metric": 0.0}

    search = build_search(
        SteadyStatePopulation(max_population_size=3, compare=compare), MagicMock()
    )
    search.objective = objective

    search.run_for(5)

    # genome 1 is the unevaluated seed, so the five evaluated genomes are 2-6
    assert evaluated == [2, 3, 4, 5, 6]
    assert search.inserted_genomes == 5


def test_a_real_search_writes_a_fixed_set_of_files(tmp_path) -> None:
    """A small search leaves one archive plus fixed-name files, however long it runs.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    def objective(genome: CircuitGenome) -> None:
        """Assigns a fitness that makes the first genome the best on both measures."""
        genome.fitness = {
            "loss": float(genome.genome_number),
            "target_metric": -float(genome.genome_number),
        }

    run_dir = tmp_path / "run"
    archive = GenomeArchive.create(str(run_dir))
    search = build_search(
        SteadyStatePopulation(max_population_size=3, compare=compare), archive
    )
    search.objective = objective

    try:
        search.run_for(6)
    finally:
        search.close()

    assert set(os.listdir(run_dir)) == {
        ARCHIVE_FILENAME,
        "best_fitness.json",
        "best_fitness.png",
        "best_target_metric.json",
        "best_target_metric.png",
        "exaqc_history.csv",
        "exaqc_curves.png",
    }

    with GenomeArchive.open_readonly(str(run_dir)) as reader:
        stored = dict(reader.iter_genome_dicts())
        assert reader.count() == len(stored) >= 1
        assert reader.run_info()["task_target"] == "iris"

    best_number = min(stored)
    restored = CircuitGenome.from_dict(stored[best_number])
    assert restored.task == "classification"
    assert (run_dir / "best_fitness.json").read_text().count(
        f'"genome_number": {best_number}'
    ) == 1
