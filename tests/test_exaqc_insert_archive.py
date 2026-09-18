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
import platform  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import MagicMock, call, patch  # noqa: E402

import pytest  # noqa: E402

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.circuits.decoder import initialize_decoder  # noqa: E402
from src.circuits.encoder import initialize_encoder  # noqa: E402
from src.circuits.pennylane_gate_specifications import (  # noqa: E402
    pennylane_gate_specifications,
)
from src.evolution import master_worker  # noqa: E402
from src.evolution.exaqc import EXAQC, MUTATION_WEIGHTS  # noqa: E402
from src.evolution.island import Island  # noqa: E402
from src.evolution.objective import evaluate_genome  # noqa: E402
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

    # how operators were drawn, so observed operator rates can be compared
    # against what the search was configured to do
    selection = info["operator_selection"]
    assert selection["mutation_weights"] == MUTATION_WEIGHTS
    assert set(selection["crossover_rates"]) == {
        "binary_crossover",
        "n_ary_crossover",
        "exponential_crossover",
    }
    assert selection["mutation_strategy"] == ["uniform", "1", "2"]
    assert selection["parent_strategy"] == ["uniform", "2", "3"]

    # a single population has no islands to describe
    assert "island_topology" not in info


def test_island_topology_is_recorded_when_the_search_starts() -> None:
    """An island search records which islands each island draws parents from."""

    archive = MagicMock()
    population = SteadyStateIslands(
        n_islands=3, max_island_size=2, compare=compare, topology=["ring"]
    )
    build_search(population, archive)

    info = archive.set_run_info.call_args.kwargs
    assert info["island_topology"] == {
        "topology": ["ring"],
        "neighbors": [[1, 2], [0, 2], [1, 0]],
    }


def test_mutations_are_drawn_from_the_weighted_list() -> None:
    """mutate draws from the weights expanded in their fixed order.

    The expansion must match the list the search always drew from, element for
    element, so recording the weights did not change which mutation a seeded
    search picks.
    """

    search = build_search(
        SteadyStatePopulation(max_population_size=4, compare=compare), MagicMock()
    )
    expected = (
        ["add_gate"] * 11
        + ["reorder_gate"] * 2
        + ["qubit_swap"] * 2
        + ["enable_gate"]
        + ["disable_gate"] * 2
        + ["clone"] * 2
        + ["mutate_some_weights"] * 2
        + ["mutate_all_weights"] * 2
    )
    drawn_from: list[list[str]] = []

    def choose(options: list[str]) -> str:
        """Records the options offered and picks clone, which always succeeds."""
        drawn_from.append(list(options))
        return "clone"

    with patch("src.evolution.exaqc.random.choice", side_effect=choose):
        child = search.mutate(search.initial_genome, {}, n_mutations=2)

    assert drawn_from == [expected, expected]
    assert child.metadata["generated_by"] == ["clone", "clone"]


def test_inserted_genomes_are_archived_in_insertion_order() -> None:
    """Every recorded genome is archived, with a population delta after each."""

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
        recorded.kwargs["step"] for recorded in archive.record_population.call_args_list
    ] == [1, 2]
    snapshot = archive.record_population.call_args_list[-1].kwargs["population"]
    assert [genome.genome_number for genome in snapshot] == [2, 1]


def test_best_files_are_rewritten_only_when_a_best_changes() -> None:
    """Fitness and target_metric bests are tracked independently."""

    archive = MagicMock()
    search = build_search(
        SteadyStatePopulation(max_population_size=2, compare=compare), archive
    )

    search.insert_genome(FakeGenome(1, loss=1.0, target_metric=0.5))
    assert best_writes(archive) == [(1, "fitness"), (1, "target_metric")]

    archive.reset_mock()
    search.insert_genome(FakeGenome(2, loss=2.0, target_metric=0.4))
    assert best_writes(archive) == []

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


def test_duplicates_of_better_genomes_are_archived_as_discarded() -> None:
    """A duplicate of a better genome is recorded, with why and to which genome it lost."""

    archive = MagicMock()
    population = SteadyStatePopulation(max_population_size=4, compare=compare)
    search = build_search(population, archive)

    search.insert_genome(FakeGenome(1, loss=0.5, target_metric=0.5, gates=(7,)))
    archive.reset_mock()

    duplicate = FakeGenome(2, loss=0.9, target_metric=0.4, gates=(7,))
    search.insert_genome(duplicate)

    archive.add_genome.assert_called_once()
    assert duplicate.metadata["insert_type"] == "discarded"
    assert duplicate.metadata["discard_reason"] == "duplicate_of_better"
    assert duplicate.metadata["lost_to"] == 1
    assert search.inserted_genomes == 2
    assert search.target_metric_best_genome.genome_number == 1
    assert [genome.genome_number for genome in population.get_population()] == [1]


def test_a_genome_worse_than_a_full_population_records_the_genome_it_failed_to_beat() -> (
    None
):
    """The discarded genome names the worst genome the population kept."""

    population = SteadyStatePopulation(max_population_size=2, compare=compare)
    search = build_search(population, MagicMock())
    for number, loss in ((1, 0.1), (2, 0.2)):
        search.insert_genome(FakeGenome(number, loss=loss, target_metric=0.0))

    worse = FakeGenome(3, loss=0.9, target_metric=0.0)
    search.insert_genome(worse)

    assert worse.metadata["insert_type"] == "discarded"
    assert worse.metadata["discard_reason"] == "worse_than_population"
    assert worse.metadata["lost_to"] == 2


def test_island_discards_record_their_reason() -> None:
    """An island records each way it discards a genome."""

    island = Island(max_size=1, id=0, compare=compare)
    island.insert_genome(FakeGenome(1, loss=0.1, target_metric=0.0, gates=(5,)))

    worse = FakeGenome(2, loss=0.5, target_metric=0.0)
    island.insert_genome(worse)
    duplicate = FakeGenome(3, loss=0.9, target_metric=0.0, gates=(5,))
    island.insert_genome(duplicate)

    # a genome generated before its island was repopulated is discarded however good it is
    island.repopulate(repopulation_genome_number=10)
    stale = FakeGenome(4, loss=0.01, target_metric=0.0)
    island.insert_genome(stale)

    assert (worse.metadata["discard_reason"], worse.metadata["lost_to"]) == (
        "worse_than_population",
        1,
    )
    assert (duplicate.metadata["discard_reason"], duplicate.metadata["lost_to"]) == (
        "duplicate_of_better",
        1,
    )
    assert stale.metadata["discard_reason"] == "generated_before_repopulation"
    assert "lost_to" not in stale.metadata


def test_evaluation_records_its_timing_and_placement() -> None:
    """Evaluating a genome records how long it took and which process ran it."""

    genome = FakeGenome(1, loss=0.0, target_metric=0.0)
    genome.metadata["timing"] = {"generated_at": 1.0}

    evaluate_genome(lambda evaluated: time.sleep(0.01), genome, rank=3)

    timing = genome.metadata["timing"]
    assert timing["generated_at"] == 1.0
    assert timing["evaluation_started_at"] <= timing["evaluation_finished_at"]
    assert timing["evaluation_seconds"] >= 0.01
    assert genome.metadata["evaluated_by"] == {
        "rank": 3,
        "host": platform.node(),
        "pid": os.getpid(),
    }


def test_generated_genomes_record_when_and_under_what_island_status_they_were_bred() -> (
    None
):
    """Every genome records the insertion it was created at and its full timing."""

    archive = MagicMock()
    population = SteadyStateIslands(n_islands=2, max_island_size=2, compare=compare)
    search = build_search(population, archive)

    def objective(genome: CircuitGenome) -> None:
        """Assigns a fitness without training, so the search runs quickly."""
        genome.fitness = {"loss": float(genome.genome_number), "target_metric": 0.0}

    search.objective = objective
    search.run_for(10)

    for recorded in archive.add_genome.call_args_list:
        genome, insertion = recorded.args[0], recorded.kwargs["insertion"]
        metadata = genome.metadata
        # evaluated serially, so each genome is inserted right after it is generated
        assert metadata["generated_at_insertion"] == insertion - 1
        timing = metadata["timing"]
        assert (
            timing["generated_at"]
            <= timing["evaluation_started_at"]
            <= timing["evaluation_finished_at"]
            <= timing["inserted_at"]
        )
        assert metadata["evaluated_by"]["rank"] == 0
        # initial genomes are mutated from the seed, not generated for an island
        if metadata["parent_genomes"] != [1]:
            assert metadata["target_island_status"] in ("full", "repopulating")


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

    # the population snapshot merges the islands into one ranking, best first
    snapshot = archive.record_population.call_args_list[-1].kwargs["population"]
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
    }

    with GenomeArchive.open_readonly(str(run_dir)) as reader:
        stored = dict(reader.iter_genome_dicts())
        assert reader.count() == len(stored) >= 1
        info = reader.run_info()
        assert info["task_target"] == "iris"
        # the operator selection survives the archive's JSON round trip intact
        assert info["operator_selection"]["mutation_weights"] == MUTATION_WEIGHTS
        # when and where each genome was created and evaluated is queryable
        rows = reader.connection.execute(
            "SELECT insertion, generated_at_insertion, evaluation_seconds, "
            "evaluated_host, evaluated_rank FROM genomes"
        ).fetchall()
        assert rows
        for insertion, generated_at, seconds, host, rank in rows:
            assert 0 <= generated_at < insertion
            assert seconds >= 0
            assert host == platform.node()
            assert rank == 0

    best_number = min(stored)
    restored = CircuitGenome.from_dict(stored[best_number])
    assert restored.task == "classification"
    assert (run_dir / "best_fitness.json").read_text().count(
        f'"genome_number": {best_number}'
    ) == 1
