"""Scenario tests for the train/validation/test split and the final test evaluation.

``tests/test_uci_split.py`` pins how a UCI dataset is divided and scaled. These
tests cover how the split is used: which test set a run loads, that scoring a
genome on it changes nothing, and that full searches -- steady-state or islands,
fixed or SHO-tuned hyperparameters, new or restarted -- record (or deliberately
skip) their test results in the archive.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.circuits.circuit import CircuitGenome
from src.datasets.classification_loaders import (
    DEFAULT_UCI_SPLIT,
    LEGACY_UCI_SPLIT,
    _split_uci_dataset,
    get_uci_dataloaders,
    get_uci_test_dataloader,
)
from src.evolution import master_worker
from src.evolution.exaqc import EXAQC
from src.examples import classification, refine_genome
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.objectives import classification_objective
from src.objectives.classification_objective import (
    ClassificationObjective,
    load_test_data,
)
from src.utils.genome_archive import GenomeArchive

#: A small, fast iris search; the population sub-command and its flags follow.
SEARCH_ARGUMENTS: list[str] = (
    "--dataset iris --input_qubits 4 --output_qubits 2 --batch_size 3 --epochs 1 "
    "-ms uniform 1 2 -ps uniform 2 2 --binary_crossover_rate 0.1 "
    "--n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 "
    "-qim amplitude -qom probs --encoding identity --decoding clipped "
    "--logging_level WARNING"
).split()

STEADY_STATE: list[str] = ["steady_state", "--max_population_size", "3"]
ISLANDS: list[str] = ["islands", "--n_islands", "2", "--max_island_size", "2"]


def data_arguments(**overrides: Any) -> SimpleNamespace:
    """Builds the dataset-loading arguments ``load_test_data`` reads.

    Args:
        **overrides: Values replacing the defaults below.

    Returns:
        A namespace shaped like the classification entry point's arguments.
    """

    values: dict[str, Any] = {
        "dataset": "iris",
        "normalization": "minmax",
        "uci_split": list(DEFAULT_UCI_SPLIT),
        "batch_size": 3,
        "validation_batch_size": None,
        "seed": 0,
        "data_dir": "data",
        "download_dataset": False,
        "num_workers": 0,
        "pin_memory": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def run_search(
    monkeypatch: pytest.MonkeyPatch,
    out_dir: Path,
    number_genomes: int,
    extra: list[str] | None = None,
    population: list[str] | None = None,
) -> None:
    """Runs the classification entry point in this process.

    Args:
        monkeypatch: Sets the command line.
        out_dir: Where the run writes.
        number_genomes: How many genomes the run should have evaluated in all.
        extra: Further search flags, before the population sub-command.
        population: The population sub-command and its flags.
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification",
            *SEARCH_ARGUMENTS,
            "--number_genomes",
            str(number_genomes),
            "--out_dir",
            str(out_dir),
            *(extra or []),
            *(population or STEADY_STATE),
        ],
    )
    classification.main()


def run_info(out_dir: Path) -> dict[str, Any]:
    """Reads a run's ``run_info``.

    Args:
        out_dir: The run's output directory.

    Returns:
        Every recorded value, keyed by name.
    """

    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        return reader.run_info()


def lowest_loss_genome(out_dir: Path) -> int:
    """Finds a run's genome with the lowest ``loss``.

    Args:
        out_dir: The run's output directory.

    Returns:
        That genome's number.
    """

    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        return int(reader.best_value("loss", higher_is_better=False)["genome_number"])


def stored_genome(out_dir: Path, genome_number: int) -> CircuitGenome:
    """Loads a genome as a run's archive stored it.

    Args:
        out_dir: The run's output directory.
        genome_number: The genome to load.

    Returns:
        The genome.
    """

    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        return CircuitGenome.from_dict(reader.get_genome_dict(genome_number))


def iris_objective(with_test: bool = True) -> ClassificationObjective:
    """Builds the objective the iris search above evaluates genomes with.

    Args:
        with_test: Whether to give it the default split's test set.

    Returns:
        The objective, on the CPU.
    """

    training, validation = get_uci_dataloaders("iris", batch_size=3, seed=0)
    test = get_uci_test_dataloader("iris", batch_size=3, seed=0) if with_test else None
    loss = torch.nn.CrossEntropyLoss(weight=training.label_weights)
    return ClassificationObjective(
        training_dataloader=training,
        validation_dataloader=validation,
        training_loss_function=loss,
        validation_loss_function=loss,
        metrics={"mean_class_accuracy": MeanClassAccuracy(training.n_labels)},
        device="cpu",
        testing_dataloader=test,
        testing_loss_function=(
            None
            if test is None
            else torch.nn.CrossEntropyLoss(weight=test.label_weights)
        ),
    )


def strip_split_from_archive(out_dir: Path) -> None:
    """Makes a run's archive look like one written before ``--uci_split`` existed.

    Args:
        out_dir: The run's output directory.

    Returns:
        None. Removes ``uci_split`` from the recorded arguments and drops the
        ``final_test_results``.
    """

    connection = sqlite3.connect(out_dir / "genomes.sqlar")
    arguments = json.loads(
        connection.execute(
            "SELECT value FROM run_info WHERE key = 'arguments'"
        ).fetchone()[0]
    )
    arguments.pop("uci_split")
    connection.execute(
        "UPDATE run_info SET value = ? WHERE key = 'arguments'",
        (json.dumps(arguments),),
    )
    connection.execute("DELETE FROM run_info WHERE key = 'final_test_results'")
    connection.commit()
    connection.close()


# --- which test set a run loads ----------------------------------------------


def test_uci_run_loads_the_test_fraction_of_its_split() -> None:
    """Checks a UCI run's test set is the held-out tenth of the dataset."""

    test = load_test_data(data_arguments())

    assert test is not None
    assert len(test.dataset) == 15
    assert not test.is_image


def test_uci_run_with_no_test_fraction_loads_no_test_set() -> None:
    """Checks a split holding nothing out gives no test set to score on."""

    assert load_test_data(data_arguments(uci_split=list(LEGACY_UCI_SPLIT))) is None


@pytest.mark.parametrize(
    ("validation_batch_size", "expected_batch_size"), [(None, 3), (16, 16)]
)
def test_image_run_loads_the_official_test_split(
    monkeypatch: pytest.MonkeyPatch,
    validation_batch_size: int | None,
    expected_batch_size: int,
) -> None:
    """Checks an image run asks for its official test split, batched for evaluation.

    Args:
        monkeypatch: Replaces the image loader so nothing is read from disk.
        validation_batch_size: The run's ``--validation_batch_size``.
        expected_batch_size: The test batch size it should lead to.
    """

    loader = MagicMock(name="test_loader")
    get_test = MagicMock(return_value=loader)
    monkeypatch.setattr(classification_objective, "get_image_test_dataloader", get_test)

    arguments = data_arguments(
        dataset="mnist",
        validation_batch_size=validation_batch_size,
        data_dir="somewhere",
        download_dataset=True,
    )

    assert load_test_data(arguments) is loader
    get_test.assert_called_once_with(
        "mnist",
        data_dir="somewhere",
        batch_size=expected_batch_size,
        download=True,
        num_workers=0,
        pin_memory=False,
    )


def test_each_seed_gives_its_own_reproducible_split() -> None:
    """Checks one seed always gives the same test set, and another seed a new one."""

    first = _split_uci_dataset("breast_cancer", "none", DEFAULT_UCI_SPLIT, 1)
    again = _split_uci_dataset("breast_cancer", "none", DEFAULT_UCI_SPLIT, 1)
    other = _split_uci_dataset("breast_cancer", "none", DEFAULT_UCI_SPLIT, 2)

    for name in ("training", "validation", "test"):
        np.testing.assert_array_equal(first[name][0], again[name][0])
    assert not np.array_equal(first["test"][0], other["test"][0])


def test_zscore_scales_held_out_samples_by_the_training_statistics() -> None:
    """Checks ``zscore`` takes its mean and spread from the training samples only."""

    raw = _split_uci_dataset("wine", "none", DEFAULT_UCI_SPLIT, 0)
    scaled = _split_uci_dataset("wine", "zscore", DEFAULT_UCI_SPLIT, 0)
    mean = raw["training"][0].mean(axis=0)
    std = raw["training"][0].std(axis=0)

    np.testing.assert_allclose(
        scaled["test"][0], (raw["test"][0] - mean) / (std + 1e-8), rtol=1e-4, atol=1e-4
    )


def test_refinement_uses_the_default_split() -> None:
    """Checks a refined genome trains on the split a new run would use."""

    defaults = refine_genome._CLASSIFICATION_DATA_DEFAULTS
    assert tuple(defaults["uci_split"]) == DEFAULT_UCI_SPLIT


# --- scoring a genome on the test set ----------------------------------------


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Runs one small steady-state search for the tests that only read it.

    Args:
        tmp_path_factory: Makes the run's directory.

    Returns:
        The run's output directory.
    """

    out_dir = tmp_path_factory.mktemp("finished") / "run"
    with pytest.MonkeyPatch.context() as monkeypatch:
        run_search(monkeypatch, out_dir, number_genomes=4)
    return out_dir


def test_objective_without_a_test_set_scores_nothing(finished_run: Path) -> None:
    """Checks an objective given no test set reports that rather than guessing.

    Args:
        finished_run: A finished run to take a trained genome from.
    """

    genome = stored_genome(finished_run, lowest_loss_genome(finished_run))

    assert iris_objective(with_test=False).test(genome) is None


def test_scoring_a_genome_leaves_its_weights_alone(finished_run: Path) -> None:
    """Checks scoring is read-only and repeatable.

    Args:
        finished_run: A finished run to take a trained genome from.
    """

    genome = stored_genome(finished_run, lowest_loss_genome(finished_run))
    objective = iris_objective()
    before = genome.to_dict()

    first = objective.test(genome)
    second = objective.test(genome)

    assert first["loss"] == pytest.approx(second["loss"])
    after = genome.to_dict()
    assert after["gates"] == before["gates"]
    assert after["encoder"] == before["encoder"]
    assert after["decoder"] == before["decoder"]


def test_test_scores_never_enter_fitness(finished_run: Path) -> None:
    """Checks the search ranked genomes on training and validation data alone.

    Args:
        finished_run: A finished run to read.
    """

    with GenomeArchive.open_readonly(str(finished_run)) as reader:
        for _, genome in reader.iter_genome_dicts():
            assert set(genome["fitness"]) == {"loss", "target_metric"}
            assert "testing_metrics" not in genome["metadata"]


def test_stored_test_score_matches_the_best_genome_by_loss(finished_run: Path) -> None:
    """Checks the archive scores the run's lowest-loss genome, as it was stored.

    Args:
        finished_run: A finished run to read.
    """

    results = run_info(finished_run)["final_test_results"]
    by_role = {role: entry for entry in results["genomes"] for role in entry["best_by"]}
    best = lowest_loss_genome(finished_run)

    assert by_role["best_fitness"]["genome_number"] == best
    rescored = iris_objective().test(stored_genome(finished_run, best))
    assert rescored["loss"] == pytest.approx(by_role["best_fitness"]["metrics"]["loss"])
    assert results["inserted_genomes"] == 4


# --- how a search drives the evaluation --------------------------------------


def test_serial_run_scores_its_best_genomes_before_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the evaluation runs after the search and before the archive closes.

    Args:
        monkeypatch: Pins the run to a single process.
    """

    calls: list[str] = []
    search = MagicMock()
    search.run_for.side_effect = lambda count: calls.append("run_for")
    search.evaluate_best_on_test.side_effect = lambda: calls.append("test")
    search.close.side_effect = lambda: calls.append("close")
    monkeypatch.setattr(
        master_worker.MPI,
        "COMM_WORLD",
        MagicMock(Get_rank=lambda: 0, Get_size=lambda: 1),
    )

    master_worker.run_evolution(
        objective=MagicMock(), build_exaqc=lambda: search, run_for=2
    )

    assert calls == ["run_for", "test", "close"]


def test_search_with_a_plain_callable_objective_skips_the_evaluation() -> None:
    """Checks a function objective, which has no ``test``, is not scored."""

    def objective(genome: CircuitGenome) -> None:
        """Scores nothing; the search under test never calls it."""

    search = EXAQC.__new__(EXAQC)
    search.objective = objective
    search.population = MagicMock()
    search.target_metric_best_genome = MagicMock(genome_number=1)
    search.archive = MagicMock()

    search.evaluate_best_on_test()

    search.archive.set_run_info.assert_not_called()


# --- full searches -----------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "population"),
    [
        ([], ISLANDS),
        (["--hyperparameter_strategy", "simplex", "--sho_genomes", "2"], STEADY_STATE),
        (["--hyperparameter_strategy", "simplex", "--sho_genomes", "2"], ISLANDS),
    ],
    ids=["islands", "sho_steady_state", "sho_islands"],
)
def test_every_search_kind_records_its_test_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    population: list[str],
) -> None:
    """Checks islands and SHO-tuned searches score and store their best genome too.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
        extra: Search flags choosing the hyperparameter strategy.
        population: The population sub-command.
    """

    out_dir = tmp_path / "run"
    run_search(
        monkeypatch, out_dir, number_genomes=5, extra=extra, population=population
    )

    results = run_info(out_dir)["final_test_results"]
    scored = {entry["genome_number"] for entry in results["genomes"]}
    roles = {role for entry in results["genomes"] for role in entry["best_by"]}

    assert lowest_loss_genome(out_dir) in scored
    assert roles == {"best_fitness", "best_target_metric"}
    for entry in results["genomes"]:
        assert 0.0 <= entry["metrics"]["mean_class_accuracy"]["mean"] <= 1.0


def test_a_custom_split_is_recorded_and_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks ``--uci_split`` reaches the archive and sizes the test set.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
    """

    out_dir = tmp_path / "run"
    run_search(
        monkeypatch,
        out_dir,
        number_genomes=2,
        extra=["--uci_split", "0.6", "0.2", "0.2"],
    )

    info = run_info(out_dir)
    assert info["arguments"]["uci_split"] == [0.6, 0.2, 0.2]
    for entry in info["final_test_results"]["genomes"]:
        per_class = entry["metrics"]["mean_class_accuracy"]
        assert sum(per_class[str(label)]["total"] for label in range(3)) == 30


def test_a_run_holding_no_test_set_records_no_test_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks a test fraction of 0 runs normally and stores nothing to report.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
    """

    out_dir = tmp_path / "run"
    run_search(
        monkeypatch, out_dir, number_genomes=2, extra=["--uci_split", "0.8", "0.2", "0"]
    )

    assert "final_test_results" not in run_info(out_dir)


@pytest.mark.parametrize(
    "split",
    [["0.5", "0.5", "0.5"], ["0", "0.5", "0.5"]],
    ids=["over_one", "no_training"],
)
def test_an_invalid_split_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: list[str]
) -> None:
    """Checks fractions that cannot divide the dataset fail before any evolution.

    Args:
        tmp_path: Where the run would write.
        monkeypatch: Sets the command line.
        split: The ``--uci_split`` values.
    """

    with pytest.raises(ValueError, match="UCI split"):
        run_search(
            monkeypatch,
            tmp_path / "run",
            number_genomes=2,
            extra=["--uci_split", *split],
        )


# --- restarts ----------------------------------------------------------------


def test_a_restart_rescores_the_continued_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks continuing a run replaces its test results with ones for the whole run.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
    """

    out_dir = tmp_path / "run"
    run_search(monkeypatch, out_dir, number_genomes=3)
    assert run_info(out_dir)["final_test_results"]["inserted_genomes"] == 3

    run_search(monkeypatch, out_dir, number_genomes=5)

    results = run_info(out_dir)["final_test_results"]
    assert results["inserted_genomes"] == 5
    assert lowest_loss_genome(out_dir) in {
        entry["genome_number"] for entry in results["genomes"]
    }


def test_a_restart_cannot_change_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks a run cannot be continued on different data than it began on.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
    """

    out_dir = tmp_path / "run"
    run_search(monkeypatch, out_dir, number_genomes=2)

    with pytest.raises(SystemExit):
        run_search(
            monkeypatch,
            out_dir,
            number_genomes=4,
            extra=["--uci_split", "0.6", "0.2", "0.2"],
        )


def test_an_archive_from_before_the_split_continues_on_its_own_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks an old run keeps its 80/20 split on restart and is not test-scored.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line, and records the split data is loaded with.
    """

    out_dir = tmp_path / "run"
    run_search(monkeypatch, out_dir, number_genomes=2)
    strip_split_from_archive(out_dir)

    splits: list[list[float]] = []
    load_data = classification.load_data

    def recording_load_data(args: argparse.Namespace) -> Any:
        """Records the split a run loads its data with, then loads it."""

        splits.append(list(args.uci_split))
        return load_data(args)

    monkeypatch.setattr(classification, "load_data", recording_load_data)
    run_search(monkeypatch, out_dir, number_genomes=3)

    assert splits == [list(LEGACY_UCI_SPLIT)]
    assert "final_test_results" not in run_info(out_dir)
    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        assert (
            reader.connection.execute("SELECT COUNT(*) FROM genomes").fetchone()[0] == 3
        )
