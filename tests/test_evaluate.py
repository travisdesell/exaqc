"""Tests for ``src.examples.evaluate``: scoring a saved genome on its test split.

For a UCI dataset the test split is rebuilt from the run's split, normalization
and seed, so these check that evaluating a genome out of a run's archive gives
exactly the score the run stored in its ``final_test_results``, and that a
genome file, an old archive and a split with no test samples are handled.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.datasets.classification_loaders import CLASSIFICATION_DATASETS
from src.examples import classification
from src.examples import evaluate as evaluate_module
from src.utils.genome_archive import GenomeArchive, load_genome_dict

#: A small, fast iris search, before its ``--out_dir`` and population.
SEARCH_ARGUMENTS: list[str] = (
    "--dataset iris --input_qubits 4 --output_qubits 2 --batch_size 3 --epochs 1 "
    "--number_genomes 3 -ms uniform 1 2 -ps uniform 2 2 --binary_crossover_rate 0.1 "
    "--n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 "
    "-qim amplitude -qom probs --encoding identity --decoding clipped "
    "--logging_level WARNING"
).split()


def run_search(
    monkeypatch: pytest.MonkeyPatch, out_dir: Path, extra: list[str] | None = None
) -> None:
    """Runs a small classification search in this process.

    Args:
        monkeypatch: Sets the command line.
        out_dir: Where the run writes.
        extra: Further search flags.
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification",
            *SEARCH_ARGUMENTS,
            *(extra or []),
            "--out_dir",
            str(out_dir),
            "steady_state",
            "--max_population_size",
            "3",
        ],
    )
    classification.main()


def stored_best(out_dir: Path) -> dict[str, Any]:
    """Finds the run's best genome by ``loss`` in its stored test results.

    Args:
        out_dir: The run's output directory.

    Returns:
        That genome's ``final_test_results`` entry.
    """

    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        results = reader.run_info()["final_test_results"]
    return next(
        entry for entry in results["genomes"] if "best_fitness" in entry["best_by"]
    )


def evaluate(arguments: list[str]) -> dict[str, Any]:
    """Runs the evaluation entry point's scoring on a command line.

    Args:
        arguments: The command-line arguments.

    Returns:
        The genome's test metrics.
    """

    parser = evaluate_module.build_parser()
    return evaluate_module.evaluate(parser.parse_args(arguments), parser)


@pytest.fixture(scope="module")
def default_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Runs one search with the default split for the tests that only read it.

    Args:
        tmp_path_factory: Makes the run's directory.

    Returns:
        The run's output directory.
    """

    out_dir = tmp_path_factory.mktemp("evaluate") / "run"
    with pytest.MonkeyPatch.context() as monkeypatch:
        run_search(monkeypatch, out_dir)
    return out_dir


def test_parser_reads_the_split_from_the_run_by_default() -> None:
    """Checks the split settings default to unset, so the archive's are used."""

    args = evaluate_module.build_parser().parse_args(["--genome_json", "genome.json"])

    assert args.dataset is None
    assert args.uci_split is None
    assert args.normalization is None
    assert args.seed is None
    assert args.batch_size == 32


def test_parser_accepts_every_classification_dataset() -> None:
    """Checks UCI datasets are accepted alongside the image ones."""

    parser = evaluate_module.build_parser()
    for dataset in CLASSIFICATION_DATASETS:
        args = parser.parse_args(["--genome_json", "g.json", "--dataset", dataset])
        assert args.dataset == dataset


def test_archive_genome_scores_as_the_run_did(default_run: Path) -> None:
    """Checks an archived genome reproduces the run's own test score.

    The dataset comes from the genome and the split from the run's arguments,
    so no data flags are needed.

    Args:
        default_run: A finished run.
    """

    best = stored_best(default_run)

    metrics = evaluate(
        ["--archive", str(default_run), "--genome_number", str(best["genome_number"])]
    )

    assert metrics["loss"] == pytest.approx(best["metrics"]["loss"])
    assert metrics["mean_class_accuracy"]["mean"] == pytest.approx(
        best["metrics"]["mean_class_accuracy"]["mean"]
    )


def test_a_custom_split_is_read_from_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks a run's own ``--uci_split`` and seed are used, not the defaults.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
    """

    out_dir = tmp_path / "run"
    run_search(
        monkeypatch, out_dir, ["--uci_split", "0.6", "0.2", "0.2", "--seed", "3"]
    )
    best = stored_best(out_dir)

    metrics = evaluate(
        ["--archive", str(out_dir), "--genome_number", str(best["genome_number"])]
    )

    assert metrics["loss"] == pytest.approx(best["metrics"]["loss"])
    per_class = metrics["mean_class_accuracy"]
    assert (
        sum(entry["total"] for entry in per_class.values() if isinstance(entry, dict))
        == 30
    )


def test_genome_file_uses_the_default_split(default_run: Path, tmp_path: Path) -> None:
    """Checks a genome file from a default run scores as the run did.

    Args:
        default_run: A finished run with the default split.
        tmp_path: Where the genome file is written.
    """

    best = stored_best(default_run)
    genome_file = tmp_path / "genome.json"
    genome_file.write_text(
        json.dumps(load_genome_dict(None, str(default_run), best["genome_number"]))
    )

    metrics = evaluate(["--genome_json", str(genome_file)])

    assert metrics["loss"] == pytest.approx(best["metrics"]["loss"])


@pytest.mark.parametrize(
    ("given", "recorded", "expected"),
    [
        # nothing given or recorded (a genome file): the classification defaults
        ({}, {}, {"uci_split": [0.8, 0.1, 0.1], "normalization": "minmax", "seed": 0}),
        # the run's own values
        (
            {},
            {"uci_split": [0.6, 0.2, 0.2], "normalization": "zscore", "seed": 3},
            {"uci_split": [0.6, 0.2, 0.2], "normalization": "zscore", "seed": 3},
        ),
        # a flag wins over what the run recorded
        (
            {"seed": 7},
            {"uci_split": [0.6, 0.2, 0.2], "normalization": "zscore", "seed": 3},
            {"uci_split": [0.6, 0.2, 0.2], "normalization": "zscore", "seed": 7},
        ),
        # a run from before --uci_split held nothing out
        (
            {},
            {"normalization": "minmax", "seed": 1},
            {"uci_split": [0.8, 0.2, 0.0], "normalization": "minmax", "seed": 1},
        ),
    ],
    ids=["defaults", "recorded", "flag_wins", "before_uci_split"],
)
def test_split_settings_precedence(
    given: dict[str, Any], recorded: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Checks a flag wins, then the run's recorded value, then the default.

    Args:
        given: Split flags on the command line.
        recorded: The arguments the run recorded.
        expected: The settings the test split should be rebuilt with.
    """

    args = evaluate_module.build_parser().parse_args(["--genome_json", "g.json"])
    for name, value in given.items():
        setattr(args, name, value)

    assert evaluate_module.uci_split_settings(args, recorded) == expected


def test_a_split_with_no_test_samples_is_refused(default_run: Path) -> None:
    """Checks there is nothing to score when the split holds nothing out.

    Args:
        default_run: A finished run.
    """

    best = stored_best(default_run)

    with pytest.raises(SystemExit):
        evaluate(
            [
                "--archive",
                str(default_run),
                "--genome_number",
                str(best["genome_number"]),
                "--uci_split",
                "0.8",
                "0.2",
                "0",
            ]
        )


def test_an_archive_from_before_the_split_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks a run that held out no test samples is not scored on its own data.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
    """

    out_dir = tmp_path / "run"
    run_search(monkeypatch, out_dir)
    best = stored_best(out_dir)

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
    connection.commit()
    connection.close()

    with pytest.raises(SystemExit):
        evaluate(
            ["--archive", str(out_dir), "--genome_number", str(best["genome_number"])]
        )


def test_image_dataset_loads_its_official_test_split(
    default_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks ``--dataset`` picks the image test split, batched as asked.

    Args:
        default_run: A finished run to take a genome from.
        monkeypatch: Stops the image test split being read from disk.
    """

    class Loaded(Exception):
        """Raised in place of loading, to end the evaluation there."""

    load = MagicMock(side_effect=Loaded)
    monkeypatch.setattr(evaluate_module, "load_test_data", load)
    best = stored_best(default_run)

    with pytest.raises(Loaded):
        evaluate(
            [
                "--archive",
                str(default_run),
                "--genome_number",
                str(best["genome_number"]),
                "--dataset",
                "mnist",
                "--batch_size",
                "64",
                "--no-download_dataset",
            ]
        )

    loaded = load.call_args.args[0]
    assert loaded.dataset == "mnist"
    assert loaded.batch_size == 64
    assert loaded.download_dataset is False
