"""Tests for the UCI train/validation/test split and the final test evaluation.

A search must never see a UCI dataset's test samples, and when it ends its best
genomes are scored on them and the results stored in the run's archive.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.circuits.circuit import CircuitGenome
from src.datasets.classification_loaders import (
    DEFAULT_UCI_SPLIT,
    LEGACY_UCI_SPLIT,
    UCI_DATASETS,
    _split_uci_dataset,
    get_uci_dataloaders,
    get_uci_dataset,
    get_uci_test_dataloader,
)
from src.evolution.exaqc import EXAQC
from src.examples import classification
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.objectives.classification_objective import ClassificationObjective
from src.utils.genome_archive import GenomeArchive


def sample_rows(features: torch.Tensor) -> set[tuple[float, ...]]:
    """Returns a dataset's samples as a set of hashable rows.

    Args:
        features: The dataset's features, one sample per row.

    Returns:
        Each sample's values as a tuple.
    """

    return {tuple(row) for row in features.tolist()}


@pytest.mark.parametrize("dataset", sorted(UCI_DATASETS))
def test_default_split_is_80_10_10_and_disjoint(dataset: str) -> None:
    """Pins the split's sizes, and that no sample index lands in two splits.

    Args:
        dataset: UCI dataset name.
    """

    n_samples = len(get_uci_dataset(dataset)[1])
    training, validation = get_uci_dataloaders(dataset, seed=3)
    test = get_uci_test_dataloader(dataset, seed=3)

    assert test is not None
    assert len(test.dataset) == round(0.1 * n_samples)
    assert len(validation.dataset) == round(0.1 * n_samples)
    assert (
        len(training.dataset) + len(validation.dataset) + len(test.dataset) == n_samples
    )

    # duplicate rows in a dataset cannot make these overlap unless the split does
    if len(np.unique(get_uci_dataset(dataset)[0], axis=0)) == n_samples:
        train_rows = sample_rows(training.dataset.x)
        validation_rows = sample_rows(validation.dataset.x)
        test_rows = sample_rows(test.dataset.x)
        assert not train_rows & test_rows
        assert not validation_rows & test_rows
        assert not train_rows & validation_rows


def test_test_split_is_stratified() -> None:
    """Checks every class is represented in proportion in the test split."""

    test = get_uci_test_dataloader("iris", seed=0)

    assert test is not None
    assert torch.bincount(test.dataset.y.long()).tolist() == [5, 5, 5]


def test_legacy_split_reproduces_the_old_division() -> None:
    """Checks a run from before the test split keeps the very same samples."""

    x, y = get_uci_dataset("iris")
    old_train, old_validation, _, _ = train_test_split(
        x, y, train_size=0.8, random_state=5, stratify=y
    )

    splits = _split_uci_dataset("iris", "none", LEGACY_UCI_SPLIT, 5)

    np.testing.assert_allclose(splits["training"][0], old_train, rtol=1e-6)
    np.testing.assert_allclose(splits["validation"][0], old_validation, rtol=1e-6)
    assert get_uci_test_dataloader("iris", split=LEGACY_UCI_SPLIT, seed=5) is None

    # minmax is still fitted on the whole dataset, so a restart sees the same features
    scaled = MinMaxScaler().fit_transform(x) * math.pi
    old_scaled, _, _, _ = train_test_split(
        scaled, y, train_size=0.8, random_state=5, stratify=y
    )
    splits = _split_uci_dataset("iris", "minmax", LEGACY_UCI_SPLIT, 5)
    np.testing.assert_array_equal(splits["training"][0], old_scaled.astype(np.float32))


@pytest.mark.parametrize("dataset", sorted(UCI_DATASETS))
def test_minmax_is_fitted_on_the_training_split_only(dataset: str) -> None:
    """Checks held-out samples do not affect the scaling.

    The training features must span exactly ``[0, pi]``, and the validation and
    test samples must be scaled by the training range rather than their own.

    Args:
        dataset: UCI dataset name.
    """

    raw = _split_uci_dataset(dataset, "none", DEFAULT_UCI_SPLIT, 2)
    scaled = _split_uci_dataset(dataset, "minmax", DEFAULT_UCI_SPLIT, 2)
    low = raw["training"][0].min(axis=0)
    high = raw["training"][0].max(axis=0)

    assert scaled["training"][0].min(axis=0) == pytest.approx(0.0, abs=1e-5)
    assert scaled["training"][0].max(axis=0) == pytest.approx(math.pi, abs=1e-5)
    for name in ("validation", "test"):
        np.testing.assert_allclose(
            scaled[name][0],
            (raw[name][0] - low) / (high - low) * math.pi,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "split", [(0.6, 0.2, 0.1), (0.0, 0.5, 0.5), (0.6, 0.6, -0.2), (0.5, 0.0, 0.5)]
)
def test_invalid_split_is_rejected(split: tuple[float, float, float]) -> None:
    """Checks fractions that cannot divide a dataset raise.

    Args:
        split: The invalid fractions.
    """

    with pytest.raises(ValueError, match="UCI split"):
        get_uci_dataloaders("iris", split=split)


def build_search(best_fitness: MagicMock, best_metric: MagicMock | None) -> EXAQC:
    """Builds just enough of a search to call ``evaluate_best_on_test``.

    Args:
        best_fitness: What the population reports as its best genome.
        best_metric: The search's best genome by ``target_metric``.

    Returns:
        The search, with a mocked archive and objective.
    """

    search = EXAQC.__new__(EXAQC)
    search.population = MagicMock()
    search.population.get_best_genome.return_value = best_fitness
    search.target_metric_best_genome = best_metric
    search.objective = MagicMock()
    search.objective.test.side_effect = lambda genome: {
        "loss": float(genome.genome_number)
    }
    search.archive = MagicMock()
    search.inserted_genomes = 7
    return search


def test_final_evaluation_records_each_best_genome() -> None:
    """Checks both best genomes are scored and stored in the archive."""

    search = build_search(MagicMock(genome_number=1), MagicMock(genome_number=2))

    search.evaluate_best_on_test()

    results = search.archive.set_run_info.call_args.kwargs["final_test_results"]
    assert results["inserted_genomes"] == 7
    assert results["genomes"] == [
        {"genome_number": 1, "best_by": ["best_fitness"], "metrics": {"loss": 1.0}},
        {
            "genome_number": 2,
            "best_by": ["best_target_metric"],
            "metrics": {"loss": 2.0},
        },
    ]


def test_final_evaluation_scores_a_shared_best_once() -> None:
    """Checks a genome that is best by both measures is scored only once."""

    genome = MagicMock(genome_number=4)
    search = build_search(genome, genome)

    search.evaluate_best_on_test()

    search.objective.test.assert_called_once_with(genome)
    results = search.archive.set_run_info.call_args.kwargs["final_test_results"]
    assert results["genomes"][0]["best_by"] == ["best_fitness", "best_target_metric"]


def test_final_evaluation_writes_nothing_without_test_data() -> None:
    """Checks an objective with no test data leaves the archive untouched."""

    search = build_search(MagicMock(genome_number=1), None)
    search.objective.test.side_effect = None
    search.objective.test.return_value = None

    search.evaluate_best_on_test()

    search.archive.set_run_info.assert_not_called()


def test_search_stores_test_results_it_can_reproduce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs a real search and checks its archive holds the test results.

    Rescoring the best genome as the archive stored it must match, which shows
    the score came from the weights the search kept.

    Args:
        tmp_path: Where the run writes.
        monkeypatch: Sets the command line.
    """

    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        (
            "classification --dataset iris --input_qubits 4 --output_qubits 2 "
            "--batch_size 3 --number_genomes 3 --epochs 1 -ms uniform 1 2 "
            "-ps uniform 2 2 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 "
            "--exponential_crossover_rate 0.1 -qim amplitude -qom probs "
            f"--encoding identity --decoding clipped --out_dir {out_dir} "
            "steady_state --max_population_size 3"
        ).split(),
    )

    classification.main()

    with GenomeArchive.open_readonly(str(out_dir)) as reader:
        info = reader.run_info()
        best = reader.best_value("loss", higher_is_better=False)

    assert info["arguments"]["uci_split"] == [0.8, 0.1, 0.1]
    results = info["final_test_results"]
    assert results["inserted_genomes"] == 3
    assert best["genome_number"] in [
        entry["genome_number"] for entry in results["genomes"]
    ]
    for entry in results["genomes"]:
        assert 0.0 <= entry["metrics"]["mean_class_accuracy"]["mean"] <= 1.0

        with GenomeArchive.open_readonly(str(out_dir)) as reader:
            stored = CircuitGenome.from_dict(
                reader.get_genome_dict(entry["genome_number"])
            )
        test = get_uci_test_dataloader("iris", batch_size=3, seed=0)
        loss = torch.nn.CrossEntropyLoss(weight=test.label_weights)
        objective = ClassificationObjective(
            training_dataloader=test,
            validation_dataloader=test,
            training_loss_function=loss,
            validation_loss_function=loss,
            metrics={"mean_class_accuracy": MeanClassAccuracy(test.n_labels)},
            device="cpu",
            testing_dataloader=test,
        )
        assert objective.test(stored)["loss"] == pytest.approx(entry["metrics"]["loss"])
