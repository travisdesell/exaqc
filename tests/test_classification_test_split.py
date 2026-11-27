"""Tests for the held-out test split of the UCI classification datasets.

``--test_fraction`` sets rows aside before the search, the search never sees
them, and every genome's best weights are scored on them once after training
(``test_loss`` / ``test_accuracy``). Normalization is fitted on the training
rows only. ``src.analysis.test_metrics_report`` reports those test metrics.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.datasets.classification_loaders import get_uci_dataloaders
from src.examples.classification import build_parser
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.objectives.classification_objective import ClassificationObjective, load_data
from tests.supervised_trainer_test_utils import (
    build_classification_genome,
    cross_entropy_on_logits,
    make_balanced_binary_dataloaders,
)


def _rows(loader) -> set[tuple[float, ...]]:
    """The rows a loader holds, as a set.

    Args:
        loader: A UCI dataloader.

    Returns:
        Each row's features, rounded so equal rows compare equal.
    """

    return {tuple(np.round(row, 5)) for row in loader.dataset.x.numpy()}


@pytest.mark.parametrize("dataset", ["iris", "wine", "seeds", "breast_cancer"])
def test_split_is_80_10_10_disjoint_and_stratified(dataset: str) -> None:
    """An 80/10/10 split has the right sizes, no shared rows, and every class.

    Args:
        dataset: The UCI dataset.
    """

    train, validation = get_uci_dataloaders(
        dataset, seed=0, test_fraction=0.1, validation_fraction=0.1
    )
    test = train.test_loader
    sizes = [len(loader.dataset) for loader in (train, validation, test)]
    total = sum(sizes)

    assert sizes[0] == pytest.approx(0.8 * total, abs=2)
    assert sizes[1] == pytest.approx(0.1 * total, abs=2)
    assert sizes[2] == pytest.approx(0.1 * total, abs=2)
    assert not (_rows(train) & _rows(test)) and not (_rows(validation) & _rows(test))
    for loader in (train, validation, test):
        assert int((loader.label_counts == 0).sum()) == 0


def test_test_rows_do_not_depend_on_the_validation_fraction() -> None:
    """The test rows are set aside first, so changing validation leaves them alone."""

    first, _ = get_uci_dataloaders(
        "wine", seed=0, test_fraction=0.1, validation_fraction=0.1
    )
    second, _ = get_uci_dataloaders(
        "wine", seed=0, test_fraction=0.1, validation_fraction=0.2
    )

    assert first.test_loader.dataset.y.tolist() == second.test_loader.dataset.y.tolist()


def test_without_a_test_set_the_split_stays_80_20() -> None:
    """test_fraction 0 keeps the old 80/20 split and has no test loader."""

    train, validation = get_uci_dataloaders("iris", seed=0)

    assert (len(train.dataset), len(validation.dataset)) == (120, 30)
    assert train.test_loader is None


def test_minmax_normalization_is_fitted_on_training_rows() -> None:
    """Training columns span exactly [0, pi]; held-out rows may fall outside it."""

    train, validation = get_uci_dataloaders(
        "breast_cancer", seed=0, test_fraction=0.1, validation_fraction=0.1
    )
    x = train.dataset.x.numpy()

    assert np.allclose(x.min(axis=0), 0.0, atol=1e-5)
    assert np.allclose(x.max(axis=0), math.pi, atol=1e-5)


def _args(**overrides: object) -> argparse.Namespace:
    """Parsed classification arguments for a small search.

    Args:
        **overrides: Flags to set, as ``name=value``.

    Returns:
        The parsed namespace.
    """

    # -ms and -ps take every value up to the next flag, so flags follow them
    argv = [
        "-ms",
        "uniform",
        "1",
        "3",
        "-ps",
        "uniform",
        "5",
        "5",
        "--dataset",
        "iris",
        "--input_qubits",
        "4",
        "--output_qubits",
        "2",
    ]
    for name, value in overrides.items():
        argv += [f"--{name}", str(value)]
    return build_parser().parse_args(argv + ["steady_state"])


def test_flag_reaches_the_loaders_and_images_are_refused() -> None:
    """--test_fraction gives load_data a test loader; image datasets refuse it."""

    assert _args().test_fraction == 0.0
    train, _ = load_data(_args(test_fraction=0.1))
    assert train.test_loader is not None and len(train.test_loader.dataset) == 15

    with pytest.raises(ValueError, match="only supported for the UCI"):
        load_data(_args(dataset="mnist", test_fraction=0.1))


def test_objective_records_test_metrics_without_changing_fitness() -> None:
    """Test metrics are added to the fitness; loss and target_metric are unchanged."""

    def build() -> tuple[object, object, object]:
        """Builds the same genome and loaders each time.

        Returns:
            The genome, the training loader and the validation loader.
        """
        genome, n_features = build_classification_genome(
            genome_number=1,
            target="pennylane",
            complexity="shallow",
            encoder_name="linear",
            decoder_name="linear",
            epochs=1,
        )
        train, validation = make_balanced_binary_dataloaders(n_features=n_features)
        return genome, train, validation

    def objective(train: object, validation: object, test: object | None):
        """Builds an objective over the loaders.

        Args:
            train: Training loader.
            validation: Validation loader.
            test: Test loader, or ``None``.

        Returns:
            The objective.
        """
        return ClassificationObjective(
            training_dataloader=train,
            validation_dataloader=validation,
            training_loss_function=cross_entropy_on_logits,
            validation_loss_function=cross_entropy_on_logits,
            metrics={"mean_class_accuracy": MeanClassAccuracy(n_labels=2)},
            test_dataloader=test,
        )

    import torch

    torch.manual_seed(0)
    with_test, train, validation = build()
    objective(train, validation, validation)(with_test)

    torch.manual_seed(0)
    without_test, train, validation = build()
    objective(train, validation, None)(without_test)

    assert set(with_test.fitness) == {
        "loss",
        "target_metric",
        "test_loss",
        "test_accuracy",
    }
    assert set(without_test.fitness) == {"loss", "target_metric"}
    assert 0.0 <= with_test.fitness["test_accuracy"] <= 1.0
    assert "testing_metrics" in with_test.metadata


def test_report_lists_best_and_front_genomes(tmp_path: Path) -> None:
    """The report picks the best genome (single objective) and the Pareto front (MOO).

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    from tests.test_exaqc_dashboard import FakeGenome, build_run

    def genome(
        number: int, loss: float, gates: int, test_accuracy: float
    ) -> FakeGenome:
        """A stored genome with test metrics.

        Args:
            number: Genome number.
            loss: Its loss.
            gates: Its n_enabled_gates.
            test_accuracy: Its test accuracy.

        Returns:
            The fake genome.
        """
        fake = FakeGenome(
            number, loss=loss, target_metric=1 - loss, parents=[0], generated_by=[]
        )
        fake.serialized["fitness"].update(
            n_enabled_gates=gates, test_loss=loss + 0.01, test_accuracy=test_accuracy
        )
        return fake

    genomes = [genome(1, 0.5, 1, 0.6), genome(2, 0.3, 5, 0.8), genome(3, 0.4, 2, 0.7)]
    build_run(tmp_path / "single", genomes)
    build_run(
        tmp_path / "moo",
        genomes,
        run_info={
            "multi_objective": {
                "algorithm": "nsga2",
                "objectives": [
                    {"name": "loss", "sign": 1.0, "direction": "minimize"},
                    {"name": "n_enabled_gates", "sign": 1.0, "direction": "minimize"},
                ],
            }
        },
    )
    csv_path = tmp_path / "report.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.test_metrics_report",
            "--runs",
            str(tmp_path / "single"),
            str(tmp_path / "moo"),
            "--csv",
            str(csv_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("run,genome,role")
    # single: genome 2 (lowest loss); moo: front of the population {1, 2, 3} is all three
    single = [line for line in lines if line.startswith("single,")]
    moo = [line for line in lines if line.startswith("moo,")]
    assert single == [line for line in single if ",2,best," in line]
    assert any(",2,best," in line for line in moo) and len(moo) == 3
    assert "2 runs, best genome per run: test accuracy 0.8000" in result.stdout
