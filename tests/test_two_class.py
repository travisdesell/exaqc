"""Tests for the 2-class (``--classes``) image classification comparison.

Covers the class filter in the image loaders, how ``--classes`` reaches the
training, validation and test loaders, ``evaluate.py``, and the 2-class Tigris
job script (``rc-scripts/exaqc_two_class_job.sh``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.datasets import classification_loaders
from src.datasets.classification_loaders import (
    _keep_classes,
    get_image_dataloaders,
    get_image_test_dataloader,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = REPO_ROOT / "rc-scripts" / "exaqc_two_class_job.sh"


class FakeImageDataset(Dataset):
    """A tiny torchvision-like dataset: 1x4x4 images, ``targets`` like MNIST."""

    def __init__(
        self, n_per_class: int = 20, n_classes: int = 10, tensor_targets: bool = True
    ):
        labels = [label for label in range(n_classes) for _ in range(n_per_class)]
        self.targets = torch.tensor(labels) if tensor_targets else list(labels)
        # each image is filled with its original label, so a sample can be traced back
        self.images = (
            torch.tensor(labels, dtype=torch.float32)
            .view(-1, 1, 1, 1)
            .expand(-1, 1, 4, 4)
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        target = self.targets[index]
        return self.images[index], int(target)


# ---------------------------------------------------------------------------
# _keep_classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tensor_targets", [True, False])
def test_keep_classes_relabels_in_given_order(tensor_targets):
    dataset = FakeImageDataset(n_per_class=5, tensor_targets=tensor_targets)
    original = torch.as_tensor(dataset.targets).clone()

    indices = _keep_classes(dataset, [6, 3])

    targets = torch.as_tensor(dataset.targets)
    assert isinstance(dataset.targets, torch.Tensor) == tensor_targets
    # only the two kept classes are returned, 5 samples each
    assert sorted(original[indices].unique().tolist()) == [3, 6]
    assert len(indices) == 10
    # relabelled in the order given: 6 -> 0, 3 -> 1; everything else -> -1
    assert torch.all(targets[original == 6] == 0)
    assert torch.all(targets[original == 3] == 1)
    assert torch.all(targets[(original != 3) & (original != 6)] == -1)


def test_keep_classes_none_keeps_everything():
    dataset = FakeImageDataset(n_per_class=3)
    before = dataset.targets.clone()

    indices = _keep_classes(dataset, None)

    np.testing.assert_array_equal(indices, np.arange(30))
    assert torch.equal(dataset.targets, before)


@pytest.mark.parametrize(
    ("classes", "message"),
    [
        ([3], "at least two distinct"),
        ([3, 3], "at least two distinct"),
        ([3, 42], "do not occur"),
    ],
)
def test_keep_classes_rejects_bad_classes(classes, message):
    with pytest.raises(ValueError, match=message):
        _keep_classes(FakeImageDataset(n_per_class=2), classes)


# ---------------------------------------------------------------------------
# Training / validation and test loaders
# ---------------------------------------------------------------------------


def _patch_training_dataset(monkeypatch):
    monkeypatch.setattr(
        classification_loaders,
        "_load_image_training_dataset",
        lambda dataset, data_dir, download, training_transform=False: FakeImageDataset(),
    )


def _all_samples(loader):
    images, labels = zip(*(batch for batch in loader))
    return torch.cat(images), torch.cat(labels)


def test_training_loaders_keep_only_two_classes(monkeypatch):
    _patch_training_dataset(monkeypatch)

    training_loader, validation_loader = get_image_dataloaders(
        "mnist", batch_size=8, validation_fraction=0.25, classes=[3, 6]
    )

    assert training_loader.n_labels == 2
    assert validation_loader.n_labels == 2
    for loader, expected_per_class in ((training_loader, 15), (validation_loader, 5)):
        images, labels = _all_samples(loader)
        # stratified split of 20 samples per class
        assert (
            sorted(labels.tolist())
            == [0] * expected_per_class + [1] * expected_per_class
        )
        # the label each image carries matches its original class: 3 -> 0, 6 -> 1
        original = images[:, 0, 0, 0].long()
        assert torch.equal(original, torch.where(labels == 0, 3, 6))


def test_training_and_validation_splits_do_not_overlap(monkeypatch):
    _patch_training_dataset(monkeypatch)

    training_loader, validation_loader = get_image_dataloaders(
        "cifar10", batch_size=8, validation_fraction=0.25, classes=[3, 5]
    )

    training = set(training_loader.dataset.indices)
    validation = set(validation_loader.dataset.indices)
    assert training.isdisjoint(validation)
    assert len(training | validation) == 40


def test_training_loaders_without_classes_keep_all(monkeypatch):
    _patch_training_dataset(monkeypatch)

    training_loader, _ = get_image_dataloaders(
        "mnist", batch_size=8, validation_fraction=0.25
    )

    assert training_loader.n_labels == 10


def test_test_loader_keeps_only_two_classes(monkeypatch):
    constructor = MagicMock(return_value=FakeImageDataset(n_per_class=10))
    monkeypatch.setattr(classification_loaders.datasets, "FashionMNIST", constructor)

    loader = get_image_test_dataloader(
        "fashion_mnist", batch_size=4, download=False, classes=[0, 3]
    )

    assert constructor.call_args.kwargs["train"] is False
    assert loader.n_labels == 2
    images, labels = _all_samples(loader)
    assert sorted(labels.tolist()) == [0] * 10 + [1] * 10
    assert torch.equal(images[:, 0, 0, 0].long(), torch.where(labels == 0, 0, 3))


def test_load_data_passes_classes_to_image_loaders(monkeypatch):
    from src.objectives import classification_objective

    image_loaders = MagicMock(return_value=("train", "val"))
    monkeypatch.setattr(
        classification_objective, "get_image_dataloaders", image_loaders
    )
    args = SimpleNamespace(
        dataset="mnist",
        data_dir="data",
        batch_size=64,
        validation_batch_size=64,
        validation_fraction=0.1,
        training_samples=None,
        validation_samples=None,
        seed=1,
        download_dataset=False,
        num_workers=0,
        pin_memory=False,
        classes=[3, 6],
    )

    assert classification_objective.load_data(args) == ("train", "val")
    assert image_loaders.call_args.kwargs["classes"] == [3, 6]

    # runs saved before --classes existed have no such attribute
    del args.classes
    classification_objective.load_data(args)
    assert image_loaders.call_args.kwargs["classes"] is None


# ---------------------------------------------------------------------------
# evaluate.py
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_evaluate(monkeypatch):
    """evaluate.py with the genome, data and trainer replaced by mocks."""
    from src.examples import evaluate

    mocks = SimpleNamespace(genome=MagicMock())
    monkeypatch.setattr(evaluate, "load_genome_dict", MagicMock(return_value={}))
    monkeypatch.setattr(
        evaluate.CircuitGenome, "from_dict", MagicMock(return_value=mocks.genome)
    )
    test_loader = SimpleNamespace(n_labels=2, label_weights=torch.ones(2))
    mocks.get_loader = MagicMock(return_value=test_loader)
    monkeypatch.setattr(evaluate, "get_image_test_dataloader", mocks.get_loader)
    mocks.trainer = MagicMock()
    mocks.trainer.return_value.test.return_value = {"mean_class_accuracy": 1.0}
    monkeypatch.setattr(evaluate, "SupervisedTrainer", mocks.trainer)
    mocks.metric = MagicMock()
    monkeypatch.setattr(evaluate, "MeanClassAccuracy", mocks.metric)

    def run(*args: str) -> None:
        monkeypatch.setattr(
            sys, "argv", ["evaluate", "--genome_json", "genome.json", *args]
        )
        evaluate.main()

    mocks.run = run
    return mocks


def test_evaluate_scores_only_the_given_classes(mocked_evaluate):
    mocked_evaluate.run(
        "--dataset",
        "cifar10",
        "--classes",
        "3",
        "5",
        "--no-download_dataset",
        "--logging_level",
        "WARNING",
    )

    call = mocked_evaluate.get_loader.call_args
    assert call.args == ("cifar10",)
    assert call.kwargs["classes"] == [3, 5]
    assert call.kwargs["download"] is False
    mocked_evaluate.metric.assert_called_once_with(2)
    mocked_evaluate.trainer.return_value.test.assert_called_once_with(
        mocked_evaluate.genome
    )


def test_evaluate_defaults_to_all_classes(mocked_evaluate):
    mocked_evaluate.run("--dataset", "mnist")

    assert mocked_evaluate.get_loader.call_args.kwargs["classes"] is None


def test_evaluate_rejects_a_single_class_flag_without_values(mocked_evaluate):
    with pytest.raises(SystemExit):
        mocked_evaluate.run("--dataset", "mnist", "--classes")


# ---------------------------------------------------------------------------
# classification.py
# ---------------------------------------------------------------------------


def test_classification_rejects_classes_for_tabular_datasets(
    monkeypatch, tmp_path, capsys
):
    from src.examples import classification

    loader = SimpleNamespace(is_image=False)
    monkeypatch.setattr(classification, "load_data", lambda args: (loader, loader))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification",
            "--dataset",
            "iris",
            "--classes",
            "0",
            "1",
            "--input_qubits",
            "2",
            "--output_qubits",
            "2",
            "--mutation_strategy",
            "uniform",
            "1",
            "2",
            "--parent_strategy",
            "uniform",
            "2",
            "2",
            "--out_dir",
            str(tmp_path / "run"),
            "steady_state",
            "--max_population_size",
            "4",
        ],
    )

    with pytest.raises(SystemExit) as error:
        classification.main()
    assert error.value.code == 2
    assert "--classes is only supported for image datasets" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# rc-scripts/exaqc_two_class_job.sh
# ---------------------------------------------------------------------------


def _dry_run(*args: str, **env: str) -> subprocess.CompletedProcess:
    environment = {**os.environ, "DRY_RUN": "1", **env}
    return subprocess.run(
        ["bash", str(JOB_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def _option(command: list[str], name: str, count: int = 1) -> list[str]:
    position = command.index(name)
    return command[position + 1 : position + 1 + count]


@pytest.mark.parametrize(
    ("dataset", "classes", "genomes", "qubits"),
    [
        ("mnist", ["3", "6"], "800", "5"),
        ("fashion_mnist", ["0", "3"], "800", "5"),
        ("cifar10", ["3", "5"], "1000", "8"),
    ],
)
def test_job_script_uses_the_aqea_qas_class_pairs(dataset, classes, genomes, qubits):
    extra = (
        {"ENCODER_CONFIG": "configs/cifar10_cnn.json"} if dataset == "cifar10" else {}
    )
    result = _dry_run(dataset, "2", **extra)

    assert result.returncode == 0, result.stderr
    command = result.stdout.split()
    assert _option(command, "--dataset") == [dataset]
    assert _option(command, "--classes", 2) == classes
    assert _option(command, "--number_genomes") == [genomes]
    assert _option(command, "--input_qubits") == [qubits]
    assert _option(command, "--output_qubits") == [qubits]
    assert _option(command, "--seed") == ["2"]
    assert _option(command, "steady_state", 2) == ["--max_population_size", "30"]
    out_dir = _option(command, "--out_dir")[0]
    assert out_dir.endswith(f"/{dataset}_c{'v'.join(classes)}_ry_q{qubits}_pop30_2")


def test_job_script_overrides():
    result = _dry_run(
        "mnist",
        "1",
        CLASSES="1 7",
        INPUT_MODE="amplitude",
        QUBITS="3",
        POP_SIZE="10",
        OUT_ROOT="/tmp/out",
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout.split()
    assert _option(command, "--classes", 2) == ["1", "7"]
    assert _option(command, "--quantum_input_mode") == ["amplitude"]
    assert _option(command, "--out_dir") == ["/tmp/out/mnist_c1v7_amplitude_q3_pop10_1"]


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (("mnist",), {}, "usage"),
        (("svhn", "1"), {}, "unknown image dataset"),
        (("mnist", "1"), {"CLASSES": "3"}, "at least two labels"),
        (("cifar10", "1"), {}, "ENCODER_CONFIG"),
    ],
)
def test_job_script_rejects_bad_input(args, env, message):
    result = _dry_run(*args, **env)

    assert result.returncode == 2
    assert message in result.stderr
