from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import src.examples.classification as classification  # noqa


def make_image_loader(
    *,
    n_samples: int = 8,
    batch_size: int = 4,
    n_classes: int = 2,
) -> DataLoader:
    """Creates a small synthetic image dataloader."""

    images = torch.randn(
        n_samples,
        1,
        8,
        8,
        dtype=torch.float32,
    )
    labels = torch.arange(n_samples, dtype=torch.long) % n_classes

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    # Custom attributes expected by the execution script.
    loader.is_image = True
    loader.input_shape = (1, 8, 8)
    loader.n_features = 64
    loader.n_labels = n_classes
    loader.label_weights = torch.ones(
        n_classes,
        dtype=torch.float32,
    )

    return loader


@pytest.fixture
def image_loaders() -> tuple[DataLoader, DataLoader]:
    """Returns synthetic training and validation image loaders."""

    return (
        make_image_loader(n_samples=8),
        make_image_loader(n_samples=4),
    )


def test_load_data_uses_image_dataloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image datasets should be routed to get_image_dataloaders."""

    training_loader = make_image_loader(n_samples=8)
    validation_loader = make_image_loader(n_samples=4)

    mocked_get_image_dataloaders = MagicMock(
        return_value=(training_loader, validation_loader)
    )

    monkeypatch.setattr(
        classification,
        "IMAGE_DATASETS",
        ["mnist"],
    )
    monkeypatch.setattr(
        classification,
        "get_image_dataloaders",
        mocked_get_image_dataloaders,
    )

    args = SimpleNamespace(
        dataset="mnist",
        data_dir="data",
        batch_size=4,
        validation_batch_size=2,
        validation_fraction=0.2,
        training_samples=8,
        validation_samples=4,
        seed=42,
        download_dataset=False,
        num_workers=0,
        pin_memory=False,
        normalization="minmax",
    )

    returned_training, returned_validation = classification.load_data(args)

    assert returned_training is training_loader
    assert returned_validation is validation_loader

    mocked_get_image_dataloaders.assert_called_once_with(
        "mnist",
        data_dir="data",
        batch_size=4,
        validation_batch_size=2,
        validation_fraction=0.2,
        training_samples=8,
        validation_samples=4,
        seed=42,
        download=False,
        num_workers=0,
        pin_memory=False,
    )


def test_load_encoder_config_reads_json(
    tmp_path,
) -> None:
    """CNN configuration should be loaded from a JSON object."""

    config_path = tmp_path / "encoder.json"
    config_path.write_text(
        json.dumps(
            {
                "activation": "relu",
                "use_batch_norm": True,
            }
        ),
        encoding="utf-8",
    )

    config = classification.load_encoder_config(str(config_path))

    assert config == {
        "activation": "relu",
        "use_batch_norm": True,
    }


def test_load_encoder_config_rejects_non_object(
    tmp_path,
) -> None:
    """The encoder configuration must contain a JSON object."""

    config_path = tmp_path / "encoder.json"
    config_path.write_text(
        json.dumps(["invalid", "config"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must contain a JSON object"):
        classification.load_encoder_config(str(config_path))


@pytest.mark.parametrize(
    "target",
    [
        "pennylane",
        "qiskit",
    ],
)
def test_main_builds_cnn_encoder_for_image_data(
    target: str,
    image_loaders: tuple[DataLoader, DataLoader],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Tests the image-data CNN path through main()."""

    training_loader, validation_loader = image_loaders

    mocked_encoder = MagicMock(name="cnn_encoder")
    mocked_decoder = MagicMock(name="decoder")

    mocked_initialize_encoder = MagicMock(return_value=mocked_encoder)
    mocked_initialize_decoder = MagicMock(return_value=mocked_decoder)
    mocked_exaqc = MagicMock(name="exaqc")
    # build_parser() asks EXAQC to register the shared search flags, so keep the
    # real classmethod on the mock; only EXAQC *construction* is stubbed out.
    mocked_exaqc.initialize_parser = classification.EXAQC.initialize_parser

    monkeypatch.setattr(
        classification,
        "load_data",
        MagicMock(
            return_value=(
                training_loader,
                validation_loader,
            )
        ),
    )
    monkeypatch.setattr(
        classification,
        "initialize_encoder",
        mocked_initialize_encoder,
    )
    monkeypatch.setattr(
        classification,
        "initialize_decoder",
        mocked_initialize_decoder,
    )
    # EXAQC is constructed by main()'s build_exaqc factory (invoked by
    # run_evolution, which runs serially here since the tests are a single
    # process), so it is mocked to capture that construction and to avoid its
    # eager strategy validation running during this encoder-wiring test. With
    # EXAQC mocked, run_evolution's serial path just calls run_for on the mock.
    monkeypatch.setattr(
        classification,
        "EXAQC",
        mocked_exaqc,
    )

    # Prevent tests from creating real log handlers.
    monkeypatch.setattr(
        classification.logger,
        "remove",
        MagicMock(),
    )
    monkeypatch.setattr(
        classification.logger,
        "add",
        MagicMock(),
    )

    encoder_config_path = tmp_path / "cnn_config.json"
    encoder_config_path.write_text(
        json.dumps(
            {
                "activation": "relu",
            }
        ),
        encoding="utf-8",
    )

    output_directory = tmp_path / target

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification.py",
            "--dataset",
            "mnist",
            "--out_dir",
            str(output_directory),
            "--mutation_strategy",
            "add_gate",
            "--parent_strategy",
            "uniform 2 3",
            "--input_qubits",
            "2",
            "--output_qubits",
            "1",
            "--target",
            target,
            "--encoding",
            "cnn",
            "--decoding",
            "linear",
            "--encoder_config",
            str(encoder_config_path),
            "--batch_size",
            "4",
            "--epochs",
            "1",
            "--number_genomes",
            "1",
            "--cnn_channels",
            "8",
            "16",
            "--cnn_pooled_size",
            "2",
            "--cnn_dropout",
            "0.1",
            "steady_state",
            "--max_population_size",
            "2",
        ],
    )

    classification.main()

    mocked_initialize_encoder.assert_called_once()

    encoder_call = mocked_initialize_encoder.call_args

    assert encoder_call.kwargs["target"] == target
    assert encoder_call.kwargs["encoding_str"] == "cnn"
    assert encoder_call.kwargs["n_inputs"] == 64

    # u3 encoding uses three encoder outputs per input qubit.
    assert encoder_call.kwargs["n_outputs"] == 6

    cnn_config = encoder_call.kwargs["config"]

    assert cnn_config == {
        "activation": "relu",
        "input_channels": 1,
        "input_height": 8,
        "input_width": 8,
        "hidden_channels": [8, 16],
        "pooled_size": 2,
        "dropout": pytest.approx(0.1),
    }

    mocked_initialize_decoder.assert_called_once_with(
        target=target,
        decoding_str="linear",
        n_inputs=2,
        n_outputs=2,
    )

    # main()'s build_exaqc factory constructs the (mocked) EXAQC directly from
    # the pieces it computes plus the shared factories (GateSpecifications /
    # PopulationStrategy, which run for real here), so the encoder/decoder,
    # registers and hyperparameters are asserted on the EXAQC construction.
    mocked_exaqc.assert_called_once()

    exaqc_call = mocked_exaqc.call_args.kwargs

    assert exaqc_call["initial_encoder"] is mocked_encoder
    assert exaqc_call["initial_decoder"] is mocked_decoder
    assert exaqc_call["task"] == "classification"
    assert exaqc_call["task_target"] == "mnist"
    # The backend comes from the gate set built by GateSpecifications.from_args.
    assert exaqc_call["gate_specifications"].target == target

    assert exaqc_call["input_registers"] == {
        "input": 2,
    }
    assert exaqc_call["output_registers"] == {
        "input": 1,
    }

    assert exaqc_call["hyperparameters"] == {
        "epochs": 1,
        "learning_rate": pytest.approx(5e-3),
        "weight_decay": pytest.approx(0.0),
        "improvement_cutoff": 3,
        "batch_size": 4,
        "quantum_input_mode": "u3",
        "quantum_output_mode": "probs",
        "quantum_dropout": False,
        "quantum_dropout_rate": 0.0,
        "quantum_dropout_type": "none",
    }

    # run_evolution runs serially in the single-process test: build_exaqc builds
    # the (mocked) EXAQC and run_for is invoked with the genome budget.
    mocked_exaqc.return_value.run_for.assert_called_once_with(1)


@pytest.mark.skip(
    reason="No longer needed since linear encoder can used with image datasets"
)
def test_main_rejects_non_cnn_encoder_for_image_data(
    image_loaders: tuple[DataLoader, DataLoader],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Image datasets must use CNN encoding."""

    training_loader, validation_loader = image_loaders

    monkeypatch.setattr(
        classification,
        "load_data",
        MagicMock(
            return_value=(
                training_loader,
                validation_loader,
            )
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification.py",
            "--dataset",
            "mnist",
            "--out_dir",
            str(tmp_path),
            "--mutation_strategy",
            "add_gate",
            "--parent_strategy",
            "uniform 2 3",
            "--input_qubits",
            "2",
            "--output_qubits",
            "1",
            "--encoding",
            "linear",
            "steady_state",
        ],
    )

    with pytest.raises(SystemExit) as error:
        classification.main()

    assert error.value.code == 2


def test_main_rejects_cnn_encoder_for_tabular_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Tabular datasets must not use CNN encoding."""

    images = torch.randn(8, 4)
    labels = torch.arange(8, dtype=torch.long) % 2

    training_loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )
    validation_loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    for loader in (training_loader, validation_loader):
        loader.is_image = False
        loader.input_shape = (4,)
        loader.n_features = 4
        loader.n_labels = 2
        loader.label_weights = torch.ones(2)

    monkeypatch.setattr(
        classification,
        "load_data",
        MagicMock(
            return_value=(
                training_loader,
                validation_loader,
            )
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification.py",
            "--dataset",
            "iris",
            "--out_dir",
            str(tmp_path),
            "--mutation_strategy",
            "add_gate",
            "--parent_strategy",
            "uniform 2 3",
            "--input_qubits",
            "2",
            "--output_qubits",
            "1",
            "--encoding",
            "cnn",
            "steady_state",
        ],
    )

    with pytest.raises(SystemExit) as error:
        classification.main()

    assert error.value.code == 2
