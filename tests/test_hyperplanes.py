# tests/test_hyperplanes.py
from __future__ import annotations

import numpy as np
import pytest
import pennylane as qml
import torch

from src.trainer import QuantumStateTrainer  # your trainer
# loss_name="mse" uses loss_obs_mse from your utils.losses via LOSS_REGISTRY


torch.set_default_dtype(torch.float64)


# -----------------------------
# Helpers: dataset extraction
# -----------------------------
def _get_first_attr(ds, names):
    for n in names:
        if hasattr(ds, n):
            return getattr(ds, n)
    return None


def _extract_hyperplanes_splits(ds):
    """Extract (X_train, y_train, X_test, y_test) from a PennyLane Dataset.

    The hyperplanes dataset has had mild naming differences across versions,
    so we try multiple possible attribute names.
    """
    X_train = _get_first_attr(ds, ["x_train", "train_x", "X_train", "train_features", "features_train"])
    y_train = _get_first_attr(ds, ["y_train", "train_y", "Y_train", "train_labels", "labels_train"])
    X_test = _get_first_attr(ds, ["x_test", "test_x", "X_test", "test_features", "features_test"])
    y_test = _get_first_attr(ds, ["y_test", "test_y", "Y_test", "test_labels", "labels_test"])

    # Some datasets store a single "features"/"labels" plus an index-based split
    if X_train is None or y_train is None or X_test is None or y_test is None:
        X_all = _get_first_attr(ds, ["x", "X", "features", "inputs"])
        y_all = _get_first_attr(ds, ["y", "Y", "labels", "targets"])
        if X_all is None or y_all is None:
            raise AttributeError(
                "Could not find expected hyperplanes dataset attributes. "
                "Tried common names like x_train/y_train/x_test/y_test and features/labels."
            )
        # fallback: deterministic split
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        n = len(X_all)
        n_train = int(0.8 * n)
        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_test, y_test = X_all[n_train:], y_all[n_train:]

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    return X_train, y_train, X_test, y_test


def _to_pm_one(y: np.ndarray) -> np.ndarray:
    """Map labels to {-1, +1} as float64."""
    y = y.astype(np.float64)
    uniq = set(np.unique(y).tolist())
    if uniq <= {0.0, 1.0}:
        return 2.0 * y - 1.0
    if uniq <= {-1.0, 1.0}:
        return y
    # If labels are not in {0,1} or {-1,1}, try threshold at 0
    return np.where(y > 0, 1.0, -1.0)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def hyperplanes_data():
    """Load hyperplanes dataset and return tensors.

    Returns:
        (X_train_t, y_train_t, X_test_t, y_test_t)
    """
    # PennyLane data module fetch
    [ds] = qml.data.load("hyperplanes")
    X_train, y_train, X_test, y_test = _extract_hyperplanes_splits(ds)

    # Convert labels to {-1, +1} to match expval(PauliZ) output range
    y_train = _to_pm_one(y_train)
    y_test = _to_pm_one(y_test)

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    return X_train_t, y_train_t, X_test_t, y_test_t


@pytest.fixture()
def vqc_model(hyperplanes_data):
    """Create a PennyLane VQC model_fn compatible with QuantumStateTrainer."""
    X_train, y_train, _, _ = hyperplanes_data
    n_features = int(X_train.shape[1])

    # Use one wire per feature (angle embedding). Keep it modest for speed.
    n_wires = n_features
    dev = qml.device("default.qubit", wires=n_wires)

    n_layers = 2

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def model_fn(weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: shape (n_features,)
        # Angle embedding (feature -> rotations)
        qml.AngleEmbedding(x, wires=range(n_wires), rotation="Y")

        # Expressive trainable block
        qml.StronglyEntanglingLayers(weights, wires=range(n_wires))

        # Binary classifier output: expectation in [-1, 1]
        return qml.expval(qml.PauliZ(0))

    # StronglyEntanglingLayers expects shape (n_layers, n_wires, 3)
    weights = torch.nn.Parameter(0.01 * torch.randn(n_layers, n_wires, 3, dtype=torch.float64))
    return model_fn, weights


# -----------------------------
# Tests
# -----------------------------
def _accuracy_from_pm_one_scores(scores: torch.Tensor, y_true: torch.Tensor) -> float:
    """Convert real-valued scores to {-1,+1} predictions and compute accuracy."""
    y_pred = torch.where(scores >= 0.0, torch.tensor(1.0, dtype=scores.dtype), torch.tensor(-1.0, dtype=scores.dtype))
    return float((y_pred == y_true).double().mean().item())


def test_hyperplanes_trains_loss_decreases(hyperplanes_data, vqc_model):
    X_train, y_train, _, _ = hyperplanes_data
    model_fn, weights = vqc_model

    # Small subset for speed + determinism in tests
    torch.manual_seed(0)
    n_train = min(24, X_train.shape[0])
    Xb = X_train[:n_train]
    yb = y_train[:n_train]

    trainer = QuantumStateTrainer(
        model_fn=lambda params, x, y: model_fn(params, x),  # model uses (params, x)
        params=weights,
        target=lambda x, y: y,  # target depends on (x, y), returns scalar label
        loss_name="mse",
        normalize_states=False,  # IMPORTANT for scalar outputs
    )

    # Evaluate initial loss on this batch (averaged by trainer.fit pattern)
    with torch.no_grad():
        init_losses = []
        for i in range(n_train):
            L, _ = trainer.compute_loss(Xb[i], yb[i])
            init_losses.append(L.detach())
        init_loss = torch.stack(init_losses).mean().item()

    # Train: average across fixed batch each step using model_args_list
    logs = trainer.fit(
        steps=120,
        lr=0.15,
        model_args_list=[(Xb[i], yb[i]) for i in range(n_train)],
        log_every=40,
    )

    with torch.no_grad():
        final_losses = []
        for i in range(n_train):
            L, _ = trainer.compute_loss(Xb[i], yb[i])
            final_losses.append(L.detach())
        final_loss = torch.stack(final_losses).mean().item()

    assert len(logs) >= 1
    assert final_loss < init_loss, f"Expected loss to decrease. init={init_loss:.6f}, final={final_loss:.6f}"


def test_hyperplanes_classifier_reaches_reasonable_accuracy(hyperplanes_data, vqc_model):
    X_train, y_train, X_test, y_test = hyperplanes_data
    model_fn, weights = vqc_model

    torch.manual_seed(1)

    # Train on a small-but-not-tiny subset for stability
    n_train = min(48, X_train.shape[0])
    Xb = X_train[:n_train]
    yb = y_train[:n_train]

    trainer = QuantumStateTrainer(
        model_fn=lambda params, x, y: model_fn(params, x),
        params=weights,
        target=lambda x, y: y,
        loss_name="mse",
        normalize_states=False,
    )

    trainer.fit(
        steps=200,
        lr=0.12,
        model_args_list=[(Xb[i], yb[i]) for i in range(n_train)],
        log_every=50,
    )

    # Evaluate on test split (may be small depending on dataset settings)
    with torch.no_grad():
        scores = torch.stack([model_fn(weights, X_test[i]) for i in range(min(64, X_test.shape[0]))])
        y_eval = y_test[: scores.shape[0]]
        acc = _accuracy_from_pm_one_scores(scores, y_eval)

    # Threshold is intentionally modest for CI robustness.
    # You can tighten once you see stable baseline accuracy on your machine.
    assert acc >= 0.65, f"Test accuracy too low: {acc:.3f}"
