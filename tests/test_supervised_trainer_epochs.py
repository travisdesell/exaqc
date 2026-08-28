"""End-to-end ``SupervisedTrainer.train`` tests over one or more epochs.

Unlike ``test_supervised_trainer_gradients.py`` (which calls
``genome.forward``/``loss.backward`` directly), these tests drive the full
``SupervisedTrainer.train`` loop: dataloader iteration, loss computation,
``optimizer.step()``, per-epoch metric bookkeeping, and writing the
best-epoch parameters back into the genome.

After the refactor to a ``CircuitGenome.hybrid_model`` (a single
``torch.nn.Module`` wrapping the encoder, quantum layer, and decoder), the
trainer no longer juggles per-target parameter dicts: it optimizes
``genome.hybrid_model.parameters()`` and snapshots/restores the best epoch
via ``genome.hybrid_model.state_dict()``. That refactor fixed the two bugs
these tests originally documented (a per-batch ``print(qnode.state_dict())``
that crashed for pennylane, and a best-parameter snapshot that only
*printed* the qiskit weights instead of saving them), so every combination
exercised here is now expected to pass.

Change-detection note: the trainer writes the best epoch's weights back into
the genome through ``CircuitGenome.set_state_dict`` (encoder/decoder tensors
copied in place, quantum weights pushed back into ``gate.parameters``). These
tests therefore snapshot the encoder/decoder parameter tensors and the
gate-parameter floats *before* training and assert at least one of them moved
afterwards, rather than reaching into the hybrid model's internals.
"""

from __future__ import annotations

import math

import torch

import pytest

from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.trainer.supervised_trainer import SupervisedTrainer

from tests.supervised_trainer_test_utils import (
    COMPLEXITY_LEVELS,
    build_classification_genome,
    cross_entropy_on_logits,
    decoder_trainable_parameters,
    encoder_trainable_parameters,
    make_balanced_binary_dataloaders,
    snapshot_gate_parameters,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")

#: (encoder_name, decoder_name, include_parametric) combinations. The
#: (identity, clipped, include_parametric=False) combination -- the only one
#: with zero trainable parameters anywhere in the genome -- is covered
#: separately in test_supervised_trainer_no_parameters.py.
CODER_AND_PARAMETRIC_COMBOS: tuple[tuple[str, str, bool], ...] = (
    ("identity", "clipped", True),
    ("linear", "linear", True),
    ("linear", "linear", False),
)


def _any_tensor_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    """Returns True if any corresponding tensor pair differs.

    Args:
        before: Tensors captured before training.
        after: Tensors captured after training, in the same order.

    Returns:
        True if at least one (before, after) pair is not ``allclose``.
    """

    return any(not torch.allclose(b, a) for b, a in zip(before, after))


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS)
@pytest.mark.parametrize(
    "encoder_name,decoder_name,include_parametric", CODER_AND_PARAMETRIC_COMBOS
)
@pytest.mark.parametrize("target", TARGETS)
def test_train_runs_epochs_and_updates_parameters(
    target: str,
    encoder_name: str,
    decoder_name: str,
    include_parametric: bool,
    complexity: str,
) -> None:
    """Trains a genome for two epochs and checks the trainer's bookkeeping.

    When training succeeds, this asserts that:

    * per-epoch training/validation metrics were recorded and are finite.
    * ``best_training_metrics``/``best_validation_metrics`` were recorded.
    * at least one trainable parameter (encoder, decoder, or circuit gate)
      actually changed value, proving gradients were computed and applied.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        encoder_name: Either ``"identity"`` or ``"linear"``.
        decoder_name: Either ``"clipped"`` or ``"linear"``.
        include_parametric: Whether the quantum circuit has trainable gates.
        complexity: One of the circuit complexity levels defined in
            ``tests.supervised_trainer_test_utils``.
    """

    n_classes = 2
    epochs = 2

    genome, n_features = build_classification_genome(
        genome_number=1,
        target=target,
        complexity=complexity,
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        include_parametric=include_parametric,
        n_classes=n_classes,
        epochs=epochs,
    )

    train_loader, val_loader = make_balanced_binary_dataloaders(n_features=n_features)

    initial_encoder = [p.detach().clone() for p in encoder_trainable_parameters(genome)]
    initial_decoder = [p.detach().clone() for p in decoder_trainable_parameters(genome)]
    initial_gate_parameters = snapshot_gate_parameters(genome)

    trainer = SupervisedTrainer(
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        training_loss_function=cross_entropy_on_logits,
        validation_loss_function=cross_entropy_on_logits,
        metrics={"mean_class_accuracy": MeanClassAccuracy(n_labels=n_classes)},
    )

    trainer.train(genome)

    training_history = genome.metadata["training_epoch_metrics"]
    validation_history = genome.metadata["validation_epoch_metrics"]

    assert len(training_history) >= 1
    assert len(validation_history) >= 1
    assert len(training_history) <= epochs
    assert len(validation_history) <= epochs

    for epoch_metrics in training_history + validation_history:
        assert "loss" in epoch_metrics
        assert math.isfinite(epoch_metrics["loss"])
        assert "epoch" in epoch_metrics
        assert "mean_class_accuracy" in epoch_metrics
        assert 0.0 <= epoch_metrics["mean_class_accuracy"]["mean"] <= 1.0

    best_training_metrics = genome.metadata["best_training_metrics"]
    best_validation_metrics = genome.metadata["best_validation_metrics"]
    assert math.isfinite(best_training_metrics["loss"])
    assert math.isfinite(best_validation_metrics["loss"])

    encoder_changed = _any_tensor_changed(
        initial_encoder, encoder_trainable_parameters(genome)
    )
    decoder_changed = _any_tensor_changed(
        initial_decoder, decoder_trainable_parameters(genome)
    )
    final_gate_parameters = snapshot_gate_parameters(genome)
    gate_params_changed = final_gate_parameters != initial_gate_parameters

    assert encoder_changed or decoder_changed or gate_params_changed, (
        "expected at least one trainable parameter (encoder, decoder, or "
        "circuit gate) to change value after training"
    )
