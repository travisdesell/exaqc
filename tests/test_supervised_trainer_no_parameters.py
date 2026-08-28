"""Tests for the "genome has no trainable parameters" branch of ``train()``.

``SupervisedTrainer.train`` has a branch for genomes with no trainable
parameters at all (``IdentityEncoder`` + ``ClippedDecoder`` + only
non-parametric quantum gates): it should skip optimization and instead just
evaluate the (untrained) genome once on the training and validation data.

After the ``hybrid_model`` refactor this branch is now correctly *reached*
for both targets -- it is guarded by
``sum(p.numel() for p in genome.hybrid_model.parameters() if p.requires_grad)
== 0``, which is true for both pennylane and qiskit when there are no
trainable weights anywhere. (This replaces the old, target-asymmetric
``len(get_torch_parameters()) == 0`` guard, which was never true for qiskit
because a ``"qiskit_parameters"`` entry was always inserted.)

After computing and storing ``best_training_metrics`` /
``best_validation_metrics`` the branch returns immediately, leaving the
per-epoch histories empty, which is exactly what this test asserts for both
targets.
"""

from __future__ import annotations

import pytest

from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.trainer.supervised_trainer import SupervisedTrainer

from tests.supervised_trainer_test_utils import (
    build_classification_genome,
    cross_entropy_on_logits,
    make_balanced_binary_dataloaders,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")


@pytest.mark.parametrize("target", TARGETS)
def test_train_with_no_trainable_parameters_only_evaluates(target: str) -> None:
    """A parameter-free genome should be evaluated once, not optimized.

    Builds a genome with an ``IdentityEncoder``, a ``ClippedDecoder``, and
    only non-parametric gates (``h``/``cx``), so
    ``genome.hybrid_model`` has zero trainable parameters and
    ``SupervisedTrainer.train`` should take its evaluation-only path --
    recording ``best_training_metrics``/``best_validation_metrics`` but
    leaving the per-epoch history empty.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    genome, n_features = build_classification_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=False,
        epochs=2,
    )

    train_loader, val_loader = make_balanced_binary_dataloaders(n_features=n_features)

    trainer = SupervisedTrainer(
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        training_loss_function=cross_entropy_on_logits,
        validation_loss_function=cross_entropy_on_logits,
        metrics={"mean_class_accuracy": MeanClassAccuracy(n_labels=2)},
    )

    trainer.train(genome)

    assert genome.metadata["training_epoch_metrics"] == []
    assert genome.metadata["validation_epoch_metrics"] == []
    assert "best_training_metrics" in genome.metadata
    assert "best_validation_metrics" in genome.metadata
