"""Tests for the "genome has no trainable parameters" branch of ``train()``.

``SupervisedTrainer.train`` has a branch for genomes with no trainable
parameters at all (``IdentityEncoder`` + ``ClippedDecoder`` + only
non-parametric quantum gates): it should skip optimization and instead just
evaluate the (untrained) genome once on the training and validation data.

The branch is guarded by ``genome.count_trainable_parameters() == 0``: the
encoder's and decoder's trainable weights plus the parameters of *enabled*
gates, which is what every trainer records as ``n_trainable_parameters``. A
genome whose parameterized gates are all disabled therefore also takes this
branch, since their parameters are never connected to the loss.

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
    assert genome.metadata["n_trainable_parameters"] == 0


@pytest.mark.parametrize("target", TARGETS)
def test_trainable_parameters_leave_out_disabled_gates(target: str) -> None:
    """Only enabled gates' parameters count towards what a genome trains.

    Disabled gates keep their entries in the quantum weight vector but are
    skipped in the forward pass, so their parameters are never trained.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    genome, _ = build_classification_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
        epochs=1,
    )
    genome.initialize_model()

    # the identity encoder and clipped decoder have no weights, so only gates count
    gate_parameters = sum(len(gate.parameters) for gate in genome.gates)
    assert gate_parameters > 0
    assert genome.count_trainable_parameters() == gate_parameters

    parameterized = [gate for gate in genome.gates if gate.parameters]
    parameterized[0].enabled = False
    assert genome.count_trainable_parameters() == gate_parameters - len(
        parameterized[0].parameters
    )

    for gate in parameterized:
        gate.enabled = False
    assert genome.count_trainable_parameters() == 0


@pytest.mark.parametrize(
    "target",
    [
        "pennylane",
        pytest.param(
            "qiskit",
            marks=pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason=(
                    "generate_qiskit_circuit leaves disabled gates' weights out of the "
                    "circuit but still passes the full weight vector to SamplerQNN, so a "
                    "qiskit genome with a disabled parameterized gate cannot be built"
                ),
            ),
        ),
    ],
)
def test_a_genome_whose_parameterized_gates_are_all_disabled_is_only_evaluated(
    target: str,
) -> None:
    """With every parameterized gate disabled nothing is trained, and zero is recorded.

    Their parameters are never connected to the loss, so training would fail in
    ``backward()``; the trainer evaluates the genome instead.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    genome, n_features = build_classification_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
        epochs=1,
    )
    for gate in genome.gates:
        if gate.parameters:
            gate.enabled = False

    train_loader, val_loader = make_balanced_binary_dataloaders(n_features=n_features)
    trainer = SupervisedTrainer(
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        training_loss_function=cross_entropy_on_logits,
        validation_loss_function=cross_entropy_on_logits,
        metrics={"mean_class_accuracy": MeanClassAccuracy(n_labels=2)},
    )

    trainer.train(genome)

    assert genome.metadata["n_trainable_parameters"] == 0
    assert genome.metadata["training_epoch_metrics"] == []
