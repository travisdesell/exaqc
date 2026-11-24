"""Tests that ``CircuitGenome.initialize_model`` can be called more than once.

Several call sites initialize a genome's model unconditionally --
``SupervisedTrainer.train`` and ``CircuitGenome.save_circuit`` both do -- so a
caller that also initializes explicitly ends up initializing twice. That used to
raise ``ValueError: Weight param weights[0] not present in circuit`` on the
qiskit target.

The cause was in ``Gate.add_to_qiskit_circuit``: it cached each gate's binding
into the weight ``ParameterVector`` on first use. Every rebuild creates a fresh
vector, and qiskit ``Parameter`` objects compare by identity rather than name,
so the rebuilt circuit held stale parameters while the QNN was told about the
new ones. Gates now rebind to whichever weight vector they are handed.

These tests pin that re-initialization is safe and produces an identical model
on both targets.
"""

from __future__ import annotations

import random

import pytest
import torch

from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.trainer.supervised_trainer import SupervisedTrainer
from tests.supervised_trainer_test_utils import (
    build_classification_genome,
    cross_entropy_on_logits,
    make_balanced_binary_dataloaders,
)

#: Every target a genome can be built for.
_TARGETS = ["pennylane", "qiskit"]


def build_genome(target: str, seed: int = 3):
    """Builds a small hybrid genome deterministically.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.
        seed: Seed applied before construction so the genome is reproducible.

    Returns:
        A tuple of the (uninitialized) genome and its input feature count.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    return build_classification_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        include_parametric=True,
    )


@pytest.mark.parametrize("target", _TARGETS)
def test_repeated_initialization_succeeds(target: str) -> None:
    """Initializing repeatedly does not raise on either target.

    Args:
        target: The quantum backend under test.
    """

    genome, _ = build_genome(target)

    for _ in range(4):
        genome.initialize_model()


@pytest.mark.parametrize("target", _TARGETS)
def test_repeated_initialization_preserves_outputs(target: str) -> None:
    """Re-initializing leaves the model computing the same function.

    Args:
        target: The quantum backend under test.
    """

    genome, n_features = build_genome(target)
    inputs = torch.rand(4, n_features, dtype=torch.float32)

    genome.initialize_model()
    with torch.no_grad():
        before = genome.forward(inputs).clone()

    for _ in range(3):
        genome.initialize_model()
    with torch.no_grad():
        after = genome.forward(inputs).clone()

    assert torch.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize("target", _TARGETS)
def test_explicit_initialization_before_training(target: str) -> None:
    """A caller may initialize before ``train()``, which initializes again.

    This is the exact sequence that used to fail on qiskit.

    Args:
        target: The quantum backend under test.
    """

    genome, n_features = build_genome(target)
    genome.hyperparameters.update(
        {"learning_rate": 0.25, "epochs": 3, "improvement_cutoff": 20}
    )

    genome.initialize_model()

    training_loader, validation_loader = make_balanced_binary_dataloaders(
        n_features=n_features,
        batch_size=4,
        n_train_per_class=8,
        n_val_per_class=4,
    )

    SupervisedTrainer(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        training_loss_function=cross_entropy_on_logits,
        validation_loss_function=cross_entropy_on_logits,
        metrics={"accuracy": MeanClassAccuracy(n_labels=2)},
    ).train(genome)

    assert genome.metadata["training_epoch_metrics"]


def test_gate_parameters_rebind_to_the_current_weight_vector() -> None:
    """Each qiskit rebuild binds gates to that build's weight ParameterVector.

    Stale bindings are what made a second initialization fail, so this checks
    the circuit's parameters really belong to the current vector.
    """

    genome, _ = build_genome("qiskit")

    genome.initialize_model()
    genome.initialize_model()

    circuit_parameters = set(genome.qiskit_circuit.parameters)
    current_weights = set(genome.weight_vector)

    # every weight of the current vector is actually present in the circuit
    assert current_weights <= circuit_parameters
