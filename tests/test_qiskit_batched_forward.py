"""Tests that a batched forward pass matches running the samples one at a time.

This guards a bug that was silent and severe on the ``qiskit`` target: every row
of a batch received the *same* output vector, so any run with a batch size above
one trained on wrong predictions (identical predictions across a batch give
wrong losses and wrong gradients), with nothing raising an error.

The cause was the name of the generated circuit's classical register.
``qiskit_machine_learning``'s ``SamplerQNN`` unpacks a batched sampler result by
looking for per-sample counts on a result data field named
:data:`~src.circuits.circuit.QISKIT_CLASSICAL_REGISTER_NAME`; when it cannot
find one it falls back to counts aggregated over the whole batch. Naming the
register to match keeps the supported per-sample path.

Because the failure mode is "batching silently disagrees with single samples",
these tests compare the two directly -- for outputs and for gradients -- on both
targets.
"""

from __future__ import annotations

import pytest
import torch

from src.circuits.circuit import QISKIT_CLASSICAL_REGISTER_NAME
from tests.supervised_trainer_test_utils import build_classification_genome

#: Every target a genome can be built for.
_TARGETS = ["pennylane", "qiskit"]


def build_genome(target: str):
    """Builds and initializes a small hybrid genome for the given target.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.

    Returns:
        A tuple of the initialized genome and its input feature count.
    """

    genome, n_features = build_classification_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        include_parametric=True,
    )
    genome.initialize_model()
    return genome, n_features


@pytest.mark.parametrize("target", _TARGETS)
def test_batched_forward_matches_single_samples(target: str) -> None:
    """A batched forward pass equals stacking the per-sample forward passes.

    Args:
        target: The quantum backend under test.
    """

    genome, n_features = build_genome(target)

    torch.manual_seed(0)
    inputs = torch.rand(4, n_features, dtype=torch.float32)

    with torch.no_grad():
        batched = genome.forward(inputs)
        one_at_a_time = torch.stack(
            [genome.forward(inputs[index : index + 1])[0] for index in range(4)]
        )

    assert batched.shape == one_at_a_time.shape
    assert torch.allclose(batched, one_at_a_time, atol=1e-5)


@pytest.mark.parametrize("target", _TARGETS)
def test_batch_rows_are_not_collapsed(target: str) -> None:
    """Distinct inputs in a batch produce distinct outputs.

    This is the direct symptom of the original bug: every row of the batch came
    back identical.

    Args:
        target: The quantum backend under test.
    """

    genome, n_features = build_genome(target)

    torch.manual_seed(0)
    inputs = torch.rand(4, n_features, dtype=torch.float32)

    with torch.no_grad():
        batched = genome.forward(inputs)

    assert not bool(
        (batched[0] == batched).all()
    ), "every row of the batch is identical, so per-sample information was lost"


@pytest.mark.parametrize("target", _TARGETS)
def test_batched_gradients_match_summed_single_gradients(target: str) -> None:
    """Gradients from a batch equal the sum of the per-sample gradients.

    A correct forward pass is not enough: the training signal has to be
    per-sample too, otherwise every sample in a batch contributes the same
    (wrong) gradient.

    Args:
        target: The quantum backend under test.
    """

    genome, n_features = build_genome(target)

    torch.manual_seed(0)
    inputs = torch.rand(4, n_features, dtype=torch.float32)

    quantum_layer = genome.hybrid_model.quantum_layer
    weight_name = "weights" if target == "pennylane" else "weight"

    genome.forward(inputs).sum().backward()
    batched_gradient = getattr(quantum_layer, weight_name).grad.clone()

    getattr(quantum_layer, weight_name).grad = None
    for index in range(4):
        genome.forward(inputs[index : index + 1]).sum().backward()
    summed_gradient = getattr(quantum_layer, weight_name).grad.clone()

    assert torch.allclose(batched_gradient, summed_gradient, atol=1e-4)


def test_qiskit_classical_register_is_named_for_sampler_unpacking() -> None:
    """The generated qiskit circuit names its classical register as required.

    ``SamplerQNN`` only exposes per-sample counts on a data field with this
    name; renaming the register silently reintroduces batch collapsing.
    """

    genome, _ = build_genome("qiskit")

    register_names = [register.name for register in genome.qiskit_circuit.cregs]

    assert register_names == [QISKIT_CLASSICAL_REGISTER_NAME]
