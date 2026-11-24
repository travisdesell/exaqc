"""Gradient-flow tests for genomes used by ``SupervisedTrainer``.

These tests exercise ``CircuitGenome.forward`` (encoder -> quantum circuit ->
decoder) directly rather than going through
``SupervisedTrainer.get_metrics``/``SupervisedTrainer.train``. This
deliberately isolates whether autograd can actually propagate a gradient
from the loss back through the decoder, the quantum circuit, and the
encoder -- independent of any bugs in the trainer's bookkeeping (several of
which are documented and exercised in ``test_supervised_trainer_epochs.py``).
"""

from __future__ import annotations

import torch

import pytest

from tests.supervised_trainer_test_utils import (
    COMPLEXITY_LEVELS,
    build_classification_genome,
    circuit_trainable_parameters,
    cross_entropy_on_logits,
    decoder_trainable_parameters,
    encoder_trainable_parameters,
    hybrid_named_parameters,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")


def _assert_all_params_received_gradient(
    parameters: list[torch.nn.Parameter], role: str
) -> None:
    """Asserts every parameter in ``parameters`` received a non-zero gradient.

    Args:
        parameters: Parameters expected to have been touched by
            ``loss.backward()``.
        role: Human-readable label (e.g. ``"encoder"``) used in assertion
            messages.

    Raises:
        AssertionError: If any parameter has no gradient, or an all-zero
            gradient.
    """

    assert parameters, f"expected at least one trainable {role} parameter to check"
    for param in parameters:
        assert param.grad is not None, f"{role} parameter received no gradient at all"
        assert torch.any(
            param.grad != 0
        ), f"{role} parameter gradient was entirely zero"


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS)
@pytest.mark.parametrize("target", TARGETS)
def test_gradients_flow_through_trainable_encoder_circuit_decoder(
    target: str, complexity: str
) -> None:
    """A LinearEncoder/LinearDecoder genome should backprop through all three stages.

    Builds a genome with a trainable (``LinearEncoder``) encoder and a
    trainable (``LinearDecoder``) decoder, does a single forward and backward
    pass, and checks -- using the single unified
    ``hybrid_model.named_parameters()`` view exposed by the refactor -- that
    the encoder, quantum layer, and decoder are all represented and that
    every one of their parameters received a non-zero gradient.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: One of the circuit complexity levels defined in
            ``tests.supervised_trainer_test_utils``.
    """

    genome, n_features = build_classification_genome(
        genome_number=1,
        target=target,
        complexity=complexity,
        encoder_name="linear",
        decoder_name="linear",
        include_parametric=True,
    )
    genome.initialize_model()

    x = torch.randn(n_features)
    y = torch.tensor(0)

    prediction = genome.forward(x)
    loss = cross_entropy_on_logits(prediction, y)
    loss.backward()

    named_parameters = hybrid_named_parameters(genome)

    # all three stages must be present in the single hybrid-model view
    assert any(
        name.startswith("encoder.") for name in named_parameters
    ), "no encoder parameters found"
    assert any(
        name.startswith("decoder.") for name in named_parameters
    ), "no decoder parameters found"
    assert (
        genome.quantum_weight_key in named_parameters
    ), "no quantum-layer weight found"

    # and every trainable parameter must have received a non-zero gradient
    _assert_all_params_received_gradient(
        list(named_parameters.values()), "hybrid model"
    )


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS)
@pytest.mark.parametrize("target", TARGETS)
def test_gradients_flow_through_circuit_with_identity_and_clipped_coders(
    target: str, complexity: str
) -> None:
    """The quantum circuit's own parameters should train even with non-trainable coders.

    ``IdentityEncoder`` and ``ClippedDecoder`` contribute no trainable
    parameters of their own, but they must still pass gradients *through*
    them so the quantum circuit's parametric gates remain trainable.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: One of the circuit complexity levels defined in
            ``tests.supervised_trainer_test_utils``.
    """

    genome, n_features = build_classification_genome(
        genome_number=2,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.initialize_model()

    assert encoder_trainable_parameters(genome) == []
    assert decoder_trainable_parameters(genome) == []

    x = torch.randn(n_features)
    y = torch.tensor(0)

    prediction = genome.forward(x)
    loss = cross_entropy_on_logits(prediction, y)
    loss.backward()

    _assert_all_params_received_gradient(
        circuit_trainable_parameters(genome), "quantum circuit"
    )


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS)
@pytest.mark.parametrize("target", TARGETS)
def test_forward_pass_has_no_trainable_parameters_without_parametric_gates(
    target: str, complexity: str
) -> None:
    """A structural-only circuit (no parametric gates) has nothing to train.

    This is a sanity check for the "no trainable parameters" scenario
    exercised end-to-end in ``test_supervised_trainer_no_parameters.py``: with
    ``IdentityEncoder``, ``ClippedDecoder``, and only non-parametric gates
    (``h``/``x``/``cx``), the forward pass should still run, but there
    should be no quantum-circuit parameters to differentiate.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: One of the circuit complexity levels defined in
            ``tests.supervised_trainer_test_utils``.
    """

    genome, n_features = build_classification_genome(
        genome_number=3,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=False,
    )
    genome.initialize_model()

    x = torch.randn(n_features)
    prediction = genome.forward(x)

    assert prediction.shape[-1] == genome.decoder.n_outputs
    assert circuit_trainable_parameters(genome) == []
