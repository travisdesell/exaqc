"""Tests that noisy PennyLane circuits train on a batch of encoder outputs.

PennyLane 0.41 (pinned on Intel macOS) raises "Can't call numpy() on Tensor
that requires grad" when ``qml.probs`` or ``qml.expval`` is measured on
``default.mixed`` for a batch of inputs that require grad, which is every
noisy training step. Noisy circuits now measure the density matrix and read
the same values off its diagonal (``_DensityMatrixTorchLayer``). These tests
check that a batched training step works and gives exactly what
``qml.probs`` / ``qml.expval`` give one sample at a time.
"""

from __future__ import annotations

import random

import pennylane as qml
import pytest
import torch

from src.circuits.circuit import CircuitGenome
from src.noise import PennyLaneNoiseModel
from tests.supervised_trainer_test_utils import build_classification_genome


def _noisy_genome(output_mode: str) -> CircuitGenome:
    """Builds and initializes a three-wire genome with depolarizing noise.

    Args:
        output_mode: ``"probs"`` or ``"expval"``.

    Returns:
        The initialized genome.
    """
    random.seed(3)
    torch.manual_seed(3)
    genome, _ = build_classification_genome(
        genome_number=1,
        target="pennylane",
        complexity="deep",
        encoder_name="linear",
        decoder_name="linear",
        quantum_output_mode=output_mode,
    )
    genome.initialize_model(
        noise_model=PennyLaneNoiseModel(noise_type="depolarizing", p=0.05)
    )
    return genome


@pytest.mark.parametrize("output_mode", ["probs", "expval"])
def test_batched_noisy_forward_and_backward(output_mode: str) -> None:
    """A batch of grad-requiring inputs runs forward and backward.

    Args:
        output_mode: The quantum readout mode under test.
    """
    genome = _noisy_genome(output_mode)
    encoder = torch.nn.Linear(4, genome.n_quantum_inputs())

    outputs = genome.torch_model(encoder(torch.rand(5, 4)))
    outputs.sum().backward()

    assert outputs.shape == (5, genome.n_quantum_outputs())
    assert genome.torch_model.weights.grad.abs().sum() > 0
    assert encoder.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("output_mode", ["probs", "expval"])
def test_density_matrix_readout_matches_direct_measurement(output_mode: str) -> None:
    """The values read off ρ equal ``qml.probs`` / ``qml.expval`` exactly.

    Args:
        output_mode: The quantum readout mode under test.
    """
    genome = _noisy_genome(output_mode)
    layer = genome.torch_model
    wires = genome.output_indexes
    inputs = torch.rand(4, genome.n_quantum_inputs())

    @qml.qnode(qml.device("default.mixed", wires=genome.total_qubits))
    def direct(sample: torch.Tensor, weights: torch.Tensor) -> object:
        """Runs the genome's own circuit, swapping only the final measurement.

        Args:
            sample: One quantum input.
            weights: The quantum weights.

        Returns:
            ``qml.probs`` or the per-wire ``qml.expval`` of Pauli-Z.
        """
        tape = qml.workflow.construct_tape(layer.qnode)(sample, weights)
        for operation in tape.operations:
            qml.apply(operation)
        if output_mode == "probs":
            return qml.probs(wires=wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    with torch.no_grad():
        batched = layer(inputs).double()
        for sample, row in zip(inputs, batched):
            expected = torch.as_tensor(direct(sample, layer.weights)).double()
            assert row.tolist() == pytest.approx(expected.tolist(), abs=1e-6)
