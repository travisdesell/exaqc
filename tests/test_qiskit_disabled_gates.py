"""Tests that a qiskit genome with a disabled or dropped gate still builds.

Evolution disables gates (``disable_gate``) and quantum dropout drops them for
a forward pass. Their parameters stay in the genome's weight vector, which
always holds every gate's parameters, but the gate is not added to the
circuit. ``SamplerQNN`` refuses weight parameters that are missing from the
circuit ("Number of circuit parameters (7) does not match the sum of number of
inputs and weights (8)"), so every qiskit run crashed as soon as a mutation
disabled a gate. The skipped parameters are now kept in the circuit as an
exact identity, ``rz(w) rz(-w)``.
"""

from __future__ import annotations

import random

import pytest
import torch

from src.circuits.circuit import CircuitGenome
from src.noise import PennyLaneNoiseModel
from tests.supervised_trainer_test_utils import build_classification_genome


def _genome(target: str, disable_first_parametric: bool) -> CircuitGenome:
    """Builds the same small genome, optionally with a parametric gate disabled.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.
        disable_first_parametric: Whether to disable the first gate that has
            parameters.

    Returns:
        The uninitialized genome.
    """
    random.seed(3)
    torch.manual_seed(3)
    genome, _ = build_classification_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
    )
    if disable_first_parametric:
        next(gate for gate in genome.gates if gate.parameters).enabled = False
    return genome


def _probs(genome: CircuitGenome) -> torch.Tensor:
    """Runs the genome's quantum circuit on a fixed input.

    Args:
        genome: An initialized genome.

    Returns:
        The output probabilities as a 1-D double tensor.
    """
    inputs = torch.linspace(0.1, 1.2, genome.n_quantum_inputs())
    with torch.no_grad():
        return genome.torch_model(inputs).double()


@pytest.mark.parametrize(
    "noise_model",
    [None, PennyLaneNoiseModel(noise_type="depolarizing", p=0.05)],
    ids=["noiseless", "noisy"],
)
def test_disabled_gate_builds_and_matches_pennylane(
    noise_model: PennyLaneNoiseModel | None,
) -> None:
    """A disabled gate no longer breaks the qiskit build, and outputs still match.

    Args:
        noise_model: Noise to run with, or ``None`` for none.
    """
    pennylane = _genome("pennylane", disable_first_parametric=True)
    pennylane.initialize_model(noise_model=noise_model)
    qiskit = _genome("qiskit", disable_first_parametric=True)
    qiskit.initialize_model(noise_model=noise_model)

    # qiskit numbers its two-bit outcomes with the first output least significant
    assert _probs(qiskit)[[0, 2, 1, 3]] == pytest.approx(
        _probs(pennylane).tolist(), abs=1e-6
    )


def test_disabled_gate_weights_keep_their_place_and_get_no_gradient() -> None:
    """The weight vector keeps every gate's parameters; disabled ones learn nothing."""
    genome = _genome("qiskit", disable_first_parametric=True)
    genome.initialize_model()
    weight = genome.hybrid_model.quantum_layer.weight
    assert weight.numel() == len(genome.get_parameters_as_list())

    disabled = next(gate for gate in genome.gates if gate.parameters)
    offset = 0
    for gate in genome.gates:
        if gate is disabled:
            break
        offset += len(gate.parameters)

    inputs = torch.linspace(0.1, 1.2, genome.n_quantum_inputs()).unsqueeze(0)
    genome.torch_model(inputs).sum().backward()
    gradient = weight.grad[offset : offset + len(disabled.parameters)]

    assert gradient.abs().max().item() < 1e-9
