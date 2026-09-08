"""Tests for genomes with no classical stages (``encoder``/``decoder`` = None).

A quantum-teacher imitation genome has nothing classical to learn: its inputs
are fed straight into the quantum circuit and its outputs are the raw circuit
readout. Such a genome carries ``encoder = None`` and ``decoder = None``, and
every stage of the pipeline has to tolerate that -- the forward pass, gradient
flow to the circuit weights, JSON serialization, crossover, and the
architecture diagram.

These tests pin that contract on both the ``pennylane`` and ``qiskit`` targets.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402
import pytest  # noqa: E402

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.circuits.registers import expand_registers  # noqa: E402
from src.circuits.pennylane_gate_specifications import (  # noqa: E402
    pennylane_gate_specifications,
)
from src.circuits.qiskit_gate_specifications import (  # noqa: E402
    qiskit_gate_specifications,
)
from src.evolution.crossover import crossover_encoder_decoder  # noqa: E402
from src.utils.draw_hybrid_model import draw_hybrid_model  # noqa: E402

#: Gate specifications per target, used to look up per-target parameter names
#: (pennylane's ``ry`` takes ``phi`` while qiskit's takes ``theta``).
_SPECIFICATIONS = {
    "pennylane": pennylane_gate_specifications,
    "qiskit": qiskit_gate_specifications,
}

#: The first eight bytes of any PNG file (the PNG signature).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def build_quantum_only_genome(
    target: str,
    genome_number: int = 1,
    n_qubits: int = 2,
) -> CircuitGenome:
    """Builds a purely quantum genome with no encoder and no decoder.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.
        genome_number: The genome's identifier.
        n_qubits: How many qubits the circuit spans.

    Returns:
        A :class:`CircuitGenome` with ``encoder``/``decoder`` set to ``None``
        and a small entangling circuit carrying one trainable rotation.
    """

    genome = CircuitGenome(
        genome_number=genome_number,
        target=target,
        input_qubits=expand_registers({"q": n_qubits}),
    )
    genome.hyperparameters = {
        "quantum_input_mode": "ry",
        "quantum_output_mode": "probs",
    }
    genome.encoder = None
    genome.decoder = None

    rotation_parameter = _SPECIFICATIONS[target]["ry"].parameters[0]

    genome.add_gate(depth=0.4, method_name="h", qubits=[("q", 0)])
    genome.add_gate(depth=0.5, method_name="cx", qubits=[("q", 0), ("q", 1)])
    genome.add_gate(
        depth=0.6,
        method_name="ry",
        qubits=[("q", 1)],
        parameters={rotation_parameter: 0.3 * genome_number},
    )
    return genome


@pytest.mark.parametrize("target", ["pennylane", "qiskit"])
def test_forward_skips_absent_classical_stages(target: str) -> None:
    """A genome with no encoder/decoder maps inputs straight through the circuit.

    Args:
        target: The quantum backend under test.
    """

    genome = build_quantum_only_genome(target)
    genome.initialize_model()

    n_inputs = genome.n_quantum_inputs()
    n_outputs = genome.n_quantum_outputs()

    batched = genome.forward(torch.rand(4, n_inputs))
    single = genome.forward(torch.rand(n_inputs))

    assert batched.shape == (4, n_outputs)
    assert single.shape == (n_outputs,)


@pytest.mark.parametrize("target", ["pennylane", "qiskit"])
def test_gradients_reach_circuit_weights(target: str) -> None:
    """Training signal still reaches the circuit weights with no classical stages.

    Args:
        target: The quantum backend under test.
    """

    genome = build_quantum_only_genome(target)
    genome.initialize_model()

    parameters = list(genome.parameters())
    assert parameters, "a purely quantum genome should still expose circuit weights"

    # The readout is a probability distribution summing to one, so differentiate
    # a single component rather than the (constant) sum.
    genome.forward(torch.rand(4, genome.n_quantum_inputs()))[:, 0].sum().backward()

    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in parameters
    )


@pytest.mark.parametrize("target", ["pennylane", "qiskit"])
def test_absent_stages_round_trip_through_serialization(target: str) -> None:
    """``to_dict``/``from_dict`` preserve absent classical stages as ``None``.

    Args:
        target: The quantum backend under test.
    """

    genome = build_quantum_only_genome(target)
    serialized = genome.to_dict()

    assert serialized["encoder"] is None
    assert serialized["decoder"] is None

    restored = CircuitGenome.from_dict(serialized)

    assert restored.encoder is None
    assert restored.decoder is None
    assert len(restored.gates) == len(genome.gates)


def test_crossover_keeps_absent_stages_absent() -> None:
    """Crossing purely quantum parents yields a purely quantum child."""

    parents = [
        build_quantum_only_genome("pennylane", genome_number=n) for n in (1, 2, 3)
    ]

    encoder, decoder = crossover_encoder_decoder(parents, r=0.5)

    assert encoder is None
    assert decoder is None


def test_diagram_renders_without_classical_stages(tmp_path) -> None:
    """The architecture diagram renders for a genome with no classical stages.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome = build_quantum_only_genome("pennylane", n_qubits=3)
    genome.initialize_model()

    draw_hybrid_model(str(tmp_path), genome, "diagram.png", quantum_circuit_fig=None)

    written = tmp_path / "diagram.png"
    assert written.is_file()
    assert written.stat().st_size > 0
    with open(written, "rb") as handle:
        assert handle.read(len(_PNG_MAGIC)) == _PNG_MAGIC
