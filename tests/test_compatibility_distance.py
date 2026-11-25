"""Unit tests for the three-component speciation compatibility distance."""

from __future__ import annotations

import math

import pytest
import torch

from src.circuits.circuit import CircuitGenome, Gate
from src.circuits.decoder import LinearDecoder
from src.circuits.encoder import IdentityEncoder, LinearEncoder
from src.circuits.registers import expand_registers
from src.evolution.compatibility_distance import (
    angular_difference,
    classical_parameter_distance,
    compatibility_distance,
    quantum_parameter_distance,
    structural_distance,
)


class _Genome:
    """A genome stub exposing the three distance inputs.

    Args:
        innovations: Enabled-gate innovation IDs.
        parameters: Enabled-gate parameter values keyed by innovation.
        classical: Trainable encoder parameters, then decoder parameters.
        names: Optional parameter names parallel to ``parameters``.
    """

    def __init__(
        self,
        innovations: list[int],
        parameters: dict[int, list[float]] | None = None,
        classical: list[float] | None = None,
        names: dict[int, list[str]] | None = None,
    ) -> None:
        self.innovations = list(innovations)
        self.parameters = {} if parameters is None else parameters
        self.classical = [] if classical is None else classical
        self.names = names

    def get_historical_gate_innovations(self) -> list[int]:
        """Returns the stub's enabled innovation IDs.

        Returns:
            The innovation IDs, in stored order.
        """

        return list(self.innovations)

    def enabled_gate_parameters(self) -> dict[int, list[float]]:
        """Returns the stub's quantum parameter vectors.

        Returns:
            Parameter values keyed by innovation.
        """

        return self.parameters

    def classical_parameters(self) -> list[float]:
        """Returns the stub's classical parameter vector.

        Returns:
            Encoder parameters followed by decoder parameters.
        """

        return list(self.classical)

    def enabled_gate_parameter_names(self) -> dict[int, list[str]] | None:
        """Returns parameter names when the stub was given them.

        Returns:
            The name map, or ``None`` when names were not supplied.
        """

        return self.names


def test_structural_distance_is_sqrt_of_the_symmetric_difference() -> None:
    """Unmatched enabled innovations contribute their square root."""

    assert structural_distance([], []) == 0.0
    assert structural_distance([1, 2, 2], [2, 1]) == 0.0
    assert structural_distance([1, 2, 5], [1, 2]) == 1.0
    assert structural_distance([1, 2], [3, 4]) == pytest.approx(math.sqrt(4))


def test_angular_difference_wraps_on_a_two_pi_circle() -> None:
    """A full turn is zero and a half turn is π."""

    assert angular_difference(0.0, 2.0 * math.pi) == pytest.approx(0.0)
    assert angular_difference(0.0, math.pi) == pytest.approx(math.pi)
    assert angular_difference(0.1, 2.0 * math.pi - 0.1) == pytest.approx(0.2)
    assert angular_difference(0.0, -0.1) == pytest.approx(0.1)


def test_quantum_distance_ignores_unmatched_and_parameter_free_gates() -> None:
    """Only matching gates with angles enter the quantum term."""

    left = {1: [0.0], 2: [], 3: [1.0]}
    right = {1: [math.pi], 2: [], 4: [0.5]}
    assert quantum_parameter_distance(left, right) == pytest.approx(math.pi)
    assert quantum_parameter_distance({}, {1: [0.2]}) == 0.0
    assert quantum_parameter_distance({5: []}, {5: []}) == 0.0


def test_quantum_distance_rejects_mismatched_parameters() -> None:
    """Different names, counts, or non-numeric values are rejected."""

    with pytest.raises(ValueError, match="parameter names"):
        quantum_parameter_distance(
            {1: [0.0]},
            {1: [0.0]},
            names_a={1: ["phi"]},
            names_b={1: ["theta"]},
        )
    with pytest.raises(ValueError, match="parameters on one genome"):
        quantum_parameter_distance({1: [0.0]}, {1: [0.0, 1.0]})
    with pytest.raises(ValueError, match="not a real number"):
        quantum_parameter_distance({1: ["no"]}, {1: ["no"]})


def test_classical_distance_is_euclidean_and_empty_is_zero() -> None:
    """Aligned classical weights use an ordinary Euclidean length."""

    assert classical_parameter_distance([], []) == 0.0
    assert classical_parameter_distance([0.0, 3.0], [0.0, 0.0]) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="classical parameter counts"):
        classical_parameter_distance([1.0], [1.0, 2.0])


def test_compatibility_distance_sums_weighted_components() -> None:
    """δ is the sum of the three weighted lengths, and it is symmetric."""

    left = _Genome([1, 2], {1: [0.0]}, [0.0, 0.0])
    right = _Genome([1, 3], {1: [math.pi]}, [3.0, 0.0])
    # d_s = sqrt(2), d_q = π, d_c = 3
    expected = 2.0 * math.sqrt(2.0) + 0.5 * math.pi + 4.0 * 3.0
    distance = compatibility_distance(left, right, 2.0, 0.5, 4.0)
    assert distance == pytest.approx(expected)
    assert compatibility_distance(right, left, 2.0, 0.5, 4.0) == pytest.approx(distance)
    assert compatibility_distance(left, left) == 0.0


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_coefficients_raise(bad: float) -> None:
    """Negative, NaN, and infinite coefficients are rejected."""

    genome = _Genome([])
    with pytest.raises(ValueError, match="structure_coefficient"):
        compatibility_distance(genome, genome, structure_coefficient=bad)
    with pytest.raises(ValueError, match="quantum_coefficient"):
        compatibility_distance(genome, genome, quantum_coefficient=bad)
    with pytest.raises(ValueError, match="classical_coefficient"):
        compatibility_distance(genome, genome, classical_coefficient=bad)


def _ry(innovation_number: int, enabled: bool, depth: float, phi: float) -> Gate:
    """Builds a pennylane ``ry`` gate.

    Args:
        innovation_number: Historical marking to stamp on the gate.
        enabled: Whether the gate is enabled.
        depth: Circuit depth in ``(0, 1)``.
        phi: The gate's angle, in radians.

    Returns:
        A single-parameter ``ry`` gate on wire ``('i', 0)``.
    """

    return Gate(
        depth=depth,
        method_name="ry",
        qubits=[("i", 0)],
        parameters={"phi": phi},
        innovation_number=innovation_number,
        target="pennylane",
        enabled=enabled,
    )


def _circuit(*gates: Gate) -> CircuitGenome:
    """Returns a pennylane genome containing ``gates`` and no classical stage.

    Args:
        gates: Gate records to insert.

    Returns:
        A genome with those gates.
    """

    genome = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"i": 2}),
        output_qubits=expand_registers({"i": 2}),
        target="pennylane",
    )
    for gate in gates:
        genome.add_existing_gate(gate)
    return genome


def _zeroed_linear(n_inputs: int, n_outputs: int, kind: str) -> torch.nn.Module:
    """Returns a linear encoder or decoder with every parameter set to zero.

    Args:
        n_inputs: Input width of the linear layer.
        n_outputs: Output width of the linear layer.
        kind: ``"encoder"`` or ``"decoder"``.

    Returns:
        The zeroed module.
    """

    module: torch.nn.Module
    if kind == "encoder":
        module = LinearEncoder(n_inputs, n_outputs)
    else:
        module = LinearDecoder(n_inputs, n_outputs)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    return module


def test_circuit_genome_distance_uses_enabled_angles_and_classical_weights() -> None:
    """A real genome ignores disabled gates and reads trained classical weights."""

    left_encoder = _zeroed_linear(2, 2, "encoder")
    right_encoder = _zeroed_linear(2, 2, "encoder")
    left_decoder = _zeroed_linear(2, 1, "decoder")
    right_decoder = _zeroed_linear(2, 1, "decoder")
    with torch.no_grad():
        right_decoder.layer.bias[0] = 1.0

    left = _circuit(_ry(1, True, 0.2, 0.0), _ry(9, False, 0.8, 1.25))
    right = _circuit(_ry(1, True, 0.3, math.pi))
    left.encoder = left_encoder
    left.decoder = left_decoder
    right.encoder = right_encoder
    right.decoder = right_decoder

    distance = compatibility_distance(left, right)
    # d_s = 0, d_q = π, d_c = 1
    assert distance == pytest.approx(math.pi + 1.0)

    left.gates[1].parameters["phi"] = 9.0
    assert compatibility_distance(left, right) == pytest.approx(distance)

    bare = _circuit(_ry(1, True, 0.2, 0.0))
    bare.encoder = IdentityEncoder(2, 2)
    bare.decoder = None
    assert bare.classical_parameters() == []
    assert compatibility_distance(bare, bare) == 0.0


def test_repeated_enabled_innovation_uses_the_shallowest_gate() -> None:
    """Parameter comparison keeps the enabled record of smallest depth."""

    genome = _circuit(
        _ry(3, True, 0.4, 1.5),
        _ry(3, True, 0.2, 0.25),
        _ry(3, False, 0.1, 9.0),
    )
    assert genome.enabled_gate_parameters() == {3: [0.25]}
    assert genome.enabled_gate_parameter_names() == {3: ["phi"]}


def test_different_gate_parameter_names_raise() -> None:
    """The same innovation with different parameter names is rejected."""

    other = CircuitGenome(
        genome_number=2,
        input_qubits=expand_registers({"i": 2}),
        output_qubits=expand_registers({"i": 2}),
        target="pennylane",
    )
    other.add_existing_gate(
        Gate(
            depth=0.3,
            method_name="u",
            qubits=[("i", 0)],
            parameters={"theta": 0.0, "phi": 0.0, "delta": 0.0},
            innovation_number=1,
            target="pennylane",
            enabled=True,
        )
    )
    left = _circuit(_ry(1, True, 0.2, 0.0))
    with pytest.raises(ValueError, match="parameter names"):
        compatibility_distance(left, other)
