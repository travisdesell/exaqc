"""Tests for the reference ("teacher") circuits built as ``CircuitGenome``.

A teacher is a known circuit that student genomes are evolved to imitate. Each
is declared once, backend-agnostically, and built into a genome for either the
``pennylane`` or ``qiskit`` target -- with the per-target rotation parameter
name (``phi`` vs ``theta``) resolved from that target's gate specifications.

These tests pin: every teacher builds and runs on both targets, teachers carry
no classical stages, they use only gates the evolutionary search can also use,
the circuits are functionally correct (checked against hand-computed truth
tables), and invalid requests fail loudly.

Note on batching: forward passes here use one sample at a time. The qiskit
backend currently returns the same output for every row of a multi-row batch,
which is a pre-existing backend issue unrelated to these circuits.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.circuits.teacher_circuits import (
    TEACHER_CIRCUITS,
    TEACHER_NAMES,
    build_teacher_genome,
)

#: Gate specifications per target framework.
_SPECIFICATIONS = {
    "pennylane": pennylane_gate_specifications,
    "qiskit": qiskit_gate_specifications,
}

#: Every target the teachers must support.
_TARGETS = ["pennylane", "qiskit"]

#: A wire layout satisfying each teacher's minimum requirements, as
#: ``(input_wires, output_wires)``. Grover is limited to three total wires
#: because a wider multi-controlled Z needs an unvalidated gate.
_LAYOUTS: dict[str, tuple[list[int], list[int]]] = {
    "identity": ([0, 1], [2, 3]),
    "x_out4": ([0, 1], [2, 3]),
    "bell_out": ([0, 1], [2, 3]),
    "copy_in_to_out": ([0, 1], [2, 3]),
    "parity012_to_out4": ([0, 1, 2], [3]),
    "input_controlled_bell": ([0], [1, 2]),
    "2layer_out_block": ([0, 1], [2, 3]),
    "grover": ([0], [1, 2]),
    "half_adder": ([0, 1], [2, 3]),
}


def run_one(genome, angles: list[float]) -> torch.Tensor:
    """Runs a single sample through a teacher genome.

    Args:
        genome: An initialized teacher genome.
        angles: One input angle per input wire.

    Returns:
        The genome's output vector for that single sample.
    """

    with torch.no_grad():
        return genome.forward(torch.tensor([angles], dtype=torch.float32))[0]


def test_every_teacher_has_a_layout() -> None:
    """The test layouts cover exactly the registered teachers."""

    assert set(_LAYOUTS) == set(TEACHER_NAMES)
    assert set(TEACHER_CIRCUITS) == set(TEACHER_NAMES)


@pytest.mark.parametrize("teacher_name", TEACHER_NAMES)
@pytest.mark.parametrize("target", _TARGETS)
def test_teacher_builds_and_runs(teacher_name: str, target: str) -> None:
    """Every teacher builds and produces a normalized distribution on each target.

    Args:
        teacher_name: The teacher circuit under test.
        target: The quantum backend under test.
    """

    input_wires, output_wires = _LAYOUTS[teacher_name]
    genome = build_teacher_genome(teacher_name, target, input_wires, output_wires)
    genome.initialize_model()

    assert genome.n_quantum_inputs() == len(input_wires)
    assert genome.n_quantum_outputs() == 2 ** len(output_wires)

    output = run_one(genome, [0.3] * len(input_wires))

    assert output.shape == (2 ** len(output_wires),)
    assert float(output.sum()) == pytest.approx(1.0, abs=1e-4)
    assert bool((output >= -1e-6).all())


@pytest.mark.parametrize("teacher_name", TEACHER_NAMES)
@pytest.mark.parametrize("target", _TARGETS)
def test_teacher_has_no_classical_stages(teacher_name: str, target: str) -> None:
    """Teachers are purely quantum: no encoder and no decoder.

    Args:
        teacher_name: The teacher circuit under test.
        target: The quantum backend under test.
    """

    input_wires, output_wires = _LAYOUTS[teacher_name]
    genome = build_teacher_genome(teacher_name, target, input_wires, output_wires)

    assert genome.encoder is None
    assert genome.decoder is None


@pytest.mark.parametrize("teacher_name", TEACHER_NAMES)
@pytest.mark.parametrize("target", _TARGETS)
def test_teacher_uses_only_search_usable_gates(teacher_name: str, target: str) -> None:
    """Teachers avoid gates the evolutionary search cannot use.

    Gates flagged ``needs_validation`` are excluded from the mutation pool, so a
    student could never reproduce a teacher built from them.

    Args:
        teacher_name: The teacher circuit under test.
        target: The quantum backend under test.
    """

    input_wires, output_wires = _LAYOUTS[teacher_name]
    genome = build_teacher_genome(teacher_name, target, input_wires, output_wires)

    for gate in genome.gates:
        specification = _SPECIFICATIONS[target][gate.method_name]
        assert not specification.needs_validation, (
            f"teacher {teacher_name!r} uses {gate.method_name!r}, which the "
            "evolutionary search cannot use"
        )


@pytest.mark.parametrize("target", _TARGETS)
def test_rotation_parameter_named_per_target(target: str) -> None:
    """Fixed rotation angles use each target's own parameter name.

    Args:
        target: The quantum backend under test.
    """

    genome = build_teacher_genome("2layer_out_block", target, [0, 1], [2, 3])

    expected_name = _SPECIFICATIONS[target]["ry"].parameters[0]
    rotations = [gate for gate in genome.gates if gate.method_name == "ry"]

    assert rotations
    for gate in rotations:
        assert list(gate.parameters) == [expected_name]

    assert sorted(gate.parameters[expected_name] for gate in rotations) == [0.7, 1.1]


@pytest.mark.parametrize("target", _TARGETS)
def test_half_adder_truth_table(target: str) -> None:
    """The half adder computes SUM = A XOR B and CARRY = A AND B.

    The readout is a distribution over ``(SUM, CARRY)``. PennyLane indexes it as
    ``2 * SUM + CARRY``; qiskit uses the opposite qubit order, so the expected
    index is bit-reversed there.

    Args:
        target: The quantum backend under test.
    """

    genome = build_teacher_genome("half_adder", target, [0, 1], [2, 3])
    genome.initialize_model()

    for addend_a in (0, 1):
        for addend_b in (0, 1):
            total = addend_a ^ addend_b
            carry = addend_a & addend_b

            if target == "pennylane":
                expected_index = 2 * total + carry
            else:
                expected_index = 2 * carry + total

            output = run_one(genome, [addend_a * math.pi, addend_b * math.pi])

            assert int(output.argmax()) == expected_index
            assert float(output[expected_index]) == pytest.approx(1.0, abs=1e-4)


def test_parity_teacher_matches_hand_computed_parity() -> None:
    """The parity teacher XORs its three input wires onto one output wire.

    A single output wire has no qubit-ordering ambiguity, so this is checked
    directly against the classical truth table.
    """

    genome = build_teacher_genome("parity012_to_out4", "pennylane", [0, 1, 2], [3])
    genome.initialize_model()

    for bits in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 0)]:
        output = run_one(genome, [bit * math.pi for bit in bits])
        assert int(output.argmax()) == sum(bits) % 2


def test_identity_teacher_has_no_gates() -> None:
    """The identity teacher applies nothing, so it declares no gates."""

    genome = build_teacher_genome("identity", "pennylane", [0, 1], [2, 3])

    assert genome.gates == []


def test_unknown_teacher_is_rejected() -> None:
    """An unknown teacher name is reported with the available choices."""

    with pytest.raises(ValueError, match="Unknown teacher"):
        build_teacher_genome("not_a_teacher", "pennylane", [0], [1])


def test_unknown_target_is_rejected() -> None:
    """An unknown target framework is rejected."""

    with pytest.raises(ValueError, match="Unknown target"):
        build_teacher_genome("bell_out", "cirq", [0], [1, 2])


def test_overlapping_wires_are_rejected() -> None:
    """Input and output wires must be disjoint."""

    with pytest.raises(ValueError, match="disjoint"):
        build_teacher_genome("bell_out", "pennylane", [0, 1], [1, 2])


def test_insufficient_wires_are_rejected() -> None:
    """A teacher that cannot fit the requested wires fails loudly."""

    with pytest.raises(ValueError, match="at least 2 output wire"):
        build_teacher_genome("bell_out", "pennylane", [0], [1])

    with pytest.raises(ValueError, match="at least 3 input wire"):
        build_teacher_genome("parity012_to_out4", "pennylane", [0], [1])


def test_grover_rejects_unsupported_wire_counts() -> None:
    """Grover refuses wire counts needing an unvalidated multi-controlled Z."""

    with pytest.raises(ValueError, match="multi-controlled Z"):
        build_teacher_genome("grover", "pennylane", [0, 1], [2, 3])


@pytest.mark.parametrize("teacher_name", TEACHER_NAMES)
def test_targets_agree_up_to_output_qubit_order(teacher_name: str) -> None:
    """Both backends implement the same circuit, modulo qubit-order convention.

    PennyLane and qiskit index a multi-qubit readout in opposite bit orders, so
    the two outputs are compared after reversing the qiskit index bits.

    Args:
        teacher_name: The teacher circuit under test.
    """

    input_wires, output_wires = _LAYOUTS[teacher_name]
    angles = [0.4, 1.2, 2.1, 0.9][: len(input_wires)]

    outputs = {}
    for target in _TARGETS:
        genome = build_teacher_genome(teacher_name, target, input_wires, output_wires)
        genome.initialize_model()
        outputs[target] = run_one(genome, angles).float()

    n_bits = len(output_wires)
    reordered = outputs["qiskit"][
        [int(format(i, f"0{n_bits}b")[::-1], 2) for i in range(2**n_bits)]
    ]

    assert torch.allclose(outputs["pennylane"], reordered, atol=1e-5)
