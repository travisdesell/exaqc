"""Tests that copying is a genuine deep copy.

Two regressions are pinned here:

* :meth:`~src.circuits.gate.Gate.copy` must carry over the ``enabled`` flag and
  copy the mutable ``qubits`` list and ``parameters`` dict, so mutating a copy
  never reaches back into the source gate.
* :meth:`~src.circuits.circuit.CircuitGenome.copy` must produce distinct gate
  objects, so mutating a copied genome's gates (e.g. during crossover or
  mutation) leaves the source genome untouched.
"""

from __future__ import annotations

import pytest

from src.circuits.circuit import CircuitGenome, Gate
from src.circuits.registers import expand_registers


def param_name(target: str) -> str:
    """Returns the single rotation-parameter name ``ry`` uses on ``target``."""

    # Only the number of parameters is validated on construction, but using the
    # backend's real name keeps the gates realistic (qiskit calls it 'theta',
    # pennylane 'phi').
    return "theta" if target == "qiskit" else "phi"


def make_gate(
    target: str,
    *,
    enabled: bool = True,
    innovation_number: int = 1,
    depth: float = 0.5,
    value: float = 0.25,
) -> Gate:
    """Builds a single-parameter ``ry`` gate on wire ``('i', 0)``."""

    return Gate(
        depth=depth,
        method_name="ry",
        qubits=[("i", 0)],
        parameters={param_name(target): value},
        innovation_number=innovation_number,
        target=target,
        enabled=enabled,
    )


def make_genome(target: str) -> CircuitGenome:
    """Builds a genome with one enabled and one disabled gate.

    The enabled gate sorts first (lower depth), so ``genome.gates[0]`` is
    enabled and ``genome.gates[1]`` is disabled.
    """

    genome = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"i": 2}),
        output_qubits=expand_registers({"o": 2}),
        target=target,
    )
    # EXAQC always stamps hyperparameters onto a genome before it is copied.
    genome.hyperparameters = {}
    genome.add_existing_gate(
        make_gate(target, enabled=True, innovation_number=1, depth=0.3)
    )
    genome.add_existing_gate(
        make_gate(target, enabled=False, innovation_number=2, depth=0.6)
    )
    return genome


# ---------------------------------------------------------------------------
# Gate.copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
@pytest.mark.parametrize("enabled", [True, False])
def test_gate_copy_preserves_enabled_status(target: str, enabled: bool) -> None:
    """A copied gate keeps the source gate's ``enabled`` flag."""

    copied = make_gate(target, enabled=enabled).copy()

    assert copied.enabled is enabled


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_gate_copy_deep_copies_qubits(target: str) -> None:
    """Mutating a copied gate's qubits does not touch the source gate."""

    gate = make_gate(target)
    copied = gate.copy()

    assert copied.qubits is not gate.qubits

    copied.qubits[0] = ("i", 1)

    assert gate.qubits == [("i", 0)]


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_gate_copy_deep_copies_parameters(target: str) -> None:
    """Mutating a copied gate's parameters does not touch the source gate."""

    gate = make_gate(target, value=0.25)
    key = param_name(target)
    copied = gate.copy()

    assert copied.parameters is not gate.parameters

    copied.parameters[key] = 9.9

    assert gate.parameters[key] == 0.25


# ---------------------------------------------------------------------------
# CircuitGenome.copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_genome_copy_has_distinct_gate_objects(target: str) -> None:
    """A copied genome's gates are new objects, not the source's gates."""

    genome = make_genome(target)
    copied = genome.copy(genome_number=2)

    assert len(copied.gates) == len(genome.gates)
    for source_gate, copied_gate in zip(genome.gates, copied.gates):
        assert copied_gate is not source_gate
        # they still describe the same gate (same innovation number)
        assert copied_gate.innovation_number == source_gate.innovation_number


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_genome_copy_mutation_does_not_affect_source(target: str) -> None:
    """Mutating a copied genome's gates leaves the source genome unchanged."""

    genome = make_genome(target)
    copied = genome.copy(genome_number=2)

    assert genome.gates[0].enabled is True

    # disable and edit the copy's first gate
    copied.gates[0].enabled = False
    copied.gates[0].qubits[0] = ("i", 1)
    key = param_name(target)
    copied.gates[0].parameters[key] = 9.9

    # the source's first gate must be unaffected
    assert genome.gates[0].enabled is True
    assert genome.gates[0].qubits == [("i", 0)]
    assert genome.gates[0].parameters[key] == 0.25


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_genome_copy_preserves_gate_enabled_status(target: str) -> None:
    """A copied genome preserves each gate's enabled/disabled status."""

    copied = make_genome(target).copy(genome_number=2)

    assert copied.gates[0].enabled is True
    assert copied.gates[1].enabled is False
