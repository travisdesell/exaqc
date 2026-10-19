"""Tests for historical versus enabled-only gate accessors."""

from __future__ import annotations

from src.circuits.circuit import CircuitGenome, Gate
from src.circuits.registers import expand_registers


def _ry(innovation_number: int, enabled: bool, depth: float) -> Gate:
    """Builds a pennylane ``ry`` gate with a fixed innovation number.

    Args:
        innovation_number: Historical marking to stamp on the gate.
        enabled: Whether the gate is enabled.
        depth: Circuit depth in ``(0, 1)``.

    Returns:
        A single-parameter ``ry`` gate on wire ``('i', 0)``.
    """

    return Gate(
        depth=depth,
        method_name="ry",
        qubits=[("i", 0)],
        parameters={"phi": 0.25},
        innovation_number=innovation_number,
        target="pennylane",
        enabled=enabled,
    )


def _genome(*gates: Gate) -> CircuitGenome:
    """Returns a genome containing ``gates``.

    Args:
        gates: Gate records to insert.

    Returns:
        A pennylane genome with those gates.
    """

    genome = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"i": 2}),
        output_qubits=expand_registers({"o": 2}),
        target="pennylane",
    )
    for gate in gates:
        genome.add_existing_gate(gate)
    return genome


def test_enabled_only_ignores_disabled_records() -> None:
    """Legacy accessors still see only enabled gates."""

    genome = _genome(_ry(1, True, 0.2), _ry(2, False, 0.6))
    assert genome.get_gate_innovations() == [1]
    assert genome.get_historical_gate_innovations() == [1, 2]
    assert genome.get_historical_gate_signature() == frozenset({(1, True), (2, False)})


def test_has_same_gates_is_unchanged() -> None:
    """Enabled-only equality ignores disabled historical records."""

    left = _genome(_ry(1, True, 0.2), _ry(9, False, 0.7))
    right = _genome(_ry(1, True, 0.3))
    assert left.has_same_gates(right)


def test_repeated_and_conflicting_enable_states() -> None:
    """Repeated IDs collapse in history; conflicting enable bits are both kept."""

    genome = _genome(
        _ry(3, True, 0.2),
        _ry(3, True, 0.4),
        _ry(3, False, 0.8),
    )
    assert genome.get_historical_gate_innovations() == [3]
    assert genome.get_historical_gate_signature() == frozenset({(3, True), (3, False)})
