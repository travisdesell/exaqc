"""Tests for per-gate complexity costs and the profiler that consumes them.

Each :class:`~src.circuits.gate_specifications.GateSpecification` carries the
``cnot_count`` / ``rot_count`` its decomposition costs, so a gate's complexity
is defined beside the rest of its definition rather than in a separate lookup
table that could drift out of sync. ``src.utils.profiler._gate_counts`` reads
those costs from the gate specifications matching the genome's ``target``.

These tests pin that contract: every declared gate carries sane costs on both
targets, the two targets agree wherever they share a gate, and the profiler
sums the costs over a genome's *enabled* gates only.
"""

from __future__ import annotations

import pytest

from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications
from src.utils.profiler import _gate_counts

#: The gate specifications under test, keyed by their target framework.
_SPECIFICATIONS = {
    "pennylane": pennylane_gate_specifications,
    "qiskit": qiskit_gate_specifications,
}


class _FakeGate:
    """Minimal gate stand-in exposing what ``_gate_counts`` reads."""

    def __init__(self, method_name: str, enabled: bool = True) -> None:
        """Initializes the fake gate.

        Args:
            method_name: The gate's method name, used to look up its spec.
            enabled: Whether the gate counts toward the genome's complexity.
        """
        self.method_name = method_name
        self.enabled = enabled


class _FakeGenome:
    """Minimal genome stand-in exposing what ``_gate_counts`` reads."""

    def __init__(self, target: str, gates: list[_FakeGate]) -> None:
        """Initializes the fake genome.

        Args:
            target: The target framework selecting the gate specifications.
            gates: The genome's gates.
        """
        self.target = target
        self.gates = gates


@pytest.mark.parametrize("target", sorted(_SPECIFICATIONS))
def test_every_gate_declares_complexity(target: str) -> None:
    """Every gate carries non-negative integer decomposition costs.

    Args:
        target: The target framework whose specifications are checked.
    """
    for method_name in _SPECIFICATIONS[target].keys():
        specification = _SPECIFICATIONS[target][method_name]

        assert isinstance(specification.cnot_count, int)
        assert isinstance(specification.rot_count, int)
        assert specification.cnot_count >= 0
        assert specification.rot_count >= 0


def test_targets_agree_on_shared_gate_complexity() -> None:
    """A gate present on both targets costs the same on each."""
    shared = set(pennylane_gate_specifications.keys()) & set(
        qiskit_gate_specifications.keys()
    )
    assert shared, "expected the two targets to share gates"

    for method_name in shared:
        pennylane = pennylane_gate_specifications[method_name]
        qiskit = qiskit_gate_specifications[method_name]

        assert (pennylane.cnot_count, pennylane.rot_count) == (
            qiskit.cnot_count,
            qiskit.rot_count,
        ), f"{method_name} complexity differs between targets"


@pytest.mark.parametrize("target", sorted(_SPECIFICATIONS))
def test_gate_counts_sums_specification_costs(target: str) -> None:
    """The profiler sums each enabled gate's declared costs.

    Args:
        target: The target framework whose specifications are used.
    """
    method_names = ["cx", "crx", "h"]
    genome = _FakeGenome(target, [_FakeGate(name) for name in method_names])

    specifications = _SPECIFICATIONS[target]
    expected_cnot = sum(specifications[name].cnot_count for name in method_names)
    expected_rot = sum(specifications[name].rot_count for name in method_names)

    counts = _gate_counts(genome)

    assert counts["gates_total"] == float(len(method_names))
    assert counts["gates_cnot"] == float(expected_cnot)
    assert counts["gates_rot"] == float(expected_rot)


def test_gate_counts_ignores_disabled_gates() -> None:
    """Disabled gates contribute nothing to any of the counts."""
    enabled_only = _FakeGenome("pennylane", [_FakeGate("cx")])
    with_disabled = _FakeGenome(
        "pennylane",
        [_FakeGate("cx"), _FakeGate("crx", enabled=False)],
    )

    assert _gate_counts(with_disabled) == _gate_counts(enabled_only)


def test_gate_counts_rejects_unknown_target() -> None:
    """An unrecognized target is reported rather than silently mis-scored."""
    with pytest.raises(ValueError, match="Unknown target"):
        _gate_counts(_FakeGenome("cirq", [_FakeGate("cx")]))
