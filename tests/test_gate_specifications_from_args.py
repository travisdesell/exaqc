"""Tests for :meth:`src.circuits.gate_specifications.GateSpecifications.from_args`.

Every entry point selects its backend gate set (and optionally restricts it)
the same way, through ``GateSpecifications.from_args``. These tests pin that
shared behavior -- backend selection by ``--target`` and gate-set restriction
by ``--use_only`` -- so the entry-point tests do not have to.
"""

from __future__ import annotations

from argparse import Namespace

from src.circuits.gate_specifications import GateSpecifications
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications


def test_from_args_selects_backend_by_target() -> None:
    """``--target`` picks the matching backend gate specifications."""

    pennylane = GateSpecifications.from_args(
        Namespace(target="pennylane", use_only=None)
    )
    qiskit = GateSpecifications.from_args(Namespace(target="qiskit", use_only=None))

    assert pennylane is pennylane_gate_specifications
    assert pennylane.target == "pennylane"
    assert qiskit is qiskit_gate_specifications
    assert qiskit.target == "qiskit"


def test_from_args_without_use_only_returns_full_set() -> None:
    """Omitting ``--use_only`` leaves the backend's full gate set intact."""

    specs = GateSpecifications.from_args(Namespace(target="pennylane", use_only=None))

    assert specs is pennylane_gate_specifications
    assert len(list(specs.keys())) == len(list(pennylane_gate_specifications.keys()))


def test_from_args_use_only_restricts_the_gate_set() -> None:
    """``--use_only`` filters the gate set to just the named methods."""

    kept = ["cx", "ry"]
    specs = GateSpecifications.from_args(Namespace(target="pennylane", use_only=kept))

    assert sorted(specs.keys()) == sorted(kept)
    assert specs.target == "pennylane"
    # a new, filtered object -- the shared module-level set is left untouched
    assert specs is not pennylane_gate_specifications
    assert "cx" in pennylane_gate_specifications.keys()
    assert len(list(pennylane_gate_specifications.keys())) > len(kept)
