import random

import pytest
from types import SimpleNamespace

from src.dropout.quantum_dropout import (
    entangling_dropout,
    gate_dropout,
    innovation_dropout,
    qubit_dropout,
    rotation_dropout,
)


class MockGate:
    """Minimal gate object for testing quantum dropout policies."""

    def __init__(
        self,
        innovation_number: int,
        method_name: str = "x",
        enabled: bool = True,
        parameters: list[float] | None = None,
        qubits: list[tuple[str, int]] | None = None,
    ):
        self.innovation_number = innovation_number
        self.method_name = method_name
        self.enabled = enabled
        self.parameters = [] if parameters is None else parameters
        self.qubits = [] if qubits is None else qubits


@pytest.fixture
def gates() -> list[MockGate]:
    """Creates a representative collection of enabled and disabled gates."""
    return [
        MockGate(
            10,
            method_name="rx",
            parameters=[0.1],
            qubits=[("q", 0)],
        ),
        MockGate(
            20,
            method_name="h",
            parameters=[],
            qubits=[("q", 1)],
        ),
        MockGate(
            30,
            method_name="ry",
            parameters=[0.2],
            qubits=[("q", 0), ("q", 1)],
        ),
        MockGate(
            40,
            method_name="cnot",
            parameters=[],
            qubits=[("q", 1), ("q", 2)],
        ),
        MockGate(
            50,
            method_name="rz",
            enabled=False,
            parameters=[0.3],
            qubits=[("q", 2)],
        ),
    ]


def test_qubit_dropout_zero_rate_drops_nothing():
    """Checks that zero qubit dropout selects no qubits."""
    genome = SimpleNamespace(
        qubits=[
            ("q", 0),
            ("q", 1),
            ("q", 2),
        ],
        output_qubits=[
            ("q", 2),
        ],
    )

    dropped = qubit_dropout(
        genome.qubits,
        dropout_rate=0.0,
    )

    assert dropped == set()


def test_qubit_dropout_full_rate_drops_all_qubits():
    """Checks that full qubit dropout selects every qubit."""
    genome = SimpleNamespace(
        qubits=[
            ("q", 0),
            ("q", 1),
            ("q", 2),
        ],
        output_qubits=[
            ("q", 2),
        ],
    )

    dropped = qubit_dropout(
        genome.qubits,
        dropout_rate=1.0,
    )

    # assert ("q", 2) not in dropped
    assert dropped == set(genome.qubits)


def test_gate_dropout_full_rate_drops_all_enabled_gates(gates):
    """Checks that full gate dropout removes every enabled gate."""
    dropped = gate_dropout(
        gates,
        dropout_rate=1.0,
    )

    assert dropped == {10, 20, 30, 40}


def test_gate_dropout_ignores_disabled_gates(gates):
    """Checks that already-disabled gates are not included in dropout."""
    dropped = gate_dropout(
        gates,
        dropout_rate=1.0,
    )

    assert 50 not in dropped


def test_rotation_dropout_only_drops_parameterized_gates(gates):
    """Checks that rotation dropout targets parameterized gates only."""
    dropped = rotation_dropout(
        gates,
        dropout_rate=1.0,
    )

    assert dropped == {10, 30}


def test_rotation_dropout_zero_rate_drops_nothing(gates):
    """Checks that zero rotation dropout leaves all gates untouched."""
    dropped = rotation_dropout(
        gates,
        dropout_rate=0.0,
    )

    assert dropped == set()


def test_entangling_dropout_only_drops_multi_qubit_gates(gates):
    """Checks that entangling dropout targets multi-qubit gates only."""
    dropped = entangling_dropout(
        gates,
        dropout_rate=1.0,
    )

    assert dropped == {30, 40}


def test_entangling_dropout_does_not_drop_single_qubit_gates(gates):
    """Checks that single-qubit gates survive entangling dropout."""
    dropped = entangling_dropout(
        gates,
        dropout_rate=1.0,
    )

    assert 10 not in dropped
    assert 20 not in dropped


def test_innovation_dropout_zero_rate_drops_nothing(gates):
    """Checks that zero innovation dropout leaves all gates active."""
    dropped = innovation_dropout(
        gates,
        dropout_rate=0.0,
        innovation_strength=0.5,
    )

    assert dropped == set()


def test_innovation_dropout_full_uniform_rate_drops_all_enabled_gates(gates):
    """Checks uniform innovation dropout at probability one."""
    dropped = innovation_dropout(
        gates,
        dropout_rate=1.0,
        innovation_strength=0.0,
    )

    assert dropped == {10, 20, 30, 40}


def test_innovation_dropout_ignores_disabled_gates(gates):
    """Checks that disabled gates are never sampled for dropout."""
    dropped = innovation_dropout(
        gates,
        dropout_rate=1.0,
        innovation_strength=0.0,
    )

    assert 50 not in dropped


def test_innovation_dropout_uniform_matches_gate_dropout(gates):
    """Checks that innovation strength zero reduces to uniform gate dropout.

    Both functions use the same random seed so they should sample the
    same enabled gates when the innovation bias is disabled.
    """
    random.seed(42)

    gate_dropped = gate_dropout(
        gates,
        dropout_rate=0.5,
    )

    random.seed(42)

    innovation_dropped = innovation_dropout(
        gates,
        dropout_rate=0.5,
        innovation_strength=0.0,
    )

    assert innovation_dropped == gate_dropped


def test_innovation_dropout_newer_gate_has_higher_probability(monkeypatch):
    """Checks that newer innovations receive stronger dropout pressure.

    The mocked random value is chosen so that the oldest innovation is
    retained while the newest innovation is dropped.
    """
    test_gates = [
        MockGate(
            10,
            parameters=[0.1],
            qubits=[("q", 0)],
        ),
        MockGate(
            20,
            parameters=[0.1],
            qubits=[("q", 1)],
        ),
        MockGate(
            30,
            parameters=[0.1],
            qubits=[("q", 2)],
        ),
    ]

    # For dropout_rate=0.4 and innovation_strength=0.5:
    #
    # innovation 10 -> p = 0.20
    # innovation 20 -> p = 0.40
    # innovation 30 -> p = 0.60
    #
    # random.random() = 0.5 therefore only the newest gate is dropped.
    monkeypatch.setattr(
        random,
        "random",
        lambda: 0.5,
    )

    dropped = innovation_dropout(
        test_gates,
        dropout_rate=0.4,
        innovation_strength=0.5,
    )

    assert dropped == {30}


def test_innovation_dropout_uses_innovation_order_not_gate_order():
    """Checks that innovation ranking is independent of list ordering."""
    ordered_gates = [
        MockGate(10),
        MockGate(20),
        MockGate(30),
    ]

    shuffled_gates = [
        MockGate(30),
        MockGate(10),
        MockGate(20),
    ]

    random.seed(123)

    ordered_result = innovation_dropout(
        ordered_gates,
        dropout_rate=0.4,
        innovation_strength=0.5,
    )

    random.seed(123)

    shuffled_result = innovation_dropout(
        shuffled_gates,
        dropout_rate=0.4,
        innovation_strength=0.5,
    )

    assert ordered_result == shuffled_result


@pytest.mark.parametrize(
    "dropout_function",
    [
        gate_dropout,
        rotation_dropout,
        entangling_dropout,
    ],
)
@pytest.mark.parametrize(
    "dropout_rate",
    [-0.1, 1.1],
)
def test_gate_based_dropout_rejects_invalid_rate(
    gates,
    dropout_function,
    dropout_rate,
):
    """Checks validation of gate-based dropout probabilities."""
    with pytest.raises(ValueError):
        dropout_function(
            gates,
            dropout_rate=dropout_rate,
        )


@pytest.mark.parametrize(
    "dropout_rate",
    [-0.1, 1.1],
)
def test_qubit_dropout_rejects_invalid_rate(dropout_rate):
    """Checks validation of qubit dropout probabilities."""
    genome = SimpleNamespace(
        qubits=[
            ("q", 0),
            ("q", 1),
            ("q", 2),
        ],
        output_qubits=[
            ("q", 2),
        ],
    )

    with pytest.raises(ValueError):
        qubit_dropout(
            genome,
            dropout_rate=dropout_rate,
        )


@pytest.mark.parametrize(
    "dropout_rate",
    [-0.1, 1.1],
)
def test_innovation_dropout_rejects_invalid_rate(
    gates,
    dropout_rate,
):
    """Checks validation of innovation dropout probabilities."""
    with pytest.raises(ValueError):
        innovation_dropout(
            gates,
            dropout_rate=dropout_rate,
            innovation_strength=0.5,
        )


@pytest.mark.parametrize(
    "innovation_strength",
    [-0.1, 1.1],
)
def test_innovation_dropout_rejects_invalid_strength(
    gates,
    innovation_strength,
):
    """Checks validation of innovation bias strength."""
    with pytest.raises(ValueError):
        innovation_dropout(
            gates,
            dropout_rate=0.1,
            innovation_strength=innovation_strength,
        )


def test_dropout_functions_do_not_modify_gate_state(gates):
    """Checks that dropout policies never alter the permanent genome gates."""
    enabled_before = {gate.innovation_number: gate.enabled for gate in gates}

    gate_dropout(
        gates,
        dropout_rate=1.0,
    )

    rotation_dropout(
        gates,
        dropout_rate=1.0,
    )

    entangling_dropout(
        gates,
        dropout_rate=1.0,
    )

    innovation_dropout(
        gates,
        dropout_rate=1.0,
        innovation_strength=0.0,
    )

    enabled_after = {gate.innovation_number: gate.enabled for gate in gates}

    assert enabled_after == enabled_before
