import pytest

from src.circuits.circuit import CircuitGenome
from src.evolution.mutation import qubit_swap


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_not_enough_qubits(target: str):
    """
    Creates a circuit genome with 3 qubits and one gate with 3
    qubits. The qubit_swap method should return swap because there
    isn't a qubit to swap to not already in use by the gate.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3}, target=target)

    qc.add_gate(
        depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)]
    )

    assert qubit_swap(qc, favor_enabled=True) is False


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_enough_qubits_1_gate(target: str):
    """
    Creates a circuit genome with 3 qubits and one gate with 1
    qubits. The qubit_swap method should return True because there
    are enough qubits to swap to. One qubit from the new gate should be
    swapped to one of the qubits not in the gate.  The original gate
    should be disabled.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3}, target=target)

    qc.add_gate(
        depth=0.40,
        method_name="p",
        qubits=[("test", 0)],
        parameters={"phi": 0.2},
    )

    assert qubit_swap(qc, favor_enabled=True) is True

    assert len(qc.gates) == 2

    enabled_count = 0
    for gate in qc.gates:
        if gate.enabled:
            enabled_count += 1

            assert len(gate.qubits) == 1
            # make sure one of the gate's qubits has swapped to on of the qubits not in the gate
            assert (("test", 0) not in gate.qubits) and (
                ("test", 1) in gate.qubits or ("test", 2) in gate.qubits
            )
        else:
            # this should be the original gate with the same qubits
            assert gate.qubits == [("test", 0)]

    assert enabled_count == 1


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_prefer_enabled_gates(target: str):
    """
    Creates a circuit genome with 3 qubits and one gate with 1
    qubits. The qubit_swap method should return True because there
    are enough qubits to swap to.

    Adds two gates to the circuit and disables one. The enabled gate
    should always be the one to be disabled.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3}, target=target)

    qc.add_gate(
        depth=0.40,
        method_name="p",
        qubits=[("test", 0)],
        parameters={"phi": 0.2},
    )
    qc.add_gate(
        depth=0.50,
        method_name="u",
        qubits=[("test", 0)],
        parameters={"theta": 0.3, "phi": 0.1, "delta": 0.7},
    )

    print(f"disabling gate: {qc.gates[1].method_name}")
    qc.gates[1].enabled = False

    assert qubit_swap(qc, favor_enabled=True) is True

    assert len(qc.gates) == 3

    enabled_count = 0
    for gate in qc.gates:
        if gate.enabled:
            enabled_count += 1

            assert len(gate.qubits) == 1
            # make sure one of the gate's qubits has swapped to on of the qubits not in the gate
            assert (("test", 0) not in gate.qubits) and (
                ("test", 1) in gate.qubits or ("test", 2) in gate.qubits
            )

            # the p gate should be the one that was copied for the mutation because
            # the u gate was disabled
            assert gate.method_name == "p"
        else:
            # this should be the original gate with the same qubits
            assert gate.qubits == [("test", 0)]

    assert enabled_count == 1


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_enough_qubits_3_gate(target: str):
    """
    Creates a circuit genome with 5 qubits and one gate with 3
    qubits. The qubit_swap method should return True because there
    are enough qubits to swap to. One qubit from the new gate should be
    swapped to one of the qubits not in the gate.  The original gate
    should be disabled.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 5}, target=target)

    qc.add_gate(
        depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)]
    )

    assert qubit_swap(qc, favor_enabled=True) is True

    assert len(qc.gates) == 2

    enabled_count = 0
    for gate in qc.gates:
        if gate.enabled:
            enabled_count += 1

            assert len(gate.qubits) == 3
            # make sure one of the gate's qubits has swapped to on of the qubits not in the gate
            assert (("test", 3) in gate.qubits and ("test", 4) not in gate.qubits) or (
                ("test", 4) in gate.qubits and ("test", 3) not in gate.qubits
            )
        else:
            # this should be the original gate with the same qubits
            assert gate.qubits == [("test", 0), ("test", 1), ("test", 2)]

    assert enabled_count == 1
