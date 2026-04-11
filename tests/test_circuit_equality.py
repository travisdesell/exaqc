import pytest

from src.circuits.circuit import CircuitGenome
from src.circuits.circuit import Gate
from src.circuits.registers import expand_registers


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_all_disabled_pennylane(target: str):
    """
    Tests the has_same_gates method to determine if two
    QuantumCircuits have the same enabled gate innovation
    numbers and are therefore equivalent.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc1 = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"i": 3}),
        output_qubits=expand_registers({"o": 3}),
        target=target,
    )

    qc2 = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"i": 3}),
        output_qubits=expand_registers({"o": 3}),
        target=target,
    )

    # create potential gates for our genomes to test if they are equal (have the same gate enabled innovation numbers)
    g1 = Gate(
        depth=0.10,
        method_name="cswap",
        qubits=[("i", 0), ("i", 1), ("i", 2)],
        innovation_number=1,
    )
    g2 = Gate(
        depth=0.20,
        method_name="cswap",
        qubits=[("i", 0), ("i", 1), ("i", 2)],
        innovation_number=2,
    )
    g3 = Gate(
        depth=0.30,
        method_name="cswap",
        qubits=[("i", 0), ("i", 1), ("i", 2)],
        innovation_number=3,
    )
    g4 = Gate(
        depth=0.40,
        method_name="cswap",
        qubits=[("i", 0), ("i", 1), ("i", 2)],
        innovation_number=4,
    )
    g5 = Gate(
        depth=0.50,
        method_name="ccz",
        qubits=[("i", 2), ("o", 1), ("o", 2)],
        innovation_number=5,
    )

    # give them the same gates
    qc1.add_existing_gate(g1.copy())
    qc1.add_existing_gate(g2.copy())
    qc1.add_existing_gate(g3.copy())
    qc1.add_existing_gate(g4.copy())

    qc2.add_existing_gate(g1.copy())
    qc2.add_existing_gate(g2.copy())
    qc2.add_existing_gate(g3.copy())
    qc2.add_existing_gate(g4.copy())

    # both should have the the same enabled gate innovation numbers
    assert qc1.has_same_gates(qc2)

    qc2.add_existing_gate(g5.copy())

    # now they should have different enable gate innovation numbers
    assert not qc1.has_same_gates(qc2)

    # disable the last gate (g5), which should result in both having
    # the same enabled innovation numbers again
    qc2.gates[4].enabled = False
    assert qc1.has_same_gates(qc2)

    # disable a gate in the first qc which should result in them not
    # having the same enabled gate innovation numbers again
    qc1.gates[0].enabled = False
    assert not qc1.has_same_gates(qc2)

    # disable a gate in the first qc which should result in them not
    # having the same enabled gate innovation numbers again
    qc2.gates[0].enabled = False
    assert qc1.has_same_gates(qc2)
