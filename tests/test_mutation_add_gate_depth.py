import pytest

from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers
from src.evolution.mutation import add_gate

from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_get_circuit_indexes(target: str):
    """
    Creates multiple gates of different types in a quantum circuit with
    multiple registers of different sizes and makes sure the methods to
    get the input and output qubit indexes in the circuit are correct.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"t1": 2, "t2": 2}),
        output_qubits=[("t1", 0), ("t2", 0)],
        target=target,
    )

    # test a 2 qubit both input and output gate
    # one should be an input and the other should be an output
    # qubit
    if target == "pennylane":
        add_gate(pennylane_gate_specifications["iswap"], qc)
    else:
        add_gate(qiskit_gate_specifications["iswap"], qc)

    # gates should be both output qubits t1: 0 and t2: 0
    assert len(qc.gates[0].qubits) == 2

    q1 = qc.gates[0].qubits[0]
    q2 = qc.gates[0].qubits[1]
    print(f"q1: {q1}, q2: {q2}")
    # they should be different
    assert q1 != q2
    # at least one of these should be in the output qubits
    assert q1 in qc.output_qubits or q2 in qc.output_qubits
    # at least one of these should be in the input qubits
    assert q1 in qc.input_qubits or q2 in qc.input_qubits

    # test a gate with 2 outputs and 1 input
    for i in range(10):
        qc = CircuitGenome(
            genome_number=1,
            input_qubits=expand_registers({"t1": 2, "t2": 2}),
            output_qubits=[("t1", 0), ("t2", 0)],
            target=target,
        )

        # first is control, second and third are target for cswap
        if target == "pennylane":
            add_gate(pennylane_gate_specifications["cswap"], qc)
        else:
            add_gate(qiskit_gate_specifications["cswap"], qc)

        # gates outputs should be both output qubits t1: 0 and t2: 0
        # gates input should be either t1:1 or t2: 1
        assert len(qc.gates[0].qubits) == 3
        q1 = qc.gates[0].qubits[0]  # input
        q2 = qc.gates[0].qubits[1]  # output
        q3 = qc.gates[0].qubits[2]  # output

        # q1 should be one of the input qubits
        assert q1 in qc.input_qubits
        # at least q2 or q3 should be in the output qubits
        assert q2 in qc.output_qubits or q3 in qc.output_qubits

    # test a gate with 1 input and 1 output
    for i in range(10):
        qc = CircuitGenome(
            genome_number=1,
            input_qubits=expand_registers({"t1": 2, "t2": 2}),
            output_qubits=[("t1", 0), ("t2", 0)],
            target=target,
        )

        # first is input, second is output
        if target == "pennylane":
            add_gate(pennylane_gate_specifications["ch"], qc)
        else:
            add_gate(qiskit_gate_specifications["ch"], qc)

        # gates outputs should be both output qubits t1: 0 and t2: 0
        # gates input should be either t1:1 or t2: 1
        assert len(qc.gates[0].qubits) == 2

        assert qc.gates[0].qubits[0] != qc.gates[0].qubits[1]
        if qc.gates[0].qubits[1] == ("t1", 0):
            # inputs could be any of the other three gates
            assert qc.gates[0].qubits[0] in [("t1", 1), ("t2", 0), ("t2", 1)]
        else:
            # inputs can't be 't2', 0 as that's the output
            assert qc.gates[0].qubits[0] in [("t1", 0), ("t1", 1), ("t2", 1)]

    # test a gate with 2 inputs and 1 output
    for i in range(10):
        qc = CircuitGenome(
            genome_number=1,
            input_qubits=expand_registers({"t1": 2, "t2": 2}),
            output_qubits=[("t1", 0), ("t2", 0), ("t3", 0)],
            target=target,
        )

        # first two are control, third is target for ccz
        if target == "pennylane":
            add_gate(pennylane_gate_specifications["ccz"], qc)
        else:
            add_gate(qiskit_gate_specifications["ccz"], qc)

        # gates outputs should be both output qubits t1: 0 and t2: 0
        # gates input should be either t1:1 or t2: 1
        assert len(qc.gates[0].qubits) == 3
        q1 = qc.gates[0].qubits[0]  # input
        q2 = qc.gates[0].qubits[1]  # input
        q3 = qc.gates[0].qubits[2]  # output

        # both q1 or q2 should be in the input qubits
        assert q1 in qc.input_qubits and q2 in qc.input_qubits
        # q3 should be an output qubit
        assert q3 in qc.output_qubits
