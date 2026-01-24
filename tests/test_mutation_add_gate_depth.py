import pytest

from src.circuits.circuit import CircuitGenome
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
    qc = CircuitGenome(genome_number=1, registers={"t1": 2, "t2": 2}, target=target, output_qubits=[0,2])

    # test a 2 qubit both input and output gate
    # both are control and target so qubits should just be both outputs
    if target == "pennylane":
        add_gate(pennylane_gate_specifications['iswap'], qc)
    else:
        add_gate(qiskit_gate_specifications['iswap'], qc)

    # gates should be both output qubits t1: 0 and t2: 0
    assert len(qc.gates[0].qubits) == 2
    assert (('t1', 0) == qc.gates[0].qubits[0] and ('t2', 0) == qc.gates[0].qubits[1]) or (('t2', 0) == qc.gates[0].qubits[0] and ('t1', 0) == qc.gates[0].qubits[1])


    # test a gate with 2 outputs and 1 input
    for i in range(10):
        qc = CircuitGenome(genome_number=1, registers={"t1": 2, "t2": 2}, target=target, output_qubits=[0,2])

        # first is control, second and third are target for cswap
        if target == "pennylane":
            add_gate(pennylane_gate_specifications['cswap'], qc)
        else:
            add_gate(qiskit_gate_specifications['cswap'], qc)

        # gates outputs should be both output qubits t1: 0 and t2: 0
        # gates input should be either t1:1 or t2: 1
        assert len(qc.gates[0].qubits) == 3
        assert (('t1', 0) == qc.gates[0].qubits[1] and ('t2', 0) == qc.gates[0].qubits[2]) or (('t2', 0) == qc.gates[0].qubits[1] and ('t1', 0) == qc.gates[0].qubits[2])
        assert ('t1', 1) == qc.gates[0].qubits[0] or ('t2', 1) == qc.gates[0].qubits[0]

    # test a gate with 1 input and 1 output
    for i in range(10):
        qc = CircuitGenome(genome_number=1, registers={"t1": 2, "t2": 2}, target=target, output_qubits=[0,2])

        # first is input, second is output
        if target == "pennylane":
            add_gate(pennylane_gate_specifications['ch'], qc)
        else:
            add_gate(qiskit_gate_specifications['ch'], qc)

        # gates outputs should be both output qubits t1: 0 and t2: 0
        # gates input should be either t1:1 or t2: 1
        assert len(qc.gates[0].qubits) == 2

        assert qc.gates[0].qubits[0] != qc.gates[0].qubits[1]
        if qc.gates[0].qubits[1] == ('t1', 0):
            # inputs could be any of the other three gates
            assert qc.gates[0].qubits[0] in [('t1', 1), ('t2', 0), ('t2', 1)]
        else:
            # inputs can't be 't2', 0 as that's the output
            assert qc.gates[0].qubits[0] in [('t1', 0), ('t1', 1), ('t2', 1)]


    # test a gate with 2 inputs and 1 output
    for i in range(10):
        qc = CircuitGenome(genome_number=1, registers={"t1": 2, "t2": 2}, target=target, output_qubits=[0,2])

        # first two are control, third is target for ccz
        if target == "pennylane":
            add_gate(pennylane_gate_specifications['ccz'], qc)
        else:
            add_gate(qiskit_gate_specifications['ccz'], qc)

        # gates outputs should be both output qubits t1: 0 and t2: 0
        # gates input should be either t1:1 or t2: 1
        assert len(qc.gates[0].qubits) == 3

        assert qc.gates[0].qubits[0] != qc.gates[0].qubits[1]
        if qc.gates[0].qubits[2] == ('t1', 0):
            # inputs could be any of the other three gates
            assert qc.gates[0].qubits[0] in [('t1', 1), ('t2', 0), ('t2', 1)]
            assert qc.gates[0].qubits[1] in [('t1', 1), ('t2', 0), ('t2', 1)]
        else:
            # inputs can't be 't2', 0 as that's the output
            assert qc.gates[0].qubits[0] in [('t1', 0), ('t1', 1), ('t2', 1)]
            assert qc.gates[0].qubits[1] in [('t1', 0), ('t1', 1), ('t2', 1)]
