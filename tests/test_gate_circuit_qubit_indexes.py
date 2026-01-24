import pytest

from src.circuits.circuit import CircuitGenome
from src.evolution.mutation import reorder_gate


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_get_circuit_indexes(target: str):
    """
    Creates multiple gates of different types in a quantum circuit with
    multiple registers of different sizes and makes sure the methods to
    get the input and output qubit indexes in the circuit are correct.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"t1": 3, "t2": 4}, target=target)

    # qubit is input and output
    qc.add_gate(
        depth=0.20, method_name="p", qubits=[("t2", 3)], parameters={"theta": 0.1}
    )
    gate = qc.gates[0]
    input_indexes = gate.get_input_circuit_indexes(qc)
    output_indexes = gate.get_output_circuit_indexes(qc)
    assert input_indexes == [6]
    assert output_indexes == [6]

    # both are inputs and outputs
    qc.add_gate(
        depth=0.30, method_name="iswap", qubits=[("t2", 0), ("t1", 1)]
    )
    gate = qc.gates[1]
    input_indexes = gate.get_input_circuit_indexes(qc)
    output_indexes = gate.get_output_circuit_indexes(qc)
    assert input_indexes == [3, 1]
    assert output_indexes == [3, 1]

    # first two are control, third is target
    qc.add_gate(
        depth=0.40, method_name="ccz", qubits=[("t1", 0), ("t1", 1), ("t2", 3)]
    )
    gate = qc.gates[2]
    input_indexes = gate.get_input_circuit_indexes(qc)
    output_indexes = gate.get_output_circuit_indexes(qc)
    assert input_indexes == [0, 1, 6]
    assert output_indexes == [6]

    # first is control, second is target
    qc.add_gate(
        depth=0.45, method_name="ch", qubits=[("t1", 2), ("t2", 1)]
    )
    gate = qc.gates[3]
    input_indexes = gate.get_input_circuit_indexes(qc)
    output_indexes = gate.get_output_circuit_indexes(qc)
    assert input_indexes == [2, 4]
    assert output_indexes == [4]

    # first is control, second and third are target
    qc.add_gate(
        depth=0.50, method_name="cswap", qubits=[("t1", 2), ("t2", 2), ("t1", 1)]
    )
    gate = qc.gates[4]
    input_indexes = gate.get_input_circuit_indexes(qc)
    output_indexes = gate.get_output_circuit_indexes(qc)
    assert input_indexes == [2, 5, 1]
    assert output_indexes == [5, 1]


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_get_circuit_indexes_by_depth(target: str):
    """
    Creates multiple gates of different types in a quantum circuit with
    multiple registers of different sizes and makes sure the methods to
    get the input and output qubit indexes in the circuit are correct.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"t1": 3, "t2": 4}, target=target, output_qubits=[0, 1])

    # test with no gates yet
    assert qc.get_possible_output_qubits(0.95) == [0, 1]

    # qubit is input and output
    qc.add_gate(
        depth=0.90, method_name="p", qubits=[("t1", 1)], parameters={"theta": 0.1}
    )
    assert qc.get_possible_output_qubits(0.85) == [0, 1]

    # first two are control, third is target
    qc.add_gate(
        depth=0.70, method_name="ccz", qubits=[("t2", 2), ("t2", 0), ("t1", 0)]
    )
    assert qc.get_possible_output_qubits(0.65) == [0, 1, 3, 5]

    # both are inputs and outputs
    qc.add_gate(
        depth=0.50, method_name="iswap", qubits=[("t2", 0), ("t2", 1)]
    )
    assert qc.get_possible_output_qubits(0.45) == [0, 1, 3, 4, 5]


    # first is control, second is target
    qc.add_gate(
        depth=0.3, method_name="ch", qubits=[("t1", 0), ("t2", 2)]
    )
    assert qc.get_possible_output_qubits(0.25) == [0, 1, 3, 4, 5]

    # first is control, second and third are target
    qc.add_gate(
        depth=0.10, method_name="cswap", qubits=[("t1", 2), ("t2", 2), ("t2", 1)]
    )
    assert qc.get_possible_output_qubits(0.25) == [0, 1, 2, 3, 4, 5]

    assert qc.output_qubits == [0, 1]
