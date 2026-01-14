import inspect
import pytest
import random
import warnings


from src.circuits.circuit import CircuitGenome
from src.circuits.gate_specifications import gate_specifications
from src.evolution.mutation import add_gate


@pytest.mark.parametrize("gate_method_name", gate_specifications)
def test_gate_creation(gate_method_name: str):
    '''
    This uses the gate specifications dict to test the add_gate mutation for
    each possible gate, individually.

    Args:
        gate_method_name: is the qiskit gate method name that can be applied
            to the QuantumCircuit object.
    '''
    print(f"testing add gate for: {gate_method_name}, type: {type(gate_method_name)}")

    qc = CircuitGenome(genome_number=1, registers={"test" : 10})
    add_gate(gate_method_name, qc)

    assert len(qc.gates) == 1

@pytest.mark.parametrize("gate_method_name", gate_specifications)
def test_qubit_requirements(gate_method_name: str):
    '''
    This tests add_gate for each gate method to make sure it returns
    false if the circuit doesn't have enough qubits for the gate.

    Args:
        gate_method_name: is the qiskit gate method name that can be applied
            to the QuantumCircuit object.
    '''

    specification = gate_specifications[gate_method_name]
    qubit_args = specification['qubits']
    n_qubits = len(qubit_args)

    # print(f"testing gate qubit requirements: {gate_method_name}, type: {type(gate_method_name)}, requires {n_qubits}")

    # iterate up to a circuit size large enough to be able to add this gate
    for i in range(n_qubits+2):
        # print(f"\ti is: {i}")
        qc = CircuitGenome(genome_number=1, registers={"test" : i})
        success = add_gate(gate_method_name, qc)
        # print(f"\tsuccess? : {success}")

        if i < n_qubits:
            assert success == False
        else:
            assert success == True


def test_all_gates_one_register():
    '''
    Creates an empty circuit and then attempts to add each possible gate to it using
    the add gate mutation.
    '''

    qc = CircuitGenome(genome_number=1, registers={"test" : 10})

    gate_count = 0
    for gate_method_name, gate_specs in gate_specifications.items():
        if not 'needs_validation' in gate_specs.keys():
            add_gate(gate_method_name, qc)
            gate_count += 1


    # we should have the same number of gates in the quantum circuit
    # as add gate mutation calls
    assert len(qc.gates) == gate_count


def test_all_gates_two_registers():
    '''
    Creates an empty circuit with two registers and then attempts to add 
    each possible gate to it using the add gate mutation.
    '''

    qc = CircuitGenome(genome_number=1, registers={"test1" : 5, "test2" : 5})

    gate_count = 0
    for gate_method_name, gate_specs in gate_specifications.items():
        if not 'needs_validation' in gate_specs.keys():
            add_gate(gate_method_name, qc)
            gate_count += 1


    # we should have the same number of gates in the quantum circuit
    # as add gate mutation calls
    assert len(qc.gates) == gate_count
