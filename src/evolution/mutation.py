import math
import random

from src.circuits.circuit import CircuitGenome
from src.circuits.gate import Gate
from src.circuits.gate_specifications import GateSpecifications


def disable_gate(circuit: CircuitGenome) -> bool:
    '''
    Selects a random enabled gate in the genome and disables it.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        True if the circuit was modified, False otherwise (e.g., if the
        genome had no enabled gates)
    '''

    enabled_gates = [gate for gate in circuit.gates if gate.enabled]

    if len(enabled_gates) == 0:
        # there were no enabled gates
        return False

    # select a random enabled gate and disable it
    random_gate = random.choice(enabled_gates)
    random_gate.enabled = False

    return True


def enable_gate(circuit: CircuitGenome) -> bool:
    '''
    Selects a random disabled gate in the genome and enables it.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        True if the circuit was modified, False otherwise (e.g., if the
        genome had no disabled gates)
    '''

    disabled_gates = [gate for gate in circuit.gates if not gate.enabled]

    if len(disabled_gates) == 0:
        # there were no disabled gates
        return False

    # select a random disabled gate and enable it
    random_gate = random.choice(disabled_gates)
    random_gate.enabled = True

    return True

def reorder_gate(circuit: CircuitGenome) -> bool:
    '''
    Selects a random gate and disables it. It then creates a copy of it 
    and inserts it (enabled) into the genome at a new random depth.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        True if the circuit was modified, False otherwise (e.g., if the
        genome had no gates)
    '''

    if len(circuit.gates) == 0:
        # there were no gates
        return False

    # select a random disabled gate and enable it
    random_gate = random.choice(circuit.gates)
    random_gate.enabled = False

    # create a copy of the randomly selected gate, enable it
    # and give it a new random depth
    new_gate = random_gate.copy()
    new_gate.enabled = True
    new_gate.depth = random.uniform(0.0, 1.0)

    circuit.add_existing_gate(new_gate)
    circuit.sort_gates()

    return True


def add_gate(gate_specifications: GateSpecifications, gate_method_name: str, circuit: CircuitGenome) -> bool:
    '''
    Adds a gate with the given method name to the given circuit genome
    at a random depth. By having the gate_method_name as the argument we can
    make multiple instances of this method to use later for dynamically
    selecting which gates to add at varying probabilities.

    Args:
        gate_method_name: is the gate method name so the gate information
            can be looked up in the gate_specifications dict.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        False if the gate method requires more qubits than are in the
        circuit, otherwise it should always return true
    '''

    specification = gate_specifications[gate_method_name]

    # get the parameter args (if any) otherwise set to an empty list
    gate_parameters = {}
    for parameter_name in specification.parameters:
        # generate a random angle as all parameter values are in radians
        gate_parameters[parameter_name] = random.uniform(-math.pi, math.pi)

    # create a register large enough for the gates input and
    # output qubits
    qubit_args = specification.qubits
    n_qubits = len(qubit_args)

    # make sure there are enough qubits in the quantum circuit to
    # be able to add the gate
    if len(circuit.qubits) < n_qubits:
        return False

    # randomly sample (without replacement) the number of qubits
    # required as inputs/outputs to this gate
    gate_qubits = random.sample(circuit.qubits, n_qubits)

    circuit.add_gate(depth=random.uniform(0.0, 1.0), method_name=gate_method_name, qubits=gate_qubits, parameters=gate_parameters)
    circuit.sort_gates()

    return True


