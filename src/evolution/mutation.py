import math
import random

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.gate_specifications import GateSpecification


def disable_gate(circuit: CircuitGenome) -> bool:
    """
    Selects a random enabled gate in the genome and disables it.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        True if the circuit was modified, False otherwise (e.g., if the
        genome had no enabled gates)
    """

    enabled_gates = [gate for gate in circuit.gates if gate.enabled]

    if len(enabled_gates) == 0:
        # there were no enabled gates
        return False

    # select a random enabled gate and disable it
    random_gate = random.choice(enabled_gates)
    random_gate.enabled = False

    return True


def enable_gate(circuit: CircuitGenome) -> bool:
    """
    Selects a random disabled gate in the genome and enables it.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        True if the circuit was modified, False otherwise (e.g., if the
        genome had no disabled gates)
    """

    disabled_gates = [gate for gate in circuit.gates if not gate.enabled]

    if len(disabled_gates) == 0:
        # there were no disabled gates
        return False

    # select a random disabled gate and enable it
    random_gate = random.choice(disabled_gates)
    random_gate.enabled = True

    return True


def qubit_swap(circuit: CircuitGenome) -> bool:
    """
    Selects a random gate and randomly replaces one qubit in the
    gate with a different qubit, not already in use by the gate. The
    gate being modified will be disabled and the new gate will be
    added with a different innovation number.  The depth of the new
    gate will be something randomly between the nearest gates on
    either side of the gate being mutated.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        True if the circuit was modified, False otherwise, e.g., the
        genome had no gates using more than one qubit, or the genome
        had no gates using more than one qubit where there were spare
        cubits to swap to.
    """
    logger.info("qubit swap mutation")

    if len(circuit.gates) == 0:
        # there were no gates
        logger.info("no gates available")
        return False

    possible_gates = []
    for gate in circuit.gates:
        # a gate can be used if it has more than one qubit and
        # there is at least one extra qubit that in the circuit which
        # it is not using (that a gate could swap to).
        if len(gate.qubits) > 0 and len(gate.qubits) < len(circuit.qubits):
            possible_gates.append(gate)

    if len(possible_gates) == 0:
        # no possible gates to mutate
        logger.info("no possible gates available")
        return False

    possible_enabled_gates = []
    possible_disabled_gates = []

    for gate in possible_gates:
        if gate.enabled:
            possible_enabled_gates.append(gate)
        else:
            possible_disabled_gates.append(gate)

    # select a random gate, make a copy of it for the
    # mutation and disable the original one. Select an
    # enabled gate if possible
    random_gate = None
    if len(possible_enabled_gates) > 0:
        random_gate = random.choice(possible_enabled_gates)
    else:
        random_gate = random.choice(possible_disabled_gates)

    random_gate.enabled = False

    # create a copy of the randomly selected gate, enable it
    # and give it a new random depth
    new_gate = random_gate.copy(new_innovation_number=True)
    new_gate.enabled = True

    logger.info(f"selected gate {new_gate.method_name} with qubits: {new_gate.qubits}")

    # select a random qubit from the gate for replacement
    replace_qubit = random.choice(new_gate.qubits)

    # select a random qubit from the circuit that was not
    # in use by the gate.
    selection_qubits = circuit.qubits.copy()
    for qubit in new_gate.qubits:
        selection_qubits.remove(qubit)

    new_qubit = random.choice(selection_qubits)
    logger.info(f"replacing qubit {replace_qubit} with {new_qubit}")
    logger.info(f"gate qubits before replace {new_gate.qubits}")

    # replace the selected qubit with the new qubit
    new_gate.qubits[new_gate.qubits.index(replace_qubit)] = new_qubit
    logger.info(f"gate qubits after replace {new_gate.qubits}")

    # determine the new depth for the new gate, as it shouldn't be the exact same
    # as the parent gate but nearby. find the gates closest in depth on either side
    # or use 0 or 1 as the bounds if a gate isn't there and select randomly between
    # those bounds.

    min_depth = 0.0
    max_depth = 1.0

    for gate in circuit.gates:
        if gate.depth > min_depth and gate.depth < new_gate.depth:
            min_depth = gate.depth

        if gate.depth < max_depth and gate.depth > new_gate.depth:
            max_depth = gate.depth

    new_gate.depth = random.uniform(min_depth, max_depth)
    logger.info(
        f"setting new gates depth randomly between {min_depth} and {max_depth}: {new_gate.depth}"
    )

    circuit.add_existing_gate(new_gate)
    circuit.sort_gates()

    return True


def reorder_gate(circuit: CircuitGenome) -> bool:
    """
    Selects a random gate and disables it. It then creates a copy of it
    and inserts it (enabled) into the genome at a new random depth.

    Args:
        circuit: is the CircuitGenome to mutate

    Returns:
        True if the circuit was modified, False otherwise (e.g., if the
        genome had no gates)
    """

    if len(circuit.gates) == 0:
        # there were no gates
        return False

    possible_enabled_gates = []
    possible_disabled_gates = []

    for gate in circuit.gates:
        if gate.enabled:
            possible_enabled_gates.append(gate)
        else:
            possible_disabled_gates.append(gate)

    # select a random gate, make a copy of it for the
    # mutation and disable the original one. Select an
    # enabled gate if possible
    random_gate = None
    if len(possible_enabled_gates) > 0:
        random_gate = random.choice(possible_enabled_gates)
    else:
        random_gate = random.choice(possible_disabled_gates)

    # disable the selected gate
    random_gate.enabled = False

    # create a copy of the randomly selected gate, enable it
    # and give it a new random depth
    new_gate = random_gate.copy(new_innovation_number=True)
    new_gate.enabled = True
    new_gate.depth = random.uniform(0.0, 1.0)

    circuit.add_existing_gate(new_gate)
    circuit.sort_gates()

    return True


def add_gate(
    gate_specification: GateSpecification,
    circuit: CircuitGenome,
) -> bool:
    """
    Adds a gate with the given method name to the given circuit genome
    at a random depth. By having the gate_method_name as the argument we can
    make multiple instances of this method to use later for dynamically
    selecting which gates to add at varying probabilities.

    Args:
        gate_specification: is the GateSpecification for the new gate to
            be added to the provided circuit
        circuit: is the CircuitGenome to mutate

    Returns:
        False if the gate method requires more qubits than are in the
        circuit, otherwise it should always return true
    """

    # get the parameter args (if any) otherwise set to an empty list
    gate_parameters = {}
    for parameter_name in gate_specification.parameters:
        # generate a random angle as all parameter values are in radians
        gate_parameters[parameter_name] = random.uniform(-math.pi, math.pi)

    # create a register large enough for the gates input and
    # output qubits
    qubit_args = gate_specification.qubits
    n_qubits = len(qubit_args)

    # make sure there are enough qubits in the quantum circuit to
    # be able to add the gate
    if len(circuit.qubits) < n_qubits:
        return False

    # randomly sample (without replacement) the number of qubits
    # required as inputs/outputs to this gate
    gate_qubits = random.sample(circuit.qubits, n_qubits)

    circuit.add_gate(
        depth=random.uniform(0.0, 1.0),
        method_name=gate_specification.method_name,
        qubits=gate_qubits,
        parameters=gate_parameters,
    )
    circuit.sort_gates()

    return True
