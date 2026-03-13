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


def qubit_swap(circuit: CircuitGenome, favor_enabled: bool = False) -> bool:
    """
    Selects a random gate and randomly replaces one qubit in the
    gate with a different qubit, not already in use by the gate. The
    gate being modified will be disabled and the new gate will be
    added with a different innovation number.  The depth of the new
    gate will be something randomly between the nearest gates on
    either side of the gate being mutated.

    Args:
        circuit: is the CircuitGenome to mutate
        favor_enabled: makes this mutation favored enabled gates
            when selecting a gate to apply on

    Returns:
        True if the circuit was modified, False otherwise, e.g., the
        genome had no gates using more than one qubit, or the genome
        had no gates using more than one qubit where there were spare
        qubits to swap to.
    """
    logger.debug("qubit swap mutation")

    if len(circuit.gates) == 0:
        # there were no gates
        logger.debug("no gates available")
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
        logger.debug("no possible gates available")
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
    if favor_enabled:
        if len(possible_enabled_gates) > 0:
            random_gate = random.choice(possible_enabled_gates)
        else:
            random_gate = random.choice(possible_disabled_gates)
    else:
        random_gate = random.choice(circuit.gates)

    random_gate.enabled = False

    # create a copy of the randomly selected gate, enable it
    # and give it a new random depth
    new_gate = random_gate.copy(new_innovation_number=True)
    new_gate.enabled = True

    logger.debug(f"selected gate {new_gate.method_name} with qubits: {new_gate.qubits}")

    # select a random qubit from the gate for replacement
    replace_qubit = random.choice(new_gate.qubits)

    # select a random qubit from the circuit that was not
    # in use by the gate.
    selection_qubits = circuit.qubits.copy()
    for qubit in new_gate.qubits:
        selection_qubits.remove(qubit)

    new_qubit = random.choice(selection_qubits)
    logger.debug(f"replacing qubit {replace_qubit} with {new_qubit}")
    logger.debug(f"gate qubits before replace {new_gate.qubits}")

    # replace the selected qubit with the new qubit
    new_gate.qubits[new_gate.qubits.index(replace_qubit)] = new_qubit
    logger.debug(f"gate qubits after replace {new_gate.qubits}")

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
    logger.debug(
        f"setting new gates depth randomly between {min_depth} and {max_depth}: {new_gate.depth}"
    )

    circuit.add_existing_gate(new_gate)
    circuit.sort_gates()

    return True


def reorder_gate(circuit: CircuitGenome, favor_enabled: bool = False) -> bool:
    """
    Selects a random gate and disables it. It then creates a copy of it
    and inserts it (enabled) into the genome at a new random depth.

    Args:
        circuit: is the CircuitGenome to mutate
        favor_enabled: makes this mutation favored enabled gates
            when selecting a gate to apply on

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
    if favor_enabled:
        if len(possible_enabled_gates) > 0:
            random_gate = random.choice(possible_enabled_gates)
        else:
            random_gate = random.choice(possible_disabled_gates)
    else:
        random_gate = random.choice(circuit.gates)

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
    depth: float = None,
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
        depth: is used by the add_gate_with_selection method, as that will
            need to predetermine the depth for gate addition.

    Returns:
        False if the gate method requires more qubits than are in the
        circuit, otherwise it should always return true
    """

    # make sure there are enough qubits in the quantum circuit to
    # be able to add the gate
    if len(circuit.qubits) < len(gate_specification.qubits):
        return False

    if depth is None:
        depth = random.uniform(0.0, 1.0)

    # get the possible output qubits so we know which gates we can use for
    # outputs
    possible_input_indexes = circuit.get_possible_input_qubits(depth=depth)
    possible_output_indexes = circuit.get_possible_output_qubits(depth=depth)

    n_qubits = len(gate_specification.qubits)
    gate_qubits = [None] * n_qubits
    if n_qubits == 1:
        # this will require selecting a qubit which is both an input and an
        # output for this gate to be effective

        possible_indexes = set(possible_input_indexes).intersection(
            possible_output_indexes
        )

        # select one of these qubits for the gate parameter
        gate_qubits[0] = circuit.qubits[random.choice(list(possible_indexes))]

    elif (
        gate_specification.input_qubit_indexes
        == gate_specification.output_qubit_indexes
    ):
        # this is a gate where all qubits are both inputs and outputs. need to make
        # sure that at least one qubit is from the inputs and one is from the outputs

        input_index = random.choice(possible_input_indexes)
        possible_input_indexes.remove(input_index)
        logger.debug(f"\tselected input index: {input_index}")
        if input_index in possible_output_indexes:
            possible_output_indexes.remove(input_index)

        if len(possible_output_indexes) == 0:
            logger.error(
                "There were not enough possible input indexes to add this gate. This shouldn't happen."
            )
            return False

        output_index = random.choice(possible_output_indexes)
        possible_output_indexes.remove(output_index)
        logger.debug(f"\tselected output index: {output_index}")
        if output_index in possible_input_indexes:
            possible_input_indexes.remove(output_index)

        indexes = [input_index, output_index]

        remaining_indexes = set(possible_input_indexes).union(possible_output_indexes)

        if len(indexes) < n_qubits:
            n_to_sample = n_qubits - len(indexes)
            indexes.extend(random.sample(remaining_indexes, n_to_sample))

        random.shuffle(indexes)

        logger.debug(f"\tselected the following indexes: {indexes}")

        for i, index in enumerate(indexes):
            gate_qubits[i] = circuit.qubits[index]

    else:
        # at least one target/output qubit should take a possible output qubit
        # and at least one control/input qubit should take a possible input qubit

        first = True

        for i in range(n_qubits):
            logger.debug(
                f"\tselecting qubit {i} of {n_qubits}, inputs: {possible_input_indexes}, "
                f"outputs: {possible_output_indexes}"
            )

            if i not in gate_specification.output_qubit_indexes:
                # this argument is a control qubit and needs to come from an input qubit
                index = random.choice(possible_input_indexes)
                logger.debug(
                    f"\t\tselected index {index} for an input only to be put at index {i}"
                )
                gate_qubits[i] = circuit.qubits[index]

                if index not in possible_input_indexes:
                    logger.error(
                        "There were not enough possible input indexes to add this gate. This shouldn't happen."
                    )
                    return False

                # remove this as a possible selection
                possible_input_indexes.remove(index)
                if index in possible_output_indexes:
                    possible_output_indexes.remove(index)
            else:
                # make sure the target gates go to possible output gates for at least
                # the first output

                index = None
                if first:
                    if len(possible_output_indexes) == 0:
                        # TODO: sometimes all the outputs get used up by the inputs by random
                        # chance, in this case we just need to try again.  there may be
                        # a smarter way to do this.
                        return False

                    index = random.choice(possible_output_indexes)
                else:
                    remaining_indexes = set(possible_input_indexes).union(
                        possible_output_indexes
                    )
                    index = random.choice(list(remaining_indexes))

                logger.debug(
                    f"\t\tselected index {index} for an output or input to be put at index {i} (first: {first})"
                )
                gate_qubits[i] = circuit.qubits[index]

                if first and index not in possible_output_indexes:
                    logger.error(
                        "There were not enough possible output indexes to add this gate. This shouldn't happen."
                    )
                    return False

                first = False

                # remove this as a possible selection
                if index in possible_output_indexes:
                    possible_output_indexes.remove(index)

                if index in possible_input_indexes:
                    possible_input_indexes.remove(index)

    # get the parameter args (if any) otherwise set to an empty list
    gate_parameters = {}
    for parameter_name in gate_specification.parameters:
        # generate a random angle as all parameter values are in radians
        gate_parameters[parameter_name] = random.uniform(-math.pi, math.pi)

    logger.debug(f"\tqubits are: {gate_qubits}")
    logger.debug(f"\tparameters are: {gate_parameters}")

    circuit.add_gate(
        depth=depth,
        method_name=gate_specification.method_name,
        qubits=gate_qubits,
        parameters=gate_parameters,
    )
    circuit.sort_gates()

    return True


def add_gate_with_selection(
    gate_specifications: list[GateSpecification],
    circuit: CircuitGenome,
) -> bool:
    """
    Adds a gate that will be effective in the quantum circuit.  This will first
    select a depth, and then given that depth determine which qubits can be
    possible inputs and possible outputs.

    Given the depth, the possible input and possible output qubits will determine
    which gates can be added. If the input and output qubits are disjoint, then
    only circuits which use multiple qubits can be added. If the input and output
    qubits overlap, then any gate could be added.

    Args:
        gate_specifications: are all the possible gate specifications for the
            gate types which could potentially be added to the circuit.
        circuit: is the CircuitGenome to mutate

    Returns:
        False if the gate method requires more qubits than are in the
        circuit, otherwise it should always return true
    """

    depth = random.uniform(0.0, 1.0)

    # get the possible input and output qubits so we know which gates we can use
    possible_input_indexes = circuit.get_possible_input_qubits(depth=depth)
    possible_output_indexes = circuit.get_possible_output_qubits(depth=depth)

    logger.debug(
        f"selecting appropriate add gates at depth {depth} with input indexes {possible_input_indexes} "
        f"and output indexes {possible_output_indexes}"
    )

    # the two sets share some qubits
    if not set(possible_input_indexes).isdisjoint(possible_output_indexes):
        logger.debug("input and output indexes share qubits, any gate allowed")
        gate_specification = random.choice(gate_specifications)

        logger.debug(f"\tattempting with selected {gate_specification}")
        return add_gate(gate_specification, circuit)

    else:
        logger.debug(
            "input and output indexes do not share qubits, use only multi-qubit gates"
        )

        possible_gate_specifications = []
        for gate in gate_specifications:
            if len(gate.qubits) > 1:
                possible_gate_specifications.append(gate)

        gate_specification = random.choice(possible_gate_specifications)

        logger.debug(f"\tattempting with selected {gate_specification}")
        return add_gate(gate_specification, circuit)
