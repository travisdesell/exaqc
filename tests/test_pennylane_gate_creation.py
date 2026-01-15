import pytest
import random
import warnings
import pennylane as qml

from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications


@pytest.mark.parametrize("gate_method_name", pennylane_gate_specifications.keys())
def test_gate_creation_pennylane(gate_method_name: str):
    """
    Test that a single PennyLane gate can be added to a circuit and executed.
    
    Args:
        gate_method_name: the PennyLane gate method name from the specification dict.
    """
    print(f"testing gate: {gate_method_name}, type: {type(gate_method_name)}")

    specification = pennylane_gate_specifications[gate_method_name]
    print(f"gate specification: {specification}")

    if getattr(specification, "needs_validation", False):
        # skip these for now
        warnings.warn(
            f"skipping gate {gate_method_name} ({specification.name}) that needs validation"
        )
        return

    # determine number of qubits (register size)
    n_qubits = len(specification.qubits)

    # create a CircuitGenome with a single register
    qc = CircuitGenome(genome_number=1, registers={"test": n_qubits})

    # create qubit tuples for add_gate
    qc_qubits = [("test", i) for i in range(n_qubits)]
    print(f"qc_qubits: {qc_qubits}")

    # generate random parameters if gate is parametric
    qc_params = {param: random.random() for param in specification.parameters}
    print(f"qc_params: {qc_params}")

    # add the gate to the circuit genome
    qc.add_gate(depth=0.5, method_name=gate_method_name, qubits=qc_qubits, parameters=qc_params)

    # generate PennyLane circuit
    dev, qnode_fn = qc.generate_pennylane_circuit(measure_registers=False)

    # run the QNode to ensure no errors occur
    state = qnode_fn()
    print(f"Output state shape: {state.shape if hasattr(state, 'shape') else type(state)}\n")
