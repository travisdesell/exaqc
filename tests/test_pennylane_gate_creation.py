import pytest
import random
import pennylane as qml

from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications


@pytest.mark.parametrize("gate_method_name", list(pennylane_gate_specifications.keys()))
def test_gate_creation_pennylane(gate_method_name: str):
    """
    Test that a single PennyLane gate can be added to a CircuitGenome and executed.
    Supports multi-qubit, controlled, parametric, and decomposed gates.
    """

    # Get the PL-friendly gate specification
    spec = pennylane_gate_specifications[gate_method_name]

    # Skip gates that require validation
    if getattr(spec, "needs_validation", False):
        pytest.skip(f"Skipping gate {gate_method_name} ({spec.name}) that needs validation")

    print(f"\nTesting gate: {gate_method_name}")
    print(f"Specification: {spec}")

    # Number of qubits needed for this gate
    n_qubits = getattr(spec, "n_qubits", 1)
    print(f"n_qubits: {n_qubits}")

    # Create a CircuitGenome with a single register
    qc = CircuitGenome(genome_number=1, registers={"test": n_qubits})

    # Build qubit tuples (always 0..n_qubits-1 for the register)
    qc_qubits = [("test", i) for i in range(n_qubits)]
    print(f"qc_qubits: {qc_qubits}")

    # Generate random parameters if gate is parametric
    qc_params = {param_name: random.random() for param_name in (spec.parameters or [])}
    print(f"qc_params: {qc_params}")

    # Add the gate to the circuit genome
    try:
        qc.add_gate(
            depth=0.5,
            method_name=gate_method_name,
            qubits=qc_qubits,
            parameters=qc_params,
        )
    except Exception as e:
        pytest.fail(f"Failed to add gate {gate_method_name}: {e}")

    # Generate PennyLane QNode
    try:
        dev, qnode_fn = qc.generate_pennylane_circuit(measure_registers=False)
    except Exception as e:
        pytest.fail(f"Failed to generate PennyLane circuit for {gate_method_name}: {e}")

    # Run the QNode to ensure execution works
    try:
        state = qnode_fn()
    except Exception as e:
        pytest.fail(f"Execution failed for gate {gate_method_name}: {e}")

    # Basic sanity check for output state
    assert state is not None
    assert hasattr(state, "shape"), "Output state has no 'shape' attribute"

    print(
        f"Gate {gate_method_name} executed successfully. "
        f"Output state shape: {state.shape}"
    )
