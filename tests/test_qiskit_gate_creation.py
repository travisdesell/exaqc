import pytest
import random
import torch
import warnings

from src.circuits.circuit import CircuitGenome
from src.circuits.encoder import initialize_encoder
from src.circuits.decoder import initialize_decoder
from src.circuits.registers import expand_registers
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications

from tests.innovation_validation import validate_innovation_numbers


@pytest.mark.parametrize("gate_method_name", qiskit_gate_specifications.keys())
def test_gate_creation(gate_method_name: str):
    """
    This uses the gate specifications dict to test creation of quantum circuits
    with each gate, individually.

    Args:
        gate_method_name: is the qiskit gate method name that can be applied
            to the QuantumCircuit object.
    """
    print(f"testing gate: {gate_method_name}, type: {type(gate_method_name)}")

    specification = qiskit_gate_specifications[gate_method_name]

    print(f"gate specification: {specification}")

    if specification.needs_validation:
        # skip these for now
        warnings.warn(
            f"skipping gate {gate_method_name} ({specification.name}) that needs validation"
        )
        return

    # create a register large enough for the gates input and
    # output qubits
    qubit_args = specification.qubits
    n_qubits = len(qubit_args)

    qc = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"test": n_qubits}),
        target="qiskit",
    )

    # set up default hyperparameters for output generation
    qc.hyperparameters = {
        "quantum_input_mode": "rz",
        "quantum_output_mode": "probs",
    }

    print(f"qubit_args: {qubit_args}")
    print(f"n_qubits: {n_qubits}")

    # create a basic encoder and decoder
    qc.encoder = initialize_encoder(
        target="pennylane", encoding_str="linear", n_inputs=n_qubits, n_outputs=n_qubits
    )
    qc.decoder = initialize_decoder(
        target="pennylane",
        decoding_str="linear",
        n_inputs=2**n_qubits,
        n_outputs=n_qubits,
    )

    # make the lost of qubit tuples for the add_gate method
    qc_qubits = []
    for i in range(len(qubit_args)):
        qc_qubits.append(("test", i))
    print(f"qc_qubits: {qc_qubits}")

    # make the dict of parameters (if there are parameters) for the add_gate_method
    qc_params = {}
    param_args = specification.parameters

    for parameter in param_args:
        qc_params[parameter] = random.random()

    print(f"qc_params: {qc_params}")

    qc.add_gate(
        depth=0.5, method_name=gate_method_name, qubits=qc_qubits, parameters=qc_params
    )

    validate_innovation_numbers(qc)

    input_bits = torch.zeros(n_qubits, dtype=torch.int64)
    print(f"input bits: {input_bits}")

    try:
        print("generating qiskit circuit")
        qc.initialize_model()
    except Exception as e:
        pytest.fail(f"Execution failed for gate {gate_method_name}: {e}")

    # Run the circuit to ensure execution works
    try:
        output = qc.forward(input_bits)
    except Exception as e:
        pytest.fail(f"Execution failed for gate {gate_method_name}: {e}")

    # Basic sanity check for output
    assert output is not None
    assert hasattr(output, "shape"), "Output output has no 'shape' attribute"

    print(
        f"Gate {gate_method_name} executed successfully. "
        f"Output output shape: {output.shape}"
    )
    print()
