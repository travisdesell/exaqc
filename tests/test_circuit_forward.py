import pytest
import torch

from src.circuits.circuit import CircuitGenome
from src.circuits.encoder import initialize_encoder
from src.circuits.decoder import initialize_decoder
from src.circuits.registers import expand_registers


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_example_circuit_full_stack(target: str):
    """
    Integration test:
    Build a non-trivial circuit using CircuitGenome and ensure
    PennyLane circuit generation + execution works end-to-end.
    """
    qc = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"a": 3, "b": 5}),
        target=target,
    )

    qc.hyperparameters = {
        "steps": 30,
        "learning_rate": 0.005,
        "log_every": 15,
        "batch_size": 12,
        "quantum_input_mode": "u3",
        "quantum_output_mode": "probs",
    }

    # create a linear encoder which also needs to serialized weights
    n_qubits = len(qc.input_indexes)
    qc.encoder = initialize_encoder(
        target=target, encoding_str="linear", n_inputs=n_qubits, n_outputs=n_qubits * 3
    )

    # create a linear decoder which also needs to serialized weights
    qc.decoder = initialize_decoder(
        target=target, decoding_str="linear", n_inputs=2**n_qubits, n_outputs=n_qubits
    )

    qc.add_gate(depth=0.05, method_name="x", qubits=[("a", 1)])
    qc.add_gate(depth=0.10, method_name="x", qubits=[("b", 1)])
    qc.add_gate(depth=0.15, method_name="x", qubits=[("b", 2)])
    qc.add_gate(depth=0.20, method_name="x", qubits=[("b", 4)])

    qc.add_gate(depth=0.25, method_name="h", qubits=[("a", 0)])
    qc.add_gate(depth=0.30, method_name="h", qubits=[("b", 1)])

    qc.add_gate(
        depth=0.31, method_name="rx", qubits=[("b", 1)], parameters={"theta": 0.2}
    )

    qc.add_gate(
        depth=0.35,
        method_name="cp",
        qubits=[("b", 3), ("a", 0)],
        parameters={"theta": 0.3},
    )

    qc.add_gate(depth=0.40, method_name="ccz", qubits=[("b", 0), ("b", 1), ("b", 3)])
    qc.add_gate(depth=0.41, method_name="cswap", qubits=[("b", 0), ("b", 1), ("b", 2)])
    qc.add_gate(depth=0.42, method_name="cswap", qubits=[("b", 2), ("b", 3), ("b", 4)])
    qc.add_gate(depth=0.43, method_name="cswap", qubits=[("b", 3), ("b", 4), ("b", 0)])

    n_qubits = len(qc.qubits)
    input_bits = torch.zeros(n_qubits, dtype=torch.int64)

    # ---- Generate circuit ----
    try:
        qc.initialize_model()

    except Exception as e:
        pytest.fail(
            f"Failed to initialize quantum circuit model for target {target}: {e}"
        )

    # ---- Perform forward pass through circuit----
    output = None
    try:
        output = qc.forward(input_bits)
    except Exception as e:
        pytest.fail(f"Forward pass on quantum circuit for target {target} failed: {e}")

    # ---- Basic sanity checks ----
    assert output is not None, "Returned output tensor is None"
    assert hasattr(
        output, "shape"
    ), "Returned object has no shape (not a state vector?)"
    assert (
        len(output) == n_qubits
    ), f"Expected output tensor of size {n_qubits}, got tensor of shape {output.shape}"

    print("\n✅ {target} example circuit executed successfully")
