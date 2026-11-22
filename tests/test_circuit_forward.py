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


def build_minimal_genome(target: str, quantum_output_mode: str) -> CircuitGenome:
    """Builds a minimal purely quantum genome with the given readout mode.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.
        quantum_output_mode: The readout mode to configure.

    Returns:
        An uninitialized :class:`CircuitGenome`.
    """

    genome = CircuitGenome(
        genome_number=0,
        target=target,
        input_qubits=expand_registers({"q": 2}),
    )
    genome.hyperparameters = {
        "quantum_input_mode": "ry",
        "quantum_output_mode": quantum_output_mode,
    }
    genome.encoder = None
    genome.decoder = None
    genome.add_gate(depth=0.5, method_name="cx", qubits=[("q", 0), ("q", 1)])
    return genome


def test_qiskit_rejects_unimplemented_output_modes() -> None:
    """The qiskit backend refuses readout modes it does not implement.

    ``expval`` is a recognized mode that only pennylane builds a model for. The
    backend used to leave ``torch_model`` unset for it, which surfaced much
    later as an opaque ``AttributeError`` on a ``None`` model.
    """

    genome = build_minimal_genome("qiskit", "expval")

    with pytest.raises(NotImplementedError, match="not implemented for the qiskit"):
        genome.initialize_model()


@pytest.mark.parametrize("target", ["pennylane", "qiskit"])
def test_unknown_output_mode_is_rejected(target: str) -> None:
    """An unrecognized readout mode is a ValueError, not a backend gap.

    ``state`` is included because the full-statevector readout was removed: it
    is now simply an unknown mode on both targets.

    Args:
        target: The quantum backend under test.
    """

    for quantum_output_mode in ("not_a_mode", "state"):
        genome = build_minimal_genome(target, quantum_output_mode)

        with pytest.raises(ValueError, match="nknown quantum_output_mode"):
            genome.initialize_model()


@pytest.mark.parametrize("target", ["pennylane", "qiskit"])
def test_probs_output_mode_still_builds(target: str) -> None:
    """The supported readout mode keeps working on both targets.

    Args:
        target: The quantum backend under test.
    """

    genome = build_minimal_genome(target, "probs")
    genome.initialize_model()

    output = genome.forward(torch.rand(genome.n_quantum_inputs()))

    assert output.shape == (genome.n_quantum_outputs(),)
