import pytest
import torch
from src.circuits.circuit import CircuitGenome


def test_pennylane_example_circuit_full_stack():
    """
    Integration test:
    Build a non-trivial circuit using CircuitGenome and ensure
    PennyLane circuit generation + execution works end-to-end.
    """
    qc = CircuitGenome(genome_number=1, registers={"a": 3, "b": 5}, target="pennylane")

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

    n_qubits = sum(qc.registers.values())
    input_bits = torch.zeros(n_qubits, dtype=torch.int64)

    torch_params = {
        f"{gate.innovation_number}:{name}": torch.tensor(value, dtype=torch.float64)
        for gate in qc.gates
        for name, value in gate.parameters.items()
    }

    # ---- Generate PennyLane circuit ----
    try:
        dev, qnode_fn = qc.generate_pennylane_circuit(measure_registers=False)
    except Exception as e:
        pytest.fail(f"Failed to generate PennyLane circuit: {e}")

    # ---- Execute ----
    try:
        state = qnode_fn(input_bits, torch_params)
    except Exception as e:
        pytest.fail(f"Execution failed: {e}")

    # ---- Basic sanity checks ----
    assert state is not None, "Returned state is None"
    assert hasattr(state, "shape"), "Returned object has no shape (not a state vector?)"
    assert (
        state.shape[0] == 2**8
    ), f"Expected statevector of size 256, got {state.shape}"

    print("\n✅ PennyLane example circuit executed successfully")
    print(f"State shape: {state.shape}")
