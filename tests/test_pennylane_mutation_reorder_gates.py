from src.circuits.circuit import CircuitGenome
from src.evolution.mutation import reorder_gate


def test_all_disabled_pennylane():
    """
    Creates a circuit genome with 3 gates which are all disabled.
    The reorder_gate method should return True as it copies one
    gate and inserts a new enabled gate at a random depth.
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3})

    qc.add_gate(
        depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)]
    )
    qc.add_gate(
        depth=0.40, method_name="ccx", qubits=[("test", 0), ("test", 2), ("test", 1)]
    )
    qc.add_gate(
        depth=0.40, method_name="cswap", qubits=[("test", 2), ("test", 1), ("test", 0)]
    )

    for gate in qc.gates:
        gate.enabled = False

    assert reorder_gate(qc) is True

    # There should now be 4 gates in the circuit
    assert len(qc.gates) == 4

    # Only one gate should be enabled
    enabled_count = sum(1 for gate in qc.gates if gate.enabled)
    assert enabled_count == 1


def test_all_enabled_pennylane():
    """
    Creates a circuit genome with 3 gates which are all enabled.
    The reorder_gate method should return True as it copies one
    gate and inserts it as a new enabled gate at a different depth.
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3})

    qc.add_gate(
        depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)]
    )
    qc.add_gate(
        depth=0.40, method_name="ccx", qubits=[("test", 0), ("test", 2), ("test", 1)]
    )
    qc.add_gate(
        depth=0.40, method_name="cswap", qubits=[("test", 2), ("test", 1), ("test", 0)]
    )

    assert reorder_gate(qc) is True

    # There should now be 4 gates in the circuit
    assert len(qc.gates) == 4

    # Three gates should remain enabled, one copied gate should be disabled
    enabled_count = sum(1 for gate in qc.gates if gate.enabled)
    assert enabled_count == 3
