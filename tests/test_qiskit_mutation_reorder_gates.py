from src.circuits.circuit import CircuitGenome
from src.evolution.mutation import reorder_gate


def test_all_disabled():
    """
    Creates a circuit genome with 3 gates which are all disabled.
    The reorder_gate method should return true as it will copy one
    and add a new enabled gate at a different depth.
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

    # there should be four gates in the circuit now
    assert len(qc.gates) == 4

    # there should be one enabled gates and three disabled gates
    enabled_count = 0
    for gate in qc.gates:
        if gate.enabled:
            enabled_count += 1

    assert enabled_count == 1


def test_all_enabled():
    """
    Creates a circuit genome with 3 gates which are all enabled.
    The reorder_gate method should return true as it will copy one
    and add a new enabled gate at a different depth.
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

    # there should be four gates in the circuit now
    assert len(qc.gates) == 4

    # there should be three enabled gates and one disabled gate
    enabled_count = 0
    for gate in qc.gates:
        if gate.enabled:
            enabled_count += 1

    assert enabled_count == 3
