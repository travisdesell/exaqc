import pytest
from src.circuits.circuit import CircuitGenome
from src.evolution.mutation import disable_gate


def test_no_gates_pennylane():
    """
    Creates a circuit genome with no gates and attempts to use the disable_gate
    mutation. Should return False.
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3})
    assert disable_gate(qc) is False


def test_all_disabled_pennylane():
    """
    Creates a circuit genome with 3 gates which are all disabled.
    The mutation should return False.
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3})

    qc.add_gate(depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)])
    qc.add_gate(depth=0.40, method_name="ccx", qubits=[("test", 0), ("test", 2), ("test", 1)])
    qc.add_gate(depth=0.40, method_name="cswap", qubits=[("test", 2), ("test", 1), ("test", 0)])

    for gate in qc.gates:
        gate.enabled = False

    assert disable_gate(qc) is False


def test_all_enabled_pennylane():
    """
    Creates a circuit genome with 3 gates which are all enabled.
    The mutation should return True and one gate should be disabled.
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3})

    qc.add_gate(depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)])
    qc.add_gate(depth=0.40, method_name="ccx", qubits=[("test", 0), ("test", 2), ("test", 1)])
    qc.add_gate(depth=0.40, method_name="cswap", qubits=[("test", 2), ("test", 1), ("test", 0)])

    assert disable_gate(qc) is True

    enabled_count = sum(1 for gate in qc.gates if gate.enabled)
    assert enabled_count == 2  # one gate should now be disabled


def test_one_enabled_pennylane():
    """
    Creates a circuit genome with 3 gates where only one is enabled.
    The mutation should return True and all gates should then be disabled.
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3})

    qc.add_gate(depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)])
    qc.add_gate(depth=0.40, method_name="ccx", qubits=[("test", 0), ("test", 2), ("test", 1)])
    qc.add_gate(depth=0.40, method_name="cswap", qubits=[("test", 2), ("test", 1), ("test", 0)])

    # Only the last gate is enabled
    qc.gates[0].enabled = False
    qc.gates[1].enabled = False

    assert disable_gate(qc) is True

    enabled_count = sum(1 for gate in qc.gates if gate.enabled)
    assert enabled_count == 0  # all gates should now be disabled
