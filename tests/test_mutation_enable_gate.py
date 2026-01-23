import pytest

from src.circuits.circuit import CircuitGenome
from src.evolution.mutation import enable_gate


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_no_gates_pennylane(target: str):
    """
    Creates a circuit genome with no gates and attempts to use the enable_gate
    mutation. Should return False.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3}, target="pennylane")
    assert enable_gate(qc) is False


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_all_disabled_pennylane(target: str):
    """
    Creates a circuit genome with 3 gates which are all disabled.
    The mutation should return True, enabling one gate.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3}, target="pennylane")

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

    assert enable_gate(qc) is True

    enabled_count = sum(1 for gate in qc.gates if gate.enabled)
    assert enabled_count == 1  # exactly one gate should now be enabled


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_all_enabled_pennylane(target: str):
    """
    Creates a circuit genome with 3 gates which are all enabled.
    The mutation should return False because no disabled gates exist.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3}, target="pennylane")

    qc.add_gate(
        depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)]
    )
    qc.add_gate(
        depth=0.40, method_name="ccx", qubits=[("test", 0), ("test", 2), ("test", 1)]
    )
    qc.add_gate(
        depth=0.40, method_name="cswap", qubits=[("test", 2), ("test", 1), ("test", 0)]
    )

    assert enable_gate(qc) is False


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_one_enabled_pennylane(target: str):
    """
    Creates a circuit genome with 3 gates, where only one is enabled.
    The mutation should return True, enabling one more gate.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 3}, target="pennylane")

    qc.add_gate(
        depth=0.40, method_name="ccz", qubits=[("test", 0), ("test", 1), ("test", 2)]
    )
    qc.add_gate(
        depth=0.40, method_name="ccx", qubits=[("test", 0), ("test", 2), ("test", 1)]
    )
    qc.add_gate(
        depth=0.40, method_name="cswap", qubits=[("test", 2), ("test", 1), ("test", 0)]
    )

    qc.gates[0].enabled = False
    qc.gates[1].enabled = False

    assert enable_gate(qc) is True

    enabled_count = sum(1 for gate in qc.gates if gate.enabled)
    assert enabled_count == 2  # now two gates should be enabled
