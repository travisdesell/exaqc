from src.circuits.circuit import CircuitGenome
from src.evolution.mutation import enable_gate

from tests.innovation_validation import validate_innovation_numbers


def test_no_gates():
    """
    Creates a circuit genome with no gates and attempts to use the disable gate
    mutation. The method should return false.
    """

    qc = CircuitGenome(genome_number=1, registers={"test": 3})

    assert enable_gate(qc) is False

    validate_innovation_numbers(qc)


def test_all_disabled():
    """
    Creates a circuit genome with 3 gates which are all disabled.
    The mutation method should return true, and there should be
    one enabled gate and two disabled gates.
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

    assert enable_gate(qc) is True

    # there should be two enabled gates and one disabled gate
    enabled_count = 0
    for gate in qc.gates:
        if gate.enabled:
            enabled_count += 1

    assert enabled_count == 1

    validate_innovation_numbers(qc)


def test_all_enabled():
    """
    Creates a circuit genome with 3 gates which are all enabled.
    The mutation method should return False as there is no
    disabled gate to enable.
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

    # the mutation should return true because it modified a gate
    assert enable_gate(qc) is False

    validate_innovation_numbers(qc)


def test_one_enabled():
    """
    Creates a circuit genome with 3 gates, where only one is enabled. The
    mutation method should return True, and two gates should then
    be enabled.
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

    qc.gates[0].enabled = False
    qc.gates[1].enabled = False

    # the mutation should return true because it modified a gate
    assert enable_gate(qc) is True

    # there should be two enabled gates and one disabled gate
    enabled_count = 0
    for gate in qc.gates:
        if gate.enabled:
            enabled_count += 1

    assert enabled_count == 2

    validate_innovation_numbers(qc)
