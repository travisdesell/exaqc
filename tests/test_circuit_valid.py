import pytest

from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers


@pytest.mark.parametrize("target", ["qiskit", "pennylane"])
def test_all_disabled_pennylane(target: str):
    """
    Creates a circuit genome with 3 gates which are all disabled.
    The mutation should return False.

    Args:
        target: is the target framework (qiskit or pennylane)
    """
    qc = CircuitGenome(
        genome_number=1,
        input_qubits=expand_registers({"i": 3}),
        output_qubits=expand_registers({"o": 3}),
        target=target,
    )

    assert not qc.is_valid()

    # cswap is one control two target
    qc.add_gate(depth=0.30, method_name="cswap", qubits=[("i", 0), ("i", 1), ("i", 2)])
    # this does not connect an input to an output so the circuit is now valid
    assert not qc.is_valid()

    # ccz is two control one target
    qc.add_gate(depth=0.40, method_name="ccz", qubits=[("i", 2), ("o", 1), ("o", 2)])

    # this connects an input to an output so the circuit is now valid
    assert qc.is_valid()
