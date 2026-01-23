import pytest
import warnings
from src.circuits.circuit import CircuitGenome
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.evolution.mutation import add_gate


@pytest.mark.parametrize("gate_method_name", pennylane_gate_specifications.keys())
def test_gate_creation_pennylane(gate_method_name: str):
    """
    Tests the add_gate mutation for each possible PennyLane gate, individually.
    """
    specification = pennylane_gate_specifications[gate_method_name]

    if getattr(specification, "needs_validation", False):
        warnings.warn(
            f"Skipping gate {gate_method_name} ({specification.name}) that needs validation"
        )
        return

    qc = CircuitGenome(genome_number=1, registers={"test": 10}, target="pennylane")
    add_gate(pennylane_gate_specifications[gate_method_name], qc)

    # one gate should have been added
    assert len(qc.gates) == 1


@pytest.mark.parametrize("gate_method_name", pennylane_gate_specifications.keys())
def test_qubit_requirements_pennylane(gate_method_name: str):
    """
    Tests add_gate for each gate method to make sure it returns False
    if the circuit doesn't have enough qubits.
    """
    specification = pennylane_gate_specifications[gate_method_name]

    if getattr(specification, "needs_validation", False):
        warnings.warn(
            f"Skipping gate {gate_method_name} ({specification.name}) that needs validation"
        )
        return

    n_qubits = len(specification.qubits)

    # iterate up to a circuit size large enough to add the gate
    for i in range(n_qubits + 2):
        qc = CircuitGenome(genome_number=1, registers={"test": i}, target="pennylane")
        success = add_gate(pennylane_gate_specifications[gate_method_name], qc)

        if i < n_qubits:
            assert success is False
        else:
            assert success is True


def test_all_gates_one_register_pennylane():
    """
    Creates a single-register circuit and adds all possible PennyLane gates.
    """
    qc = CircuitGenome(genome_number=1, registers={"test": 10}, target="pennylane")

    gate_count = 0
    for gate_method_name, gate_specs in pennylane_gate_specifications.items():
        if not getattr(gate_specs, "needs_validation", False):
            add_gate(pennylane_gate_specifications[gate_method_name], qc)
            gate_count += 1

    # the number of gates in the circuit should match the number of mutation calls
    assert len(qc.gates) == gate_count


def test_all_gates_two_registers_pennylane():
    """
    Creates a two-register circuit and adds all possible PennyLane gates.
    """
    qc = CircuitGenome(
        genome_number=1, registers={"test1": 5, "test2": 5}, target="pennylane"
    )

    gate_count = 0
    for gate_method_name, gate_specs in pennylane_gate_specifications.items():
        if not getattr(gate_specs, "needs_validation", False):
            add_gate(pennylane_gate_specifications[gate_method_name], qc)
            gate_count += 1

    assert len(qc.gates) == gate_count
