"""The unitary a genome's evolved gates implement.

A genome's model is ``readout(A · E(x))``: the input encoding ``E`` is applied
first and only there (the data is never re-uploaded), then every enabled gate in
depth order, then the measurement. The evolved ansatz ``A`` therefore does not
depend on the data, and it is what this module computes -- the encoding and the
measurement are left out, and quantum dropout (a training-time mask) is ignored.

The gates are replayed through the very methods the model is built with
(:meth:`~src.circuits.gate.Gate.add_to_pennylane_circuit` and
:meth:`~src.circuits.gate.Gate.add_to_qiskit_circuit`), with the parameter
values the genome was saved with. Trainers write their best weights back into
the gates before a genome is archived, so for a trained genome these are its
trained values.

Matrices use one convention whatever the genome's target: the first qubit of the
qubit list is the most significant bit (PennyLane's ordering; qiskit's
little-endian operators are reversed to match).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from src.circuits.gate import Gate

#: A qubit, as a genome names it: its register name and index in the register.
Qubit = tuple[str, int]

#: The most qubits a unitary is computed over. A unitary over ``n`` qubits holds
#: ``4**n`` complex values (1 MiB at 8 qubits), and every genome of a run needs
#: one, so larger circuits are refused rather than exhausting memory.
MAX_UNITARY_QUBITS = 8


def genome_qubits(genome: dict[str, Any]) -> list[Qubit]:
    """Lists the qubits a serialized genome's circuit runs on.

    Mirrors :class:`~src.circuits.circuit.CircuitGenome`: the sorted union of
    its input and output qubits.

    Args:
        genome: A serialized genome (``CircuitGenome.to_dict``).

    Returns:
        The qubits, sorted.
    """

    qubits = {tuple(qubit) for qubit in genome.get("input_qubits") or []}
    qubits.update(tuple(qubit) for qubit in genome.get("output_qubits") or [])
    return sorted(qubits)


def enabled_gates(genome: dict[str, Any]) -> list[Gate]:
    """Rebuilds a serialized genome's enabled gates in the order they are applied.

    Args:
        genome: A serialized genome (``CircuitGenome.to_dict``).

    Returns:
        The enabled gates, sorted by depth and then innovation number, as
        :meth:`~src.circuits.circuit.CircuitGenome.sort_gates` orders them.
    """

    # Imported here so that the rest of the package (and the dashboard that
    # uses it) does not load the quantum frameworks until a unitary is needed.
    from src.circuits.gate import Gate

    gates = [
        Gate.from_dict({**gate, "qubits": [tuple(qubit) for qubit in gate["qubits"]]})
        for gate in genome.get("gates") or []
        if gate.get("enabled", True)
    ]
    gates.sort(key=lambda gate: (gate.depth, gate.innovation_number))
    return gates


def circuit_unitary(
    genome: dict[str, Any], qubits: list[Qubit] | None = None
) -> np.ndarray:
    """Computes the unitary of a serialized genome's enabled gates.

    Args:
        genome: A serialized genome (``CircuitGenome.to_dict``).
        qubits: The qubits to compute the unitary over, in matrix order (the
            first is the most significant bit). Qubits the genome does not use
            are acted on by the identity, which is how genomes on different
            qubit sets are brought to one size. Defaults to the genome's own
            qubits (:func:`genome_qubits`).

    Returns:
        The ``2**n x 2**n`` complex unitary, ``n = len(qubits)``.

    Raises:
        ValueError: If a gate acts on a qubit missing from ``qubits``, if there
            are more than :data:`MAX_UNITARY_QUBITS` qubits, or if the genome's
            target is neither ``"pennylane"`` nor ``"qiskit"``.
    """

    qubits = [tuple(qubit) for qubit in (qubits or genome_qubits(genome))]
    if len(qubits) > MAX_UNITARY_QUBITS:
        raise ValueError(
            f"The circuit runs on {len(qubits)} qubits; unitaries are only computed "
            f"for up to {MAX_UNITARY_QUBITS}."
        )

    gates = enabled_gates(genome)
    known = set(qubits)
    for gate in gates:
        missing = [qubit for qubit in gate.qubits if qubit not in known]
        if missing:
            raise ValueError(
                f"Gate {gate.innovation_number} ({gate.method_name}) acts on "
                f"{missing}, which are not among the qubits {qubits}."
            )

    if not gates:
        return np.eye(2 ** len(qubits), dtype=complex)

    target = genome.get("target")
    if target not in ("pennylane", "qiskit"):
        raise ValueError(f"Unknown target {target!r}: expected pennylane or qiskit.")

    # the gates log every one they add at debug level, which for a whole run's
    # genomes would bury everything else the process logs
    logger.disable("src.circuits.gate")
    try:
        if target == "pennylane":
            return _pennylane_unitary(gates, qubits)
        return _qiskit_unitary(gates, qubits)
    finally:
        logger.enable("src.circuits.gate")


def _pennylane_unitary(gates: list[Gate], qubits: list[Qubit]) -> np.ndarray:
    """Computes the unitary of PennyLane gates.

    Args:
        gates: The gates to apply, in order.
        qubits: The qubits, in matrix order; qubit ``i`` is wire ``i``.

    Returns:
        The unitary.
    """

    import pennylane as qml

    def apply() -> None:
        """Queues every gate, each with its own saved parameter values."""
        for gate in gates:
            gate.add_to_pennylane_circuit(
                qubits, weights=list(gate.parameters.values()), offset=0
            )

    matrix = qml.matrix(apply, wire_order=list(range(len(qubits))))()
    return np.asarray(matrix, dtype=complex)


def _qiskit_unitary(gates: list[Gate], qubits: list[Qubit]) -> np.ndarray:
    """Computes the unitary of qiskit gates.

    Args:
        gates: The gates to apply, in order.
        qubits: The qubits, in matrix order; each gets its own one-qubit register,
            as :meth:`~src.circuits.circuit.CircuitGenome.generate_qiskit_circuit`
            builds them.

    Returns:
        The unitary, with the first qubit as the most significant bit.
    """

    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Operator

    registers = {
        qubit: QuantumRegister(1, name=f"{qubit[0]}-{qubit[1]}") for qubit in qubits
    }
    circuit = QuantumCircuit(*registers.values())

    values = [value for gate in gates for value in gate.parameters.values()]
    weights = ParameterVector("weights", length=len(values))
    offset = 0
    for gate in gates:
        gate.add_to_qiskit_circuit(registers, circuit, weights, offset)
        offset += len(gate.parameters)
    if values:
        circuit = circuit.assign_parameters(dict(zip(weights, values)))

    # qiskit numbers its first qubit as the least significant bit
    return np.asarray(Operator(circuit.reverse_bits()).data, dtype=complex)
