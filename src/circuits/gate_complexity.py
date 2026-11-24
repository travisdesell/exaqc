"""Circuit-complexity counts derived from each gate's declared costs.

Every :class:`~src.circuits.gate_specifications.GateSpecification` carries the
``cnot_count``/``rot_count`` its decomposition costs, so a gate's complexity is
defined in the same place as the rest of the gate rather than in a separate
table that could drift out of sync. This module turns those per-gate costs into
per-genome totals.

The counts are recorded on each genome's summary row as it is archived, so a
run's complexity over time can be queried without reopening every genome's JSON.
Nothing here imports a quantum framework -- the gate specifications are plain
data -- so a reader can load it as cheaply as a writer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.circuits.gate_specifications import GateSpecifications
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome

#: Gate specifications per target framework, holding the decomposition costs
#: used to score a genome's circuit complexity.
GATE_SPECIFICATIONS: dict[str, GateSpecifications] = {
    "pennylane": pennylane_gate_specifications,
    "qiskit": qiskit_gate_specifications,
}


def gate_counts_from_serialized(
    target: str, gates: list[dict[str, Any]]
) -> dict[str, float]:
    """Computes enabled-gate complexity statistics from a serialized genome.

    This works on the dictionaries
    :meth:`~src.circuits.circuit.CircuitGenome.to_dict` produces, so a genome can
    be scored while it is being archived without rebuilding it.

    Args:
        target: The genome's target framework, naming which gate specifications
            declare the costs.
        gates: The genome's serialized gates; a gate without an ``enabled`` key
            counts as enabled.

    Returns:
        A dictionary of float-valued gate counts with keys ``"gates_total"``,
        ``"gates_cnot"`` and ``"gates_rot"``.

    Raises:
        ValueError: If ``target`` has no known gate specifications.
    """

    if target not in GATE_SPECIFICATIONS:
        raise ValueError(
            f"Unknown target {target!r} for gate complexity; "
            f"choices: {sorted(GATE_SPECIFICATIONS)}"
        )
    specifications = GATE_SPECIFICATIONS[target]

    total = 0
    cnot = 0
    rot = 0

    for gate in gates:
        if not gate.get("enabled", True):
            continue

        specification = specifications[str(gate.get("method_name", "")).lower()]

        total += 1
        cnot += specification.cnot_count
        rot += specification.rot_count

    return {
        "gates_total": float(total),
        "gates_cnot": float(cnot),
        "gates_rot": float(rot),
    }


def gate_counts(genome: CircuitGenome) -> dict[str, float]:
    """Computes enabled-gate complexity statistics for a genome.

    Args:
        genome: Circuit genome whose gates should be counted.

    Returns:
        A dictionary of float-valued gate counts with keys ``"gates_total"``,
        ``"gates_cnot"`` and ``"gates_rot"``.

    Raises:
        ValueError: If the genome's ``target`` has no known gate specifications.
    """

    return gate_counts_from_serialized(
        getattr(genome, "target", "pennylane"),
        [
            {
                "method_name": getattr(gate, "method_name", ""),
                "enabled": getattr(gate, "enabled", True),
            }
            for gate in getattr(genome, "gates", [])
        ],
    )


def num_enabled_gates(genome: CircuitGenome) -> float:
    """Returns the total number of enabled gates in a genome.

    Args:
        genome: Circuit genome to inspect.

    Returns:
        Number of enabled gates as a float.

    Raises:
        ValueError: If the genome's ``target`` has no known gate specifications.
    """

    return gate_counts(genome)["gates_total"]
