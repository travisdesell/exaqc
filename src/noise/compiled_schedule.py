"""Translate a genome into the chip's own gates, for chip noise.

A chip runs only a few simple gates (on IBM's Heron chips ``sx``, ``rz``,
``x`` and ``cz``), and ``cz`` only between qubits that are wired together.
Before IBM runs a circuit, its translator (Qiskit's ``transpile``) rewrites the
circuit into those gates and adds moves (SWAPs) wherever two qubits are not
wired. :func:`compile_genome` does the same translation for a genome, once, so
the noise model can charge each genome gate for the chip gates it really
becomes.

The simulator still runs the genome's own gates (PennyLane); this translation
only says how much noise to add after each of them, and on which wires
(noise placement "A2"). Qiskit is used here only as the translator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome

DEFAULT_TRANSPILE_LEVEL = 2
"""How hard the translator shrinks the circuit; Qiskit's own default."""

DEFAULT_TRANSPILE_SEED = 7
"""Seed for the translator's random choices, so a genome always translates the same way."""

NOT_CHIP_GATES = ("id", "xslow", "delay", "measure", "reset")
"""Entries of a chip's ``basis_gates`` that the translator should not use."""


@dataclass
class ChipGate:
    """One gate the chip really runs.

    Attributes:
        name: ``"sx"``, ``"x"``, ``"rz"`` or ``"cz"``.
        slots: Which chosen chip qubits it runs on, as positions in
            ``--noise_qubits`` (0 is the first listed qubit).
        wires: Which genome wires' states sit on those qubits when it runs.
    """

    name: str
    slots: list[int]
    wires: list[int]


@dataclass
class CompiledSchedule:
    """A genome translated into chip gates.

    Attributes:
        per_gate: For each enabled genome gate (keyed by ``id(gate)``), the
            chip gates it became, routing moves included, in order.
        final_slot: ``final_slot[wire]`` is the slot holding that wire's state
            at the end, which is where its readout happens.
        encoding: The chip gates the data-loading step became, or ``None``
            when the encoding mode depends on each data sample (``basis``,
            ``amplitude``), in which case each input qubit gets one dose.
    """

    per_gate: dict[int, list[ChipGate]] = field(default_factory=dict)
    final_slot: list[int] = field(default_factory=list)
    encoding: list[ChipGate] | None = None


def _placeholder_angle(k: int) -> float:
    """Returns a fixed, ordinary angle to stand in for a data value or weight.

    The translation must not depend on the real values (they change every
    training step and every data sample), and an angle of exactly 0, pi/2
    or pi would let Qiskit simplify the gate. Steps of the golden ratio from
    0.1 never land on such angles. Symbols would also work, but Qiskit cannot
    translate some gates (e.g. ``rv``) while their angles are symbols.

    Args:
        k: The angle's position among all angles in the circuit.

    Returns:
        The placeholder angle, in radians.
    """
    return 0.1 + 0.6180339887 * k


EXACT_ENCODINGS = ("u3", "rx", "ry", "rz")
"""Encoding modes whose gates are the same for every data sample (only the
angles come from the data), so they can be translated once per genome."""


def chip_gates_from_basis(basis_gates: list[str]) -> list[str]:
    """Returns the gates the translator may use, from a chip's ``basis_gates``.

    Args:
        basis_gates: The chip's ``basis_gates`` from the calibration file,
            e.g. ``["cz", "id", "rz", "sx", "x", "xslow"]`` for Fez.

    Returns:
        The chip's gates without the do-nothing ``id``, the slow ``xslow`` and
        non-gate instructions, e.g. ``["cz", "rz", "sx", "x"]``.
    """
    return [g for g in basis_gates if g not in NOT_CHIP_GATES]


def _swap_in_chip_gates(chip_gates: list[str]) -> list[tuple[str, list[int]]]:
    """Returns one SWAP between slots 0 and 1, written in chip gates.

    Translated with optimization level 0: at level 2 and above Qiskit deletes
    a lone SWAP and just relabels the qubits, which would charge a routing
    move no noise at all. A move inside a circuit is real on the chip.

    Args:
        chip_gates: The gates the translator may use.

    Returns:
        Each chip gate as ``(name, slots)``, with slots 0 and 1 standing for
        the two qubits being swapped (3 ``cz`` plus one-qubit gates).
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap

    swap = QuantumCircuit(2)
    swap.swap(0, 1)
    translated = transpile(
        swap,
        basis_gates=chip_gates,
        coupling_map=CouplingMap([[0, 1], [1, 0]]),
        initial_layout=[0, 1],
        optimization_level=0,
    )
    return [
        (ins.operation.name, [translated.find_bit(q).index for q in ins.qubits])
        for ins in translated.data
    ]


def _add_gate(circuit: Any, gate: Any, registers: dict, angles: list[float]) -> None:
    """Adds one genome gate to a Qiskit circuit, angles passed by position.

    EXAQC's PennyLane gates name some angles differently from Qiskit (e.g.
    ``phi`` for ``rx``/``ry``/``rzz``, ``delta`` for ``u``), so passing them
    by name fails; both use the same order, so they are passed by position.
    Multi-controlled gates (qubit names ending in ``...``) take their
    controls as one list, then the target. A genome ``swap`` is written as
    three ``cx``: at optimization level 2 Qiskit would otherwise delete it and
    relabel the qubits, which the wire tracking in :func:`compile_genome`
    cannot see, so later gates' noise could land on the wrong wires.

    Args:
        circuit: The Qiskit ``QuantumCircuit`` being built.
        gate: The genome gate (``method_name``, ``qubits``, ``specs``).
        registers: One Qiskit register per genome qubit.
        angles: The gate's angles, in order.

    Returns:
        None. Appends the gate to ``circuit``.
    """
    bits = [registers[q][0] for q in gate.qubits]
    if gate.method_name == "swap":
        a, b = bits
        circuit.cx(a, b)
        circuit.cx(b, a)
        circuit.cx(a, b)
        return
    method = getattr(circuit, gate.method_name)
    if any(name.endswith("...") for name in gate.specs.qubits):
        method(*angles, bits[:-1], bits[-1])
    else:
        method(*angles, *bits)


def compile_genome(
    genome: CircuitGenome,
    wired_pairs: list[tuple[int, int]],
    chip_gates: list[str],
    level: int = DEFAULT_TRANSPILE_LEVEL,
    seed: int = DEFAULT_TRANSPILE_SEED,
) -> CompiledSchedule:
    """Translates a genome into the chip gates each of its gates becomes.

    Three steps: build the genome as a Qiskit circuit, starting with its
    data-loading gates when the encoding mode allows (:data:`EXACT_ENCODINGS`),
    with a barrier after every gate and all angles set to fixed placeholder
    values (:func:`_placeholder_angle`, so the translation never depends on
    the data or the trained values); translate
    it onto the chosen qubits with Qiskit's
    ``transpile``; then walk the result one genome gate at a time, writing
    each routing move out as its chip gates and tracking which wire's state
    sits on which qubit.

    Wire ``i`` starts on slot ``i`` (the ``i``-th qubit in ``--noise_qubits``).

    Args:
        genome: The genome to translate; it is not changed.
        wired_pairs: The chosen qubits wired together, as slot pairs, e.g.
            ``[(0, 1), (1, 2), (2, 3)]`` for a chain of four.
        chip_gates: The gates the chip runs, from :func:`chip_gates_from_basis`.
        level: The translator's optimization level; must match the level used
            when circuits are sent to the real chip.
        seed: The translator's seed.

    Returns:
        The translation: each enabled genome gate's chip gates, and where
        each wire's state ends up.

    Raises:
        ValueError: If the translation does not keep one barrier per genome
            gate, so gates could not be matched to their chip gates.
    """
    from qiskit import QuantumCircuit, QuantumRegister, transpile
    from qiskit.transpiler import CouplingMap

    # 1. The genome as a Qiskit circuit: one register per wire, placeholder
    #    angles, and a barrier after each gate to mark where it ends.
    registers = {q: QuantumRegister(1, name=f"{q[0]}-{q[1]}") for q in genome.qubits}
    circuit = QuantumCircuit(*registers.values())

    # The data-loading gates first (as in generate_qiskit_circuit), then a
    # barrier; only for modes whose gates do not depend on the data sample.
    mode = genome.hyperparameters.get("quantum_input_mode")
    exact_encoding = mode in EXACT_ENCODINGS
    angle_count = 0
    if exact_encoding:
        wire_registers = list(registers.values())
        for w in genome.input_indexes:
            n_angles = 3 if mode == "u3" else 1
            angles = [_placeholder_angle(angle_count + j) for j in range(n_angles)]
            angle_count += n_angles
            getattr(circuit, "u" if mode == "u3" else mode)(*angles, wire_registers[w])
        circuit.barrier()

    kept = []
    for gate in genome.gates:
        if gate.enabled:
            angles = [
                _placeholder_angle(angle_count + j) for j in range(len(gate.parameters))
            ]
            angle_count += len(gate.parameters)
            _add_gate(circuit, gate, registers, angles)
            circuit.barrier()
            kept.append(gate)

    # 2. Translate onto the chosen qubits. "swap" stays allowed so routing
    #    moves come back as one visible "swap" that step 3 can track.
    edges = [[a, b] for a, b in wired_pairs] + [[b, a] for a, b in wired_pairs]
    routed = transpile(
        circuit,
        basis_gates=list(chip_gates) + ["swap"],
        coupling_map=CouplingMap(edges),
        initial_layout=list(range(len(genome.qubits))),
        optimization_level=level,
        seed_transpiler=seed,
    )

    # 3. Split at the barriers (one group per genome gate) and track states.
    groups: list[list] = [[]]
    for ins in routed.data:
        if ins.operation.name == "barrier":
            groups.append([])
        else:
            groups[-1].append(ins)
    groups.pop()  # everything after the last barrier (nothing, by construction)
    encoding_group = groups.pop(0) if exact_encoding else None
    if len(groups) != len(kept):
        raise ValueError(
            f"translation kept {len(groups)} gate boundaries for {len(kept)} genome gates."
        )

    swap_pieces = _swap_in_chip_gates(chip_gates)
    if sum(name == "cz" for name, _ in swap_pieces) != 3:
        raise ValueError(f"a SWAP should take 3 cz on this chip, got {swap_pieces}.")
    where = list(range(len(genome.qubits)))  # where[slot] = wire whose state is there

    def to_chip_gates(group: list) -> list[ChipGate]:
        """Turns one gate's translated instructions into chip gates.

        Writes each routing ``swap`` out as its chip gates and updates
        ``where`` as states move.

        Args:
            group: The translated instructions between two barriers.

        Returns:
            The chip gates, each with its slots and the wires on them.
        """
        chip: list[ChipGate] = []
        for ins in group:
            slots = [routed.find_bit(q).index for q in ins.qubits]
            if ins.operation.name == "swap":
                a, b = slots
                for name, piece_slots in swap_pieces:
                    piece = [[a, b][k] for k in piece_slots]  # 0 -> a, 1 -> b
                    chip.append(ChipGate(name, piece, [where[s] for s in piece]))
                where[a], where[b] = where[b], where[a]
            else:
                chip.append(
                    ChipGate(ins.operation.name, slots, [where[s] for s in slots])
                )
        return chip

    schedule = CompiledSchedule()
    if encoding_group is not None:
        schedule.encoding = to_chip_gates(encoding_group)
    for gate, group in zip(kept, groups):
        schedule.per_gate[id(gate)] = to_chip_gates(group)

    schedule.final_slot = [where.index(wire) for wire in range(len(genome.qubits))]
    return schedule
