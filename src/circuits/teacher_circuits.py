"""Reference ("teacher") quantum circuits expressed as :class:`CircuitGenome`.

EXAQC's teacher-imitation task evolves a student circuit to reproduce the
outputs of a known reference circuit. Both sides are ordinary
:class:`~src.circuits.circuit.CircuitGenome` instances, so a teacher runs on the
same ``pennylane`` or ``qiskit`` backend as the students that imitate it, and is
drawn, serialized, and executed by exactly the same code.

A teacher is declared as a backend-agnostic list of :class:`TeacherGate` entries
(a gate method name, the wires it acts on, and an optional fixed rotation
angle). Nothing in a declaration is target-specific: gate *method names* are
shared by both targets, and the per-target **parameter name** for a rotation
(pennylane's ``phi`` versus qiskit's ``theta``) is resolved from that target's
gate specifications when the genome is built.

Teachers use only gates that are validated on both targets. The multi-controlled
family (``mcp``, ``mcx``, ``mcrx``, ...) is deliberately avoided: those are
flagged ``needs_validation`` and are excluded from the evolutionary mutation
pool, so a student could never reproduce a teacher built from them.

Teacher genomes carry no encoder and no decoder -- there is nothing classical to
learn. Inputs are fed straight into the circuit through the genome's own
``quantum_input_mode`` and the outputs are the raw circuit readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.circuits.circuit import CircuitGenome
from src.circuits.gate_specifications import GateSpecifications
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.qiskit_gate_specifications import qiskit_gate_specifications

#: Gate specifications per target framework, used to resolve a rotation's
#: per-target parameter name when a teacher genome is built.
_GATE_SPECIFICATIONS: dict[str, GateSpecifications] = {
    "pennylane": pennylane_gate_specifications,
    "qiskit": qiskit_gate_specifications,
}

#: Default register name for the wires a teacher circuit spans.
DEFAULT_REGISTER_NAME = "q"


@dataclass(frozen=True)
class TeacherGate:
    """One gate in a teacher circuit, independent of the target framework.

    Attributes:
        method_name: The gate's method name (e.g. ``"cx"``), which is shared by
            the pennylane and qiskit gate specifications.
        wires: The wire indices the gate acts on, in the order the gate's
            specification lists its qubit arguments.
        parameter: The fixed rotation angle for a parameterized gate, or
            ``None`` for a gate that takes no parameter. Teachers are constant
            circuits, so these angles are fixed rather than trainable.
    """

    method_name: str
    wires: tuple[int, ...]
    parameter: float | None = None


#: A teacher declaration: given the input and output wires, return the gates.
TeacherBuilder = Callable[[list[int], list[int]], list[TeacherGate]]


def _require_wires(
    teacher_name: str,
    input_wires: list[int],
    output_wires: list[int],
    n_inputs: int = 0,
    n_outputs: int = 0,
) -> None:
    """Checks that a teacher has enough input and output wires to be built.

    Args:
        teacher_name: Name of the teacher being built, used in the error.
        input_wires: The wires driven by the classical inputs.
        output_wires: The wires whose output is imitated.
        n_inputs: Minimum number of input wires the teacher needs.
        n_outputs: Minimum number of output wires the teacher needs.

    Returns:
        None. Returns normally when the teacher has enough wires.

    Raises:
        ValueError: If either wire list is shorter than required.
    """

    if len(input_wires) < n_inputs:
        raise ValueError(
            f"teacher {teacher_name!r} needs at least {n_inputs} input wire(s), "
            f"but received {len(input_wires)}: {input_wires}"
        )
    if len(output_wires) < n_outputs:
        raise ValueError(
            f"teacher {teacher_name!r} needs at least {n_outputs} output wire(s), "
            f"but received {len(output_wires)}: {output_wires}"
        )


def _teacher_identity(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds the identity teacher, which applies no operation at all.

    Declared as an empty gate list rather than an explicit ``id`` gate: an
    identity gate is a no-op by definition, and ``id`` is flagged
    ``needs_validation`` on both targets. The teacher's output is therefore the
    state produced by the input encoding alone.

    Args:
        input_wires: The wires driven by the classical inputs (unused).
        output_wires: The wires whose output is imitated (unused).

    Returns:
        An empty list of gates.
    """

    return []


def _teacher_x_out4(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds a teacher that flips the first output wire.

    Args:
        input_wires: The wires driven by the classical inputs (unused).
        output_wires: The wires whose output is imitated.

    Returns:
        A single Pauli-X on the first output wire.

    Raises:
        ValueError: If there is no output wire.
    """

    _require_wires("x_out4", input_wires, output_wires, n_outputs=1)
    return [TeacherGate("x", (output_wires[0],))]


def _teacher_bell_out(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds a teacher preparing a Bell pair across the first two output wires.

    Args:
        input_wires: The wires driven by the classical inputs (unused).
        output_wires: The wires whose output is imitated.

    Returns:
        A Hadamard followed by a CNOT across the first two output wires.

    Raises:
        ValueError: If there are fewer than two output wires.
    """

    _require_wires("bell_out", input_wires, output_wires, n_outputs=2)
    return [
        TeacherGate("h", (output_wires[0],)),
        TeacherGate("cx", (output_wires[0], output_wires[1])),
    ]


def _teacher_copy_in_to_out(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds a teacher copying the first two input wires onto the output wires.

    Args:
        input_wires: The wires driven by the classical inputs.
        output_wires: The wires whose output is imitated.

    Returns:
        Two CNOTs, copying each input wire onto the matching output wire.

    Raises:
        ValueError: If there are fewer than two input or two output wires.
    """

    _require_wires("copy_in_to_out", input_wires, output_wires, n_inputs=2, n_outputs=2)
    return [
        TeacherGate("cx", (input_wires[0], output_wires[0])),
        TeacherGate("cx", (input_wires[1], output_wires[1])),
    ]


def _teacher_parity012_to_out4(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds a teacher computing the parity of three inputs onto one output.

    Args:
        input_wires: The wires driven by the classical inputs.
        output_wires: The wires whose output is imitated.

    Returns:
        Three CNOTs accumulating the parity of the first three input wires onto
        the first output wire.

    Raises:
        ValueError: If there are fewer than three input wires or no output wire.
    """

    _require_wires(
        "parity012_to_out4", input_wires, output_wires, n_inputs=3, n_outputs=1
    )
    return [
        TeacherGate("cx", (input_wires[index], output_wires[0])) for index in range(3)
    ]


def _teacher_input_controlled_bell(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds a Bell pair whose phase is controlled by the first input wire.

    Args:
        input_wires: The wires driven by the classical inputs.
        output_wires: The wires whose output is imitated.

    Returns:
        A Bell-pair preparation followed by a CNOT from the first input wire.

    Raises:
        ValueError: If there is no input wire or fewer than two output wires.
    """

    _require_wires(
        "input_controlled_bell", input_wires, output_wires, n_inputs=1, n_outputs=2
    )
    return [
        TeacherGate("h", (output_wires[0],)),
        TeacherGate("cx", (output_wires[0], output_wires[1])),
        TeacherGate("cx", (input_wires[0], output_wires[0])),
    ]


def _teacher_2layer_out_block(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds a fixed-angle two-layer block on the first two output wires.

    The angles are constants, so this teacher is a fixed unitary independent of
    the classical inputs on the output wires.

    Args:
        input_wires: The wires driven by the classical inputs (unused).
        output_wires: The wires whose output is imitated.

    Returns:
        Two RY rotations, an entangling CNOT, then two RX rotations.

    Raises:
        ValueError: If there are fewer than two output wires.
    """

    _require_wires("2layer_out_block", input_wires, output_wires, n_outputs=2)
    return [
        TeacherGate("ry", (output_wires[0],), 0.7),
        TeacherGate("ry", (output_wires[1],), 1.1),
        TeacherGate("cx", (output_wires[0], output_wires[1])),
        TeacherGate("rx", (output_wires[0],), 0.4),
        TeacherGate("rx", (output_wires[1],), -0.9),
    ]


def _teacher_grover(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds one Grover diffusion operator across every wire.

    Applies Hadamards and Pauli-X gates to all wires, a multi-controlled Z, then
    undoes the X and Hadamard layers.

    The multi-controlled Z is expressed with ``cz`` (two wires) or ``ccz``
    (three wires). Wider circuits would need the ``mcp`` gate, which is flagged
    ``needs_validation`` on both targets and is excluded from the evolutionary
    mutation pool -- a student could never reproduce such a teacher -- so this
    raises rather than emitting a gate the search cannot use.

    Args:
        input_wires: The wires driven by the classical inputs.
        output_wires: The wires whose output is imitated.

    Returns:
        The gates of one Grover diffusion operator.

    Raises:
        ValueError: If the circuit does not span exactly two or three wires.
    """

    wires = list(input_wires) + list(output_wires)

    if len(wires) == 2:
        controlled_z = TeacherGate("cz", (wires[0], wires[1]))
    elif len(wires) == 3:
        controlled_z = TeacherGate("ccz", (wires[0], wires[1], wires[2]))
    else:
        raise ValueError(
            f"teacher 'grover' spans {len(wires)} wires, but a multi-controlled Z "
            "is only expressible with validated gates for 2 wires (cz) or 3 "
            "wires (ccz). Use 2 or 3 total input+output wires."
        )

    gates = [TeacherGate("h", (wire,)) for wire in wires]
    gates += [TeacherGate("x", (wire,)) for wire in wires]
    gates.append(controlled_z)
    gates += [TeacherGate("x", (wire,)) for wire in wires]
    gates += [TeacherGate("h", (wire,)) for wire in wires]
    return gates


def _teacher_half_adder(
    input_wires: list[int], output_wires: list[int]
) -> list[TeacherGate]:
    """Builds a half adder over two input and two output wires.

    Computes ``SUM = A XOR B`` onto the first output wire and
    ``CARRY = A AND B`` onto the second, assuming both output wires start in
    the zero state.

    Args:
        input_wires: The wires carrying the addends ``A`` and ``B``.
        output_wires: The wires receiving ``SUM`` and ``CARRY``.

    Returns:
        Two CNOTs computing the sum plus a Toffoli computing the carry.

    Raises:
        ValueError: If there are fewer than two input or two output wires, or
            if the four wires used are not distinct.
    """

    _require_wires("half_adder", input_wires, output_wires, n_inputs=2, n_outputs=2)

    addend_a, addend_b = input_wires[0], input_wires[1]
    total, carry = output_wires[0], output_wires[1]

    used = [addend_a, addend_b, total, carry]
    if len(set(used)) != 4:
        raise ValueError(
            "teacher 'half_adder' requires 4 distinct wires (A, B, SUM, CARRY), "
            f"but received {used}"
        )

    return [
        TeacherGate("cx", (addend_a, total)),
        TeacherGate("cx", (addend_b, total)),
        TeacherGate("ccx", (addend_a, addend_b, carry)),
    ]


#: Every teacher circuit, keyed by the name used on the command line.
TEACHER_CIRCUITS: dict[str, TeacherBuilder] = {
    "identity": _teacher_identity,
    "x_out4": _teacher_x_out4,
    "bell_out": _teacher_bell_out,
    "copy_in_to_out": _teacher_copy_in_to_out,
    "parity012_to_out4": _teacher_parity012_to_out4,
    "input_controlled_bell": _teacher_input_controlled_bell,
    "2layer_out_block": _teacher_2layer_out_block,
    "grover": _teacher_grover,
    "half_adder": _teacher_half_adder,
}

#: Teacher names offered as command-line choices, in registry order.
TEACHER_NAMES: tuple[str, ...] = tuple(TEACHER_CIRCUITS)


def build_teacher_genome(
    teacher_name: str,
    target: str,
    input_wires: list[int],
    output_wires: list[int],
    quantum_input_mode: str = "ry",
    quantum_output_mode: str = "probs",
    genome_number: int = 0,
    register_name: str = DEFAULT_REGISTER_NAME,
) -> CircuitGenome:
    """Builds a teacher circuit as a :class:`CircuitGenome`.

    The returned genome has no encoder and no decoder, so its inputs are fed
    straight into the circuit through ``quantum_input_mode`` and its outputs are
    the raw circuit readout. Its model is not initialized; call
    ``initialize_model()`` before running a forward pass.

    Args:
        teacher_name: One of :data:`TEACHER_NAMES`.
        target: ``"pennylane"`` or ``"qiskit"``.
        input_wires: The wires driven by the classical inputs.
        output_wires: The wires whose output is imitated.
        quantum_input_mode: How classical inputs are encoded onto the input
            wires. Students must use the same mode for imitation to be
            well-posed.
        quantum_output_mode: How the circuit is read out (e.g. ``"probs"``).
        genome_number: Identifier assigned to the teacher genome.
        register_name: Register name used for the teacher's wires.

    Returns:
        The teacher :class:`CircuitGenome`.

    Raises:
        ValueError: If ``teacher_name`` or ``target`` is unknown, if the wire
            lists are empty or overlap, or if the teacher cannot be built from
            the given wires.
    """

    if teacher_name not in TEACHER_CIRCUITS:
        raise ValueError(
            f"Unknown teacher {teacher_name!r}; choices: {list(TEACHER_NAMES)}"
        )
    if target not in _GATE_SPECIFICATIONS:
        raise ValueError(
            f"Unknown target {target!r}; choices: {sorted(_GATE_SPECIFICATIONS)}"
        )
    if not input_wires:
        raise ValueError("A teacher circuit needs at least one input wire.")
    if not output_wires:
        raise ValueError("A teacher circuit needs at least one output wire.")

    overlapping = set(input_wires) & set(output_wires)
    if overlapping:
        raise ValueError(
            "A teacher's input and output wires must be disjoint, but these are "
            f"in both: {sorted(overlapping)}"
        )

    gates = TEACHER_CIRCUITS[teacher_name](list(input_wires), list(output_wires))

    genome = CircuitGenome(
        genome_number=genome_number,
        target=target,
        input_qubits=[(register_name, wire) for wire in input_wires],
        output_qubits=[(register_name, wire) for wire in output_wires],
    )
    genome.hyperparameters = {
        "quantum_input_mode": quantum_input_mode,
        "quantum_output_mode": quantum_output_mode,
    }

    # A teacher has nothing classical to learn.
    genome.encoder = None
    genome.decoder = None

    specifications = _GATE_SPECIFICATIONS[target]

    for index, gate in enumerate(gates):
        # Depths must lie strictly between 0 and 1 and preserve the declared
        # gate order.
        depth = (index + 1) / (len(gates) + 1)

        parameters: dict[str, float] = {}
        if gate.parameter is not None:
            # pennylane and qiskit name the same rotation's parameter
            # differently ('phi' vs 'theta'), so resolve it from the target.
            parameter_name = specifications[gate.method_name].parameters[0]
            parameters[parameter_name] = gate.parameter

        genome.add_gate(
            depth=depth,
            method_name=gate.method_name,
            qubits=[(register_name, wire) for wire in gate.wires],
            parameters=parameters,
        )

    return genome
