"""Device cards: per-gate noise from optimal control (Dr. Vogt's method).

A *device card* is a JSON file written by the optimal-control side of the
project (outside this repository, see ``arch_aware/README.md`` on the
``DrVogt/noise-model`` branch). For each gate of a small universal set it
records the duration of the gate's optimised control pulse (``T_ns``) and the
Pauli-twirled error channel that pulse leaves behind (``pauli_error``), under
one open-system model of one modelled device (a transmon pair). It also
records the channel a qubit suffers while it waits, at a grid of idle
durations (``idle``).

Device-card noise charges a genome for the card gates it really becomes:

1. Each genome gate is translated into the card's native gates
   (:data:`CARD_NATIVE_GATES`: ``h``, ``x``, ``cx`` and a virtual ``rz``) with
   :func:`src.noise.compiled_schedule.compile_genome`, the same translator chip
   noise uses, so ``rx`` becomes ``h rz h`` and pays for two ``h`` errors.
2. Each card gate gets its own Pauli channel from the card, on the wires it
   acts on.
3. An as-soon-as-possible timeline of those card gates (:func:`card_timeline`)
   says how long each wire waits before each gate and before measurement; the
   card's idle channel for that long is applied there, on the waiting wire.

The card describes one qubit's gates and one coupled pair's ``cx``; every
qubit and every wired pair of the circuit gets those same numbers (a device
built from identical copies). Which pairs are wired is either all of them, or
the pairs an IBM calibration's coupling map wires among the chosen physical
qubits (``--noise_calibration`` with ``--noise_qubits``).

The same timeline also gives three simulation-free objectives
(:func:`hardware_metrics`): ``p_fail``, ``total_time_ns`` and
``n_entanglers``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.noise.compiled_schedule import ChipGate, CompiledSchedule

#: Where the device cards live: ``device_cards/`` at the repository root
#: (copied unchanged from ``arch_aware/device_cards`` on ``DrVogt/noise-model``).
DEVICE_CARD_DIR = Path(__file__).resolve().parents[2] / "device_cards"

#: The ``--noise_device_card`` value that builds the card from the chip named
#: by ``--noise_calibration`` and ``--noise_qubits`` instead of reading a card
#: file (see :mod:`src.noise.calibration_card`).
CALIBRATION_CARD = "calibration"

#: The gates genomes are translated into for device-card noise. ``rz`` is a
#: virtual frame change (no pulse, no error, no time), as on real transmon
#: hardware; the card's driven ``z``/``s``/``t`` entries are therefore never
#: used, since every phase gate becomes an ``rz``.
CARD_NATIVE_GATES: list[str] = ["h", "x", "cx", "rz"]

#: The card gates :data:`CARD_NATIVE_GATES` needs a channel for.
REQUIRED_CARD_GATES: tuple[str, ...] = ("h", "x", "cx")

#: The fitness keys :func:`hardware_metrics` produces.
HARDWARE_OBJECTIVES: tuple[str, ...] = ("p_fail", "total_time_ns", "n_entanglers")

_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def resolve_device_card(spec: str) -> Path:
    """Finds a device card from a name or a path.

    Args:
        spec: A card name such as ``transmon_detuned_mixed`` (looked up as
            ``device_cards/device_card_<name>.json``), a file name
            in that folder, or a path to a card file.

    Returns:
        The card file's path.

    Raises:
        ValueError: If no such card exists; the message lists the cards
            available.
    """

    candidates = [
        Path(spec),
        DEVICE_CARD_DIR / spec,
        DEVICE_CARD_DIR / f"device_card_{spec}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    available = sorted(
        path.stem.removeprefix("device_card_")
        for path in DEVICE_CARD_DIR.glob("device_card_*.json")
    )
    raise ValueError(
        f"no device card {spec!r}; give a path or one of: {', '.join(available) or 'none found'}."
    )


def load_device_card(spec: str) -> dict[str, Any]:
    """Reads a device card into the compact form genomes carry.

    Only what the noise model and the objectives use is kept: for each gate
    its duration, its Pauli error probabilities and how many wires it acts on,
    and for each idle duration the idle channel's Pauli transfer diagonal.
    The run ids and solver settings stay in the card file, which is recorded
    by name.

    Args:
        spec: The card, as for :func:`resolve_device_card`.

    Returns:
        ``name``, ``file``, ``architecture``, ``model``, ``gates`` (gate ->
        ``T_ns``, ``wires``, ``pauli_error``) and ``idle`` (duration in ns,
        as text -> ``ptm_error_diag``), plus ``wired_pairs`` set to ``None``
        (every pair wired; see :func:`wired_slot_pairs`).

    Raises:
        ValueError: If the card cannot be found, or lacks a gate in
            :data:`REQUIRED_CARD_GATES`.
    """

    path = resolve_device_card(spec)
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    gates = raw.get("gates", {})
    missing = [gate for gate in REQUIRED_CARD_GATES if gate not in gates]
    if missing:
        raise ValueError(
            f"device card {path} has no entry for {', '.join(missing)}; it needs "
            f"{', '.join(REQUIRED_CARD_GATES)}."
        )
    return {
        "name": path.stem.removeprefix("device_card_"),
        "file": str(path),
        "architecture": raw.get("architecture"),
        "model": raw.get("model"),
        "gates": {
            name: {
                "T_ns": float(entry["T_ns"]),
                "wires": int(entry.get("wires", 1)),
                "pauli_error": {k: float(v) for k, v in entry["pauli_error"].items()},
            }
            for name, entry in gates.items()
        },
        "idle": {
            str(float(duration)): {
                k: float(v) for k, v in entry["ptm_error_diag"].items()
            }
            for duration, entry in raw.get("idle", {}).items()
        },
        "wired_pairs": None,
    }


def wired_slot_pairs(
    raw_calibration: dict[str, Any], physical_qubits: list[int]
) -> list[list[int]]:
    """Lists which circuit wires an IBM chip wires together.

    Only the chip's coupling map is used, never its error rates: device-card
    noise takes every rate and duration from the card.

    Args:
        raw_calibration: A stored calibration file's contents (it records the
            backend's ``coupling_map``).
        physical_qubits: The chip qubits wires 0, 1, ... sit on.

    Returns:
        Each wired pair as ``[i, j]`` wire positions with ``i < j``, sorted.

    Raises:
        ValueError: If the file records no coupling map, or a qubit is not on
            the chip.
    """

    backend = raw_calibration.get("backend", {})
    coupling = backend.get("coupling_map")
    if not coupling:
        raise ValueError("the calibration file records no coupling_map.")
    n_qubits = int(backend.get("n_qubits", 0))
    for qubit in physical_qubits:
        if not 0 <= qubit < n_qubits:
            raise ValueError(
                f"qubit {qubit} is not on {backend.get('name')} (0..{n_qubits - 1})."
            )
    position = {qubit: i for i, qubit in enumerate(physical_qubits)}
    pairs = {
        tuple(sorted((position[a], position[b])))
        for a, b in coupling
        if a in position and b in position
    }
    return [list(pair) for pair in sorted(pairs)]


def all_pairs(n_wires: int) -> list[tuple[int, int]]:
    """Lists every pair of wires, for a device with every pair wired.

    Args:
        n_wires: The number of circuit wires.

    Returns:
        Every ``(i, j)`` with ``i < j``.
    """

    return list(combinations(range(n_wires), 2))


def pauli_matrix(label: str) -> np.ndarray:
    """Builds the matrix of a multi-qubit Pauli string.

    Args:
        label: The Pauli string, first wire first, e.g. ``"XZ"``.

    Returns:
        The tensor product of the single-qubit Paulis, ``2**n`` square.
    """

    matrix = np.array([[1.0 + 0.0j]])
    for letter in label:
        matrix = np.kron(matrix, _PAULI_MATRICES[letter])
    return matrix


def kraus_from_pauli_probabilities(probabilities: dict[str, float]) -> list[np.ndarray]:
    """Kraus operators ``sqrt(p_P) P`` of a Pauli channel.

    The card's twirled probabilities sum to ``1 - trace_loss`` rather than 1
    (leakage out of the computational subspace is missing from them), and
    channels that are not Pauli-dominated can have small negative entries.
    Negative entries are clipped and the rest renormalised, so the channel is
    trace preserving (as Dr. Vogt's ``DeviceCardNoiseModel`` does).

    Args:
        probabilities: Pauli string -> probability, e.g. ``{"I": 0.999, ...}``.

    Returns:
        The Kraus operators, one per Pauli with positive probability; just the
        identity when every probability is clipped to zero.
    """

    labels = list(probabilities)
    weights = np.array([max(float(probabilities[label]), 0.0) for label in labels])
    total = weights.sum()
    if total <= 0.0:
        return [pauli_matrix("I" * len(labels[0]))]
    weights = weights / total
    return [
        math.sqrt(weight) * pauli_matrix(label)
        for label, weight in zip(labels, weights)
        if weight > 0.0
    ]


def gate_error_probability(card: dict[str, Any], gate: str) -> float:
    """The chance a card gate suffers any Pauli error.

    Args:
        card: A card from :func:`load_device_card`.
        gate: The card gate, e.g. ``"h"`` or ``"cx"``; ``"rz"`` is virtual.

    Returns:
        ``1 - p_identity``, at least 0; 0 for ``rz``.

    Raises:
        KeyError: If the card has no such gate.
    """

    if gate == "rz":
        return 0.0
    entry = card["gates"][gate]
    identity = "I" * int(entry["wires"])
    return max(0.0, 1.0 - float(entry["pauli_error"][identity]))


def gate_duration_ns(card: dict[str, Any], gate: str) -> float:
    """How long a card gate's pulse takes.

    Args:
        card: A card from :func:`load_device_card`.
        gate: The card gate; ``"rz"`` is virtual and takes no time.

    Returns:
        The duration in nanoseconds.

    Raises:
        KeyError: If the card has no such gate.
    """

    return 0.0 if gate == "rz" else float(card["gates"][gate]["T_ns"])


def idle_ptm_diagonal(
    card: dict[str, Any], duration_ns: float
) -> dict[str, float] | None:
    """The idle channel's Pauli transfer diagonal for one wait.

    Within the card's grid of idle durations the diagonal is interpolated
    log-linearly (it decays multiplicatively); outside it, the nearest grid
    point is raised to the power ``duration / grid_duration``, which is exact
    only for a Markovian channel. This follows Dr. Vogt's
    ``DeviceCard.idle_ptm_diagonal``.

    Args:
        card: A card from :func:`load_device_card`.
        duration_ns: How long the wire waits.

    Returns:
        Pauli label -> diagonal entry, or ``None`` when the wait is not
        positive or the card has no idle data (the naive cards).
    """

    if duration_ns <= 0.0 or not card["idle"]:
        return None
    grid = sorted((float(key), value) for key, value in card["idle"].items())
    for grid_duration, diagonal in grid:
        if math.isclose(grid_duration, duration_ns):
            return {k: max(v, 0.0) for k, v in diagonal.items()}
    lower = [(t, d) for t, d in grid if t < duration_ns]
    upper = [(t, d) for t, d in grid if t > duration_ns]
    if lower and upper:
        (a, da), (b, db) = lower[-1], upper[0]
        w = (duration_ns - a) / (b - a)
        return {
            k: float(
                math.exp(
                    (1.0 - w) * math.log(max(da[k], 1e-300))
                    + w * math.log(max(db[k], 1e-300))
                )
            )
            for k in da
        }
    base_duration, diagonal = lower[-1] if lower else upper[0]
    power = duration_ns / base_duration
    return {k: max(v, 0.0) ** power for k, v in diagonal.items()}


def idle_pauli_probabilities(
    card: dict[str, Any], duration_ns: float
) -> dict[str, float] | None:
    """The idle channel for one wait, as Pauli error probabilities.

    Inverts the single-qubit twirl, ``p_P = (1/4) sum_Q chi(P, Q) R_QQ``.

    Args:
        card: A card from :func:`load_device_card`.
        duration_ns: How long the wire waits.

    Returns:
        ``{"I": ..., "X": ..., "Y": ..., "Z": ...}``, or ``None`` when there
        is no idle noise (see :func:`idle_ptm_diagonal`).
    """

    diagonal = idle_ptm_diagonal(card, duration_ns)
    if diagonal is None:
        return None
    labels = ["I", "X", "Y", "Z"]
    chi = np.array(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float
    )
    probabilities = chi @ np.array([diagonal[label] for label in labels]) / 4.0
    return dict(zip(labels, (float(p) for p in probabilities)))


def idle_error_probability(card: dict[str, Any], duration_ns: float) -> float:
    """The chance a wire waiting ``duration_ns`` suffers any Pauli error.

    Args:
        card: A card from :func:`load_device_card`.
        duration_ns: How long the wire waits.

    Returns:
        ``1 - p_identity`` of the idle channel, at least 0; 0 without idle
        data.
    """

    probabilities = idle_pauli_probabilities(card, duration_ns)
    if probabilities is None:
        return 0.0
    return max(0.0, 1.0 - probabilities["I"])


@dataclass
class CardTimeline:
    """When each translated card gate runs, and how long each wire waits.

    Built by :func:`card_timeline` from a genome's translation. A gate starts
    as soon as every wire it acts on is free (as soon as possible), in the
    genome's gate order.

    Attributes:
        idle_before: For each enabled genome gate (keyed by ``id(gate)``), how
            long each of its card gates' wires waited before them, summed per
            wire: the idle noise added just before that genome gate.
        idle_at_end: For each wire, how long it waits between its last gate
            and the end of the circuit (measurement).
        makespan_ns: When the last gate finishes.
        gates: Every card gate in the translation, encoding first.
    """

    idle_before: dict[int, dict[int, float]] = field(default_factory=dict)
    idle_at_end: list[float] = field(default_factory=list)
    makespan_ns: float = 0.0
    gates: list[ChipGate] = field(default_factory=list)


def card_timeline(
    schedule: CompiledSchedule, card: dict[str, Any], n_wires: int
) -> CardTimeline:
    """Schedules a genome's card gates as soon as possible.

    Waits inside one genome gate's own decomposition (for example between the
    two ``cx`` of a ``crx``) are charged before that genome gate, together
    with the wait before its first card gate: the noise model can only add
    channels between genome gates.

    Args:
        schedule: The genome's translation into card gates.
        card: A card from :func:`load_device_card`.
        n_wires: The number of circuit wires.

    Returns:
        The timeline.
    """

    free_at = [0.0] * n_wires
    timeline = CardTimeline()

    def place(chip_gate: ChipGate) -> dict[int, float]:
        """Starts one card gate as soon as its wires are free.

        Args:
            chip_gate: The card gate.

        Returns:
            How long each of its wires waited for it.
        """
        start = max(free_at[w] for w in chip_gate.wires)
        end = start + gate_duration_ns(card, chip_gate.name)
        waits = {}
        for w in chip_gate.wires:
            if start - free_at[w] > 1e-9:
                waits[w] = start - free_at[w]
            free_at[w] = end
        timeline.gates.append(chip_gate)
        return waits

    for chip_gate in schedule.encoding or []:
        place(chip_gate)
    for gate_key, chip_gates in schedule.per_gate.items():
        waits: dict[int, float] = {}
        for chip_gate in chip_gates:
            for wire, wait in place(chip_gate).items():
                waits[wire] = waits.get(wire, 0.0) + wait
        timeline.idle_before[gate_key] = waits

    timeline.makespan_ns = max(free_at, default=0.0)
    timeline.idle_at_end = [timeline.makespan_ns - t for t in free_at]
    return timeline


def hardware_metrics(timeline: CardTimeline, card: dict[str, Any]) -> dict[str, float]:
    """Simulation-free hardware objectives for a translated genome.

    ``p_fail`` is the chance that at least one card gate or one idle period
    errs, assuming independent errors: one minus the product of every gate's
    and every wire's idle success probability (each wire idles for its total
    waiting time). ``1 / (1 - p_fail)`` is the expected number of shots per
    error-free run. This follows Dr. Vogt's ``hardware_metrics``, counted over
    the translated card gates (routing moves and data loading included).

    Args:
        timeline: The genome's card timeline (:func:`card_timeline`).
        card: A card from :func:`load_device_card`.

    Returns:
        ``p_fail``, ``total_time_ns`` (the makespan, i.e. depth in
        nanoseconds) and ``n_entanglers`` (the number of ``cx``).
    """

    log_success = 0.0
    n_entanglers = 0
    for chip_gate in timeline.gates:
        if len(chip_gate.wires) >= 2:
            n_entanglers += 1
        log_success += _log_one_minus(gate_error_probability(card, chip_gate.name))

    waiting = [0.0] * len(timeline.idle_at_end)
    for waits in timeline.idle_before.values():
        for wire, wait in waits.items():
            waiting[wire] += wait
    for wire, wait in enumerate(timeline.idle_at_end):
        waiting[wire] += wait
    for wait in waiting:
        log_success += _log_one_minus(idle_error_probability(card, wait))

    return {
        "p_fail": 1.0 - math.exp(log_success),
        "total_time_ns": float(timeline.makespan_ns),
        "n_entanglers": float(n_entanglers),
    }


def _log_one_minus(p: float) -> float:
    """``log(1 - p)``, with ``p`` clamped so the result stays finite.

    Args:
        p: A probability.

    Returns:
        ``log1p(-p)`` for ``p`` clamped to ``[0, 1 - 1e-16]``.
    """

    return math.log1p(-min(max(p, 0.0), 1.0 - 1e-16))
