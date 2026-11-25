"""Device cards built from an IBM chip's calibration, for when no card exists.

Dr. Vogt's device cards come from simulating each gate's control pulse
(:mod:`src.noise.device_card`). Without one, a card of the same shape can be
built from what IBM measures on a real chip every day: each qubit's T1 and T2,
and each gate's error and duration. Device-card noise and its hardware
objectives (``p_fail``, ``total_time_ns``, ``n_entanglers``) then run against
that chip instead.

The card treats every qubit alike, so it takes the median of the chosen
qubits' values (``--noise_qubits``) and of the gates between them. Each card
gate's channel is the one chip noise uses -- the qubits relax (T1, T2) for the
gate's duration, and a depolarizing remainder makes the total equal IBM's
measured error -- Pauli-twirled into the card's form:

- ``h`` and ``x`` are one ``sx`` (or ``x``) pulse on the chip (``h`` is
  ``rz sx rz``), so they get the ``sx`` error and duration.
- ``cx`` is ``cz`` with an ``sx``-sized gate before and after on the target
  (``h cz h``), so it gets the ``cz`` duration plus two ``sx`` durations, and
  the error of all three.
- Idling for ``t`` relaxes a qubit by ``exp(-t/T2)`` in X and Y and
  ``exp(-t/T1)`` in Z.

Unlike a simulated card, this one knows only IBM's averages: no leakage, no
memory effects, and an even split of the unexplained error over the Paulis.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

#: The idle durations (ns) a calibration card records, as Dr. Vogt's cards do.
IDLE_GRID_NS: tuple[float, ...] = tuple(50.0 * k for k in range(1, 25))


def _relaxation_pauli_probabilities(
    t1: float, t2: float, duration: float
) -> dict[str, float]:
    """The Pauli-twirled thermal-relaxation channel of one qubit.

    Args:
        t1: Relaxation time in seconds.
        t2: Dephasing time in seconds (capped at ``2 * t1``).
        duration: How long the qubit relaxes, in seconds.

    Returns:
        ``{"I", "X", "Y", "Z"}`` probabilities: ``p_X = p_Y = (1 - e^(-t/T1)) / 4``
        and ``p_Z = (1 - e^(-t/T2)) / 2 - p_X``.
    """

    t2 = min(t2, 2.0 * t1)
    amplitude = 1.0 - math.exp(-duration / t1)
    p_x = amplitude / 4.0
    p_z = max(0.0, (1.0 - math.exp(-duration / t2)) / 2.0 - p_x)
    return {"I": 1.0 - 2.0 * p_x - p_z, "X": p_x, "Y": p_x, "Z": p_z}


def _gate_pauli_probabilities(
    gate_error: float, relaxations: list[dict[str, float]]
) -> dict[str, float]:
    """A gate's Pauli channel: relaxation plus a depolarizing remainder.

    The relaxation of each wire is combined into one Pauli channel; the error
    IBM measured but relaxation does not explain is spread evenly over the
    non-identity Paulis, so the channel's average gate infidelity equals
    ``gate_error`` (the rule Qiskit Aer and chip noise use).

    Args:
        gate_error: IBM's average gate error for the gate.
        relaxations: Each wire's relaxation channel, first wire first.

    Returns:
        Pauli string (one letter per wire) -> probability.
    """

    combined = {"": 1.0}
    for channel in relaxations:
        combined = {
            label + letter: p * q
            for label, p in combined.items()
            for letter, q in channel.items()
        }
    n_wires = len(relaxations)
    dim = 2**n_wires
    identity = "I" * n_wires
    # average gate infidelity -> process (entanglement) infidelity
    target_process_error = min(gate_error * (dim + 1) / dim, 1.0 - 1.0 / dim**2)
    relaxation_error = 1.0 - combined[identity]
    if target_process_error <= relaxation_error:
        return combined
    # depolarize the relaxed channel by lam: p_I' = (1 - lam) p_I + lam / dim^2
    lam = (target_process_error - relaxation_error) / (
        combined[identity] - 1.0 / dim**2
    )
    return {label: (1.0 - lam) * p + lam / dim**2 for label, p in combined.items()}


def card_from_chip(
    per_qubit: dict[str, Any], chip: str, date: str | None
) -> dict[str, Any]:
    """Builds a device card from chip noise's per-qubit calibration values.

    Args:
        per_qubit: The chosen qubits' values, as
            :meth:`src.noise.chip_channels.ChipChannels.from_calibration` reads
            them (``t1``, ``t2``, ``gate_error_1q``, ``gate_time_1q`` per qubit,
            and ``pairs`` with ``gate_error_2q`` and ``gate_time_2q``).
        chip: The chip's name, e.g. ``"ibm_fez"``.
        date: The calibration's date.

    Returns:
        A card in the compact form of
        :func:`src.noise.device_card.load_device_card`, named
        ``calibration_<chip>_<date>``, with ``wired_pairs`` set to ``None``
        (the caller sets the wiring).

    Raises:
        ValueError: If no wired pair of the chosen qubits has a two-qubit gate.
    """

    if not per_qubit["pairs"]:
        raise ValueError(
            "none of the chosen qubits are wired together, so the chip has no "
            "two-qubit gate to build the card's cx from."
        )
    t1 = statistics.median(per_qubit["t1"])
    t2 = statistics.median(per_qubit["t2"])
    error_1q = statistics.median(per_qubit["gate_error_1q"])
    time_1q = statistics.median(per_qubit["gate_time_1q"])
    error_2q = statistics.median(
        p["gate_error_2q"] for p in per_qubit["pairs"].values()
    )
    time_2q = statistics.median(p["gate_time_2q"] for p in per_qubit["pairs"].values())

    relax_1q = _relaxation_pauli_probabilities(t1, t2, time_1q)
    one_qubit = _gate_pauli_probabilities(error_1q, [relax_1q])

    time_cx = time_2q + 2.0 * time_1q
    error_cx = 1.0 - (1.0 - error_2q) * (1.0 - error_1q) ** 2
    relax_cx = _relaxation_pauli_probabilities(t1, t2, time_cx)
    two_qubit = _gate_pauli_probabilities(error_cx, [relax_cx, relax_cx])

    idle = {}
    for duration_ns in IDLE_GRID_NS:
        seconds = duration_ns * 1e-9
        decay_xy = math.exp(-seconds / min(t2, 2.0 * t1))
        decay_z = math.exp(-seconds / t1)
        idle[str(duration_ns)] = {"I": 1.0, "X": decay_xy, "Y": decay_xy, "Z": decay_z}

    # the day is enough to name it; the full timestamp is kept below
    name = f"calibration_{chip}" + (f"_{date[:10]}" if date else "")
    return {
        "name": name,
        "file": None,
        "architecture": chip,
        "model": "ibm_calibration",
        "gates": {
            "h": {"T_ns": time_1q * 1e9, "wires": 1, "pauli_error": one_qubit},
            "x": {"T_ns": time_1q * 1e9, "wires": 1, "pauli_error": dict(one_qubit)},
            "cx": {"T_ns": time_cx * 1e9, "wires": 2, "pauli_error": two_qubit},
        },
        "idle": idle,
        "wired_pairs": None,
        "calibration": {
            "chip": chip,
            "date": date,
            "qubits": list(per_qubit["qubits"]),
            "t1": t1,
            "t2": t2,
            "gate_error_1q": error_1q,
            "gate_time_1q": time_1q,
            "gate_error_2q": error_2q,
            "gate_time_2q": time_2q,
        },
    }
