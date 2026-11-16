"""Chip noise: simulate one real IBM chip, qubit by qubit.

IBM measures every qubit of a chip each day (how long it holds its state,
how often each gate fails, how often a measurement reads wrong) and
``src.examples.collect_calibration`` stores that as one JSON file per chip and
day. A run picks one file with ``--noise_calibration`` and the chip's qubits
its circuit sits on with ``--noise_qubits`` (wire 0 on the first listed
qubit, and so on). The code here reads those qubits' numbers and adds noise
of exactly that size while the circuit is simulated, so a circuit is scored
as it would run on those qubits that day.

Flow: :meth:`ChipChannels.from_calibration` reads the file once;
:meth:`~ChipChannels._values_for_wires` looks up a gate's numbers; the
``apply`` methods insert the noise. Hand-set noise for parameter sweeps lives
in :mod:`src.noise.synthetic_channels`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pennylane as qml

from src.noise.base_noise_model import BaseNoiseModel
from src.noise.calibration.extract import UNUSABLE_ERROR

if TYPE_CHECKING:
    from src.noise.compiled_schedule import ChipGate

_PAULIS = [
    np.eye(2),
    np.array([[0.0, 1.0], [1.0, 0.0]]),
    np.array([[0.0, -1.0j], [1.0j, 0.0]]),
    np.array([[1.0, 0.0], [0.0, -1.0]]),
]


def _two_qubit_depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Kraus operators of a two-qubit depolarizing channel.

    With probability ``p`` one of the 15 non-identity two-qubit Pauli errors
    happens, each equally likely; otherwise nothing. This is PennyLane's
    ``p`` convention for two qubits (``p = 15 lam / 16`` for Qiskit Aer's
    ``depolarizing_error(lam, 2)``), written out because PennyLane has no
    built-in two-qubit depolarizing channel.

    Args:
        p: Total error probability, between 0 and 1.

    Returns:
        The 16 Kraus operators, 4x4 each: ``sqrt(1 - p) I`` and
        ``sqrt(p / 15) P_a (x) P_b`` for the 15 non-identity Pauli pairs.
    """
    kraus = []
    for a, pa in enumerate(_PAULIS):
        for b, pb in enumerate(_PAULIS):
            weight = 1.0 - p if a == b == 0 else p / 15.0
            kraus.append(math.sqrt(weight) * np.kron(pa, pb))
    return kraus


class ChipChannels(BaseNoiseModel):
    """The chip-noise half of the noise model (``noise_type="ibm_backend"``).

    Not used on its own: :class:`src.noise.pennylane_noise_model.PennyLaneNoiseModel`
    inherits it, and the circuit builder calls its hooks.
    """

    @classmethod
    def from_calibration(
        cls,
        raw: dict[str, Any],
        physical_qubits: list[int],
        noise_type: str = "ibm_backend",
    ) -> Self:
        """Reads the chosen qubits' numbers from one calibration file.

        Runs once, at the start of a run. For each qubit in
        ``physical_qubits`` it keeps that qubit's own T1 and T2 (how long it
        holds its state), ``sx`` error and duration (its one-qubit gate),
        readout error in both directions and start-up error; for every pair
        of chosen qubits the chip wires together, the ``cz`` error and
        duration (its two-qubit gate). Nothing is averaged. Neighbours in the
        list must be wired together.

        It stops with a clear message, rather than simulating a missing or
        meaningless number, if a qubit lacks a value or IBM marked a needed
        gate unusable (error exactly 1.0). Stopping here costs nothing: no
        genome has been evolved yet.

        Args:
            raw: The calibration file's contents, from ``load_raw_calibration``.
            physical_qubits: The chip's qubits the circuit sits on, in wire
                order, e.g. ``[10, 11, 12, 13]``.
            noise_type: The noise channel these numbers drive.

        Returns:
            The noise model, with the numbers in ``per_qubit`` (one list
            entry per wire, pairs keyed like ``"10-11"``) and the file's date
            and chip name recorded.

        Raises:
            ValueError: If a qubit is not on the chip, neighbours in the list
                are not wired together, a needed value is missing, or a
                needed gate is marked unusable.
        """
        from src.noise.calibration.backends import properties_from_raw

        props = properties_from_raw(raw)
        chip, n_qubits = raw["backend"]["name"], raw["backend"]["n_qubits"]

        when = raw["calibration_date"]

        for q in physical_qubits:
            if not 0 <= q < n_qubits:
                raise ValueError(f"qubit {q} is not on {chip} (0..{n_qubits - 1}).")

        def qubit_value(q: int, name: str) -> float:
            """Returns one of a qubit's calibrated values, or stops clearly.

            Args:
                q: The physical qubit.
                name: The property, e.g. ``"T1"``, ``"T2"`` or
                    ``"readout_error"``.

            Returns:
                The value, in SI units (seconds for T1 and T2).

            Raises:
                ValueError: If the file has no such value for the qubit.
            """
            try:
                return float(props.qubit_property(q, name)[0])
            except Exception:
                raise ValueError(
                    f"qubit {q} has no {name} in the {chip} calibration of {when}; "
                    "choose other qubits or another day."
                )

        def gate_values(gate: str, qubits: list[int]) -> tuple[float, float]:
            """Returns a gate's calibrated error and duration, or stops clearly.

            Args:
                gate: The gate name, ``"sx"`` or ``"cz"``.
                qubits: The physical qubit, or pair, it acts on.

            Returns:
                ``(error, duration)``, the duration in seconds.

            Raises:
                ValueError: If the file has no such gate on those qubits
                    (for ``cz``: they are not wired together), or marks it
                    unusable with an error of exactly
                    :data:`~src.noise.calibration.extract.UNUSABLE_ERROR`.
            """
            if len(qubits) == 1:
                where, verb = f"qubit {qubits[0]}", "has"
            else:
                where, verb = f"qubits {qubits[0]} and {qubits[1]}", "have"
            try:
                error = float(props.gate_error(gate, qubits))
                length = float(props.gate_length(gate, qubits))
            except Exception:
                raise ValueError(f"{where} {verb} no calibrated {gate} on {chip}.")
            if error >= UNUSABLE_ERROR:
                raise ValueError(
                    f"the {gate} on {where} is marked unusable (error 1.0) in the "
                    f"{chip} calibration of {when}; choose other qubits or another day."
                )
            return error, length

        pairs: dict[str, dict[str, float]] = {}
        for a, b in zip(physical_qubits, physical_qubits[1:]):
            error, length = gate_values("cz", [a, b])
            pairs[f"{a}-{b}"] = {"gate_error_2q": error, "gate_time_2q": length}

        # Every other pair of listed qubits the chip wires together, keyed
        # with the earlier-listed qubit first, as the consecutive pairs are.
        position = {q: i for i, q in enumerate(physical_qubits)}
        for a, b in raw["backend"]["coupling_map"]:
            if a not in position or b not in position or position[a] > position[b]:
                continue
            if f"{a}-{b}" in pairs:
                continue
            error, length = gate_values("cz", [a, b])
            pairs[f"{a}-{b}"] = {"gate_error_2q": error, "gate_time_2q": length}

        def readout_direction(q: int, name: str) -> float:
            """Returns one direction of a qubit's readout error.

            Args:
                q: The physical qubit.
                name: ``"prob_meas1_prep0"`` (reads 1 when it was 0) or
                    ``"prob_meas0_prep1"`` (reads 0 when it was 1).

            Returns:
                That probability, or the qubit's averaged ``readout_error``
                when the file does not list it (Qiskit Aer's fallback).
            """
            try:
                return float(props.qubit_property(q, name)[0])
            except Exception:
                return qubit_value(q, "readout_error")

        def start_up_error(q: int) -> float:
            """Returns a qubit's start-up error, or 0 when IBM lists none.

            Args:
                q: The physical qubit.

            Returns:
                The chance it starts in 1 instead of 0 (``init_error``); IBM
                omits it for some qubits (36 of Fez's 156 on 2026-11-06), so
                no reported error counts as none.
            """
            try:
                return float(props.qubit_property(q, "init_error")[0])
            except Exception:
                return 0.0

        sx = [gate_values("sx", [q]) for q in physical_qubits]
        model = cls(noise_type=noise_type)
        model.per_qubit = {
            "qubits": list(physical_qubits),
            "t1": [qubit_value(q, "T1") for q in physical_qubits],
            "t2": [qubit_value(q, "T2") for q in physical_qubits],
            "readout_error": [qubit_value(q, "readout_error") for q in physical_qubits],
            "prob_meas1_prep0": [
                readout_direction(q, "prob_meas1_prep0") for q in physical_qubits
            ],
            "prob_meas0_prep1": [
                readout_direction(q, "prob_meas0_prep1") for q in physical_qubits
            ],
            "init_error": [start_up_error(q) for q in physical_qubits],
            "gate_error_1q": [error for error, _ in sx],
            "gate_time_1q": [length for _, length in sx],
            "pairs": pairs,
        }
        model.calibration_date = raw["calibration_date"]
        model.ibm_backend_name = chip
        # the chip's own gates, for translating genomes (compiled_schedule)
        from src.noise.compiled_schedule import chip_gates_from_basis

        model.chip_gates = chip_gates_from_basis(raw["backend"]["basis_gates"])
        return model

    def _values_for_wires(self, wires: Sequence[int]) -> dict[str, Any]:
        """Looks up the noise numbers for the qubit, or pair, a gate acts on.

        The noise channels decide what kind of noise to add; this says how
        much. With a calibration file loaded, wire ``i`` is physical qubit
        ``per_qubit["qubits"][i]`` and the numbers are that qubit's (one
        wire) or that pair's (two wires). Without one, every wire gets the
        hand-set values from the flags.

        Args:
            wires: The gate's circuit wires: one for a one-qubit gate, two for
                a two-qubit gate.

        Returns:
            ``gate_error`` and ``gate_time`` for the gate (the qubit's ``sx``
            or the pair's ``cz``), plus one entry per wire, in the order
            given, for ``t1``, ``t2``, ``readout_error`` (average),
            ``prob_meas1_prep0`` (reads 1 when it was 0), ``prob_meas0_prep1``
            (reads 0 when it was 1) and ``init_error`` (start-up). Hand-set
            values have no readout or start-up error, so those are 0.

        Raises:
            ValueError: If no wires are given, chip noise has no file loaded,
                a wire is beyond the chosen qubits, the gate has more than two
                wires (it must first be split into the chip's own gates), or
                its two qubits are not wired together (it needs routing).
        """
        wires = list(wires)
        n_wires = len(wires)
        if n_wires == 0:
            raise ValueError("a gate needs at least one wire.")

        if self.per_qubit is None:
            # Backup to the mode check in ``from_args``: chip noise must not
            # quietly fall back to the hand-set defaults (e.g. a genome whose
            # saved settings lost the calibration values).
            if self.noise_type == "ibm_backend":
                raise ValueError(
                    "noise_type 'ibm_backend' has no calibration values loaded; "
                    "build the model with --noise_calibration and --noise_qubits."
                )
            return {
                "gate_error": float(self.probability_for_n_wires(n_wires)),
                "gate_time": float(
                    self.gate_time_1q if n_wires == 1 else self.gate_time_2q
                ),
                "t1": [float(self.t1)] * n_wires,
                "t2": [float(self.t2)] * n_wires,
                "readout_error": [0.0] * n_wires,
                "prob_meas1_prep0": [0.0] * n_wires,
                "prob_meas0_prep1": [0.0] * n_wires,
                "init_error": [0.0] * n_wires,
            }

        qubits = self.per_qubit["qubits"]
        for w in wires:
            if not 0 <= w < len(qubits):
                raise ValueError(
                    f"wire {w} is not one of the {len(qubits)} calibrated wires."
                )
        if n_wires > 2:
            raise ValueError(
                f"a gate on {n_wires} wires has no calibrated error; split it "
                "into the chip's own one- and two-qubit gates first."
            )

        if n_wires == 1:
            gate_error = self.per_qubit["gate_error_1q"][wires[0]]
            gate_time = self.per_qubit["gate_time_1q"][wires[0]]
        else:
            a, b = qubits[wires[0]], qubits[wires[1]]
            pairs = self.per_qubit["pairs"]
            pair = pairs.get(f"{a}-{b}") or pairs.get(f"{b}-{a}")
            if pair is None:
                raise ValueError(
                    f"wires {wires[0]} and {wires[1]} sit on qubits {a} and {b}, "
                    "which are not wired together; the gate needs routing."
                )
            gate_error = pair["gate_error_2q"]
            gate_time = pair["gate_time_2q"]

        return {
            "gate_error": float(gate_error),
            "gate_time": float(gate_time),
            "t1": [float(self.per_qubit["t1"][w]) for w in wires],
            "t2": [float(self.per_qubit["t2"][w]) for w in wires],
            "readout_error": [float(self.per_qubit["readout_error"][w]) for w in wires],
            "prob_meas1_prep0": [
                float(self.per_qubit["prob_meas1_prep0"][w]) for w in wires
            ],
            "prob_meas0_prep1": [
                float(self.per_qubit["prob_meas0_prep1"][w]) for w in wires
            ],
            "init_error": [float(self.per_qubit["init_error"][w]) for w in wires],
        }

    @staticmethod
    def _relaxation_entanglement_fidelity(t1: float, t2: float, tg: float) -> float:
        """Scores how much of a qubit's state survives decay during one gate.

        1 means untouched. The formula is the process fidelity of thermal
        relaxation, ``(1 + e^(-tg/T1) + 2 e^(-tg/T2)) / 4``; it matches Qiskit's
        exact calculation. For a two-qubit gate, multiply the two qubits'
        scores, since each decays on its own.

        Args:
            t1: Relaxation time in seconds.
            t2: Dephasing time in seconds.
            tg: Gate duration in seconds.

        Returns:
            The entanglement fidelity, between 0.25 and 1.
        """
        return (1.0 + math.exp(-tg / t1) + 2.0 * math.exp(-tg / t2)) / 4.0

    def _apply_ibm_backend(self, wires: list[int]) -> None:
        """Adds a gate's real chip error after it, sized to what IBM measured.

        Two parts, the same rule Qiskit Aer uses for its chip models:

        1. Decay: each qubit loses some of its state for the gate's duration,
           with its own T1 and T2.
        2. Remainder: IBM's measured gate error also includes causes other
           than decay, so the rest is added as depolarizing (random) noise,
           sized so decay plus remainder equals the measured error exactly.

        Example, ``sx`` on Fez qubit 11 on 2026-11-06: decay explains 0.0003
        of the measured 0.0015, the remainder adds the other 0.0012. The
        strength is converted from Aer's convention to PennyLane's
        (``3/4`` for one qubit, ``15/16`` for two).

        Args:
            wires: The wires the gate acted on.

        Returns:
            None. Inserts the relaxation and depolarizing channels into the
            QNode being built, or nothing for no wires or a non-positive
            time.

        Raises:
            ValueError: From :meth:`_values_for_wires`, e.g. a gate on more
                than two wires or on two qubits that are not wired together.
        """
        if len(wires) == 0:
            return
        self._apply_calibrated(self._values_for_wires(wires), list(wires))

    def _apply_chip_gate(self, gate: ChipGate) -> None:
        """Adds the noise of one gate the chip really runs (from the translation).

        ``rz`` is virtual on IBM chips (error and duration 0), so it adds
        nothing. ``sx`` and ``x`` get their qubit's ``sx`` error and duration
        (IBM reports the same error for both); ``cz`` gets its pair's error
        and duration. The numbers are the chip qubits' own (``gate.slots``),
        and the noise goes on the genome wires whose states sit there
        (``gate.wires``), which differ after a routing move.

        Args:
            gate: One entry of a :class:`~src.noise.compiled_schedule.CompiledSchedule`.

        Returns:
            None. Inserts decay plus the depolarizing remainder, or nothing
            for ``rz``.
        """
        if gate.name == "rz":
            return
        self._apply_calibrated(self._values_for_wires(gate.slots), gate.wires)

    def _apply_calibrated(self, values: dict[str, Any], wires: list[int]) -> None:
        """Inserts decay plus the depolarizing remainder sized by ``values``.

        The shared core of :meth:`_apply_ibm_backend` and
        :meth:`_apply_chip_gate` (see the former for the rule).

        Args:
            values: The gate's numbers from :meth:`_values_for_wires`.
            wires: The wires to put the noise on, one per entry of
                ``values["t1"]``.

        Returns:
            None. Inserts the channels, or nothing for a non-positive time.
        """
        n_wires = len(wires)
        tg = values["gate_time"]
        pe = min(max(float(self.excited_state_population), 0.0), 1.0)
        if tg <= 0.0:
            return

        # Clamp each qubit's values to physical ranges, checking all of them
        # before inserting anything.
        per_wire: list[tuple[float, float, float]] = []
        for t1, t2 in zip(values["t1"], values["t2"]):
            if t1 <= 0.0 or t2 <= 0.0:
                return
            t2 = min(t2, 2.0 * t1)
            per_wire.append((t1, t2, min(tg, 0.999 * min(t1, t2))))

        # Each qubit relaxes with its own T1/T2 for the gate's duration.
        relax_process_fidelity = 1.0
        for w, (t1, t2, tg_w) in zip(wires, per_wire):
            qml.ThermalRelaxationError(pe, t1, t2, tg_w, wires=w)
            relax_process_fidelity *= self._relaxation_entanglement_fidelity(
                t1, t2, tg_w
            )

        # How much of the measured error the relaxation already explains.
        dim = 2**n_wires
        relax_fidelity = (dim * relax_process_fidelity + 1.0) / (dim + 1.0)
        gate_error = min(values["gate_error"], dim / (dim + 1.0))
        if gate_error <= 1.0 - relax_fidelity:
            return

        # The rest as depolarizing noise (Aer's rule, PennyLane's convention).
        aer_lambda = (
            dim * (gate_error - (1.0 - relax_fidelity)) / (dim * relax_fidelity - 1.0)
        )
        if n_wires == 1:
            qml.DepolarizingChannel(min(0.75 * aer_lambda, 1.0), wires=wires[0])
        else:
            # PennyLane has no built-in two-qubit depolarizing channel, so it
            # is written out from its Kraus operators.
            qml.QubitChannel(
                _two_qubit_depolarizing_kraus(min(aer_lambda * 15.0 / 16.0, 1.0)),
                wires=list(wires),
            )

    def apply_initialization_error(self, wires: Sequence[int]) -> None:
        """Adds each qubit's start-up error, before the data is loaded.

        A real qubit does not always start exactly in 0. IBM reports the
        chance it starts in 1 as ``init_error`` (Fez on 2026-11-06: 0.0002
        to 0.021, median 0.002). Every qubit starts in 0, so a bit flip with that chance is
        exact. Qiskit Aer's chip models leave this out, so circuits come out
        slightly noisier here than in Aer.

        Args:
            wires: The circuit's wires.

        Returns:
            None. Inserts one bit flip per wire with a non-zero start-up
            error.

        Raises:
            ValueError: From :meth:`_values_for_wires`, e.g. if chip noise has
                no calibration file loaded.
        """
        for w in wires:
            p = self._values_for_wires([w])["init_error"][0]
            if p > 0.0:
                qml.BitFlip(p, wires=w)

    def apply_readout_error(
        self,
        measurement_wires: Sequence[int],
        slots: Sequence[int] | None = None,
    ) -> None:
        """Adds each measured qubit's readout error, just before measuring.

        A real chip sometimes reads a 0 as 1 and a 1 as 0, at different rates
        (``prob_meas1_prep0`` and ``prob_meas0_prep1`` in the file; Fez qubit
        11: 0.025 and 0.018). Each measured wire gets a channel flipping its
        result with those two chances, as Qiskit Aer does. The channel also
        erases superpositions, which is harmless because a measurement in the
        0/1 basis follows at once.

        Args:
            measurement_wires: Wires being measured.
            slots: For each measured wire, the chip qubit (position in
                ``--noise_qubits``) its state ends on, whose readout error
                applies; after routing moves that can differ from the wire.
                ``None`` means each wire's own position.

        Returns:
            None. Inserts one readout channel per measured wire into the
            QNode being built.

        Raises:
            ValueError: From :meth:`_values_for_wires`, e.g. if
                ``noise_type`` is ``"ibm_backend"`` but no calibration values
                are loaded.
        """
        wires = list(measurement_wires)
        slots = wires if slots is None else list(slots)
        for w, s in zip(wires, slots):
            values = self._values_for_wires([s])
            a = values["prob_meas1_prep0"][0]
            b = values["prob_meas0_prep1"][0]
            kraus = [
                np.array([[math.sqrt(1.0 - a), 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [math.sqrt(a), 0.0]]),
                np.array([[0.0, math.sqrt(b)], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, math.sqrt(1.0 - b)]]),
            ]
            qml.QubitChannel(kraus, wires=w)
