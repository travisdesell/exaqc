"""The PennyLane noise model: the parts shared by every noise family.

The channels themselves live in one file per family:

- :mod:`src.noise.synthetic_channels`: hand-set values from the
  ``--noise_type`` flags, for parameter sweeps (depolarizing, bit flip, phase
  flip, amplitude and phase damping, mixed, thermal relaxation).
- :mod:`src.noise.chip_channels`: per-qubit values from one stored IBM
  calibration file, named with ``--noise_calibration`` and
  ``--noise_qubits`` (``from_calibration``, ``ibm_backend``, readout error).
- :mod:`src.noise.device_card_channels`: per-gate channels and idle noise from
  an optimal-control device card, named with ``--noise_device_card``
  (``device_card``).

This file joins them into :class:`PennyLaneNoiseModel` and holds what they
share: the device choice, the dispatch from ``noise_type`` to a channel, and the
hooks the circuit builder calls.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Sequence

from src.noise.chip_channels import ChipChannels
from src.noise.device_card_channels import DeviceCardChannels
from src.noise.synthetic_channels import SyntheticChannels

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome
    from src.circuits.gate import Gate
    from src.noise.compiled_schedule import ChipGate


class PennyLaneNoiseModel(SyntheticChannels, ChipChannels, DeviceCardChannels):
    """PennyLane implementation of the backend-agnostic noise model.

    This class provides PennyLane-native quantum noise injection utilities.
    Noise channels are inserted directly into PennyLane QNodes and can be
    applied after input encoding, after evolved quantum gates, or immediately
    before measurement.

    The channels come from two families, inherited from one file each:

    - Synthetic (:class:`~src.noise.synthetic_channels.SyntheticChannels`):
      depolarizing, bit flip, phase flip, amplitude damping, phase damping,
      mixed and thermal relaxation, all from hand-set values.
    - Chip (:class:`~src.noise.chip_channels.ChipChannels`): calibrated IBM
      gate errors (``ibm_backend``: thermal relaxation plus the depolarizing
      remainder, as Qiskit Aer builds them) and readout error, from one
      calibration file's per-qubit values.
    - Device card (:class:`~src.noise.device_card_channels.DeviceCardChannels`):
      each translated card gate's own Pauli channel, plus idle noise on
      waiting wires (``device_card``).

    The methods defined here are shared by every family.

    Notes:
        PennyLane noisy simulations generally require the
        ``default.mixed`` backend instead of ``default.qubit``.
    """

    def backend_name(self) -> str:
        """Return the backend identifier.

        Returns:
            The string ``"pennylane"``.
        """
        return "pennylane"

    def device_name(self) -> str:
        """Return the PennyLane device required for this noise model.

        Returns:
            ``"default.mixed"`` if noise is enabled, otherwise
            ``"default.qubit"``.
        """
        return "default.mixed" if self.is_noisy() else "default.qubit"

    def _as_wires(self, wires: Sequence[int] | int) -> list[int]:
        """Normalize wire specifications into a list format.

        Args:
            wires:
                Single wire index or iterable of wire indices.

        Returns:
            List of integer wire indices.
        """
        if isinstance(wires, int):
            return [wires]

        return list(wires)

    def apply_to_wires(self, wires: Sequence[int] | int) -> None:
        """Apply the configured noise process to selected wires.

        This method dispatches the configured ``noise_type`` to the
        corresponding PennyLane quantum channel implementation.

        Supported noise models include:

        - ``depolarizing``
        - ``bit_flip``
        - ``phase_flip``
        - ``amplitude_damping``
        - ``phase_damping``
        - ``thermal_relaxation``
        - ``mixed``
        - ``ibm_backend`` (relaxation plus a depolarizing remainder matching
          the calibrated gate error, see :meth:`_apply_ibm_backend`)

        ``device_card`` has no per-wire channel: its noise depends on the
        gate, so it is added by the hooks from the genome's translation.

        Args:
            wires:
                Single wire or collection of wires where the noise
                process should be inserted.

        Returns:
            None. Inserts the chosen channel into the QNode being built, or
            nothing when the model is noiseless.

        Raises:
            ValueError:
                If an unsupported noise type is specified, or the type is
                ``device_card``.
        """
        if not self.is_noisy():
            return
        if self.noise_type == "device_card":
            raise ValueError(
                "device_card noise depends on the gate; it is added by the hooks "
                "from the genome's translation (begin_circuit), not per wire."
            )

        wires = self._as_wires(wires)

        noise_handlers = {
            "depolarizing": self._apply_depolarizing,
            "bit_flip": self._apply_bit_flip,
            "phase_flip": self._apply_phase_flip,
            "amplitude_damping": self._apply_amplitude_damping,
            "phase_damping": self._apply_phase_damping,
            "thermal_relaxation": self._apply_thermal_relaxation,
            "mixed": self._apply_mixed,
            "ibm_backend": self._apply_ibm_backend,
        }

        if self.noise_type in noise_handlers:
            noise_handlers[self.noise_type](wires)
        else:
            raise ValueError(f"Unknown noise_type={self.noise_type}")

    def before_input_encoding(self, wires: Sequence[int]) -> None:
        """Apply noise to the qubits' starting state, before the data is loaded.

        Chip noise (``ibm_backend``) only: each qubit's start-up error from
        the calibration file (:meth:`apply_initialization_error`). Synthetic
        noise and no noise add nothing here.

        Args:
            wires: All of the circuit's wires.

        Returns:
            None. Inserts the start-up channels into the QNode being built,
            or nothing.
        """
        if self.noise_type == "ibm_backend":
            self.apply_initialization_error(wires)

    def after_input_encoding(self, input_wires: Sequence[int]) -> None:
        """Apply noise after classical input encoding.

        Chip noise (``ibm_backend``): always adds one gate's worth of chip
        noise to each input wire separately, with that qubit's own ``sx``
        error, duration, T1 and T2 from the calibration file, whatever
        ``--noise_after_encoding`` says, since on a chip the encoding gates
        are noisy like any other.

        Device-card noise: always adds the card channels of the gates the
        encoding was translated into (``u3``/``rx``/``ry``: two ``h`` per input
        qubit; ``rz``: virtual, none), or one ``h`` worth per input qubit for
        ``basis`` and ``amplitude``.

        Synthetic noise: when ``--noise_after_encoding`` is on (off by
        default), applies the configured channel once to the input wires;
        otherwise nothing.

        For chip noise the amount is exact when the encoding's gates do not
        depend on the data sample (``u3``, ``rx``, ``ry``, ``rz``): the
        translation in :meth:`begin_circuit` turns them into chip gates
        (``u3``/``rx``/``ry``: 2 ``sx`` per input qubit; ``rz``: virtual, no
        noise) and each chip gate's noise is added. For ``basis`` and
        ``amplitude``, whose gates change with each data sample, each input
        qubit gets one ``sx`` worth instead.

        Args:
            input_wires:
                Wires used for classical data encoding.

        Returns:
            None. Inserts the noise channel on the input wires into the QNode
            being built, or nothing when encoding noise is off.
        """
        if self.noise_type == "device_card":
            schedule = getattr(self, "_schedule", None)
            if schedule is not None and schedule.encoding is not None:
                for chip_gate in schedule.encoding:
                    self._apply_card_gate(chip_gate)
            else:
                for w in input_wires:
                    self._apply_card_gate_named("h", w)
            return
        if self.noise_type == "ibm_backend":
            schedule = getattr(self, "_schedule", None)
            if schedule is not None and schedule.encoding is not None:
                # exact: the chip gates the encoding was translated into
                for chip_gate in schedule.encoding:
                    self._apply_chip_gate(chip_gate)
                return
            # otherwise one sx worth per input qubit, one wire at a time (the
            # chip channel charges a qubit's own sx, and refuses > 2 wires)
            for w in input_wires:
                self.apply_to_wires([w])
        elif self.apply_after_input_encoding:
            self.apply_to_wires(input_wires)

    def before_gate(self, gate_wires: Sequence[int], gate: Gate | None = None) -> None:
        """Apply noise just before an evolved quantum gate runs.

        Device-card noise only (with idle noise on): each wire this gate's card
        gates act on gets the card's idle channel for as long as it waited
        for them in the genome's as-soon-as-possible timeline
        (:func:`src.noise.device_card.card_timeline`). Adding it here, on the
        state the wire held while it waited, rather than all at the end, is
        what makes the idle model right. Every other noise type adds nothing.

        Args:
            gate_wires: Wires associated with the evolved gate.
            gate: The gate about to run; used to find its waits.

        Returns:
            None. Inserts the idle channels into the circuit being built, or
            nothing.
        """
        if self.noise_type != "device_card" or gate is None:
            return
        timeline = getattr(self, "_card_timeline", None)
        if timeline is None:
            return
        for wire, wait in sorted(timeline.idle_before.get(id(gate), {}).items()):
            self._apply_idle(wire, wait)

    def after_gate(self, gate_wires: Sequence[int], gate: Gate | None = None) -> None:
        """Apply noise after an evolved quantum gate.

        Chip noise, once :meth:`begin_circuit` has translated the genome: adds
        the noise of every chip gate this genome gate became, routing moves
        included, each on its own chip qubits' numbers (nothing for ``rz``,
        the qubit's ``sx`` error for ``sx`` or ``x``, the pair's ``cz`` error
        for ``cz``). So a ``cx`` between unwired qubits, or a three-qubit
        gate, is charged its real cost instead of stopping the run.

        Device-card noise: the same, with each card gate's own Pauli channel
        from the card (nothing for the virtual ``rz``).

        Otherwise (synthetic noise, or no translation): applies the
        configured channel once to the gate's wires.

        Args:
            gate_wires:
                Wires associated with the evolved gate.
            gate:
                The gate that just ran; used to find its chip gates in the
                translation.

        Returns:
            None. Inserts the noise channels into the QNode being built.
        """
        if not self.apply_after_gates:
            return
        schedule = getattr(self, "_schedule", None)
        if schedule is not None and gate is not None and id(gate) in schedule.per_gate:
            for chip_gate in schedule.per_gate[id(gate)]:
                self._apply_compiled_gate(chip_gate)
        elif self.noise_type != "device_card":
            self.apply_to_wires(gate_wires)

    def _apply_compiled_gate(self, chip_gate: ChipGate) -> None:
        """Adds the noise of one gate of a genome's translation.

        Args:
            chip_gate: One translated gate.

        Returns:
            None. Inserts the card's channel for device-card noise, or the
            chip's calibrated noise otherwise.
        """
        if self.noise_type == "device_card":
            self._apply_card_gate(chip_gate)
        else:
            self._apply_chip_gate(chip_gate)

    def before_measurement(self, measurement_wires: Sequence[int]) -> None:
        """Apply noise immediately before measurement.

        Chip noise (``ibm_backend``, reality item 4): always adds each
        measured qubit's own readout error from the calibration file
        (:meth:`apply_readout_error`), whatever ``--noise_before_measurement``
        says, since a real chip always misreads at its measured rates.

        Device-card noise (idle on): every wire gets the card's idle channel
        for the time between its last gate and the end of the circuit. The
        card has no readout error.

        Synthetic noise: when ``--noise_before_measurement`` is on (off by
        default), applies the configured channel once more on the measured
        wires, as a test setting; otherwise nothing.

        Args:
            measurement_wires:
                Wires being measured.

        Returns:
            None. Inserts the readout channels (chip noise), the final idle
            channels (device-card noise) or the configured channel
            (synthetic noise, switch on) into the QNode being built, or
            nothing.
        """
        if self.noise_type == "device_card":
            timeline = getattr(self, "_card_timeline", None)
            if timeline is not None:
                for wire, wait in enumerate(timeline.idle_at_end):
                    self._apply_idle(wire, wait)
        elif self.noise_type == "ibm_backend":
            # read each wire out on the chip qubit its state ended on
            schedule = getattr(self, "_schedule", None)
            slots = (
                None
                if schedule is None
                else [schedule.final_slot[w] for w in measurement_wires]
            )
            self.apply_readout_error(measurement_wires, slots)
        elif self.apply_before_measurement:
            self.apply_to_wires(measurement_wires)

    def begin_circuit(self, genome: CircuitGenome) -> None:
        """Prepares a genome's circuit; for chip noise, translates it into chip gates.

        Called once each time a genome's circuit is built, not on every
        training step. For chip noise (``ibm_backend``) it translates the
        genome into the chip's own gates on the chosen qubits
        (:func:`src.noise.compiled_schedule.compile_genome`) and keeps the
        result, which :meth:`after_gate` and :meth:`before_measurement` then
        use. For device-card noise it translates the genome into the card's
        gates on the card's wiring and schedules them
        (:meth:`_translate_for_card`), for :meth:`before_gate`,
        :meth:`after_gate` and :meth:`before_measurement`. For synthetic noise
        and no noise it does nothing.

        Args:
            genome: The genome whose circuit is being built.

        Returns:
            None. Sets ``self._schedule`` (the translation, or ``None``) and
            ``self._card_timeline`` (device-card noise's timeline, or
            ``None``).
        """
        self._schedule = None
        self._card_timeline = None
        if self.noise_type == "device_card":
            self._schedule, self._card_timeline = self._translate_for_card(genome)
            return
        if self.noise_type != "ibm_backend" or self.per_qubit is None:
            return
        from src.noise.compiled_schedule import compile_genome

        slot = {q: i for i, q in enumerate(self.per_qubit["qubits"])}
        wired_pairs = [
            (slot[int(a)], slot[int(b)])
            for a, b in (key.split("-") for key in self.per_qubit["pairs"])
        ]
        self._schedule = compile_genome(
            genome,
            wired_pairs,
            self.chip_gates,
            level=self.transpile_level,
            seed=self.transpile_seed,
        )


def noise_hyperparameters_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Returns the genome ``noise_*`` settings requested on the command line.

    Shared by the entry points that evolve circuits (classification and
    reinforcement learning): each calls this early in ``main``, so bad noise
    flags fail before any work, and merges the result into the genome
    ``hyperparameters``, so every genome carries the noise settings to the
    workers and into the archive. The PennyLane noise model is used because
    the base class cannot be created.

    Args:
        args: The parsed command-line arguments, including the noise flags
            added by ``PennyLaneNoiseModel.initialize_parser`` and the
            ``--input_qubits`` / ``--output_qubits`` counts.

    Returns:
        The ``noise_*`` hyperparameters; they describe no noise when no noise
        flag was given.

    Raises:
        ValueError: If a noise flag is out of range, the calibration file or chosen qubits cannot
            be used (see ``BaseNoiseModel.from_args``), or ``--noise_qubits``
            does not list exactly one chip qubit per circuit wire.
    """
    hp = PennyLaneNoiseModel.from_args(args).to_hyperparameters()

    # Chip noise needs one chosen chip qubit per circuit wire. Inputs and
    # outputs share the "input" register (see build_exaqc), so a circuit has
    # max(--input_qubits, --output_qubits) wires, fixed for the whole run.
    per_qubit = hp.get("noise_per_qubit")
    if per_qubit is not None:
        n_wires = max(args.input_qubits, args.output_qubits)
        n_chosen = len(per_qubit["qubits"])
        if n_chosen != n_wires:
            raise ValueError(
                f"--noise_qubits lists {n_chosen} qubits but each circuit has "
                f"{n_wires} (the larger of --input_qubits {args.input_qubits} and "
                f"--output_qubits {args.output_qubits}); list exactly {n_wires}."
            )
    return hp
