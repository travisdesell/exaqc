"""Device-card noise: insert a card's gate and idle channels into a circuit.

The card and the translation are explained in :mod:`src.noise.device_card`.
:class:`DeviceCardChannels` is the noise-model half that turns them into
PennyLane channels; :class:`src.noise.pennylane_noise_model.PennyLaneNoiseModel`
inherits it and calls it from its hooks, and the Qiskit noise model captures
the same channels as Aer errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml

from src.noise.base_noise_model import BaseNoiseModel
from src.noise.device_card import (
    CARD_NATIVE_GATES,
    CardTimeline,
    all_pairs,
    card_timeline,
    idle_pauli_probabilities,
    kraus_from_pauli_probabilities,
)

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome
    from src.noise.compiled_schedule import ChipGate, CompiledSchedule


class DeviceCardChannels(BaseNoiseModel):
    """The device-card half of the noise model (``noise_type="device_card"``).

    Not used on its own: :class:`src.noise.pennylane_noise_model.PennyLaneNoiseModel`
    inherits it, and the circuit builder calls its hooks.
    """

    def _translate_for_card(
        self, genome: CircuitGenome
    ) -> tuple[CompiledSchedule, CardTimeline]:
        """Translates a genome into card gates and schedules them.

        Args:
            genome: The genome whose circuit is being built.

        Returns:
            The translation (:func:`src.noise.compiled_schedule.compile_genome`
            onto :data:`~src.noise.device_card.CARD_NATIVE_GATES`) and its
            as-soon-as-possible timeline.

        Raises:
            ValueError: If no card is loaded.
        """
        from src.noise.compiled_schedule import compile_genome

        card = self._card()
        n_wires = len(genome.qubits)
        wired = card.get("wired_pairs")
        pairs = all_pairs(n_wires) if wired is None else [tuple(p) for p in wired]
        schedule = compile_genome(
            genome,
            pairs,
            CARD_NATIVE_GATES,
            level=self.transpile_level,
            seed=self.transpile_seed,
        )
        return schedule, card_timeline(schedule, card, n_wires)

    def _card(self) -> dict:
        """Returns the loaded card, or stops clearly.

        Returns:
            The card, as built by :func:`src.noise.device_card.load_device_card`.

        Raises:
            ValueError: If no card is loaded, e.g. a genome whose saved
                settings lost it.
        """
        if self.device_card is None:
            raise ValueError(
                "noise_type 'device_card' has no card loaded; build the model with "
                "--noise_device_card."
            )
        return self.device_card

    def _apply_card_gate(self, chip_gate: ChipGate) -> None:
        """Adds the card's error channel for one translated card gate.

        ``rz`` is virtual and adds nothing; ``h``, ``x`` and ``cx`` add their
        own Pauli channel from the card on the genome wires whose states sit
        on the gate's qubits.

        Args:
            chip_gate: One gate of the translation.

        Returns:
            None. Inserts the channel into the circuit being built, or
            nothing for ``rz`` or a perfect gate.

        Raises:
            KeyError: If the card has no entry for the gate.
        """
        if chip_gate.name == "rz":
            return
        entry = self._card()["gates"][chip_gate.name]
        kraus = kraus_from_pauli_probabilities(entry["pauli_error"])
        if len(kraus) > 1:
            qml.QubitChannel(kraus, wires=list(chip_gate.wires))

    def _apply_card_gate_named(self, name: str, wire: int) -> None:
        """Adds one card gate's channel on one wire, without a translation.

        Used for data loading whose gates change with each sample
        (``basis``, ``amplitude``), which cannot be translated once.

        Args:
            name: The one-qubit card gate, e.g. ``"h"``.
            wire: The wire.

        Returns:
            None. Inserts the channel, or nothing for a perfect gate.
        """
        kraus = kraus_from_pauli_probabilities(
            self._card()["gates"][name]["pauli_error"]
        )
        if len(kraus) > 1:
            qml.QubitChannel(kraus, wires=[wire])

    def _apply_idle(self, wire: int, duration_ns: float) -> None:
        """Adds the card's idle channel to a wire that waited.

        Args:
            wire: The waiting wire.
            duration_ns: How long it waited.

        Returns:
            None. Inserts the idle channel, or nothing when idle noise is off,
            the wait is not positive, or the card has no idle data.
        """
        if not self.apply_idle:
            return
        probabilities = idle_pauli_probabilities(self._card(), duration_ns)
        if probabilities is None:
            return
        kraus = kraus_from_pauli_probabilities(probabilities)
        if len(kraus) > 1:
            qml.QubitChannel([np.asarray(k) for k in kraus], wires=[wire])
