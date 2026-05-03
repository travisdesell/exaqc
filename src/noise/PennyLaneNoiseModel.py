from __future__ import annotations

from typing import Sequence

import pennylane as qml

from src.noise.BaseNoiseModel import BaseNoiseModel


class PennyLaneNoiseModel(BaseNoiseModel):
    """PennyLane implementation of the backend-agnostic noise model.

    This class provides PennyLane-native quantum noise injection utilities.
    Noise channels are inserted directly into PennyLane QNodes and can be
    applied after input encoding, after evolved quantum gates, or immediately
    before measurement.

    The model supports multiple standard quantum noise channels including:

    - Depolarizing noise
    - Bit-flip noise
    - Phase-flip noise
    - Amplitude damping
    - Phase damping
    - Mixed noise channels

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

    def _apply_depolarizing(self, wires: list[int]) -> None:
        """Apply depolarizing noise channels.

        Args:
            wires:
                Target wires for the depolarizing channel.
        """
        for w in wires:
            qml.DepolarizingChannel(
                self.probability_for_n_wires(1),
                wires=w,
            )

    def _apply_bit_flip(self, wires: list[int]) -> None:
        """Apply bit-flip noise channels.

        Args:
            wires:
                Target wires for the bit-flip channel.
        """
        for w in wires:
            qml.BitFlip(
                self.probability_for_n_wires(1),
                wires=w,
            )

    def _apply_phase_flip(self, wires: list[int]) -> None:
        """Apply phase-flip noise channels.

        Args:
            wires:
                Target wires for the phase-flip channel.
        """
        for w in wires:
            qml.PhaseFlip(
                self.probability_for_n_wires(1),
                wires=w,
            )

    def _apply_amplitude_damping(self, wires: list[int]) -> None:
        """Apply amplitude damping channels.

        Args:
            wires:
                Target wires for amplitude damping.
        """
        for w in wires:
            qml.AmplitudeDamping(
                self.gamma,
                wires=w,
            )

    def _apply_phase_damping(self, wires: list[int]) -> None:
        """Apply phase damping channels.

        Args:
            wires:
                Target wires for phase damping.
        """
        for w in wires:
            qml.PhaseDamping(
                self.gamma,
                wires=w,
            )

    def _apply_mixed(self, wires: list[int]) -> None:
        """Apply a mixed noise model.

        The mixed model combines depolarizing noise and amplitude damping.

        Args:
            wires:
                Target wires for the mixed noise process.
        """
        for w in wires:
            qml.DepolarizingChannel(
                self.probability_for_n_wires(1),
                wires=w,
            )

            qml.AmplitudeDamping(
                self.gamma,
                wires=w,
            )

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
        - ``mixed``

        Args:
            wires:
                Single wire or collection of wires where the noise
                process should be inserted.

        Raises:
            NotImplementedError:
                If ``thermal_relaxation`` is requested.

            ValueError:
                If an unsupported noise type is specified.
        """
        if not self.is_noisy():
            return

        wires = self._as_wires(wires)

        noise_handlers = {
            "depolarizing": self._apply_depolarizing,
            "bit_flip": self._apply_bit_flip,
            "phase_flip": self._apply_phase_flip,
            "amplitude_damping": self._apply_amplitude_damping,
            "phase_damping": self._apply_phase_damping,
            "mixed": self._apply_mixed,
        }

        if self.noise_type in noise_handlers:
            noise_handlers[self.noise_type](wires)

        elif self.noise_type == "thermal_relaxation":
            raise NotImplementedError(
                "Thermal relaxation is backend/version dependent in PennyLane. "
                "Use amplitude_damping/phase_damping or implement a custom channel."
            )

        else:
            raise ValueError(f"Unknown noise_type={self.noise_type}")

    def after_input_encoding(self, input_wires: Sequence[int]) -> None:
        """Apply noise after classical input encoding.

        Args:
            input_wires:
                Wires used for classical data encoding.
        """
        if self.apply_after_input_encoding:
            self.apply_to_wires(input_wires)

    def after_gate(self, gate_wires: Sequence[int]) -> None:
        """Apply noise after an evolved quantum gate.

        Args:
            gate_wires:
                Wires associated with the evolved gate.
        """
        if self.apply_after_gates:
            self.apply_to_wires(gate_wires)

    def before_measurement(self, measurement_wires: Sequence[int]) -> None:
        """Apply noise immediately before measurement.

        Args:
            measurement_wires:
                Wires being measured.
        """
        if self.apply_before_measurement:
            self.apply_to_wires(measurement_wires)
