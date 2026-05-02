from __future__ import annotations

from typing import Sequence

import pennylane as qml

from src.noise.BaseNoiseModel import BaseNoiseModel


class PennyLaneNoiseModel(BaseNoiseModel):
    """PennyLane implementation of backend-agnostic noise model."""

    def backend_name(self) -> str:
        return "pennylane"

    def device_name(self) -> str:
        """Return PennyLane device required by this noise model."""
        return "default.mixed" if self.is_noisy() else "default.qubit"

    def _as_wires(self, wires: Sequence[int] | int) -> list[int]:
        if isinstance(wires, int):
            return [wires]
        return list(wires)

    def _apply_depolarizing(self, wires: list[int]) -> None:
        for w in wires:
            qml.DepolarizingChannel(self.probability_for_n_wires(1), wires=w)

    def _apply_bit_flip(self, wires: list[int]) -> None:
        for w in wires:
            qml.BitFlip(self.probability_for_n_wires(1), wires=w)

    def _apply_phase_flip(self, wires: list[int]) -> None:
        for w in wires:
            qml.PhaseFlip(self.probability_for_n_wires(1), wires=w)

    def _apply_amplitude_damping(self, wires: list[int]) -> None:
        for w in wires:
            qml.AmplitudeDamping(self.gamma, wires=w)

    def _apply_phase_damping(self, wires: list[int]) -> None:
        for w in wires:
            qml.PhaseDamping(self.gamma, wires=w)

    def _apply_mixed(self, wires: list[int]) -> None:
        for w in wires:
            qml.DepolarizingChannel(self.probability_for_n_wires(1), wires=w)
            qml.AmplitudeDamping(self.gamma, wires=w)

    def apply_to_wires(self, wires: Sequence[int] | int) -> None:
        """Apply configured PennyLane channel to selected wires."""
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
        if self.apply_after_input_encoding:
            self.apply_to_wires(input_wires)

    def after_gate(self, gate_wires: Sequence[int]) -> None:
        if self.apply_after_gates:
            self.apply_to_wires(gate_wires)

    def before_measurement(self, measurement_wires: Sequence[int]) -> None:
        if self.apply_before_measurement:
            self.apply_to_wires(measurement_wires)
