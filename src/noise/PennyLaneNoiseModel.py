from __future__ import annotations

from typing import Sequence
from pathlib import Path
import os

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

    # @classmethod
    # def from_ibm_backend(cls, backend, *, verbose: bool = False):
    #     import pennylane as qml
    #     from qiskit_aer.noise import NoiseModel

    #     model = cls(noise_type="ibm_backend")
    #     qiskit_noise = NoiseModel.from_backend(backend)
    #     model.pennylane_noise_model = qml.from_qiskit_noise(
    #         qiskit_noise,
    #         verbose=verbose,
    #     )
    #     model.imported_noise = True
    #     return model

    @classmethod
    def from_ibm_backend(
        cls,
        backend,
        *,
        noise_type: str = "thermal_relaxation",
        apply_after_input_encoding: bool = False,
        apply_after_gates: bool = True,
        apply_before_measurement: bool = False,
    ) -> "PennyLaneNoiseModel":
        """Create a manual PennyLane noise model from IBM backend summary stats."""
        props = backend.properties()
        config = backend.configuration()

        one_qubit_errors = []
        two_qubit_errors = []
        one_qubit_lengths = []
        two_qubit_lengths = []

        t1s = []
        t2s = []

        for q in range(config.n_qubits):
            try:
                t1s.append(float(props.t1(q)))
            except Exception:
                pass

            try:
                t2s.append(float(props.t2(q)))
            except Exception:
                pass

        for gate in props.gates:
            gate_name = gate.gate
            qubits = tuple(gate.qubits)

            try:
                err = float(props.gate_error(gate_name, qubits))
            except Exception:
                err = None

            try:
                length = float(props.gate_length(gate_name, qubits))
            except Exception:
                length = None

            if len(qubits) <= 1:
                if err is not None:
                    one_qubit_errors.append(err)
                if length is not None:
                    one_qubit_lengths.append(length)
            else:
                if err is not None:
                    two_qubit_errors.append(err)
                if length is not None:
                    two_qubit_lengths.append(length)

        def mean_or_default(xs, default):
            return float(sum(xs) / len(xs)) if xs else default

        model = cls(
            noise_type=noise_type,
            p_1q=mean_or_default(one_qubit_errors, 0.001),
            p_2q=mean_or_default(two_qubit_errors, 0.01),
            t1=mean_or_default(t1s, 50e-6),
            t2=mean_or_default(t2s, 70e-6),
            gate_time_1q=mean_or_default(one_qubit_lengths, 50e-9),
            gate_time_2q=mean_or_default(two_qubit_lengths, 300e-9),
            apply_after_input_encoding=apply_after_input_encoding,
            apply_after_gates=apply_after_gates,
            apply_before_measurement=apply_before_measurement,
        )

        model.ibm_backend_name = backend.name() if callable(backend.name) else str(backend.name)
        return model
    
    def get_imported_noise_model(self):
        return getattr(self, "pennylane_noise_model", None)

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
    
    def save_noise_profile(self, path: str | Path) -> None:
        """Save PennyLane noise model summary as text.

        Args:
            path:
                Output text file path.
        """
        if self.pennylane_noise_model is None:
            raise ValueError("No PennyLane noise model available to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(str(self.pennylane_noise_model))

    def save_noise_summary(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            f.write(f"backend={getattr(self, 'ibm_backend_name', None)}\n")
            f.write(f"noise_type={self.noise_type}\n")
            f.write(f"p={self.p}\n")
            f.write(f"p_1q={self.p_1q}\n")
            f.write(f"p_2q={self.p_2q}\n")
            f.write(f"gamma={self.gamma}\n")
            f.write(f"t1={getattr(self, 't1', None)}\n")
            f.write(f"t2={getattr(self, 't2', None)}\n")
            f.write(f"gate_time_1q={getattr(self, 'gate_time_1q', None)}\n")
            f.write(f"gate_time_2q={getattr(self, 'gate_time_2q', None)}\n")
            f.write(f"after_encoding={self.apply_after_input_encoding}\n")
            f.write(f"after_gates={self.apply_after_gates}\n")
            f.write(f"before_measurement={self.apply_before_measurement}\n")
            

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
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.DepolarizingChannel(
                p,
                wires=w,
            )

    def _apply_bit_flip(self, wires: list[int]) -> None:
        """Apply bit-flip noise channels.

        Args:
            wires:
                Target wires for the bit-flip channel.
        """
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.BitFlip(
                p,
                wires=w,
            )

    def _apply_phase_flip(self, wires: list[int]) -> None:
        """Apply phase-flip noise channels.

        Args:
            wires:
                Target wires for the phase-flip channel.
        """
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.PhaseFlip(
                p,
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
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.DepolarizingChannel(
                p,
                wires=w,
            )

            qml.AmplitudeDamping(
                self.gamma,
                wires=w,
            )

    def _apply_thermal_relaxation(self, wires: list[int]) -> None:
        """Apply PennyLane thermal relaxation error to target wires.
        
        Args:
            wires:
                Target wires for the mixed noise process.
        """
        n_wires = len(wires)
        gate_time = self.gate_time_1q if n_wires <= 1 else self.gate_time_2q

        for w in wires:
            qml.ThermalRelaxationError(
                self.excited_state_population,
                self.t1,
                self.t2,
                gate_time,
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
            "thermal_relaxation": self._apply_thermal_relaxation,
            "mixed": self._apply_mixed,
        }

        if self.noise_type in noise_handlers:
            noise_handlers[self.noise_type](wires)

        # elif self.noise_type == "thermal_relaxation":
        #     raise NotImplementedError(
        #         "Thermal relaxation is backend/version dependent in PennyLane. "
        #         "Use amplitude_damping/phase_damping or implement a custom channel."
        #     )

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
