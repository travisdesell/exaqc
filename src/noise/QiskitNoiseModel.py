from __future__ import annotations

from src.noise.BaseNoiseModel import BaseNoiseModel


class QiskitNoiseModel(BaseNoiseModel):
    """Qiskit/Aer implementation of backend-agnostic noise model."""

    def backend_name(self) -> str:
        return "qiskit"

    def _apply_noise_model(self, noise_model, noise_type_handlers):
        """Apply noise model based on noise type."""
        if noise_type_handlers.get(self.noise_type):
            noise_type_handlers[self.noise_type](noise_model)
        else:
            raise ValueError(f"Unsupported Qiskit noise_type={self.noise_type}")

    def to_qiskit_noise_model(self):
        """Create a qiskit_aer.noise.NoiseModel."""
        if not self.is_noisy():
            return None

        try:
            from qiskit_aer.noise import (
                NoiseModel,
                depolarizing_error,
                pauli_error,
                amplitude_damping_error,
                phase_damping_error,
            )
        except Exception as e:
            raise ImportError(
                "Qiskit noise support requires qiskit-aer."
            ) from e

        noise_model = NoiseModel()

        one_qubit_gates = ["x", "sx", "rz", "rx", "ry", "h", "s", "sdg", "t", "tdg"]
        two_qubit_gates = ["cx", "cz", "swap", "rzz"]

        p1 = self.probability_for_n_wires(1)
        p2 = self.probability_for_n_wires(2)

        def apply_depolarizing():
            if p1 > 0:
                noise_model.add_all_qubit_quantum_error(
                    depolarizing_error(p1, 1),
                    one_qubit_gates,
                )
            if p2 > 0:
                noise_model.add_all_qubit_quantum_error(
                    depolarizing_error(p2, 2),
                    two_qubit_gates,
                )

        def apply_bit_flip():
            error = pauli_error([("X", p1), ("I", 1.0 - p1)])
            noise_model.add_all_qubit_quantum_error(error, one_qubit_gates)

        def apply_phase_flip():
            error = pauli_error([("Z", p1), ("I", 1.0 - p1)])
            noise_model.add_all_qubit_quantum_error(error, one_qubit_gates)

        def apply_amplitude_damping():
            error = amplitude_damping_error(self.gamma)
            noise_model.add_all_qubit_quantum_error(error, one_qubit_gates)

        def apply_phase_damping():
            error = phase_damping_error(self.gamma)
            noise_model.add_all_qubit_quantum_error(error, one_qubit_gates)

        def apply_mixed():
            if p1 > 0:
                noise_model.add_all_qubit_quantum_error(
                    depolarizing_error(p1, 1),
                    one_qubit_gates,
                )
            if self.gamma > 0:
                noise_model.add_all_qubit_quantum_error(
                    amplitude_damping_error(self.gamma),
                    one_qubit_gates,
                )

        noise_type_handlers = {
            "depolarizing": apply_depolarizing,
            "bit_flip": apply_bit_flip,
            "phase_flip": apply_phase_flip,
            "amplitude_damping": apply_amplitude_damping,
            "phase_damping": apply_phase_damping,
            "mixed": apply_mixed,
        }

        self._apply_noise_model(noise_model, noise_type_handlers)

        return noise_model