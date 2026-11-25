"""The noise model for circuits built on the Qiskit target.

It inserts exactly the noise the PennyLane target does: every hook
(:meth:`before_input_encoding`, :meth:`after_input_encoding`,
:meth:`before_gate`, :meth:`after_gate`, :meth:`before_measurement`) runs the
PennyLane model's own
rule, captures the channels it would add, and appends each one to the Qiskit
circuit as a Qiskit Aer ``QuantumError`` built from the same Kraus operators.
Synthetic noise and calibrated chip noise therefore share one set of formulas
across both targets, and a genome scores the same under either.

A noisy circuit runs on Qiskit Aer's density-matrix simulator with exact
(shot-free) probabilities, so, like ``default.mixed`` on PennyLane, training
sees no sampling noise on top of the modelled noise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pennylane as qml

from src.noise.pennylane_noise_model import PennyLaneNoiseModel

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit
    from qiskit.primitives import BaseSampler

    from src.circuits.gate import Gate


class QiskitNoiseModel(PennyLaneNoiseModel):
    """Applies the PennyLane noise rules to a Qiskit circuit being built.

    Call :meth:`bind_circuit` once with the circuit and its qubits (wire ``i``
    is ``qubits[i]``), then the same hooks, in the same order, as the
    PennyLane circuit builder calls them; each appends its noise to the bound
    circuit. Run the result with :meth:`sampler`.
    """

    def backend_name(self) -> str:
        """Return the backend identifier.

        Returns:
            The string ``"qiskit"``.
        """
        return "qiskit"

    def bind_circuit(self, circuit: QuantumCircuit, qubits: Sequence[Qubit]) -> None:
        """Sets the circuit the hooks append their noise to.

        Args:
            circuit: The Qiskit circuit being built.
            qubits: The circuit's qubits in wire order: hook wire ``i`` is
                ``qubits[i]``.

        Returns:
            None. Sets ``self._circuit`` and ``self._qubits``.
        """
        self._circuit = circuit
        self._qubits = list(qubits)

    def _emit(self, hook: Callable[..., None], *args: Any) -> None:
        """Runs a PennyLane noise hook and appends what it adds to the circuit.

        The hook is run inside a PennyLane queuing context, so the channels it
        creates are recorded instead of entering a QNode. Each is turned into
        a Qiskit Aer ``QuantumError`` from its Kraus operators. PennyLane
        orders a multi-wire matrix with the first wire most significant and
        Qiskit with the first qubit least significant, so the wires are
        passed to Qiskit reversed.

        Args:
            hook: The PennyLane model's hook, e.g. ``PennyLaneNoiseModel.after_gate``
                bound to this model.
            *args: The hook's arguments.

        Returns:
            None. Appends one ``QuantumError`` per recorded channel to the
            bound circuit.

        Raises:
            RuntimeError: If no circuit is bound (:meth:`bind_circuit`).
            TypeError: If the hook records an operation that is not a noise
                channel.
        """
        circuit = getattr(self, "_circuit", None)
        qubits = getattr(self, "_qubits", None)
        if circuit is None or qubits is None:
            raise RuntimeError("bind_circuit must be called before adding noise.")
        from qiskit.quantum_info import Kraus
        from qiskit_aer.noise import QuantumError

        with qml.queuing.AnnotatedQueue() as queue:
            hook(*args)
        for op in queue.queue:
            if not isinstance(op, qml.operation.Channel):
                raise TypeError(
                    f"noise hook recorded {op.name}, which is not a channel."
                )
            kraus = [np.asarray(k, dtype=complex) for k in op.kraus_matrices()]
            targets = [qubits[int(w)] for w in reversed(op.wires.tolist())]
            circuit.append(QuantumError(Kraus(kraus)), targets)

    def before_input_encoding(self, wires: Sequence[int]) -> None:
        """Appends each qubit's start-up error (chip noise only).

        Args:
            wires: The circuit's wires.

        Returns:
            None. Appends to the bound circuit what
            :meth:`PennyLaneNoiseModel.before_input_encoding` adds.
        """
        self._emit(super().before_input_encoding, wires)

    def after_input_encoding(self, input_wires: Sequence[int]) -> None:
        """Appends the noise that follows the input encoding.

        Args:
            input_wires: The wires the inputs were encoded on.

        Returns:
            None. Appends to the bound circuit what
            :meth:`PennyLaneNoiseModel.after_input_encoding` adds.
        """
        self._emit(super().after_input_encoding, input_wires)

    def before_gate(self, gate_wires: Sequence[int], gate: Gate | None = None) -> None:
        """Appends the noise that comes just before one evolved gate.

        Args:
            gate_wires: The wires the gate acts on.
            gate: The gate itself, which device-card noise uses to look up how
                long its wires waited.

        Returns:
            None. Appends to the bound circuit what
            :meth:`PennyLaneNoiseModel.before_gate` adds (device-card idle
            noise, or nothing).
        """
        self._emit(super().before_gate, gate_wires, gate)

    def after_gate(self, gate_wires: Sequence[int], gate: Gate | None = None) -> None:
        """Appends the noise that follows one evolved gate.

        Args:
            gate_wires: The wires the gate acted on.
            gate: The gate itself, which chip noise uses to look up the chip
                gates it was translated into.

        Returns:
            None. Appends to the bound circuit what
            :meth:`PennyLaneNoiseModel.after_gate` adds.
        """
        self._emit(super().after_gate, gate_wires, gate)

    def before_measurement(self, measurement_wires: Sequence[int]) -> None:
        """Appends the noise just before measurement, including readout error.

        Args:
            measurement_wires: The wires about to be measured.

        Returns:
            None. Appends to the bound circuit what
            :meth:`PennyLaneNoiseModel.before_measurement` adds.
        """
        self._emit(super().before_measurement, measurement_wires)

    def sampler(self) -> BaseSampler | None:
        """Returns the sampler that runs a circuit built with this model.

        Returns:
            ``None`` when no noise is configured, so the caller keeps its
            default (noiseless) sampler; otherwise a Qiskit Aer sampler on the
            density-matrix method with exact probabilities (no shots), which
            simulates the appended noise channels.
        """
        if not self.is_noisy():
            return None
        from qiskit_aer.primitives import Sampler

        return Sampler(
            backend_options={"method": "density_matrix"},
            run_options={"shots": None},
        )
