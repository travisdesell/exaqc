# tests/test_noise_models.py

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.noise import (
    BaseNoiseModel,
    PennyLaneNoiseModel,
    QiskitNoiseModel,
)


def test_pennylane_noise_model_none_uses_default_qubit():
    noise = PennyLaneNoiseModel(noise_type="none")

    assert noise.backend_name() == "pennylane"
    assert noise.is_noisy() is False
    assert noise.device_name() == "default.qubit"


def test_pennylane_noise_model_noisy_uses_default_mixed():
    noise = PennyLaneNoiseModel(noise_type="depolarizing", p=0.01)

    assert noise.is_noisy() is True
    assert noise.device_name() == "default.mixed"


def test_probability_for_n_wires_uses_default_p():
    noise = PennyLaneNoiseModel(noise_type="depolarizing", p=0.05)

    assert noise.probability_for_n_wires(1) == pytest.approx(0.05)
    assert noise.probability_for_n_wires(2) == pytest.approx(0.05)


def test_probability_for_n_wires_uses_1q_and_2q_overrides():
    noise = PennyLaneNoiseModel(
        noise_type="depolarizing",
        p=0.05,
        p_1q=0.001,
        p_2q=0.01,
    )

    assert noise.probability_for_n_wires(1) == pytest.approx(0.001)
    assert noise.probability_for_n_wires(2) == pytest.approx(0.01)


def test_from_hyperparameters_constructs_expected_model():
    hp = {
        "noise_type": "amplitude_damping",
        "noise_p": 0.02,
        "noise_p_1q": 0.001,
        "noise_p_2q": 0.01,
        "noise_gamma": 0.03,
        "noise_after_encoding": True,
        "noise_after_gates": False,
        "noise_before_measurement": True,
    }

    noise = PennyLaneNoiseModel.from_hyperparameters(hp)

    assert noise.noise_type == "amplitude_damping"
    assert noise.p == pytest.approx(0.02)
    assert noise.p_1q == pytest.approx(0.001)
    assert noise.p_2q == pytest.approx(0.01)
    assert noise.gamma == pytest.approx(0.03)
    assert noise.apply_after_input_encoding is True
    assert noise.apply_after_gates is False
    assert noise.apply_before_measurement is True


@pytest.mark.parametrize(
    "noise_type",
    [
        "depolarizing",
        "bit_flip",
        "phase_flip",
        "amplitude_damping",
        "phase_damping",
        "mixed",
    ],
)
def test_pennylane_apply_to_wires_does_not_raise(noise_type):
    qml = pytest.importorskip("pennylane")

    noise = PennyLaneNoiseModel(
        noise_type=noise_type,
        p=0.01,
        gamma=0.01,
    )

    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        noise.apply_to_wires([0])
        return qml.probs(wires=[0, 1])

    probs = circuit()

    assert probs.shape[0] == 4


def test_pennylane_after_gate_respects_flag():
    qml = pytest.importorskip("pennylane")

    noise = PennyLaneNoiseModel(
        noise_type="depolarizing",
        p=0.01,
        apply_after_gates=False,
    )

    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        noise.after_gate([0])
        return qml.probs(wires=[0])

    probs = circuit()

    assert probs.shape[0] == 2


def test_pennylane_thermal_relaxation_runs():
    """thermal_relaxation is implemented and inserts a channel on default.mixed."""
    qml = pytest.importorskip("pennylane")

    noise = PennyLaneNoiseModel(noise_type="thermal_relaxation")
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circuit() -> Any:
        """Excites the qubit and lets it relax for one gate time.

        Returns:
            The probabilities of reading 0 and 1.
        """
        qml.PauliX(wires=0)
        noise.apply_to_wires([0])
        return qml.probs(wires=[0])

    probs = circuit()

    assert 0.0 < float(probs[0]) < 0.01


def test_qiskit_noise_model_none_keeps_default_sampler():
    """Without noise the Qiskit model leaves SamplerQNN's own sampler in place."""
    noise = QiskitNoiseModel(noise_type="none")

    assert noise.backend_name() == "qiskit"
    assert noise.is_noisy() is False
    assert noise.sampler() is None


def test_qiskit_noise_model_needs_a_bound_circuit():
    """Adding noise before bind_circuit is refused rather than silently dropped."""
    noise = QiskitNoiseModel(noise_type="depolarizing", p=0.1)

    with pytest.raises(RuntimeError, match="bind_circuit"):
        noise.after_gate([0])


def _pennylane_probs(noise: PennyLaneNoiseModel, build: Any) -> np.ndarray:
    """Runs a two-wire test circuit with noise on PennyLane's default.mixed.

    Args:
        noise: The noise model whose hooks ``build`` calls.
        build: Called with ``(noise, gate_x, gate_cnot, gate_h)``; it adds the
            gates through those callables and the noise through ``noise``.

    Returns:
        The probabilities over both wires, wire 0 most significant.
    """
    qml = pytest.importorskip("pennylane")
    dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def circuit() -> Any:
        """Builds the test circuit.

        Returns:
            The probabilities over both wires.
        """
        build(
            noise,
            lambda w: qml.PauliX(wires=w),
            lambda a, b: qml.CNOT(wires=[a, b]),
            lambda w: qml.Hadamard(wires=w),
        )
        return qml.probs(wires=[0, 1])

    return np.asarray(circuit(), dtype=float)


def _qiskit_probs(noise: QiskitNoiseModel, build: Any) -> np.ndarray:
    """Runs the same two-wire test circuit with noise on Qiskit Aer.

    Args:
        noise: The noise model whose hooks ``build`` calls.
        build: As for :func:`_pennylane_probs`.

    Returns:
        The probabilities over both wires, wire 0 most significant (reordered
        from Qiskit's little-endian bit strings to PennyLane's order).
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    registers = [QuantumRegister(1, f"q{i}") for i in range(2)]
    classical = ClassicalRegister(2, "meas")
    circuit = QuantumCircuit(*registers, classical)
    noise.bind_circuit(circuit, [register[0] for register in registers])
    build(
        noise,
        lambda w: circuit.x(w),
        lambda a, b: circuit.cx(a, b),
        lambda w: circuit.h(w),
    )
    circuit.measure([0, 1], [0, 1])

    quasi = noise.sampler().run(circuit).result().quasi_dists[0]
    probs = np.zeros(4)
    for outcome, p in quasi.items():
        bit0, bit1 = outcome & 1, (outcome >> 1) & 1
        probs[2 * bit0 + bit1] += p
    return probs


@pytest.mark.parametrize(
    "noise_type",
    [
        "depolarizing",
        "bit_flip",
        "phase_flip",
        "amplitude_damping",
        "phase_damping",
        "thermal_relaxation",
        "mixed",
    ],
)
def test_qiskit_and_pennylane_noise_match(noise_type: str) -> None:
    """Both targets give the same noisy probabilities for the same circuit."""
    pytest.importorskip("qiskit_aer")
    settings = {
        "noise_type": noise_type,
        "noise_p_1q": 0.05,
        "noise_p_2q": 0.1,
        "noise_gamma": 0.2,
        "noise_after_encoding": True,
        "noise_before_measurement": True,
    }

    def build(noise: PennyLaneNoiseModel, x: Any, cnot: Any, h: Any) -> None:
        """Adds X on wire 0, a CNOT, and an H-noise-H sandwich on wire 1.

        The sandwich turns phase noise into bit flips, so every noise type
        changes the measured probabilities.

        Args:
            noise: The noise model.
            x: Adds an X gate on a wire.
            cnot: Adds a CNOT from the first wire to the second.
            h: Adds a Hadamard gate on a wire.

        Returns:
            None. Adds the gates and noise to the circuit being built.
        """
        x(0)
        h(1)
        noise.after_input_encoding([0, 1])
        h(1)
        cnot(0, 1)
        noise.after_gate([0, 1])
        noise.before_measurement([0, 1])

    noiseless = _pennylane_probs(PennyLaneNoiseModel(), build)
    expected = _pennylane_probs(
        PennyLaneNoiseModel.from_hyperparameters(settings), build
    )
    actual = _qiskit_probs(QiskitNoiseModel.from_hyperparameters(settings), build)

    assert not np.allclose(expected, noiseless)  # the noise did something
    assert actual == pytest.approx(expected, abs=1e-9)


def test_qiskit_two_wire_channel_keeps_wire_order():
    """A channel that treats its two wires differently lands on the right qubits.

    PennyLane writes a two-wire matrix with the first wire most significant
    and Qiskit with the first qubit least significant; a CNOT written as a
    one-Kraus channel only gives the same result when the order is converted.
    """
    pytest.importorskip("qiskit_aer")
    qml = pytest.importorskip("pennylane")
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )

    def build(noise: PennyLaneNoiseModel, x: Any, _cnot: Any, _h: Any) -> None:
        """Adds X on wire 0, then a CNOT(0 -> 1) as a noise channel.

        Args:
            noise: The noise model.
            x: Adds an X gate on a wire.
            _cnot: Unused.
            _h: Unused.

        Returns:
            None. Adds the gate and the channel to the circuit being built.
        """
        x(0)
        if isinstance(noise, QiskitNoiseModel):
            noise._emit(lambda: qml.QubitChannel([cnot], wires=[0, 1]))
        else:
            qml.QubitChannel([cnot], wires=[0, 1])

    settings = {"noise_type": "depolarizing", "noise_p": 0.0}
    expected = _pennylane_probs(
        PennyLaneNoiseModel.from_hyperparameters(settings), build
    )
    actual = _qiskit_probs(QiskitNoiseModel.from_hyperparameters(settings), build)

    assert expected == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-9)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_base_noise_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseNoiseModel()
