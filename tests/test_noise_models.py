# tests/test_noise_models.py

from __future__ import annotations

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


def test_pennylane_thermal_relaxation_not_implemented():
    noise = PennyLaneNoiseModel(noise_type="thermal_relaxation", gamma=0.01)

    with pytest.raises(NotImplementedError):
        noise.apply_to_wires([0])


@pytest.mark.skip(reason="Qiskit not yet implemented")
def test_qiskit_noise_model_none_returns_none():
    pytest.importorskip("qiskit_aer")

    noise = QiskitNoiseModel(noise_type="none")

    assert noise.backend_name() == "qiskit"
    assert noise.is_noisy() is False
    assert noise.to_qiskit_noise_model() is None


@pytest.mark.skip(reason="Qiskit not yet implemented")
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
def test_qiskit_noise_model_creation(noise_type):
    pytest.importorskip("qiskit_aer")

    noise = QiskitNoiseModel(
        noise_type=noise_type,
        p=0.01,
        p_1q=0.001,
        p_2q=0.01,
        gamma=0.01,
    )

    qiskit_noise = noise.to_qiskit_noise_model()

    assert qiskit_noise is not None


def test_base_noise_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseNoiseModel()
