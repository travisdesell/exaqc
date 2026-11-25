"""Tests that a noise model passed to a genome works the same on both targets.

``CircuitGenome.initialize_model(noise_model=...)`` overrides the genome's
recorded ``noise_*`` settings, so one genome can be run with or without noise.
The Qiskit target appends the very channels the PennyLane target inserts, so a
genome gives the same noisy probabilities on either. Qiskit numbers its output
bit strings with the first measured qubit least significant and PennyLane with
it most significant, so the Qiskit result is reordered before comparing.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch

from src.circuits.circuit import CircuitGenome
from src.noise import BaseNoiseModel, PennyLaneNoiseModel, QiskitNoiseModel
from tests.supervised_trainer_test_utils import build_classification_genome


def _build_genome(
    target: str, hyperparameters: dict[str, Any] | None = None
) -> CircuitGenome:
    """Builds the same small two-wire genome for either target.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.
        hyperparameters: Extra genome hyperparameters (e.g. ``noise_*``
            settings) merged in, or ``None``.

    Returns:
        The uninitialized genome.
    """
    random.seed(3)
    torch.manual_seed(3)
    genome, _ = build_classification_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
    )
    genome.hyperparameters.update(hyperparameters or {})
    return genome


def _quantum_probs(
    target: str,
    noise_model: BaseNoiseModel | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> np.ndarray:
    """Runs the genome's quantum circuit on a fixed input.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.
        noise_model: Passed to ``initialize_model``.
        hyperparameters: Extra genome hyperparameters, as for
            :func:`_build_genome`.

    Returns:
        The output probabilities, first output wire most significant.
    """
    genome = _build_genome(target, hyperparameters)
    genome.initialize_model(noise_model=noise_model)
    inputs = torch.linspace(0.1, 1.2, genome.n_quantum_inputs())
    with torch.no_grad():
        probs = genome.torch_model(inputs).double().numpy()
    if target == "qiskit":
        n_bits = len(genome.output_indexes)
        order = [int(format(i, f"0{n_bits}b")[::-1], 2) for i in range(2**n_bits)]
        probs = probs[order]
    return probs


def _chip_noise_settings(n_wires: int) -> dict[str, Any]:
    """Chip-noise settings for a chain of qubits on an offline IBM test chip.

    Uses the calibration packaged with Qiskit's ``FakeTorino``, read through
    the same path as a stored IBM calibration file, so no network is needed.

    Args:
        n_wires: How many physical qubits to choose (``0, 1, ...``, wired in
            a line on Torino).

    Returns:
        The genome ``noise_*`` settings for those qubits.
    """
    fake_provider = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    from src.noise.calibration.backends import fetch_raw_calibration

    raw = fetch_raw_calibration(fake_provider.FakeTorino())
    model = PennyLaneNoiseModel.from_calibration(raw, list(range(n_wires)))
    return model.to_hyperparameters()


@pytest.mark.parametrize(
    "noise_model",
    [
        PennyLaneNoiseModel(noise_type="depolarizing", p=0.05),
        PennyLaneNoiseModel(noise_type="amplitude_damping", gamma=0.1),
        QiskitNoiseModel(
            noise_type="thermal_relaxation", apply_before_measurement=True
        ),
    ],
    ids=["depolarizing", "amplitude_damping", "thermal_relaxation"],
)
def test_passed_noise_matches_across_targets(noise_model: BaseNoiseModel) -> None:
    """The same passed-in noise gives the same probabilities on both targets.

    Args:
        noise_model: The noise to pass; a model made for either target.
    """
    noiseless = _quantum_probs("pennylane")
    pennylane = _quantum_probs("pennylane", noise_model)
    qiskit = _quantum_probs("qiskit", noise_model)

    assert not np.allclose(pennylane, noiseless)
    assert qiskit == pytest.approx(pennylane, abs=1e-6)


def test_chip_noise_from_hyperparameters_matches_across_targets() -> None:
    """Calibrated chip noise recorded on the genome is the same on both targets."""
    settings = _chip_noise_settings(len(_build_genome("pennylane").qubits))

    noiseless = _quantum_probs("pennylane")
    pennylane = _quantum_probs("pennylane", hyperparameters=settings)
    qiskit = _quantum_probs("qiskit", hyperparameters=settings)

    assert not np.allclose(pennylane, noiseless)
    assert qiskit == pytest.approx(pennylane, abs=1e-6)


@pytest.mark.parametrize("target", ["pennylane", "qiskit"])
def test_passing_a_noiseless_model_overrides_recorded_noise(target: str) -> None:
    """A genome that records noise runs noiselessly when given a no-noise model.

    Args:
        target: The quantum backend under test.
    """
    noisy_settings = PennyLaneNoiseModel(
        noise_type="depolarizing", p=0.2
    ).to_hyperparameters()

    noiseless = _quantum_probs(target)
    recorded = _quantum_probs(target, hyperparameters=noisy_settings)
    overridden = _quantum_probs(
        target, PennyLaneNoiseModel(), hyperparameters=noisy_settings
    )

    assert not np.allclose(recorded, noiseless)
    assert overridden == pytest.approx(noiseless, abs=1e-9)
