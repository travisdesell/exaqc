"""Tests for device-card noise (Dr. Vogt's optimal-control gate channels).

A small hand-written card stands in for the real ones in
``device_cards/``, so the numbers below are easy to check by hand and
the tests do not depend on which cards are checked in.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.noise import PennyLaneNoiseModel
from src.noise.compiled_schedule import ChipGate, CompiledSchedule
from src.noise.device_card import (
    card_timeline,
    hardware_metrics,
    idle_error_probability,
    idle_ptm_diagonal,
    kraus_from_pauli_probabilities,
    load_device_card,
    resolve_device_card,
    wired_slot_pairs,
)
from tests.test_noise_targets import _build_genome, _quantum_probs


def _two_qubit_paulis(p_error: float) -> dict[str, float]:
    """A two-qubit Pauli channel with ``p_error`` spread over the 15 errors.

    Args:
        p_error: The total error probability.

    Returns:
        Pauli pair -> probability.
    """

    labels = [a + b for a in "IXYZ" for b in "IXYZ"]
    return {
        label: (1.0 - p_error if label == "II" else p_error / 15.0) for label in labels
    }


def _write_card(directory: Path, idle: bool = True) -> Path:
    """Writes a small device card.

    ``h`` and ``x`` take 50 ns and err with probability 0.01 (an X error), and
    ``cx`` takes 200 ns and errs with probability 0.03. The idle channel's
    transfer diagonal is 0.99 (X, Y) and 0.98 (Z) at 50 ns and the squares at
    100 ns.

    Args:
        directory: Where to write it.
        idle: Whether to include idle data.

    Returns:
        The card file's path.
    """

    one_qubit = {"I": 0.99, "X": 0.01, "Y": 0.0, "Z": 0.0}
    card = {
        "architecture": "test",
        "model": "unit",
        "gates": {
            "h": {"T_ns": 50.0, "wires": 1, "pauli_error": one_qubit},
            "x": {"T_ns": 50.0, "wires": 1, "pauli_error": one_qubit},
            "cx": {"T_ns": 200.0, "wires": 2, "pauli_error": _two_qubit_paulis(0.03)},
        },
        "idle": (
            {
                "50": {"ptm_error_diag": {"I": 1.0, "X": 0.99, "Y": 0.99, "Z": 0.98}},
                "100": {
                    "ptm_error_diag": {
                        "I": 1.0,
                        "X": 0.99**2,
                        "Y": 0.99**2,
                        "Z": 0.98**2,
                    }
                },
            }
            if idle
            else {}
        ),
    }
    path = directory / "device_card_unit.json"
    path.write_text(json.dumps(card))
    return path


def _parse(flags: list[str]) -> argparse.Namespace:
    """Parses noise flags.

    Args:
        flags: The command-line flags.

    Returns:
        The parsed namespace.
    """

    parser = argparse.ArgumentParser()
    PennyLaneNoiseModel.initialize_parser(parser)
    return parser.parse_args(flags)


def test_load_device_card_keeps_what_the_noise_needs(tmp_path: Path) -> None:
    """The compact card keeps durations, channels and idle data, by name.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    card = load_device_card(str(_write_card(tmp_path)))

    assert card["name"] == "unit"
    assert card["gates"]["cx"]["T_ns"] == 200.0
    assert card["gates"]["h"]["pauli_error"]["X"] == 0.01
    assert set(card["idle"]) == {"50.0", "100.0"}
    assert card["wired_pairs"] is None
    json.dumps(card)  # it travels in genome hyperparameters


def test_missing_cards_and_gates_are_reported(tmp_path: Path) -> None:
    """An unknown card lists what exists; a card without cx is refused.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    with pytest.raises(ValueError, match="no device card"):
        resolve_device_card("no_such_card")

    path = _write_card(tmp_path)
    raw = json.loads(path.read_text())
    del raw["gates"]["cx"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cx"):
        load_device_card(str(path))


def test_kraus_operators_are_trace_preserving() -> None:
    """Negative twirl entries are clipped and the rest renormalized."""

    kraus = kraus_from_pauli_probabilities(
        {"I": 0.98, "X": 0.015, "Y": -0.001, "Z": 0.004}
    )
    total = sum(k.conj().T @ k for k in kraus)

    assert len(kraus) == 3
    assert np.allclose(total, np.eye(2))


def test_idle_channel_interpolates_and_extrapolates(tmp_path: Path) -> None:
    """Grid points are exact, between them log-linear, beyond them a power.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    card = load_device_card(str(_write_card(tmp_path)))

    assert idle_ptm_diagonal(card, 50.0)["Z"] == pytest.approx(0.98)
    assert idle_ptm_diagonal(card, 75.0)["Z"] == pytest.approx(0.98**1.5)
    assert idle_ptm_diagonal(card, 200.0)["Z"] == pytest.approx(0.98**4)
    assert idle_ptm_diagonal(card, 0.0) is None
    # p_I = (1 + 0.99 + 0.99 + 0.98) / 4 at 50 ns
    assert idle_error_probability(card, 50.0) == pytest.approx(1 - 3.96 / 4)

    no_idle = load_device_card(str(_write_card(tmp_path, idle=False)))
    assert idle_error_probability(no_idle, 500.0) == 0.0


def test_timeline_and_hardware_metrics(tmp_path: Path) -> None:
    """Waits, makespan and p_fail for a hand-built translation.

    Wire 0 runs ``h`` (0-50 ns) and wire 1 ``x`` (0-50 ns); a ``cx`` on wires
    0 and 1 follows (50-250 ns). Wire 2 is free from the start, so its ``h``
    runs at 0-50 ns and it then idles 200 ns until the end.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    card = load_device_card(str(_write_card(tmp_path)))
    first, second, third = object(), object(), object()
    schedule = CompiledSchedule(
        per_gate={
            id(first): [ChipGate("h", [0], [0]), ChipGate("rz", [0], [0])],
            id(second): [ChipGate("x", [1], [1]), ChipGate("cx", [0, 1], [0, 1])],
            id(third): [ChipGate("h", [2], [2])],
        },
        final_slot=[0, 1, 2],
    )

    timeline = card_timeline(schedule, card, n_wires=3)

    assert timeline.makespan_ns == 250.0
    assert timeline.idle_before[id(second)] == {}  # both wires free at 50 ns
    assert timeline.idle_at_end == [0.0, 0.0, 200.0]

    metrics = hardware_metrics(timeline, card)
    success = 0.99**3 * 0.97 * (1 - idle_error_probability(card, 200.0))
    assert metrics["p_fail"] == pytest.approx(1 - success)
    assert metrics["total_time_ns"] == 250.0
    assert metrics["n_entanglers"] == 1.0


def test_flags_build_device_card_noise(tmp_path: Path) -> None:
    """--noise_device_card selects the type, and the settings round-trip.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    card = str(_write_card(tmp_path))
    model = PennyLaneNoiseModel.from_args(_parse(["--noise_device_card", card]))

    assert model.noise_type == "device_card"
    assert model.device_name() == "default.mixed"
    assert model.apply_idle
    again = PennyLaneNoiseModel.from_hyperparameters(model.to_hyperparameters())
    assert again.device_card == model.device_card

    off = PennyLaneNoiseModel.from_args(
        _parse(["--noise_device_card", card, "--no-noise_idle"])
    )
    assert not off.apply_idle

    with pytest.raises(ValueError, match="needs --noise_device_card"):
        PennyLaneNoiseModel.from_args(_parse(["--noise_type", "device_card"]))
    with pytest.raises(ValueError, match="only works with --noise_type device_card"):
        PennyLaneNoiseModel.from_args(
            _parse(["--noise_device_card", card, "--noise_type", "depolarizing"])
        )


def test_chip_wiring_comes_from_the_coupling_map() -> None:
    """Only the chosen qubits' couplings become wired wire pairs."""

    raw = {
        "backend": {
            "name": "fake",
            "n_qubits": 5,
            "coupling_map": [[0, 1], [1, 0], [1, 2], [2, 3], [3, 4], [4, 0]],
        }
    }

    assert wired_slot_pairs(raw, [2, 1, 0]) == [[0, 1], [1, 2]]
    assert wired_slot_pairs(raw, [0, 4, 3]) == [[0, 1], [1, 2]]
    with pytest.raises(ValueError, match="not on fake"):
        wired_slot_pairs(raw, [7])


def test_calibration_supplies_only_the_wiring(tmp_path: Path) -> None:
    """With a calibration file, the card keeps its rates and takes the chip's wiring.

    Uses the calibration packaged with Qiskit's offline ``FakeTorino``, saved
    as a stored calibration file. Torino qubits 0-1-2 are a line, so wires 0
    and 2 are not wired, and a gate between them needs routing moves.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    fake_provider = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    from src.noise.calibration.backends import (
        fetch_raw_calibration,
        save_raw_calibration,
    )

    raw = fetch_raw_calibration(fake_provider.FakeTorino())
    calibration = save_raw_calibration(raw, tmp_path / "calibration")
    card = str(_write_card(tmp_path))

    model = PennyLaneNoiseModel.from_args(
        _parse(
            [
                "--noise_device_card",
                card,
                "--noise_calibration",
                str(calibration),
                "--noise_qubits",
                "0,1,2",
            ]
        )
    )

    assert model.noise_type == "device_card"
    assert model.per_qubit is None  # no chip rates
    assert model.device_card["wired_pairs"] == [[0, 1], [1, 2]]
    assert model.device_card["wiring"]["qubits"] == [0, 1, 2]
    assert model.device_card["gates"]["cx"]["T_ns"] == 200.0

    with pytest.raises(ValueError, match="go together"):
        PennyLaneNoiseModel.from_args(
            _parse(
                ["--noise_device_card", card, "--noise_calibration", str(calibration)]
            )
        )


def _card_settings(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Device-card ``noise_*`` genome settings for the unit card.

    Args:
        tmp_path: Where to write the card.
        **overrides: Settings to replace, e.g. ``noise_idle=False``.

    Returns:
        The settings.
    """

    model = PennyLaneNoiseModel.from_args(
        _parse(["--noise_device_card", str(_write_card(tmp_path))])
    )
    settings = model.to_hyperparameters()
    settings.update(overrides)
    return settings


def test_device_card_noise_matches_across_targets(tmp_path: Path) -> None:
    """Card noise recorded on a genome gives the same output on both targets.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    settings = _card_settings(tmp_path)

    noiseless = _quantum_probs("pennylane")
    pennylane = _quantum_probs("pennylane", hyperparameters=settings)
    qiskit = _quantum_probs("qiskit", hyperparameters=settings)

    assert not np.allclose(pennylane, noiseless)
    assert qiskit == pytest.approx(pennylane, abs=1e-6)


def test_idle_noise_changes_the_output(tmp_path: Path) -> None:
    """Turning idle noise off removes some noise but keeps the gate noise.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    noiseless = _quantum_probs("pennylane")
    with_idle = _quantum_probs("pennylane", hyperparameters=_card_settings(tmp_path))
    without_idle = _quantum_probs(
        "pennylane", hyperparameters=_card_settings(tmp_path, noise_idle=False)
    )

    assert not np.allclose(without_idle, noiseless)
    distance_with = np.abs(with_idle - noiseless).sum()
    distance_without = np.abs(without_idle - noiseless).sum()
    assert not math.isclose(distance_with, distance_without)


def test_every_card_gate_is_charged(tmp_path: Path) -> None:
    """The model inserts one channel per noisy card gate and per positive wait.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    import pennylane as qml

    genome = _build_genome("pennylane", _card_settings(tmp_path))
    model = PennyLaneNoiseModel.from_hyperparameters(genome.hyperparameters)
    model.begin_circuit(genome)
    timeline = model._card_timeline

    with qml.queuing.AnnotatedQueue() as queue:
        model.after_input_encoding(genome.input_indexes)
        for gate in genome.gates:
            if gate.enabled:
                wires = gate.get_pennylane_wires(genome.qubits)
                model.before_gate(wires, gate)
                model.after_gate(wires, gate)
        model.before_measurement(genome.output_indexes)

    noisy_gates = sum(gate.name != "rz" for gate in timeline.gates)
    waits = sum(len(waits) for waits in timeline.idle_before.values()) + sum(
        wait > 1e-9 for wait in timeline.idle_at_end
    )
    assert len(queue.queue) == noisy_gates + waits


def test_multi_objective_fitness_adds_hardware_keys(tmp_path: Path) -> None:
    """classification_moo records the card's p_fail, duration and cx count.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    from src.examples.classification_moo import hardware_fitness, parse_objectives

    noisy = _build_genome("pennylane", _card_settings(tmp_path))
    metrics = hardware_fitness(noisy)

    assert set(metrics) == {"p_fail", "total_time_ns", "n_entanglers"}
    assert 0.0 < metrics["p_fail"] < 1.0
    assert metrics["total_time_ns"] > 0.0
    assert hardware_fitness(_build_genome("pennylane")) == {}

    parsed = parse_objectives(["loss", "p_fail", "total_time_ns:1"])
    assert [o.name for o in parsed] == ["loss", "p_fail", "total_time_ns"]


def test_hardware_objectives_need_a_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """Choosing p_fail without --noise_device_card stops before any work.

    Args:
        monkeypatch: Used to set the command line.
    """

    import sys

    from src.examples import classification_moo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification_moo",
            "--dataset",
            "iris",
            "--input_qubits",
            "4",
            "--output_qubits",
            "2",
            "-ms",
            "uniform",
            "1",
            "3",
            "-ps",
            "uniform",
            "5",
            "5",
            "--objectives",
            "loss",
            "p_fail",
            "--out_dir",
            "unused",
            "nsga2",
        ],
    )
    with pytest.raises(SystemExit):
        classification_moo.main()
