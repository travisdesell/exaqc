"""Tests for the quantum-teacher imitation entry point.

``src.examples.teacher`` mirrors the classification and reinforcement-learning
entry points, but evolves *purely quantum* genomes: there is nothing classical
to learn, so it seeds no encoder and no decoder and exposes no
``--encoding``/``--decoding`` options.

These tests pin the wiring that the entry point is responsible for -- the wire
layout, the fitness it records, the arguments it hands to ``master_worker``, and
the configurations it refuses -- without running the MPI driver.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch

import src.examples.teacher as teacher
from src.circuits.teacher_circuits import DEFAULT_REGISTER_NAME, TEACHER_NAMES
from src.metrics.teacher_losses import TEACHER_LOSS_NAMES


def base_argv(out_dir, **overrides) -> list[str]:
    """Builds a minimal valid command line for the teacher entry point.

    Args:
        out_dir: Directory to pass as ``--out_dir``.
        **overrides: Flag values to replace, keyed without the leading dashes.

    Returns:
        The argv list, including the required ``steady_state`` sub-command.
    """

    flags = {
        "teacher": "bell_out",
        "input_qubits": "1",
        "output_qubits": "2",
        "loss": "fidelity",
        "epochs": "1",
        "number_genomes": "1",
        "n_training_samples": "4",
        "n_validation_samples": "2",
        "batch_size": "2",
        "out_dir": str(out_dir),
    }
    flags.update({key: str(value) for key, value in overrides.items()})

    argv = ["teacher.py"]
    for key, value in flags.items():
        argv += [f"--{key}", value]
    argv += ["-ms", "uniform", "1", "2", "-ps", "uniform", "2", "3"]
    argv += ["--binary_crossover_rate", "0.1"]
    argv += ["steady_state", "--max_population_size", "2"]
    return argv


def test_teacher_wires_are_disjoint_and_ordered() -> None:
    """Input wires come first and output wires follow, never overlapping."""

    input_wires, output_wires = teacher.teacher_wires(2, 3)

    assert input_wires == [0, 1]
    assert output_wires == [2, 3, 4]
    assert set(input_wires).isdisjoint(output_wires)


def test_parser_offers_the_expected_choices() -> None:
    """The parser exposes every teacher and loss, and no classical stages."""

    parser = teacher.build_parser()
    actions = {action.dest: action for action in parser._actions}

    assert set(actions["teacher"].choices) == set(TEACHER_NAMES)
    assert set(actions["loss"].choices) == set(TEACHER_LOSS_NAMES)
    assert set(actions["quantum_output_mode"].choices) == {"probs", "expval"}

    # a purely quantum search has no classical stages to configure
    assert "encoding" not in actions
    assert "decoding" not in actions

    # the shared search and population flags are present
    assert "mutation_strategy" in actions
    assert "binary_crossover_rate" in actions


def test_compare_orders_by_loss() -> None:
    """``compare`` ranks the lower-loss genome first."""

    better = MagicMock()
    better.fitness = {"loss": 0.1}
    worse = MagicMock()
    worse.fitness = {"loss": 0.4}

    assert teacher.compare(better, worse) < 0
    assert teacher.compare(worse, better) > 0
    assert teacher.compare(better, better) == 0


def test_objective_records_loss_and_fidelity_fitness() -> None:
    """The objective writes the fitness keys the analysis tooling reads."""

    objective = teacher.TeacherObjective(
        training_dataloader=MagicMock(),
        validation_dataloader=MagicMock(),
        loss_name="fidelity",
    )
    objective.trainer = MagicMock()

    genome = MagicMock()
    genome.metadata = {
        "best_training_metrics": {"loss": 0.2, "fidelity": {"mean": 0.8}},
        "best_validation_metrics": {"loss": 0.4, "fidelity": {"mean": 0.6}},
    }

    objective(genome)

    objective.trainer.train.assert_called_once_with(genome)
    assert genome.fitness["loss"] == pytest.approx(0.3)
    assert genome.fitness["target_metric"] == pytest.approx(0.7)


def test_main_seeds_a_purely_quantum_search(monkeypatch, tmp_path) -> None:
    """``main`` hands ``master_worker`` no encoder, no decoder, disjoint wires.

    Args:
        monkeypatch: Used to replace ``master_worker`` and ``sys.argv``.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    mocked_master_worker = MagicMock()
    monkeypatch.setattr(teacher, "master_worker", mocked_master_worker)
    monkeypatch.setattr(teacher.logger, "remove", MagicMock())
    monkeypatch.setattr(teacher.logger, "add", MagicMock())
    monkeypatch.setattr(
        sys, "argv", base_argv(tmp_path, teacher="half_adder", input_qubits="2")
    )

    teacher.main()

    mocked_master_worker.assert_called_once()
    call = mocked_master_worker.call_args.kwargs

    # nothing classical is seeded
    assert call["initial_encoder"] is None
    assert call["initial_decoder"] is None

    # explicit, disjoint qubit lists rather than overlapping registers
    assert call["input_qubits"] == [
        (DEFAULT_REGISTER_NAME, 0),
        (DEFAULT_REGISTER_NAME, 1),
    ]
    assert call["output_qubits"] == [
        (DEFAULT_REGISTER_NAME, 2),
        (DEFAULT_REGISTER_NAME, 3),
    ]
    assert set(call["input_qubits"]).isdisjoint(call["output_qubits"])

    assert call["run_for"] == 1
    assert call["target"] == "pennylane"
    assert call["hyperparameters"]["quantum_input_mode"] == "ry"
    assert call["hyperparameters"]["quantum_output_mode"] == "probs"
    assert call["hyperparameters"]["epochs"] == 1


def test_main_builds_loaders_sized_to_the_wires(monkeypatch, tmp_path) -> None:
    """The generated dataset matches the requested wire layout.

    Args:
        monkeypatch: Used to replace ``master_worker`` and ``sys.argv``.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    captured = {}

    def capture_objective(*args, **kwargs):
        """Records the loaders the objective is constructed with."""
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(teacher, "master_worker", MagicMock())
    monkeypatch.setattr(teacher, "TeacherObjective", capture_objective)
    monkeypatch.setattr(teacher.logger, "remove", MagicMock())
    monkeypatch.setattr(teacher.logger, "add", MagicMock())
    monkeypatch.setattr(
        sys, "argv", base_argv(tmp_path, teacher="half_adder", input_qubits="2")
    )

    teacher.main()

    training_loader = captured["training_dataloader"]
    assert training_loader.n_features == 2
    assert training_loader.n_targets == 4
    assert training_loader.input_wires == [0, 1]
    assert training_loader.output_wires == [2, 3]

    inputs, targets = next(iter(training_loader))
    assert inputs.shape[-1] == 2
    assert targets.shape[-1] == 4
    assert targets.dtype == torch.float32


@pytest.mark.parametrize("loss_name", ["fidelity", "angle", "kl"])
def test_distribution_losses_require_probs(loss_name, monkeypatch, tmp_path) -> None:
    """Distribution losses are refused with the expval readout.

    Args:
        loss_name: The distribution loss under test.
        monkeypatch: Used to replace ``sys.argv``.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    monkeypatch.setattr(teacher, "master_worker", MagicMock())
    monkeypatch.setattr(teacher.logger, "remove", MagicMock())
    monkeypatch.setattr(teacher.logger, "add", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        base_argv(tmp_path, loss=loss_name, quantum_output_mode="expval"),
    )

    with pytest.raises(SystemExit) as error:
        teacher.main()

    assert error.value.code == 2


def test_teacher_that_cannot_fit_the_wires_is_reported(monkeypatch, tmp_path) -> None:
    """A teacher needing more wires than requested fails with a clear message.

    Args:
        monkeypatch: Used to replace ``sys.argv``.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    monkeypatch.setattr(teacher, "master_worker", MagicMock())
    monkeypatch.setattr(teacher.logger, "remove", MagicMock())
    monkeypatch.setattr(teacher.logger, "add", MagicMock())
    # half_adder needs two input and two output wires
    monkeypatch.setattr(
        sys,
        "argv",
        base_argv(tmp_path, teacher="half_adder", input_qubits="1", output_qubits="1"),
    )

    with pytest.raises(SystemExit) as error:
        teacher.main()

    assert error.value.code == 2
