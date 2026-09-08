"""Tests for genome task recording and the refinement entry point.

EXAQC stamps every genome it generates with the ``task`` it was evolved for and
the ``task_target`` it was run against (the dataset, teacher circuit, or
environment). A saved genome is therefore self-describing, and
``src.examples.refine_genome`` can reload one and continue training it without
being told anything beyond the file path.

These tests pin that the fields survive serialization, that EXAQC applies them,
and that the refinement entry point routes on them -- including refusing a
genome that predates the fields rather than guessing.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from src.circuits.circuit import CircuitGenome
from src.examples import refine_genome


def build_genome(
    task: str | None = "teacher",
    task_target: str | None = "bell_out",
    genome_number: int = 1,
) -> CircuitGenome:
    """Builds a minimal purely quantum genome carrying task information.

    Args:
        task: The task to record, or ``None`` to leave it unset.
        task_target: The task target to record, or ``None`` to leave it unset.
        genome_number: The genome's identifier.

    Returns:
        A :class:`CircuitGenome` with one gate and teacher-shaped hyperparameters.
    """

    genome = CircuitGenome(
        genome_number=genome_number,
        target="pennylane",
        input_qubits=[("q", 0)],
        output_qubits=[("q", 1)],
        task=task,
        task_target=task_target,
    )
    genome.hyperparameters = {
        "epochs": 3,
        "learning_rate": 0.005,
        "batch_size": 4,
        "quantum_input_mode": "ry",
        "quantum_output_mode": "probs",
    }
    genome.encoder = None
    genome.decoder = None
    genome.add_gate(depth=0.5, method_name="cx", qubits=[("q", 0), ("q", 1)])
    return genome


def write_genome(genome: CircuitGenome, tmp_path, name: str = "genome.json") -> str:
    """Serializes a genome to a JSON file.

    Args:
        genome: The genome to write.
        tmp_path: Directory to write into.
        name: File name to use.

    Returns:
        The path written to.
    """

    path = tmp_path / name
    path.write_text(json.dumps(genome.to_dict()), encoding="utf-8")
    return str(path)


def test_task_fields_round_trip_through_serialization() -> None:
    """``task`` and ``task_target`` survive ``to_dict``/``from_dict``."""

    genome = build_genome(task="teacher", task_target="half_adder")

    serialized = genome.to_dict()
    assert serialized["task"] == "teacher"
    assert serialized["task_target"] == "half_adder"

    restored = CircuitGenome.from_dict(serialized)
    assert restored.task == "teacher"
    assert restored.task_target == "half_adder"


def test_task_fields_default_to_none_for_older_genomes() -> None:
    """A genome file without the fields loads with them unset, not crashing."""

    serialized = build_genome().to_dict()
    del serialized["task"]
    del serialized["task_target"]

    restored = CircuitGenome.from_dict(serialized)

    assert restored.task is None
    assert restored.task_target is None


def test_load_genome_reads_the_task(tmp_path) -> None:
    """A saved genome is loaded back with its task information intact.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = write_genome(build_genome("teacher", "bell_out"), tmp_path)

    genome = refine_genome.load_genome(path)

    assert genome.task == "teacher"
    assert genome.task_target == "bell_out"
    assert len(genome.gates) == 1


def test_load_genome_rejects_a_genome_without_a_task(tmp_path) -> None:
    """A genome predating task recording is refused with a clear message.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    serialized = build_genome().to_dict()
    serialized["task"] = None
    serialized["task_target"] = None
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(serialized), encoding="utf-8")

    with pytest.raises(ValueError, match="does not record the task"):
        refine_genome.load_genome(str(path))


def test_load_genome_rejects_a_non_genome_file(tmp_path) -> None:
    """A file that is not a genome is reported rather than half-loaded.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = tmp_path / "not_a_genome.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not look like a genome file"):
        refine_genome.load_genome(str(path))


@pytest.mark.parametrize(
    "override, key, expected",
    [
        ("epochs=200", "epochs", 200),
        ("learning_rate=1e-3", "learning_rate", 0.001),
        ("quantum_input_mode=rx", "quantum_input_mode", "rx"),
    ],
)
def test_overrides_are_coerced_to_the_existing_type(override, key, expected) -> None:
    """An override keeps the type the hyperparameter already had.

    Args:
        override: The raw ``key=value`` string.
        key: The hyperparameter it targets.
        expected: The value expected after coercion.
    """

    genome = build_genome()

    refine_genome.apply_overrides(genome, [override])

    assert genome.hyperparameters[key] == expected
    assert isinstance(genome.hyperparameters[key], type(expected))


@pytest.mark.parametrize(
    "override, message",
    [
        ("epochs", "not in key=value form"),
        ("not_a_hyperparameter=3", "has no hyperparameter"),
        ("epochs=many", "Could not read"),
    ],
)
def test_bad_overrides_are_rejected(override, message) -> None:
    """Malformed, unknown, or uncoercible overrides fail loudly.

    Args:
        override: The raw ``key=value`` string.
        message: Substring expected in the raised error.
    """

    with pytest.raises(ValueError, match=message):
        refine_genome.apply_overrides(build_genome(), [override])


def test_every_task_has_an_objective_builder() -> None:
    """The refinement entry point can rebuild all three tasks."""

    assert set(refine_genome.OBJECTIVE_BUILDERS) == {
        "classification",
        "teacher",
        "reinforcement_learning",
    }


def test_main_routes_on_the_genomes_recorded_task(monkeypatch, tmp_path) -> None:
    """``main`` picks the objective from the genome, needing only the path.

    Args:
        monkeypatch: Used to replace the objective builder and ``sys.argv``.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = write_genome(build_genome("teacher", "bell_out"), tmp_path)

    trained = {}

    def fake_builder(genome, device):
        """Records the genome it was asked to build an objective for."""
        trained["genome"] = genome

        def objective(target_genome):
            """Pretends to train, writing a fitness."""
            target_genome.fitness = {"loss": 0.1, "target_metric": 0.9}

        return objective

    monkeypatch.setitem(refine_genome.OBJECTIVE_BUILDERS, "teacher", fake_builder)
    monkeypatch.setattr(refine_genome.logger, "remove", MagicMock())
    monkeypatch.setattr(refine_genome.logger, "add", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refine_genome.py",
            "--genome",
            path,
            "--out_dir",
            str(tmp_path / "out"),
            "--no-save_circuit",
            "--set",
            "epochs=42",
        ],
    )

    refine_genome.main()

    # routed on the genome's own task, with the override applied
    assert trained["genome"].task == "teacher"
    assert trained["genome"].task_target == "bell_out"
    assert trained["genome"].hyperparameters["epochs"] == 42

    # and the refined genome is written back out, still self-describing
    refined_path = tmp_path / "out" / "refined_genome_1.json"
    assert refined_path.is_file()
    refined = json.loads(refined_path.read_text())
    assert refined["task"] == "teacher"
    assert refined["task_target"] == "bell_out"
    assert refined["fitness"]["loss"] == pytest.approx(0.1)


def test_main_rejects_an_unknown_task(monkeypatch, tmp_path) -> None:
    """A genome recording a task the entry point cannot rebuild is refused.

    Args:
        monkeypatch: Used to replace ``sys.argv``.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = write_genome(build_genome("some_future_task", "thing"), tmp_path)

    monkeypatch.setattr(refine_genome.logger, "remove", MagicMock())
    monkeypatch.setattr(refine_genome.logger, "add", MagicMock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["refine_genome.py", "--genome", path, "--out_dir", str(tmp_path / "out")],
    )

    with pytest.raises(SystemExit) as error:
        refine_genome.main()

    assert error.value.code == 2


def test_exaqc_stamps_generated_genomes(tmp_path) -> None:
    """EXAQC records its task and target on every genome it generates.

    This is the integration point that makes a saved genome self-describing, so
    it is checked against a real search rather than a hand-built genome.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    from src.circuits.encoder import initialize_encoder
    from src.circuits.decoder import initialize_decoder
    from src.circuits.pennylane_gate_specifications import (
        pennylane_gate_specifications,
    )
    from src.evolution.exaqc import EXAQC
    from src.evolution.steady_state_population import SteadyStatePopulation

    search = EXAQC(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=4, compare=lambda a, b: 0, out_dir=str(tmp_path)
        ),
        objective=lambda genome: None,
        initial_encoder=initialize_encoder(
            target="pennylane",
            encoding_str="linear",
            n_inputs=4,
            n_outputs=2,
            quantum_input_mode="ry",
            n_input_qubits=2,
        ),
        initial_decoder=initialize_decoder(
            target="pennylane", decoding_str="linear", n_inputs=4, n_outputs=2
        ),
        hyperparameters={
            "quantum_input_mode": "ry",
            "quantum_output_mode": "probs",
            "epochs": 1,
            "learning_rate": 0.01,
        },
        mutation_strategy=["uniform", "1", "2"],
        parent_strategy=["uniform", "2", "3"],
        input_registers={"input": 2},
        output_registers={"input": 2},
        target="pennylane",
        task="classification",
        task_target="iris",
    )

    child = search.generate_genome()

    assert child.task == "classification"
    assert child.task_target == "iris"
    # and it survives a round trip, which is what refinement relies on
    assert CircuitGenome.from_dict(child.to_dict()).task_target == "iris"
