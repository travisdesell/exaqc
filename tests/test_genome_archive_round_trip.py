"""Round-trip tests: a genome read back from the archive is identical to the one written.

Every tool that reloads a genome -- refinement, evaluation, RL visualization,
the EXAQC dashboard and the analysis scripts -- now reads it out of a run's
``genomes.sqlar`` instead of a loose JSON file. These tests build real
:class:`~src.circuits.circuit.CircuitGenome` objects for both quantum backends,
write them into an archive, read them back out, and check that nothing was
lost or changed: every serialized field, the gates and their trained
parameters, the exact encoder/decoder weights, and the model's outputs.
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend before anything imports pyplot.
import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402
import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.utils.genome_archive import (  # noqa: E402
    GenomeArchive,
    genome_member_name,
    iter_run_genome_dicts,
    load_genome_dict,
)
from tests.supervised_trainer_test_utils import (  # noqa: E402
    build_classification_genome,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")


def add_search_metadata(genome: CircuitGenome) -> CircuitGenome:
    """Gives a genome the task, fitness and metadata a search would record.

    Args:
        genome: The genome to decorate.

    Returns:
        The same genome, for chaining.
    """

    genome.task = "classification"
    genome.task_target = "iris"
    genome.fitness = {"loss": 0.123456789, "target_metric": 0.87654321}
    genome.metadata = {
        "parent_genomes": [3, 5],
        "generated_by": ["n_ary_crossover", "add_gate"],
        "insert_type": "global_best",
        "best_epoch": 1,
        "n_trainable_parameters": 42,
        "best_training_metrics": {
            "loss": 0.1,
            "mean_class_accuracy": {"mean": 0.9, "per_class": [0.8, 1.0]},
        },
        "best_validation_metrics": {
            "loss": 0.2,
            "mean_class_accuracy": {"mean": 0.85, "per_class": [0.7, 1.0]},
        },
        "training_epoch_metrics": [
            {
                "epoch": epoch,
                "loss": 1.0 / (epoch + 1),
                "mean_class_accuracy": {"mean": 0.5},
            }
            for epoch in range(3)
        ],
    }
    return genome


def build_quantum_only_genome() -> CircuitGenome:
    """Builds a purely quantum (teacher-style) genome with no encoder or decoder.

    Returns:
        The genome, carrying a parametric and a non-parametric gate.
    """

    genome = CircuitGenome(
        genome_number=9,
        target="pennylane",
        input_qubits=[("q", 0)],
        output_qubits=[("q", 1)],
        task="teacher",
        task_target="bell_out",
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
    genome.add_gate(
        depth=0.25, method_name="ry", qubits=[("q", 0)], parameters={"theta": 0.321}
    )
    genome.add_gate(depth=0.5, method_name="cx", qubits=[("q", 0), ("q", 1)])
    genome.fitness = {"loss": 0.5, "target_metric": 0.5}
    genome.metadata = {
        "parent_genomes": [1],
        "generated_by": ["add_gate"],
        "insert_type": "inserted",
    }
    return genome


def write_and_read_back(genome: CircuitGenome, tmp_path) -> tuple[dict, CircuitGenome]:
    """Writes a genome into a fresh archive, closes it, and reads the genome back.

    Args:
        genome: The genome to round-trip.
        tmp_path: Directory to create the run output directory in.

    Returns:
        The serialized dict read from the archive, and the genome rebuilt from it.
    """

    run_dir = tmp_path / "run"
    with GenomeArchive.create(str(run_dir)) as archive:
        assert archive.add_genome(genome, insertion=1) is True

    with GenomeArchive.open_readonly(str(run_dir)) as reader:
        serialized = reader.get_genome_dict(genome.genome_number)

    return serialized, CircuitGenome.from_dict(serialized)


def assert_identical_genomes(original: CircuitGenome, restored: CircuitGenome) -> None:
    """Asserts two genomes are identical in every recorded respect.

    Args:
        original: The genome that was written.
        restored: The genome read back.
    """

    assert restored.to_dict() == original.to_dict()

    assert restored.genome_number == original.genome_number
    assert restored.task == original.task
    assert restored.task_target == original.task_target
    assert restored.target == original.target
    assert restored.fitness == original.fitness
    assert restored.metadata == original.metadata
    assert restored.hyperparameters == original.hyperparameters
    assert restored.input_qubits == original.input_qubits
    assert restored.output_qubits == original.output_qubits
    assert restored.qubits == original.qubits
    assert restored.input_indexes == original.input_indexes
    assert restored.output_indexes == original.output_indexes

    assert restored.get_gate_innovations() == original.get_gate_innovations()
    assert restored.get_parameters_as_list() == original.get_parameters_as_list()
    assert len(restored.gates) == len(original.gates)
    for restored_gate, original_gate in zip(restored.gates, original.gates):
        assert restored_gate.innovation_number == original_gate.innovation_number
        assert restored_gate.depth == original_gate.depth
        assert restored_gate.method_name == original_gate.method_name
        assert restored_gate.qubits == original_gate.qubits
        assert restored_gate.parameters == original_gate.parameters
        assert restored_gate.target == original_gate.target
        assert restored_gate.enabled == original_gate.enabled

    for original_stage, restored_stage in (
        (original.encoder, restored.encoder),
        (original.decoder, restored.decoder),
    ):
        if original_stage is None:
            assert restored_stage is None
            continue
        assert type(restored_stage) is type(original_stage)
        if isinstance(original_stage, torch.nn.Module):
            original_state = original_stage.state_dict()
            restored_state = restored_stage.state_dict()
            assert restored_state.keys() == original_state.keys()
            for name, tensor in original_state.items():
                assert restored_state[name].dtype == tensor.dtype
                assert torch.equal(restored_state[name], tensor), name


@pytest.mark.parametrize("complexity", ["shallow", "multi_param"])
@pytest.mark.parametrize("decoder_name", ["clipped", "linear"])
@pytest.mark.parametrize("target", TARGETS)
def test_archived_genome_is_identical_to_the_written_genome(
    target: str, decoder_name: str, complexity: str, tmp_path
) -> None:
    """A hybrid genome read back from the archive matches the one written exactly.

    Args:
        target: The quantum backend.
        decoder_name: The decoder to build (``linear`` carries trained weights).
        complexity: The circuit complexity level to build.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome, n_features = build_classification_genome(
        genome_number=11,
        target=target,
        complexity=complexity,
        encoder_name="linear",
        decoder_name=decoder_name,
        include_parametric=True,
    )
    add_search_metadata(genome)

    serialized, restored = write_and_read_back(genome, tmp_path)

    # the stored JSON is exactly what serializing the genome produces
    assert serialized == json.loads(json.dumps(genome.to_dict()))
    assert_identical_genomes(genome, restored)

    # and the rebuilt model computes the same outputs
    torch.manual_seed(0)
    inputs = torch.rand(3, n_features)
    genome.initialize_model()
    restored.initialize_model()
    with torch.no_grad():
        assert torch.allclose(
            restored.forward(inputs), genome.forward(inputs), atol=1e-6
        )


def test_archived_quantum_only_genome_is_identical(tmp_path) -> None:
    """A genome without an encoder or decoder also survives the round trip.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome = build_quantum_only_genome()

    _, restored = write_and_read_back(genome, tmp_path)

    assert_identical_genomes(genome, restored)


@pytest.mark.parametrize("target", TARGETS)
def test_every_way_of_reading_an_archive_returns_the_same_genome(
    target: str, tmp_path
) -> None:
    """The single-genome loader, the analysis iterator and the archive agree.

    Args:
        target: The quantum backend.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome, _ = build_classification_genome(
        genome_number=5,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
    )
    add_search_metadata(genome)
    serialized, _ = write_and_read_back(genome, tmp_path)

    run_dir = str(tmp_path / "run")
    assert load_genome_dict(archive=run_dir, genome_number=5) == serialized
    ((_, iterated),) = list(iter_run_genome_dicts(run_dir))
    assert iterated == serialized


@pytest.mark.parametrize("target", TARGETS)
def test_rewriting_a_read_back_genome_stores_identical_bytes(
    target: str, tmp_path
) -> None:
    """Writing, reading and writing again is lossless down to the stored bytes.

    Args:
        target: The quantum backend.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome, _ = build_classification_genome(
        genome_number=6,
        target=target,
        complexity="multi_param",
        encoder_name="linear",
        decoder_name="linear",
    )
    add_search_metadata(genome)
    _, restored = write_and_read_back(genome, tmp_path)

    def stored_member(
        archive_genome: CircuitGenome, run_name: str
    ) -> tuple[int, bytes]:
        """Stores a genome in its own archive and returns the raw member row."""
        with GenomeArchive.create(str(tmp_path / run_name)) as archive:
            archive.add_genome(archive_genome, insertion=1)
            return archive.connection.execute(
                "SELECT sz, data FROM sqlar WHERE name = ?",
                (genome_member_name(archive_genome.genome_number),),
            ).fetchone()

    assert stored_member(restored, "rewritten") == stored_member(genome, "original")


def test_non_finite_fitness_values_survive_the_round_trip(tmp_path) -> None:
    """NaN and infinite fitness values are kept in the stored genome.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome = build_quantum_only_genome()
    genome.fitness = {"loss": float("nan"), "target_metric": float("-inf")}

    _, restored = write_and_read_back(genome, tmp_path)

    assert math.isnan(restored.fitness["loss"])
    assert restored.fitness["target_metric"] == float("-inf")

    # the summary used for sorting stores them as missing values instead
    with GenomeArchive.open_readonly(str(tmp_path / "run")) as reader:
        assert reader.get_summary(genome.genome_number)["fitness"] == {
            "loss": None,
            "target_metric": None,
        }
