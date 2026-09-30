"""Tests for projecting a run's genomes into a search space (``src.analysis.search_space``).

They check that a genome's unitary is the product of its enabled gates on both
targets (and padded with the identity onto a larger qubit set), that the
distance metrics measure what they claim (the Fubini–Study angle ignores global
phase), that the projections place genomes faithfully and can be aligned, and
that a run's archive is turned into a projection with its links and global-best
path, reusing computed features as the run grows.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.analysis.search_space import (
    DISTANCE_METRICS,
    PROJECTIONS,
    DistanceMetric,
    SearchSpaceBuilder,
    UnsupportedGenome,
    best_path,
    get_distance,
    get_projection,
    island_separation,
    options,
    register_distance,
    silhouette,
)
from src.analysis.search_space.projections import align, classical_mds
from src.analysis.search_space.unitary import circuit_unitary
from src.utils.genome_archive import GenomeArchive

#: One-qubit gate matrices, for building expected unitaries by hand.
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_I = np.eye(2)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def _ry(theta: float) -> np.ndarray:
    """Returns the RY rotation matrix.

    Args:
        theta: The rotation angle.

    Returns:
        The 2x2 matrix.
    """

    return np.array(
        [
            [math.cos(theta / 2), -math.sin(theta / 2)],
            [math.sin(theta / 2), math.cos(theta / 2)],
        ]
    )


def _gate(
    innovation: int,
    method: str,
    qubits: list[int],
    parameters: dict[str, float] | None = None,
    target: str = "pennylane",
    enabled: bool = True,
) -> dict[str, Any]:
    """Builds a serialized gate on ``input`` qubits.

    Args:
        innovation: The gate's innovation number, which also orders it (its depth
            is ``innovation / 10``).
        method: The gate method.
        qubits: The indexes of the ``input`` qubits it acts on.
        parameters: Its parameters.
        target: ``"pennylane"`` or ``"qiskit"``.
        enabled: Whether it is enabled.

    Returns:
        The serialized gate.
    """

    return {
        "innovation_number": innovation,
        "method_name": method,
        "qubits": [["input", index] for index in qubits],
        "parameters": parameters or {},
        "depth": innovation / 10,
        "enabled": enabled,
        "target": target,
    }


def _genome(
    gates: list[dict[str, Any]], n_qubits: int = 2, target: str = "pennylane"
) -> dict[str, Any]:
    """Builds a serialized genome on ``n_qubits`` input qubits.

    Args:
        gates: Its serialized gates.
        n_qubits: How many input qubits it has (the first is also its output).
        target: ``"pennylane"`` or ``"qiskit"``.

    Returns:
        The serialized genome.
    """

    return {
        "target": target,
        "input_qubits": [["input", index] for index in range(n_qubits)],
        "output_qubits": [["input", 0]],
        "gates": gates,
    }


class _StoredGenome:
    """A genome the archive can store: a serialized dict plus its metadata."""

    def __init__(
        self,
        genome_number: int,
        loss: float,
        parents: list[int],
        gates: list[dict[str, Any]],
        insert_type: str = "inserted",
    ) -> None:
        """Creates the genome.

        Args:
            genome_number: Its number.
            loss: Its ``fitness["loss"]``.
            parents: Its parent genome numbers.
            gates: Its serialized gates.
            insert_type: How it was inserted.
        """

        self.serialized = {
            **_genome(gates),
            "genome_number": genome_number,
            "task": "classification",
            "task_target": "iris",
            "fitness": {"loss": loss, "target_metric": 1 - loss},
            "hyperparameters": {},
            "metadata": {
                "parent_genomes": parents,
                "generated_by": ["add_gate"],
                "insert_type": insert_type,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializes the genome.

        Returns:
            The serialized genome.
        """

        return self.serialized


def _bell_gates(target: str = "pennylane") -> list[dict[str, Any]]:
    """Gates of H on qubit 0, CNOT 0 -> 1, then RY(0.3) on qubit 1.

    Args:
        target: ``"pennylane"`` or ``"qiskit"``.

    Returns:
        The serialized gates.
    """

    return [
        _gate(1, "h", [0], target=target),
        _gate(2, "cx", [0, 1], target=target),
        _gate(3, "ry", [1], {"theta": 0.3}, target=target),
    ]


def test_unitary_is_the_product_of_the_enabled_gates_on_both_targets() -> None:
    """Both targets give the hand-built product, with qubit 0 most significant."""

    expected = np.kron(_I, _ry(0.3)) @ _CNOT @ np.kron(_H, _I)
    for target in ("pennylane", "qiskit"):
        unitary = circuit_unitary(_genome(_bell_gates(target), target=target))
        np.testing.assert_allclose(unitary, expected, atol=1e-10)

    # a disabled gate is left out, and gates are applied by depth, not list order
    gates = _bell_gates()
    gates[2]["enabled"] = False
    np.testing.assert_allclose(
        circuit_unitary(_genome(list(reversed(gates)))),
        _CNOT @ np.kron(_H, _I),
        atol=1e-10,
    )


def test_unitary_is_padded_with_the_identity_onto_more_qubits() -> None:
    """Qubits a genome does not use are acted on by the identity."""

    genome = _genome([_gate(1, "h", [0])], n_qubits=1)
    qubits = [("input", 0), ("input", 1)]
    np.testing.assert_allclose(
        circuit_unitary(genome, qubits), np.kron(_H, _I), atol=1e-10
    )
    np.testing.assert_allclose(circuit_unitary(_genome([], n_qubits=2)), np.eye(4))

    with pytest.raises(ValueError, match="not among the qubits"):
        circuit_unitary(_genome([_gate(1, "cx", [0, 1])]), [("input", 0)])


def test_fubini_study_ignores_global_phase() -> None:
    """Equal unitaries up to phase are at 0, orthogonal ones at pi / 2."""

    metric = get_distance("fubini_study")
    base = np.kron(_H, _I)
    features = [
        base.reshape(-1),
        (np.exp(1j * 0.7) * base).reshape(-1),
        (
            np.kron(np.array([[0, 1], [1, 0]]), _I) @ np.kron(np.diag([1, -1]), _I)
        ).reshape(-1),
        np.eye(4).reshape(-1),
    ]
    distances = metric.pairwise(features)
    assert distances.shape == (4, 4)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0)
    assert distances[0, 1] == pytest.approx(0, abs=1e-6)
    # X Z has zero trace, so it is as far from the identity as a unitary can be
    assert distances[2, 3] == pytest.approx(math.pi / 2)

    context = metric.context([_genome([], n_qubits=1), _genome([], n_qubits=2)])
    assert context == (("input", 0), ("input", 1))


def test_jaccard_distance_counts_enabled_innovations() -> None:
    """Jaccard compares enabled innovation numbers only."""

    metric = get_distance("jaccard")
    genomes = [
        _genome([_gate(1, "h", [0]), _gate(2, "h", [1])]),
        _genome(
            [_gate(1, "h", [0]), _gate(3, "h", [1]), _gate(4, "h", [1], enabled=False)]
        ),
        _genome([]),
    ]
    features = [metric.featurize(genome, None) for genome in genomes]
    assert features[1] == frozenset({1, 3})
    distances = metric.pairwise(features)
    assert distances[0, 1] == pytest.approx(1 - 1 / 3)
    assert distances[0, 2] == pytest.approx(1)
    assert distances[2, 2] == 0


def test_readout_distance_ignores_what_is_not_measured() -> None:
    """Phases after the circuit and gates on unmeasured qubits do not count."""

    metric = get_distance("readout")
    genomes = [
        _genome([_gate(1, "h", [0])]),
        # a Z after the Hadamard only changes phases the probabilities never see
        _genome([_gate(1, "h", [0]), _gate(2, "z", [0])]),
        # qubit 1 is never measured
        _genome([_gate(1, "h", [0]), _gate(3, "ry", [1], {"theta": 1.0})]),
        # an X swaps which outcome every input lands on
        _genome([_gate(1, "h", [0]), _gate(4, "x", [0])]),
    ]
    context = metric.context(genomes)
    assert context == ((("input", 0), ("input", 1)), (("input", 0),), "probs")
    distances = metric.pairwise(
        [metric.featurize(genome, context) for genome in genomes]
    )
    np.testing.assert_allclose(distances[0, 1:3], 0, atol=1e-6)
    assert distances[0, 3] == pytest.approx(1, abs=1e-6)
    assert np.all(distances <= 1 + 1e-6)

    # the full unitary sees all three differences
    unitary = get_distance("fubini_study")
    full = unitary.pairwise(
        [unitary.featurize(genome, unitary.context(genomes)) for genome in genomes]
    )
    assert np.all(full[0, 1:] > 0.1)


def _classifier(genome_number: int, gates: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds a serialized iris classifier: angle-encoded, with no classical weights.

    Args:
        genome_number: Its number.
        gates: Its serialized gates, on ``input`` qubits 0-3.

    Returns:
        The serialized genome.
    """

    return {
        **_genome(gates, n_qubits=4),
        "output_qubits": [["input", 0], ["input", 1]],
        "genome_number": genome_number,
        "task": "classification",
        "task_target": "iris",
        "fitness": {"loss": 1.0},
        "metadata": {},
        "hyperparameters": {
            "quantum_input_mode": "ry",
            "quantum_output_mode": "probs",
        },
        "encoder": {
            "class": "IdentityEncoder",
            "args": {"n_inputs": 4, "n_outputs": 4},
        },
        "decoder": {"class": "ClippedDecoder", "args": {"n_inputs": 4, "n_outputs": 3}},
    }


def test_behaviour_distance_compares_classifications_on_the_run_data() -> None:
    """Models that classify alike are at 0, and the distance is a bounded metric."""

    metric = get_distance("behaviour")
    run_info = {
        "task": "classification",
        "arguments": {"dataset": "iris", "normalization": "minmax", "seed": 0},
    }
    genomes = [
        _classifier(1, []),
        # a Z before measurement changes no probability, so it classifies alike
        _classifier(2, [_gate(1, "z", [0])]),
        _classifier(3, [_gate(2, "x", [0])]),
        _classifier(4, [_gate(3, "cx", [2, 0]), _gate(4, "ry", [1], {"theta": 0.7})]),
    ]
    context = metric.context(genomes, run_info)
    assert context == ("classification", "iris", "minmax", 0)
    features = [metric.featurize(genome, context) for genome in genomes]
    assert features[0].shape == (150, 3)
    np.testing.assert_allclose(features[0].sum(axis=1), 1)

    distances = metric.pairwise(features)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0)
    assert distances[0, 1] == pytest.approx(0, abs=1e-9)
    assert distances[0, 2] > 0.01 and distances[0, 3] > 0.01
    assert np.all(distances <= 1)
    # the square root of the Jensen-Shannon divergence obeys the triangle inequality
    assert distances[0, 2] <= distances[0, 3] + distances[3, 2] + 1e-12

    teacher = metric.context(genomes, {"task": "teacher"})
    with pytest.raises(UnsupportedGenome, match="classification"):
        metric.featurize(genomes[0], teacher)
    with pytest.raises(UnsupportedGenome, match="could not be run"):
        metric.featurize(
            {**genomes[0], "encoder": {"class": "Nope", "args": {}}}, context
        )


def test_classical_mds_recovers_euclidean_placements() -> None:
    """Distances between points in the plane come back exactly, with no stress."""

    points = np.random.default_rng(0).normal(size=(12, 2))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    result = get_projection("classical_mds").project(distances, 2)
    placed = np.linalg.norm(
        result.coordinates[:, None] - result.coordinates[None], axis=-1
    )
    np.testing.assert_allclose(placed, distances, atol=1e-8)
    assert result.quality["explained"] == pytest.approx(1)
    assert result.quality["stress"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("name", sorted(PROJECTIONS))
@pytest.mark.parametrize("dimensions", [2, 3])
def test_every_projection_places_every_genome(name: str, dimensions: int) -> None:
    """Every registered projection returns finite coordinates, even for tiny runs.

    Args:
        name: The projection.
        dimensions: How many dimensions to project to.
    """

    points = np.random.default_rng(1).normal(size=(15, 4))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    projection = get_projection(name)
    result = projection.project(distances, dimensions, seed=0)
    assert result.coordinates.shape == (15, dimensions)
    assert np.isfinite(result.coordinates).all()
    assert result.quality["stress"] >= 0

    for count in (0, 1, 2):
        tiny = projection.project(distances[:count, :count], dimensions)
        assert tiny.coordinates.shape == (count, dimensions)

    with pytest.raises(ValueError):
        projection.project(distances, 4)


def test_align_undoes_a_rotation_and_shift() -> None:
    """A rotated, reflected and shifted placement is mapped back onto the original."""

    reference = np.random.default_rng(2).normal(size=(10, 2))
    angle = 1.1
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    moved = (reference @ rotation) * [1, -1] + [3.0, -2.0]
    transform = align(moved, reference)
    np.testing.assert_allclose(
        moved @ transform[:-1] + transform[-1], reference, atol=1e-10
    )

    # too few shared points to fix a rotation leaves the placement alone
    np.testing.assert_allclose(
        align(moved[:2], reference[:2]), np.vstack([np.eye(2), np.zeros((1, 2))])
    )
    assert classical_mds(np.zeros((1, 1)), 2).coordinates.shape == (1, 2)


def test_best_path_follows_insertion_order() -> None:
    """The path keeps each genome that beat every one inserted before it."""

    genome_numbers = [1, 2, 3, 4, 5, 6]
    insertions = [1, 3, 2, 4, None, 5]
    values = [0.5, 0.3, 0.4, 0.35, 0.1, float("nan")]
    assert best_path(genome_numbers, insertions, values, higher=False) == [1, 3, 2]
    assert best_path(genome_numbers, insertions, values, higher=True) == [1]
    assert best_path([], [], [], higher=False) == []


def test_silhouette_scores_how_far_apart_groups_are() -> None:
    """Separate groups score near 1, mixed ones near 0, and undefined cases None."""

    points = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
    distances = np.abs(points[:, None] - points[None, :])
    assert silhouette(distances, [0, 0, 0, 1, 1, 1]) > 0.9
    assert silhouette(distances, [0, 1, 0, 1, 0, 1]) < 0.1
    assert silhouette(distances, [0] * 6) is None
    assert silhouette(distances[:2, :2], [0, 1]) is None


def test_island_separation_replays_the_population() -> None:
    """Separation is measured over the living population, and extinctions are found."""

    # islands 0 and 1 sit apart; genome 5 (island 1) later lands among island 0's
    positions = {1: 0.0, 2: 0.1, 3: 10.0, 4: 10.1, 5: 0.05, 6: 10.2}
    numbers = sorted(positions)
    values = np.array([positions[number] for number in numbers])
    distances = np.abs(values[:, None] - values[None, :])
    islands = [0, 0, 1, 1, 1, 1]
    insertions = [1, 2, 3, 4, 5, 6]
    events = [
        (1, [1], []),
        (2, [2], []),
        (3, [3], []),
        (4, [4], []),
        # island 1 goes extinct and is repopulated with 5 and 6
        (5, [5], [3, 4]),
        (6, [6], []),
    ]
    separation = island_separation(
        distances, numbers, islands, insertions, events, checkpoints=6
    )
    assert separation["basis"] == "population"
    assert separation["insertion"] == [1, 2, 3, 4, 5, 6]
    assert separation["genomes"] == [1, 2, 3, 4, 3, 4]
    # one genome, then one island only: undefined; a lone genome on island 1 scores 0
    assert separation["silhouette"][:2] == [None, None]
    assert 0 < separation["silhouette"][2] < separation["silhouette"][3]
    assert separation["silhouette"][3] > 0.9
    # after the extinction island 1's genome 5 sits among island 0's
    assert separation["silhouette"][4] < 0
    assert separation["extinctions"] == [{"insertion": 5, "island": 1}]

    assert island_separation(distances, numbers, [None] * 6, insertions, events) is None


def _build_archive(directory: Path, genomes: list[_StoredGenome]) -> str:
    """Writes an archive holding the genomes, in insertion order.

    Args:
        directory: The run directory.
        genomes: The genomes to store.

    Returns:
        The archive's path.
    """

    with GenomeArchive.create(str(directory)) as archive:
        for insertion, genome in enumerate(genomes, start=1):
            archive.add_genome(genome, insertion=insertion, island=insertion % 2)
        return archive.path


def test_building_a_search_space_from_an_archive(tmp_path: Path, monkeypatch) -> None:
    """A run becomes a projection with links and the global-best path, reusing work.

    Args:
        tmp_path: pytest per-test temporary directory.
        monkeypatch: Used to count how many genomes are featurized.
    """

    genomes = [
        _StoredGenome(1, 0.9, [0], [_gate(1, "h", [0])]),
        _StoredGenome(2, 0.5, [1], [_gate(1, "h", [0]), _gate(2, "cx", [0, 1])]),
        _StoredGenome(3, 0.7, [1], _bell_gates()),
        # the same circuit as genome 2, so it lands on the same point
        _StoredGenome(4, 0.4, [2, 3], [_gate(1, "h", [0]), _gate(2, "cx", [0, 1])]),
    ]
    archive_path = _build_archive(tmp_path / "run", genomes[:3])

    featurized: list[int] = []
    metric_class = DISTANCE_METRICS["fubini_study"]
    original = metric_class.featurize

    def counting(
        self: DistanceMetric, genome: dict[str, Any], context: Hashable
    ) -> Any:
        """Records which genome is featurized, then featurizes it."""
        featurized.append(genome["genome_number"])
        return original(self, genome, context)

    monkeypatch.setattr(metric_class, "featurize", counting)

    builder = SearchSpaceBuilder()
    payload = builder.build(archive_path)
    assert payload["metric"] == "fubini_study"
    assert payload["projection"] == "classical_mds"
    assert payload["genome_number"] == [1, 2, 3]
    assert len(payload["coordinates"]) == 2
    assert all(len(axis) == 3 for axis in payload["coordinates"])
    assert payload["fitness"] == [0.9, 0.5, 0.7]
    assert payload["insertion"] == [1, 2, 3]
    assert payload["island"] == [1, 0, 1]
    assert payload["best_path"] == [1, 2]
    assert payload["higher_is_better"] is False
    # the seed genome (0) is not stored, so its link is left out
    assert list(zip(payload["links"]["child"], payload["links"]["parent"])) == [
        (2, 1),
        (3, 1),
    ]
    assert payload["skipped"] == []
    assert payload["context"] == {"qubits": [["input", 0], ["input", 1]]}
    # no population changes were recorded, so separation is measured over every
    # genome inserted so far, on islands 1, 0, 1 (the archive's insertion % 2)
    separation = payload["island_separation"]
    assert separation["basis"] == "inserted"
    assert separation["insertion"] == [1, 2, 3]
    assert separation["genomes"] == [1, 2, 3]
    assert separation["silhouette"][:2] == [None, None]
    assert separation["extinctions"] == []
    assert featurized == [1, 2, 3]

    with GenomeArchive.create(str(tmp_path / "run")) as archive:
        archive.add_genome(genomes[3], insertion=4, island=0)
    payload = builder.build(archive_path)
    assert featurized == [1, 2, 3, 4]
    assert payload["best_path"] == [1, 2, 4]
    coordinates = np.array(payload["coordinates"]).T
    np.testing.assert_allclose(coordinates[1], coordinates[3], atol=1e-6)

    by_target = builder.build(
        archive_path, metric="jaccard", fitness_key="target_metric"
    )
    assert by_target["best_path"] == [1, 2, 4]
    assert by_target["higher_is_better"] is True
    assert by_target["context"] == {}

    with pytest.raises(ValueError, match="Unknown distance metric"):
        builder.build(archive_path, metric="nope")
    with pytest.raises(ValueError, match="Unknown projection"):
        builder.build(archive_path, projection="nope")
    with pytest.raises(ValueError, match="2 or 3"):
        builder.build(archive_path, dimensions=4)


def test_genomes_a_metric_cannot_place_are_reported(tmp_path: Path) -> None:
    """A genome whose circuit cannot be simulated is skipped with the reason.

    Args:
        tmp_path: pytest per-test temporary directory.
    """

    broken = _StoredGenome(2, 0.5, [1], [_gate(1, "h", [0])])
    broken.serialized["gates"][0]["method_name"] = "not_a_gate"
    broken.serialized["target"] = "not_a_target"
    archive_path = _build_archive(
        tmp_path / "run", [_StoredGenome(1, 0.9, [0], [_gate(1, "h", [0])]), broken]
    )
    payload = SearchSpaceBuilder().build(archive_path)
    assert payload["genome_number"] == [1]
    assert [entry["genome_number"] for entry in payload["skipped"]] == [2]


def test_new_metrics_can_be_registered() -> None:
    """A registered metric is offered and used like the built-in ones."""

    @register_distance
    class GateCountDistance(DistanceMetric):
        """Distance by how many gates two genomes have."""

        name = "test_gate_count"
        label = "Gate count"
        description = "Difference in gate counts."

        def featurize(self, genome: dict[str, Any], context: Hashable) -> int:
            """Counts a genome's gates.

            Args:
                genome: A serialized genome.
                context: Unused.

            Returns:
                The number of gates.
            """

            return len(genome["gates"])

        def pairwise(self, features: list[int]) -> np.ndarray:
            """Measures gate-count differences.

            Args:
                features: Each genome's gate count.

            Returns:
                The absolute differences.
            """

            counts = np.array(features, dtype=float)
            return np.abs(counts[:, None] - counts[None, :])

    try:
        names = [entry["name"] for entry in options()["metrics"]]
        assert names[:4] == ["behaviour", "readout", "fubini_study", "jaccard"]
        assert "test_gate_count" in names
        metric = get_distance("test_gate_count")
        assert metric.pairwise([1, 3])[0, 1] == 2
        with pytest.raises(ValueError, match="already registered"):
            register_distance(type("Other", (GateCountDistance,), {}))
    finally:
        DISTANCE_METRICS.pop("test_gate_count", None)
