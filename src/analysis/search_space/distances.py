"""Distances between genomes, for projecting a run's search space.

A :class:`DistanceMetric` turns each serialized genome into a *feature* (a set of
gate innovations, a unitary, a vector of outputs...) and then measures every pair
of features at once. Splitting the two steps is what makes the search-space view
cheap to keep current on a live run: features depend on one genome only, so they
are cached and only new genomes are featurized, while the pairwise step is a
single vectorized pass.

A metric may also depend on the run as a whole -- the unitary metric brings every
genome to the same set of qubits -- which it declares through :meth:`DistanceMetric
.context`: a hashable summary of the genomes, and cached features are reused only
while it is unchanged.

New metrics are added by subclassing :class:`DistanceMetric` and decorating the
class with :func:`register_distance`; the dashboard lists every registered metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any, ClassVar

import numpy as np

from src.analysis.search_space.unitary import (
    MAX_UNITARY_QUBITS,
    circuit_unitary,
    genome_qubits,
)

#: Every registered distance metric, by name.
DISTANCE_METRICS: dict[str, type[DistanceMetric]] = {}

#: The most complex values the unitary metric stacks in double precision (512 MiB);
#: larger runs are compared in single precision, halving the memory.
_DOUBLE_PRECISION_LIMIT = 2**25


class UnsupportedGenome(ValueError):
    """Raised when a metric cannot place a genome, such as a circuit too large to simulate."""


class DistanceMetric(ABC):
    """Measures how far apart genomes are.

    Attributes:
        name: The metric's identifier, as the API and dashboard name it.
        label: A short human-readable name.
        description: One or two sentences on what the distance measures.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str]

    def context(self, genomes: list[dict[str, Any]]) -> Hashable:
        """Summarizes what the metric needs to know about the run as a whole.

        Features computed under one context are reused only while the context
        stays the same, so a metric whose features depend on more than their own
        genome must return everything they depend on here.

        Args:
            genomes: Every serialized genome being placed.

        Returns:
            A hashable summary; ``None`` (the default) when features depend on
            their own genome only.
        """

        return None

    @abstractmethod
    def featurize(self, genome: dict[str, Any], context: Hashable) -> Any:
        """Computes the feature one genome is compared by.

        Args:
            genome: A serialized genome (``CircuitGenome.to_dict``).
            context: What :meth:`context` returned for the run.

        Returns:
            The genome's feature, in whatever form :meth:`pairwise` takes.

        Raises:
            UnsupportedGenome: If the genome cannot be placed by this metric.
        """

    @abstractmethod
    def pairwise(self, features: list[Any]) -> np.ndarray:
        """Measures the distance between every pair of features.

        Args:
            features: One feature per genome, from :meth:`featurize`.

        Returns:
            The symmetric ``n x n`` distance matrix, with a zero diagonal.
        """

    @classmethod
    def describe(cls) -> dict[str, str]:
        """Describes the metric for the dashboard's picker.

        Returns:
            Its ``name``, ``label`` and ``description``.
        """

        return {"name": cls.name, "label": cls.label, "description": cls.description}


def register_distance(metric: type[DistanceMetric]) -> type[DistanceMetric]:
    """Registers a distance metric under its name (usable as a class decorator).

    Args:
        metric: The metric class.

    Returns:
        The same class, so this can decorate it.

    Raises:
        ValueError: If another metric is already registered under the name.
    """

    if metric.name in DISTANCE_METRICS and DISTANCE_METRICS[metric.name] is not metric:
        raise ValueError(
            f"A distance metric named {metric.name!r} is already registered."
        )
    DISTANCE_METRICS[metric.name] = metric
    return metric


def get_distance(name: str) -> DistanceMetric:
    """Creates a registered distance metric.

    Args:
        name: The metric's name.

    Returns:
        A new instance of the metric.

    Raises:
        ValueError: If no metric is registered under the name.
    """

    if name not in DISTANCE_METRICS:
        raise ValueError(
            f"Unknown distance metric {name!r}; choose from {', '.join(sorted(DISTANCE_METRICS))}."
        )
    return DISTANCE_METRICS[name]()


@register_distance
class FubiniStudyDistance(DistanceMetric):
    """The phase-invariant angle between two circuits' unitaries.

    ``theta(U, V) = arccos(|Tr(U^dagger V)| / d)``, with ``d = 2**n``: zero for
    circuits implementing the same operation up to a global phase, and at most
    ``pi / 2``. It is a true metric (the geodesic distance between unitaries once
    the global phase is factored out) and maps one to one onto the average gate
    fidelity, ``F = (d cos^2(theta) + 1) / (d + 1)``.

    Every genome is brought to the union of the run's qubits, with the identity
    on the qubits it does not use, so genomes on different qubit sets are
    comparable. It measures the whole operation: differences on qubits that are
    never measured, or that the input encoding never reaches, still count.
    """

    name = "fubini_study"
    label = "Unitary (Fubini–Study)"
    description = (
        "Angle between the circuits' unitaries, ignoring global phase: 0 when two "
        "circuits implement the same operation, at most π/2."
    )

    def context(self, genomes: list[dict[str, Any]]) -> Hashable:
        """Collects the qubits every genome is brought to.

        Args:
            genomes: Every serialized genome being placed.

        Returns:
            The sorted union of the genomes' qubits, as a tuple.
        """

        qubits: set[tuple[str, int]] = set()
        for genome in genomes:
            qubits.update(genome_qubits(genome))
        return tuple(sorted(qubits))

    def featurize(self, genome: dict[str, Any], context: Hashable) -> np.ndarray:
        """Computes a genome's unitary over the run's qubits.

        Args:
            genome: A serialized genome.
            context: The run's qubits, from :meth:`context`.

        Returns:
            The unitary, flattened.

        Raises:
            UnsupportedGenome: If the circuit is too large or cannot be simulated.
        """

        qubits = list(context) if context else None
        if qubits is not None and len(qubits) > MAX_UNITARY_QUBITS:
            raise UnsupportedGenome(
                f"the run's circuits span {len(qubits)} qubits; unitaries are only "
                f"computed for up to {MAX_UNITARY_QUBITS}"
            )
        try:
            return circuit_unitary(genome, qubits).reshape(-1)
        except (KeyError, ValueError, TypeError) as error:
            raise UnsupportedGenome(str(error)) from error

    def pairwise(self, features: list[np.ndarray]) -> np.ndarray:
        """Measures the angle between every pair of unitaries.

        ``|Tr(U^dagger V)|`` is the magnitude of the inner product of the
        flattened matrices, so all pairs come from one matrix product.

        Args:
            features: Each genome's flattened unitary, all the same size.

        Returns:
            The angles, in radians.
        """

        if not features:
            return np.zeros((0, 0))
        size = features[0].size
        dimension = int(round(np.sqrt(size)))
        dtype = (
            np.complex128
            if len(features) * size <= _DOUBLE_PRECISION_LIMIT
            else np.complex64
        )
        stacked = np.stack(features).astype(dtype, copy=False)
        overlaps = np.abs(stacked.conj() @ stacked.T) / dimension
        distances = np.arccos(np.clip(overlaps, 0.0, 1.0)).astype(float)
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0.0)
        return distances


@register_distance
class JaccardDistance(DistanceMetric):
    """The share of two genomes' enabled gates they do not have in common.

    ``1 - |A & B| / |A | B|`` over the innovation numbers of each genome's enabled
    gates: the structural (genotype) distance of NEAT-style searches. Innovation
    numbers are shared within a run, so this compares genomes of the same run
    only. Gate parameters are ignored.
    """

    name = "jaccard"
    label = "Structure (Jaccard)"
    description = (
        "Share of the two genomes' enabled gates (by innovation number) that only "
        "one of them has: 0 for the same gates, 1 for none in common."
    )

    def featurize(self, genome: dict[str, Any], context: Hashable) -> frozenset[int]:
        """Collects a genome's enabled gate innovation numbers.

        Args:
            genome: A serialized genome.
            context: Unused.

        Returns:
            The innovation numbers.
        """

        return frozenset(
            int(gate["innovation_number"])
            for gate in genome.get("gates") or []
            if gate.get("enabled", True) and gate.get("innovation_number") is not None
        )

    def pairwise(self, features: list[frozenset[int]]) -> np.ndarray:
        """Measures the Jaccard distance between every pair of gate sets.

        Args:
            features: Each genome's enabled innovation numbers.

        Returns:
            The distances, in ``[0, 1]``; two empty circuits are at distance 0.
        """

        count = len(features)
        innovations = sorted(set().union(*features)) if features else []
        column = {innovation: i for i, innovation in enumerate(innovations)}
        membership = np.zeros((count, len(innovations)), dtype=np.float64)
        for row, feature in enumerate(features):
            membership[row, [column[innovation] for innovation in feature]] = 1.0

        shared = membership @ membership.T
        sizes = membership.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - shared
        with np.errstate(invalid="ignore", divide="ignore"):
            distances = np.where(union > 0, 1.0 - shared / union, 0.0)
        np.fill_diagonal(distances, 0.0)
        return distances
