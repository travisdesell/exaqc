"""Projections of a genome distance matrix into two or three dimensions.

A :class:`Projection` places every genome so that the distances between points
approximate the distances between genomes. Methods differ in what they keep:
classical and metric MDS keep the distances themselves (so how far the search
moved can be read off the plot), while t-SNE keeps neighborhoods (so clusters
stand out, but the gaps between them mean little).

New projections are added by subclassing :class:`Projection` and decorating the
class with :func:`register_projection`; the dashboard lists every registered one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

#: Every registered projection, by name.
PROJECTIONS: dict[str, type[Projection]] = {}


@dataclass
class ProjectionResult:
    """Where a projection placed the genomes.

    Attributes:
        coordinates: One row per genome, one column per dimension.
        quality: How faithful the placement is, as the method measures it (for
            example the share of variance classical MDS kept, or a stress).
    """

    coordinates: np.ndarray
    quality: dict[str, float] = field(default_factory=dict)


class Projection(ABC):
    """Places genomes in a low-dimensional space from their pairwise distances.

    Attributes:
        name: The projection's identifier, as the API and dashboard name it.
        label: A short human-readable name.
        description: One or two sentences on what the projection preserves.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str]

    def project(
        self, distances: np.ndarray, dimensions: int = 2, seed: int = 0
    ) -> ProjectionResult:
        """Places genomes from their pairwise distances.

        Runs too small to project (fewer genomes than ``dimensions + 1``) are
        placed by classical MDS padded with zeros, whatever the method.

        Args:
            distances: The symmetric ``n x n`` distance matrix.
            dimensions: How many dimensions to place them in (2 or 3).
            seed: Seeds any randomness in the method.

        Returns:
            The coordinates, and the method's own quality measures plus
            ``stress`` (see :func:`kruskal_stress`), which every method reports
            so placements can be compared.

        Raises:
            ValueError: If ``dimensions`` is not 2 or 3, or ``distances`` is not
                square.
        """

        if dimensions not in (2, 3):
            raise ValueError(
                f"Can only project to 2 or 3 dimensions, not {dimensions}."
            )
        distances = np.asarray(distances, dtype=float)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError(
                f"Expected a square distance matrix, got shape {distances.shape}."
            )
        if distances.shape[0] <= dimensions + 1:
            result = classical_mds(distances, dimensions)
        else:
            result = self._project(distances, dimensions, seed)
        result.quality["stress"] = kruskal_stress(distances, result.coordinates)
        return result

    @abstractmethod
    def _project(
        self, distances: np.ndarray, dimensions: int, seed: int
    ) -> ProjectionResult:
        """Places genomes from their pairwise distances (enough of them to project).

        Args:
            distances: The symmetric ``n x n`` distance matrix, ``n > dimensions + 1``.
            dimensions: How many dimensions to place them in.
            seed: Seeds any randomness in the method.

        Returns:
            The coordinates and the method's quality measures.
        """

    @classmethod
    def describe(cls) -> dict[str, str]:
        """Describes the projection for the dashboard's picker.

        Returns:
            Its ``name``, ``label`` and ``description``.
        """

        return {"name": cls.name, "label": cls.label, "description": cls.description}


def register_projection(projection: type[Projection]) -> type[Projection]:
    """Registers a projection under its name (usable as a class decorator).

    Args:
        projection: The projection class.

    Returns:
        The same class, so this can decorate it.

    Raises:
        ValueError: If another projection is already registered under the name.
    """

    if (
        projection.name in PROJECTIONS
        and PROJECTIONS[projection.name] is not projection
    ):
        raise ValueError(
            f"A projection named {projection.name!r} is already registered."
        )
    PROJECTIONS[projection.name] = projection
    return projection


def get_projection(name: str) -> Projection:
    """Creates a registered projection.

    Args:
        name: The projection's name.

    Returns:
        A new instance of the projection.

    Raises:
        ValueError: If no projection is registered under the name.
    """

    if name not in PROJECTIONS:
        raise ValueError(
            f"Unknown projection {name!r}; choose from {', '.join(sorted(PROJECTIONS))}."
        )
    return PROJECTIONS[name]()


def kruskal_stress(distances: np.ndarray, coordinates: np.ndarray) -> float:
    """Measures how badly a placement keeps the distances (Kruskal's stress-1).

    ``sqrt(sum((d - d_hat)^2) / sum(d^2))`` over every pair, where ``d_hat`` is
    the pair's distance in the placement: 0 for a placement that keeps every
    distance exactly, and scale-free, so it compares methods with each other.
    Distances are compared as they are, so methods that rescale the space (such
    as t-SNE) score worse by design.

    Args:
        distances: The symmetric ``n x n`` distance matrix.
        coordinates: The placement, one row per genome.

    Returns:
        The stress; 0 when there are no distances to keep.
    """

    from scipy.spatial.distance import pdist, squareform

    if distances.shape[0] < 2:
        return 0.0
    kept = squareform(distances, checks=False)
    placed = pdist(coordinates)
    total = float((kept**2).sum())
    if total <= 0:
        return 0.0
    return float(np.sqrt(((kept - placed) ** 2).sum() / total))


def classical_mds(distances: np.ndarray, dimensions: int) -> ProjectionResult:
    """Places points by classical (Torgerson) multidimensional scaling.

    Double-centers the squared distances and keeps the leading eigenvectors,
    which is deterministic and exact whenever the distances are Euclidean. Each
    axis's sign is fixed so its largest-magnitude coordinate is positive, so the
    same distances always give the same picture.

    Args:
        distances: The symmetric ``n x n`` distance matrix.
        dimensions: How many dimensions to keep; axes beyond the positive
            eigenvalues (and beyond ``n``) are zero.

    Returns:
        The coordinates, and ``explained``: the share of the positive eigenvalue
        mass the kept axes hold (1.0 when nothing is lost).
    """

    count = distances.shape[0]
    coordinates = np.zeros((count, dimensions))
    if count < 2:
        return ProjectionResult(coordinates, {"explained": 1.0})

    squared = distances**2
    centering = np.eye(count) - np.full((count, count), 1.0 / count)
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    kept = min(dimensions, count)
    for axis in range(kept):
        if eigenvalues[axis] <= 0:
            break
        values = eigenvectors[:, axis] * np.sqrt(eigenvalues[axis])
        if values[np.argmax(np.abs(values))] < 0:
            values = -values
        coordinates[:, axis] = values

    positive = eigenvalues[eigenvalues > 0]
    total = float(positive.sum())
    explained = (
        float(np.clip(eigenvalues[:kept], 0, None).sum() / total) if total > 0 else 1.0
    )
    return ProjectionResult(coordinates, {"explained": explained})


def align(
    coordinates: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Rotates, reflects and shifts a placement onto an earlier one.

    Projections are only defined up to rotation and reflection, so a run
    re-projected as genomes arrive could otherwise spin or flip between
    refreshes. Orthogonal Procrustes finds the rotation (or reflection) and
    shift that best match the genomes both placements hold; the scale is kept,
    so distances are unchanged.

    Args:
        coordinates: The new placement of the shared genomes (``m x k``).
        reference: Their earlier placement, row for row (``m x k``).

    Returns:
        The transform, as a ``(k + 1) x k`` matrix ``T`` whose first ``k`` rows
        are the rotation and last row the shift: apply it to any placement ``X``
        of the new projection as ``X @ T[:-1] + T[-1]``. The identity when fewer
        than ``k + 1`` genomes are shared, too few to fix a rotation.
    """

    dimensions = coordinates.shape[1]
    identity = np.vstack([np.eye(dimensions), np.zeros((1, dimensions))])
    if coordinates.shape[0] <= dimensions:
        return identity
    new_center = coordinates.mean(axis=0)
    reference_center = reference.mean(axis=0)
    left, _, right = np.linalg.svd(
        (coordinates - new_center).T @ (reference - reference_center)
    )
    rotation = left @ right
    shift = reference_center - new_center @ rotation
    return np.vstack([rotation, shift])


@register_projection
class ClassicalMDS(Projection):
    """Classical (Torgerson) MDS: exact for Euclidean distances, and deterministic."""

    name = "classical_mds"
    label = "Classical MDS"
    description = (
        "Keeps the distances as well as a linear projection can; deterministic, so "
        "the layout only changes as genomes are added."
    )

    def _project(
        self, distances: np.ndarray, dimensions: int, seed: int
    ) -> ProjectionResult:
        """Places genomes by classical MDS.

        Args:
            distances: The distance matrix.
            dimensions: How many dimensions to place them in.
            seed: Unused; classical MDS is deterministic.

        Returns:
            The coordinates and the share of variance kept (``explained``).
        """

        return classical_mds(distances, dimensions)


@register_projection
class MetricMDS(Projection):
    """Metric MDS (SMACOF), started from classical MDS and minimizing stress."""

    name = "metric_mds"
    label = "Metric MDS"
    description = (
        "Moves points to match the distances as closely as possible (minimizing "
        "stress), starting from classical MDS. Slower; better for non-Euclidean distances."
    )

    def _project(
        self, distances: np.ndarray, dimensions: int, seed: int
    ) -> ProjectionResult:
        """Places genomes by SMACOF, from the classical MDS placement.

        Args:
            distances: The distance matrix.
            dimensions: How many dimensions to place them in.
            seed: Seeds scikit-learn's solver.

        Returns:
            The coordinates (their ``stress`` is added by :meth:`Projection.project`).
        """

        import inspect

        from sklearn.manifold import MDS

        start = classical_mds(distances, dimensions).coordinates
        if "metric_mds" in inspect.signature(MDS).parameters:
            # scikit-learn 1.8+: ``metric`` names the input's metric and
            # ``metric_mds`` chooses metric scaling
            model = MDS(
                n_components=dimensions,
                metric="precomputed",
                metric_mds=True,
                init="classical_mds",
                n_init=1,
                random_state=seed,
            )
        else:
            model = MDS(
                n_components=dimensions,
                dissimilarity="precomputed",
                metric=True,
                n_init=1,
                random_state=seed,
                normalized_stress=False,
            )
        coordinates = model.fit_transform(distances, init=start)
        return ProjectionResult(np.asarray(coordinates))


@register_projection
class TSNEProjection(Projection):
    """t-SNE on the precomputed distances: keeps neighborhoods, not distances."""

    name = "tsne"
    label = "t-SNE"
    description = (
        "Keeps each genome's nearest neighbors together, so clusters stand out; "
        "distances between clusters are not meaningful."
    )

    def _project(
        self, distances: np.ndarray, dimensions: int, seed: int
    ) -> ProjectionResult:
        """Places genomes by t-SNE.

        Args:
            distances: The distance matrix.
            dimensions: How many dimensions to place them in.
            seed: Seeds the random initial placement.

        Returns:
            The coordinates and the final ``kl_divergence``.
        """

        from sklearn.manifold import TSNE

        count = distances.shape[0]
        model = TSNE(
            n_components=dimensions,
            metric="precomputed",
            init="random",
            perplexity=float(max(1.0, min(30.0, (count - 1) / 3))),
            random_state=seed,
        )
        coordinates = model.fit_transform(distances)
        return ProjectionResult(
            np.asarray(coordinates), {"kl_divergence": float(model.kl_divergence_)}
        )
