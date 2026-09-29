"""Projects a run's genomes into a low-dimensional search space.

:func:`build_search_space` reads a run's ``genomes.sqlar`` archive (read-only),
measures every pair of genomes with a :mod:`~src.analysis.search_space.distances`
metric, places them with a :mod:`~src.analysis.search_space.projections`
method, and returns plain JSON-safe lists: each genome's coordinates alongside
its fitness, insertion, island and operators, every parent link, and the path
the search's global best took through the space.

A :class:`SearchSpaceBuilder` keeps what it has computed between calls, so a run
still being written only has its new genomes read and featurized, and a
re-projection is rotated onto the previous one rather than spinning or flipping
as genomes arrive.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.analysis.search_space.distances import (
    DISTANCE_METRICS,
    UnsupportedGenome,
    get_distance,
)
from src.analysis.search_space.projections import (
    PROJECTIONS,
    align,
    get_projection,
)
from src.utils.genome_archive import GenomeArchive

#: The metric and projection used when none is asked for.
DEFAULT_METRIC = "fubini_study"
DEFAULT_PROJECTION = "classical_mds"

#: The most runs a builder keeps computed features for; the least recently
#: used is dropped first.
MAX_CACHED_RUNS = 8


def higher_is_better(key: str) -> bool:
    """Decides which direction of a fitness key is an improvement.

    Matches the dashboard: loss-like keys are minimized, everything else
    (accuracies, fidelities, returns) is maximized.

    Args:
        key: A fitness key.

    Returns:
        False for keys containing ``loss``, True otherwise.
    """

    return "loss" not in key.lower()


def best_path(
    genome_numbers: list[int],
    insertions: list[int | None],
    values: list[float | None],
    higher: bool,
) -> list[int]:
    """Traces the global best through a run, in the order genomes were inserted.

    Walks the genomes by insertion (then genome number) and keeps each one that
    improves on every genome inserted before it: the successive global bests,
    ending at the run's best genome.

    Args:
        genome_numbers: The genomes.
        insertions: Each genome's insertion; genomes without one are skipped.
        values: Each genome's fitness value; genomes without a finite value are
            skipped.
        higher: Whether larger values are better.

    Returns:
        The genome numbers that were the global best, in the order they became it.
    """

    order = sorted(
        (
            (insertion, genome_number, value)
            for genome_number, insertion, value in zip(
                genome_numbers, insertions, values
            )
            if insertion is not None
            and isinstance(value, (int, float))
            and math.isfinite(value)
        ),
    )
    path: list[int] = []
    best: float | None = None
    for _, genome_number, value in order:
        if best is None or (value > best if higher else value < best):
            best = value
            path.append(genome_number)
    return path


def options() -> dict[str, list[dict[str, str]]]:
    """Lists the metrics and projections a search space can be built with.

    Returns:
        ``metrics`` and ``projections``, each entry's ``name``, ``label`` and
        ``description``, defaults first.
    """

    def first(names: list[str], default: str) -> list[str]:
        """Moves the default to the front of a list of names."""
        return [default] + [name for name in names if name != default]

    return {
        "metrics": [
            DISTANCE_METRICS[name].describe()
            for name in first(list(DISTANCE_METRICS), DEFAULT_METRIC)
        ],
        "projections": [
            PROJECTIONS[name].describe()
            for name in first(list(PROJECTIONS), DEFAULT_PROJECTION)
        ],
    }


@dataclass
class _FeatureCache:
    """A metric's features for one run.

    Attributes:
        context: The context the features were computed under.
        features: Each placed genome's feature.
        skipped: Why each genome the metric could not place was left out.
    """

    context: Hashable = None
    features: dict[int, Any] = field(default_factory=dict)
    skipped: dict[int, str] = field(default_factory=dict)


@dataclass
class _RunCache:
    """What a builder keeps about one run.

    Attributes:
        genomes: Every serialized genome read so far.
        features: Each metric's features, by metric name.
        layouts: The last placement returned for each ``(metric, projection,
            dimensions)``, by genome number, to align the next one to.
    """

    genomes: dict[int, dict[str, Any]] = field(default_factory=dict)
    features: dict[str, _FeatureCache] = field(default_factory=dict)
    layouts: dict[tuple[str, str, int], dict[int, np.ndarray]] = field(
        default_factory=dict
    )


class SearchSpaceBuilder:
    """Builds search-space projections, reusing work between calls.

    It is safe to share between threads: builds run one at a time.
    """

    def __init__(self, max_runs: int = MAX_CACHED_RUNS) -> None:
        """Creates an empty builder.

        Args:
            max_runs: How many runs to keep computed features for.
        """

        self._runs: OrderedDict[str, _RunCache] = OrderedDict()
        self._max_runs = max_runs
        self._lock = threading.Lock()

    def _run_cache(self, archive_path: str) -> _RunCache:
        """Returns (creating it if needed) the cache for a run.

        Args:
            archive_path: The run's archive.

        Returns:
            The run's cache, marked as the most recently used.
        """

        cache = self._runs.get(archive_path)
        if cache is None:
            cache = self._runs[archive_path] = _RunCache()
            while len(self._runs) > self._max_runs:
                self._runs.popitem(last=False)
        self._runs.move_to_end(archive_path)
        return cache

    def build(
        self,
        archive_path: str,
        metric: str = DEFAULT_METRIC,
        projection: str = DEFAULT_PROJECTION,
        dimensions: int = 2,
        fitness_key: str = "loss",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Projects every genome of a run.

        Args:
            archive_path: The run's ``genomes.sqlar``.
            metric: The distance metric's name (see :func:`options`).
            projection: The projection's name (see :func:`options`).
            dimensions: 2 or 3.
            fitness_key: The fitness key (or summary column) each genome's
                ``fitness`` value, and the global-best path, are taken from.
            seed: Seeds any randomness in the projection.

        Returns:
            ``metric``, ``projection``, ``dimensions`` and ``fitness_key`` (as
            used); ``higher_is_better`` for the key; parallel per-genome lists
            ``genome_number``, ``coordinates`` (one list per dimension),
            ``fitness``, ``insertion``, ``island``, ``insert_type`` and
            ``operator`` (the first generating operator); ``links`` (parallel
            ``child``/``parent`` lists between placed genomes); ``best_path``
            (the successive global bests, see :func:`best_path`); ``skipped``
            (each genome left out, with the reason); ``quality`` (the
            projection's own measures); ``distance`` (the ``min``/``max``/``mean``
            off-diagonal distance); and ``context`` (what the metric compared
            the genomes over, e.g. the qubits).

        Raises:
            ValueError: If the metric, projection, dimensions or fitness key is
                not valid.
        """

        distance = get_distance(metric)
        method = get_projection(projection)
        if dimensions not in (2, 3):
            raise ValueError(
                f"Can only project to 2 or 3 dimensions, not {dimensions}."
            )

        with self._lock:
            cache = self._run_cache(archive_path)
            with GenomeArchive.open_readonly(archive_path) as reader:
                points = reader.points(fitness_key)
                links = reader.parent_links()
                for genome_number in points["genome_number"]:
                    if genome_number not in cache.genomes:
                        cache.genomes[genome_number] = reader.get_genome_dict(
                            genome_number
                        )

            genome_numbers = points["genome_number"]
            genomes = [cache.genomes[number] for number in genome_numbers]
            context = distance.context(genomes)
            features = cache.features.get(metric)
            if features is None or features.context != context:
                features = cache.features[metric] = _FeatureCache(context=context)
            for number, genome in zip(genome_numbers, genomes):
                if number in features.features or number in features.skipped:
                    continue
                try:
                    features.features[number] = distance.featurize(genome, context)
                except UnsupportedGenome as error:
                    features.skipped[number] = str(error)

            placed = [
                i
                for i, number in enumerate(genome_numbers)
                if number in features.features
            ]
            placed_numbers = [genome_numbers[i] for i in placed]
            matrix = distance.pairwise([features.features[n] for n in placed_numbers])
            result = method.project(matrix, dimensions, seed)
            coordinates = self._aligned(
                cache,
                (metric, projection, dimensions),
                placed_numbers,
                result.coordinates,
            )

        higher = higher_is_better(fitness_key)
        values = [points["y"][i] for i in placed]
        insertions = [points["insertion"][i] for i in placed]
        placed_set = set(placed_numbers)
        link_pairs = [
            (child, parent)
            for child, parent in zip(links["child"], links["parent"])
            if child in placed_set and parent in placed_set
        ]
        off_diagonal = matrix[~np.eye(len(placed_numbers), dtype=bool)]

        return {
            "metric": metric,
            "projection": projection,
            "dimensions": dimensions,
            "fitness_key": fitness_key,
            "higher_is_better": higher,
            "genome_number": placed_numbers,
            "coordinates": [
                coordinates[:, axis].tolist() for axis in range(dimensions)
            ],
            "fitness": values,
            "insertion": insertions,
            "island": [points["island"][i] for i in placed],
            "insert_type": [points["insert_type"][i] for i in placed],
            "operator": [points["operator"][i] for i in placed],
            "links": {
                "child": [child for child, _ in link_pairs],
                "parent": [parent for _, parent in link_pairs],
            },
            "best_path": best_path(placed_numbers, insertions, values, higher),
            "skipped": [
                {"genome_number": number, "reason": reason}
                for number, reason in sorted(features.skipped.items())
                if number in cache.genomes
            ],
            "quality": result.quality,
            "distance": {
                "min": float(off_diagonal.min()) if off_diagonal.size else None,
                "max": float(off_diagonal.max()) if off_diagonal.size else None,
                "mean": float(off_diagonal.mean()) if off_diagonal.size else None,
            },
            "context": _describe_context(metric, context),
        }

    @staticmethod
    def _aligned(
        cache: _RunCache,
        key: tuple[str, str, int],
        genome_numbers: list[int],
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """Aligns a placement onto the last one returned for the same view.

        Args:
            cache: The run's cache, whose ``layouts`` are read and updated.
            key: ``(metric, projection, dimensions)``.
            genome_numbers: The placed genomes, row for row.
            coordinates: Their new placement.

        Returns:
            The placement, rotated and shifted onto the previous one where the
            two share enough genomes (unchanged otherwise). The result is stored
            as the view's latest layout.
        """

        previous = cache.layouts.get(key)
        if previous:
            shared = [
                i for i, number in enumerate(genome_numbers) if number in previous
            ]
            if shared:
                transform = align(
                    coordinates[shared],
                    np.stack([previous[genome_numbers[i]] for i in shared]),
                )
                coordinates = coordinates @ transform[:-1] + transform[-1]
        cache.layouts[key] = {
            number: coordinates[i].copy() for i, number in enumerate(genome_numbers)
        }
        return coordinates


def _describe_context(metric: str, context: Hashable) -> dict[str, Any]:
    """Describes what a metric compared genomes over, for display.

    Args:
        metric: The metric's name.
        context: The metric's context.

    Returns:
        ``qubits`` (as ``[name, index]`` pairs) for the unitary metric, and an
        empty dict for metrics without a context.
    """

    if metric == "fubini_study" and context:
        return {"qubits": [list(qubit) for qubit in context]}
    return {}


#: The builder :func:`build_search_space` uses, shared by every call in a process.
_BUILDER = SearchSpaceBuilder()


def build_search_space(
    archive_path: str,
    metric: str = DEFAULT_METRIC,
    projection: str = DEFAULT_PROJECTION,
    dimensions: int = 2,
    fitness_key: str = "loss",
    seed: int = 0,
) -> dict[str, Any]:
    """Projects every genome of a run with the process's shared builder.

    This is a module-level function (rather than a method) so a worker process
    can run it and keep its builder's cache between requests.

    Args:
        archive_path: The run's ``genomes.sqlar``.
        metric: The distance metric's name.
        projection: The projection's name.
        dimensions: 2 or 3.
        fitness_key: The fitness key genomes are valued by.
        seed: Seeds any randomness in the projection.

    Returns:
        See :meth:`SearchSpaceBuilder.build`.

    Raises:
        ValueError: If the metric, projection, dimensions or fitness key is not
            valid.
    """

    return _BUILDER.build(
        archive_path,
        metric=metric,
        projection=projection,
        dimensions=dimensions,
        fitness_key=fitness_key,
        seed=seed,
    )
