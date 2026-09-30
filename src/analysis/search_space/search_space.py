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

import json
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

#: The metric used when none is asked for, for runs other than classification
#: (see :func:`default_metric`), and the projection used when none is asked for.
DEFAULT_METRIC = "fubini_study"
DEFAULT_PROJECTION = "classical_mds"

#: The metric used when none is asked for on a classification run.
CLASSIFICATION_METRIC = "behaviour"

#: The order metrics are offered in; metrics registered later follow.
METRIC_ORDER = ("behaviour", "readout", "fubini_study", "jaccard")

#: How many points of a run the island separation is measured at, at most.
SEPARATION_CHECKPOINTS = 50

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


def silhouette(distances: np.ndarray, labels: list[Any]) -> float | None:
    """Scores how well groups of genomes separate (the mean silhouette).

    Each genome's silhouette is ``(b - a) / max(a, b)``, where ``a`` is its mean
    distance to the other genomes of its group and ``b`` its mean distance to the
    genomes of the nearest other group; the score is their mean over every
    genome. Around 0 the groups overlap, above about 0.25 they occupy visibly
    different regions, and 1 means they are fully separate. A genome alone in its
    group scores 0, as scikit-learn defines it.

    Args:
        distances: The symmetric distance matrix between the genomes.
        labels: Each genome's group, row for row.

    Returns:
        The mean silhouette, or ``None`` when it is undefined: fewer than two
        groups, or no group with more than one genome.
    """

    groups = sorted(set(labels), key=str)
    count = len(labels)
    if len(groups) < 2 or len(groups) >= count:
        return None

    from sklearn.metrics import silhouette_score

    return float(silhouette_score(distances, labels, metric="precomputed"))


def island_separation(
    distances: np.ndarray,
    genome_numbers: list[int],
    islands: list[int | None],
    insertions: list[int | None],
    events: list[tuple[int, list[int], list[int]]],
    checkpoints: int = SEPARATION_CHECKPOINTS,
) -> dict[str, Any] | None:
    """Measures how separate a run's islands are as the search goes on.

    At evenly spaced insertions, the islands' silhouette (see :func:`silhouette`)
    is computed over the genomes then in the population -- replayed from the
    run's recorded population changes -- using the full distances, not a
    projection. A run that recorded no population changes is measured over every
    genome inserted so far instead. Island extinctions (an event removing every
    genome an island held) are listed so they can be marked on the time axis.

    Args:
        distances: The distances between the placed genomes.
        genome_numbers: The placed genomes, row for row.
        islands: Each placed genome's island.
        insertions: Each placed genome's insertion.
        events: The run's population changes, as ``(step, added, removed)`` in
            step order (a step is an insertion).
        checkpoints: How many insertions to measure at, at most.

    Returns:
        ``None`` for a run without islands; otherwise ``basis`` (``"population"``
        or ``"inserted"``: what each point was measured over), parallel lists
        ``insertion``, ``silhouette`` (``None`` where undefined) and ``genomes``
        (how many genomes each point measured), ``overall`` (the silhouette of
        every placed genome together) and ``extinctions`` (each extinction's
        ``insertion`` and ``island``).
    """

    if not any(island is not None for island in islands):
        return None

    row_of = {number: row for row, number in enumerate(genome_numbers)}
    island_of = {
        number: island
        for number, island in zip(genome_numbers, islands)
        if island is not None
    }
    known = [insertion for insertion in insertions if insertion is not None]
    if not known:
        return None
    first, last = min(known), max(known)
    steps = sorted(
        {
            int(round(first + (last - first) * k / max(checkpoints - 1, 1)))
            for k in range(checkpoints)
        }
    )

    def score(members: list[int]) -> float | None:
        """Scores the islands among the given (placed, island-labelled) genomes."""
        rows = [row_of[number] for number in members]
        return silhouette(
            distances[np.ix_(rows, rows)], [island_of[number] for number in members]
        )

    points: list[tuple[int, float | None, int]] = []
    extinctions: list[dict[str, int]] = []
    if events:
        basis = "population"
        members: set[int] = set()
        cursor = 0
        for step in steps:
            while cursor < len(events) and events[cursor][0] <= step:
                event_step, added, removed = events[cursor]
                held: dict[int, set[int]] = {}
                for number in members:
                    if number in island_of:
                        held.setdefault(island_of[number], set()).add(number)
                gone = set(removed)
                for island, numbers in sorted(held.items()):
                    if numbers and numbers <= gone:
                        extinctions.append({"insertion": event_step, "island": island})
                members.difference_update(removed)
                members.update(added)
                cursor += 1
            present = sorted(number for number in members if number in island_of)
            points.append((step, score(present), len(present)))
    else:
        basis = "inserted"
        for step in steps:
            present = sorted(
                number
                for number, insertion in zip(genome_numbers, insertions)
                if insertion is not None and insertion <= step and number in island_of
            )
            points.append((step, score(present), len(present)))

    return {
        "basis": basis,
        "insertion": [step for step, _, _ in points],
        "silhouette": [value for _, value, _ in points],
        "genomes": [count for _, _, count in points],
        "overall": score(sorted(island_of)),
        "extinctions": extinctions,
    }


def default_metric(run_info: dict[str, Any] | None) -> str:
    """Chooses the metric a run is projected with when none is asked for.

    Args:
        run_info: What the run recorded in its archive.

    Returns:
        :data:`CLASSIFICATION_METRIC` (the behaviour distance, which measures
        what the task sees) for classification runs, and :data:`DEFAULT_METRIC`
        (the unitary distance) otherwise.
    """

    if (run_info or {}).get("task") == "classification":
        return CLASSIFICATION_METRIC
    return DEFAULT_METRIC


def options() -> dict[str, list[dict[str, str]]]:
    """Lists the metrics and projections a search space can be built with.

    Returns:
        ``metrics`` and ``projections``, each entry's ``name``, ``label`` and
        ``description``: metrics in :data:`METRIC_ORDER` (then any registered
        later), projections with the default first.
    """

    def first(names: list[str], default: str) -> list[str]:
        """Moves the default to the front of a list of names."""
        return [default] + [name for name in names if name != default]

    ordered = [name for name in METRIC_ORDER if name in DISTANCE_METRICS]
    ordered += [name for name in DISTANCE_METRICS if name not in ordered]
    return {
        "metrics": [DISTANCE_METRICS[name].describe() for name in ordered],
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
        metric: str | None = None,
        projection: str = DEFAULT_PROJECTION,
        dimensions: int = 2,
        fitness_key: str = "loss",
        seed: int = 0,
    ) -> dict[str, Any]:
        """Projects every genome of a run.

        Args:
            archive_path: The run's ``genomes.sqlar``.
            metric: The distance metric's name (see :func:`options`); when not
                given, the run's :func:`default_metric`.
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
            off-diagonal distance); ``context`` (what the metric compared
            the genomes over, e.g. the qubits); and ``island_separation`` (how
            separate the islands were as the run went on, see
            :func:`island_separation`; ``None`` for a run without islands).

        Raises:
            ValueError: If the metric, projection, dimensions or fitness key is
                not valid.
        """

        if metric is not None:
            get_distance(metric)
        method = get_projection(projection)
        if dimensions not in (2, 3):
            raise ValueError(
                f"Can only project to 2 or 3 dimensions, not {dimensions}."
            )

        with self._lock:
            cache = self._run_cache(archive_path)
            with GenomeArchive.open_readonly(archive_path) as reader:
                run_info = reader.run_info()
                points = reader.points(fitness_key)
                links = reader.parent_links()
                events = [
                    (int(step), json.loads(added or "[]"), json.loads(removed or "[]"))
                    for step, added, removed in reader.connection.execute(
                        "SELECT step, added, removed FROM population_events ORDER BY step"
                    )
                ]
                for genome_number in points["genome_number"]:
                    if genome_number not in cache.genomes:
                        cache.genomes[genome_number] = reader.get_genome_dict(
                            genome_number
                        )

            metric = metric or default_metric(run_info)
            distance = get_distance(metric)
            genome_numbers = points["genome_number"]
            genomes = [cache.genomes[number] for number in genome_numbers]
            context = distance.context(genomes, run_info)
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
            "island_separation": island_separation(
                matrix,
                placed_numbers,
                [points["island"][i] for i in placed],
                insertions,
                events,
            ),
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
        ``qubits`` (as ``[name, index]`` pairs) for the unitary metrics, with the
        measured ``output_qubits`` and ``output_mode`` for the readout metric;
        the ``dataset``, ``normalization`` and ``seed`` for the behaviour metric;
        and an empty dict otherwise.
    """

    if metric == "fubini_study" and context:
        return {"qubits": [list(qubit) for qubit in context]}
    if metric == "readout" and context:
        qubits, outputs, mode = context
        return {
            "qubits": [list(qubit) for qubit in qubits],
            "output_qubits": [list(qubit) for qubit in outputs],
            "output_mode": mode,
        }
    if metric == "behaviour" and context and context[0] == "classification":
        _, dataset, normalization, seed = context
        return {"dataset": dataset, "normalization": normalization, "seed": seed}
    return {}


#: The builder :func:`build_search_space` uses, shared by every call in a process.
_BUILDER = SearchSpaceBuilder()


def build_search_space(
    archive_path: str,
    metric: str | None = None,
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
        metric: The distance metric's name; when not given, the run's
            :func:`default_metric`.
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
