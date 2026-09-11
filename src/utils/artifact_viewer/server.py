"""Local web server behind ``python3 -m src.examples.exaqc_artifacts``.

It serves a single-page viewer (the files in ``static/``) and a small read-only
JSON API over one or more EXAQC runs. Every request reads a run straight from
its ``genomes.sqlar`` archive with short read-only queries, so a run can be
browsed while its search is still writing to it. Architecture diagrams and
training plots are rendered on demand in a background worker process -- which
keeps matplotlib and the quantum frameworks out of the request threads -- and
cached in memory.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import shlex
import sqlite3
import statistics
import threading
import webbrowser
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from loguru import logger

from src.utils.genome_archive import ARCHIVE_FILENAME, GenomeArchive
from src.utils.search_history import (
    HISTORY_FILENAME,
    aggregate_history,
    history_columns,
    load_history_csv,
)

#: Directory holding the viewer's HTML, JavaScript, CSS and vendored uPlot.
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"

#: The images a genome can be rendered as.
IMAGE_KINDS = ("diagram", "training")

#: Datasets ``src.examples.evaluate`` can test a genome on (its ``--dataset``
#: choices); the viewer only offers an evaluate command for these.
EVALUATE_DATASETS = ("mnist", "fashion_mnist", "cifar10")

#: The most genome rows one listing request may return.
MAX_PAGE_SIZE = 500

#: How many generations of ancestors a genome's ancestry graph shows by default,
#: and at most.
DEFAULT_ANCESTRY_DEPTH = 5
MAX_ANCESTRY_DEPTH = 20

#: Gate fields compared when two genomes are diffed.
GATE_FIELDS = ("method_name", "qubits", "depth", "parameters", "enabled")

#: History CSV columns that describe the row rather than the search's progress.
_HISTORY_INDEX_COLUMNS = frozenset({"step", "current_time", "inserted_genomes"})

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}

_STATIC_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def higher_is_better(key: str) -> bool:
    """Decides which direction of a fitness key is an improvement.

    Loss-like keys (the value every task's population is ranked by) are
    minimized; accuracies, fidelities and returns are maximized.

    Args:
        key: A fitness key.

    Returns:
        False for keys containing ``loss``, True otherwise.
    """

    return "loss" not in key.lower()


def json_safe(value: Any) -> Any:
    """Converts a value into something ``json.dumps(allow_nan=False)`` accepts.

    Args:
        value: Any value built from dicts, lists, tuples, sets, strings, numbers
            (including numpy scalars and arrays) and ``None``.

    Returns:
        The value with non-finite floats replaced by ``None``, numpy values made
        native, tuples and sets turned into lists, and dict keys made strings.
    """

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return json_safe(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def summary_statistics(values: list[float]) -> dict[str, float | int | None]:
    """Summarizes a list of numbers, ignoring missing and non-finite ones.

    Args:
        values: The numbers to summarize.

    Returns:
        ``n``, ``min``, ``mean``, ``max`` and ``std`` (population standard
        deviation); every statistic is ``None`` when no finite values remain.
    """

    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not finite:
        return {"n": 0, "min": None, "mean": None, "max": None, "std": None}
    return {
        "n": len(finite),
        "min": min(finite),
        "mean": statistics.fmean(finite),
        "max": max(finite),
        "std": statistics.pstdev(finite),
    }


@dataclass
class Run:
    """One run the viewer serves.

    Attributes:
        index: The run's position in the viewer's run list, used in URLs.
        name: A short display name (its path relative to the served runs).
        directory: The run's output directory.
        archive_path: The path of its ``genomes.sqlar``.
        groups: The ``--groups`` substrings found in its path.
    """

    index: int
    name: str
    directory: str
    archive_path: str
    groups: list[str] = field(default_factory=list)


def discover_runs(paths: list[str]) -> list[Run]:
    """Finds the runs to serve.

    An archive file is served as given. A directory is searched recursively, so
    a run directory, or a directory of many runs (e.g. ten repeats of an
    experiment), can be passed.

    Args:
        paths: Archive files and directories.

    Returns:
        The runs found, without duplicates, sorted by path.

    Raises:
        FileNotFoundError: If a path does not exist.
    """

    archives: set[str] = set()
    for path in paths:
        if os.path.isfile(path):
            archives.add(os.path.abspath(path))
        elif os.path.isdir(path):
            for root, directories, files in os.walk(path):
                directories.sort()
                if ARCHIVE_FILENAME in files:
                    archives.add(os.path.abspath(os.path.join(root, ARCHIVE_FILENAME)))
        else:
            raise FileNotFoundError(f"{path!r} does not exist.")

    ordered = sorted(archives)
    directories = [os.path.dirname(archive) for archive in ordered]
    common = os.path.commonpath(directories) if len(directories) > 1 else ""

    runs = []
    for index, (archive_path, directory) in enumerate(zip(ordered, directories)):
        name = (
            os.path.relpath(directory, common)
            if common
            else os.path.basename(directory)
        )
        if name in ("", "."):
            name = os.path.basename(directory) or directory
        runs.append(
            Run(index=index, name=name, directory=directory, archive_path=archive_path)
        )
    return runs


def assign_groups(runs: list[Run], groups: list[str] | None) -> dict[str, list[int]]:
    """Groups runs by substrings of their paths, as the analysis scripts do.

    A run joins every group whose substring appears in its directory path.

    Args:
        runs: The runs being served; each run's ``groups`` list is filled in.
        groups: The group substrings, or ``None`` for no groups.

    Returns:
        The indexes of each group's runs, keyed by group, in the order given.
    """

    members: dict[str, list[int]] = {}
    for group in groups or []:
        members[group] = []
        for run in runs:
            if group in run.directory:
                run.groups.append(group)
                members[group].append(run.index)
    return members


def render_genome_image(
    archive_path: str, genome_number: int, kind: str
) -> bytes | None:
    """Renders one of a stored genome's images.

    This runs in the render worker process, so it imports the plotting and
    quantum stacks itself.

    Args:
        archive_path: The archive holding the genome.
        genome_number: The genome to draw.
        kind: ``"diagram"`` or ``"training"``.

    Returns:
        The PNG bytes, or ``None`` if the image could not be drawn (for example
        a training plot for a genome that recorded no training metrics).

    Raises:
        KeyError: If the archive holds no such genome.
        ValueError: If ``kind`` is not one of :data:`IMAGE_KINDS`.
    """

    if kind not in IMAGE_KINDS:
        raise ValueError(f"Unknown image kind {kind!r}.")

    import matplotlib

    matplotlib.use("Agg")

    from src.circuits.circuit import CircuitGenome
    from src.utils.genome_rendering import render_diagram_png, render_training_png

    with GenomeArchive.open_readonly(archive_path) as reader:
        genome = CircuitGenome.from_dict(reader.get_genome_dict(genome_number))

    return (
        render_diagram_png(genome) if kind == "diagram" else render_training_png(genome)
    )


class RenderService:
    """Renders genome images away from the request threads, caching the results.

    Renders go through a single worker process (matplotlib is not thread-safe),
    identical concurrent requests share one render, and recent images are kept
    in a bounded in-memory cache.
    """

    def __init__(self, processes: int = 1, cache_size: int = 256) -> None:
        """Starts the render service.

        Args:
            processes: Worker processes to render in. ``0`` renders in the
                calling thread instead, one image at a time (useful for tests).
            cache_size: How many rendered images to keep.
        """

        self._executor = (
            ProcessPoolExecutor(
                max_workers=processes, mp_context=multiprocessing.get_context("spawn")
            )
            if processes > 0
            else None
        )
        self._cache: OrderedDict[tuple[str, int, str], bytes | None] = OrderedDict()
        self._pending: dict[tuple[str, int, str], Future] = {}
        self._lock = threading.Lock()
        self._inline_lock = threading.Lock()
        self._cache_size = cache_size

    def render(self, archive_path: str, genome_number: int, kind: str) -> bytes | None:
        """Returns a genome image, rendering it if it is not cached.

        Args:
            archive_path: The archive holding the genome.
            genome_number: The genome to draw.
            kind: ``"diagram"`` or ``"training"``.

        Returns:
            The PNG bytes, or ``None`` if the image could not be drawn.

        Raises:
            KeyError: If the archive holds no such genome.
            ValueError: If ``kind`` is not an image kind.
        """

        key = (archive_path, int(genome_number), kind)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            future = self._pending.get(key)
            if future is None:
                future = self._submit(key)
                self._pending[key] = future

        try:
            image = future.result()
        finally:
            with self._lock:
                self._pending.pop(key, None)

        with self._lock:
            self._cache[key] = image
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return image

    def _submit(self, key: tuple[str, int, str]) -> Future:
        """Starts rendering an image.

        Args:
            key: ``(archive_path, genome_number, kind)``.

        Returns:
            A future resolving to the rendered image.
        """

        if self._executor is not None:
            return self._executor.submit(render_genome_image, *key)

        future: Future = Future()
        with self._inline_lock:
            try:
                future.set_result(render_genome_image(*key))
            except BaseException as error:
                future.set_exception(error)
        return future

    def close(self) -> None:
        """Stops the worker process, abandoning renders still queued.

        Returns:
            None.
        """

        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)


def compare_gates(
    gates_a: list[dict[str, Any]], gates_b: list[dict[str, Any]]
) -> dict[str, Any]:
    """Diffs two genomes' gates, matched by innovation number.

    Args:
        gates_a: The first genome's serialized gates.
        gates_b: The second genome's serialized gates.

    Returns:
        ``only_a`` and ``only_b`` (gates found in just one genome),
        ``changed`` (gates in both whose :data:`GATE_FIELDS` differ, with each
        differing field's two values) and ``unchanged`` (how many are identical).
    """

    by_a = {gate["innovation_number"]: gate for gate in gates_a}
    by_b = {gate["innovation_number"]: gate for gate in gates_b}

    changed = []
    unchanged = 0
    for innovation in sorted(by_a.keys() & by_b.keys()):
        differences = {
            name: [by_a[innovation].get(name), by_b[innovation].get(name)]
            for name in GATE_FIELDS
            if by_a[innovation].get(name) != by_b[innovation].get(name)
        }
        if differences:
            changed.append(
                {
                    "innovation_number": innovation,
                    "method_name": by_a[innovation].get("method_name"),
                    "differences": differences,
                }
            )
        else:
            unchanged += 1

    return {
        "only_a": [
            by_a[innovation] for innovation in sorted(by_a.keys() - by_b.keys())
        ],
        "only_b": [
            by_b[innovation] for innovation in sorted(by_b.keys() - by_a.keys())
        ],
        "changed": changed,
        "unchanged": unchanged,
    }


def compare_values(
    values_a: dict[str, Any] | None, values_b: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Lines up two dicts (such as fitness or hyperparameters) key by key.

    Args:
        values_a: The first genome's values.
        values_b: The second genome's values.

    Returns:
        One entry per key of either dict, sorted by key, with both values and
        whether they are equal.
    """

    values_a = values_a or {}
    values_b = values_b or {}
    return [
        {
            "key": key,
            "a": values_a.get(key),
            "b": values_b.get(key),
            "same": values_a.get(key) == values_b.get(key),
        }
        for key in sorted(values_a.keys() | values_b.keys())
    ]


def genome_commands(run: Run, genome: dict[str, Any]) -> dict[str, str]:
    """Builds ready-to-run commands for working with a stored genome.

    Args:
        run: The run the genome belongs to.
        genome: The serialized genome.

    Returns:
        Shell commands keyed by entry point: always ``refine_genome``, plus
        ``visualize_rl`` for reinforcement-learning genomes and ``evaluate`` for
        genomes evolved on an image dataset.
    """

    number = genome["genome_number"]
    source = f"--archive {shlex.quote(run.directory)} --genome_number {number}"
    commands = {"refine_genome": f"python3 -m src.examples.refine_genome {source}"}

    if genome.get("task") == "reinforcement_learning":
        commands["visualize_rl"] = (
            f"python3 -m src.examples.visualize_rl {source} --episodes 3 --output_file genome_{number}.gif"
        )

    if (
        genome.get("task") == "classification"
        and genome.get("task_target") in EVALUATE_DATASETS
    ):
        commands["evaluate"] = (
            f"python3 -m src.examples.evaluate {source} --dataset {genome['task_target']}"
        )

    return commands


def _query_int(
    query: dict[str, str],
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Reads an integer query parameter.

    Args:
        query: The request's query parameters.
        name: The parameter to read.
        default: The value when the parameter is absent.
        minimum: The smallest value allowed, if any.
        maximum: The largest value allowed; larger values are clamped to it.

    Returns:
        The parameter's value.

    Raises:
        ValueError: If the value is not an integer or is below ``minimum``.
    """

    if name not in query or query[name] == "":
        return default
    try:
        value = int(query[name])
    except ValueError as error:
        raise ValueError(f"Query parameter {name!r} must be an integer.") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"Query parameter {name!r} must be at least {minimum}.")
    return min(value, maximum) if maximum is not None else value


class ArtifactViewer:
    """The data behind the viewer's JSON API.

    Attributes:
        runs: The runs being served, indexed by ``Run.index``.
        groups: The run indexes in each ``--groups`` group.
        renderer: Renders genome images.
    """

    def __init__(
        self, runs: list[Run], groups: dict[str, list[int]], renderer: RenderService
    ) -> None:
        """Creates the viewer.

        Args:
            runs: The runs to serve.
            groups: The run indexes in each group (see :func:`assign_groups`).
            renderer: The image render service.
        """

        self.runs = runs
        self.groups = groups
        self.renderer = renderer

    def run(self, index: int) -> Run:
        """Looks up a served run.

        Args:
            index: The run's index.

        Returns:
            The run.

        Raises:
            KeyError: If there is no run with that index.
        """

        if not 0 <= index < len(self.runs):
            raise KeyError(f"There is no run {index}.")
        return self.runs[index]

    def _summary(self, run: Run) -> dict[str, Any]:
        """Summarizes a run for the run list.

        Args:
            run: The run to summarize.

        Returns:
            The run's identity, recorded run information, genome count, latest
            genome, last write time and best ``loss`` and ``target_metric``.
            An unreadable archive is reported under ``error`` instead.
        """

        summary: dict[str, Any] = {
            "index": run.index,
            "name": run.name,
            "directory": run.directory,
            "groups": run.groups,
        }
        try:
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                info = reader.run_info()
                summary.update(
                    {
                        "task": info.get("task"),
                        "task_target": info.get("task_target"),
                        "target": info.get("target"),
                        "population_strategy": info.get("population_strategy"),
                        "command_line": info.get("command_line"),
                        "start_time": info.get("start_time"),
                        "genomes": reader.count(),
                        "max_genome_number": reader.max_genome_number(),
                        "last_saved_at": reader.last_saved_at(),
                        "best_loss": reader.best_value("loss", higher_is_better=False),
                        "best_target_metric": reader.best_value(
                            "target_metric", higher_is_better=True
                        ),
                    }
                )
        except sqlite3.DatabaseError as error:
            logger.warning("Could not read {}: {}", run.archive_path, error)
            summary["error"] = str(error)
        return summary

    def runs_payload(self) -> dict[str, Any]:
        """Lists every served run.

        Returns:
            ``runs`` (each run's summary) and ``groups`` (the group names).
        """

        return {
            "runs": [self._summary(run) for run in self.runs],
            "groups": list(self.groups),
        }

    def run_payload(self, index: int) -> dict[str, Any]:
        """Describes one run in full.

        Args:
            index: The run's index.

        Returns:
            The run's summary plus its fitness keys, the values its genomes can
            be filtered by, the ``unarchived_parents`` (parents of stored
            genomes that are not stored themselves, i.e. the seed genome), and
            whether it recorded a search history.

        Raises:
            KeyError: If there is no such run.
        """

        run = self.run(index)
        payload = self._summary(run)
        with GenomeArchive.open_readonly(run.archive_path) as reader:
            payload["fitness_keys"] = reader.fitness_keys()
            payload["filter_options"] = reader.filter_options()
            payload["unarchived_parents"] = reader.unarchived_parents()
        payload["has_history"] = os.path.isfile(
            os.path.join(run.directory, HISTORY_FILENAME)
        )
        return payload

    def genomes_payload(self, index: int, query: dict[str, str]) -> dict[str, Any]:
        """Lists a page of a run's genomes.

        Args:
            index: The run's index.
            query: ``sort`` (a fitness key or summary column, default ``loss``),
                ``desc`` (``1`` to sort descending), ``offset``, ``limit``, the
                filters ``insert_type``, ``generated_by``, ``crossover_type`` and
                ``island``, and ``max_genome``: list only genomes numbered at
                most this. A client paging through a live run passes the first
                page's ``max_genome_number`` back, so genomes saved in between
                don't shift later pages (repeating or skipping rows).

        Returns:
            ``total`` (matching genomes), ``offset``, ``limit``,
            ``max_genome_number`` (the highest genome number the listing
            includes: ``max_genome`` if given, otherwise the archive's highest
            at the time of the request) and ``rows`` (genome summaries).

        Raises:
            KeyError: If there is no such run.
            ValueError: If a parameter is invalid.
        """

        run = self.run(index)
        offset = _query_int(query, "offset", 0, minimum=0)
        limit = _query_int(query, "limit", 50, minimum=1, maximum=MAX_PAGE_SIZE)
        max_genome = (
            _query_int(query, "max_genome", -1, minimum=0)
            if query.get("max_genome")
            else None
        )
        filters: dict[str, Any] = {
            "insert_type": query.get("insert_type") or None,
            "generated_by": query.get("generated_by") or None,
            "crossover_type": query.get("crossover_type") or None,
            "island": (
                _query_int(query, "island", -1, minimum=0)
                if query.get("island")
                else None
            ),
        }

        with GenomeArchive.open_readonly(run.archive_path) as reader:
            snapshot = reader.max_genome_number() if max_genome is None else max_genome
            filters["max_genome_number"] = snapshot
            return {
                "total": reader.count(filters),
                "offset": offset,
                "limit": limit,
                "max_genome_number": snapshot,
                "rows": reader.list_genomes(
                    sort_key=query.get("sort") or "loss",
                    descending=query.get("desc") in ("1", "true"),
                    filters=filters,
                    offset=offset,
                    limit=limit,
                ),
            }

    def points_payload(self, index: int, y_key: str) -> dict[str, Any]:
        """Returns every genome of a run as a point for the progress chart.

        Args:
            index: The run's index.
            y_key: The fitness key plotted on the y axis.

        Returns:
            Parallel arrays: ``genome_number``, ``insertion``, ``y``,
            ``insert_type``, ``generated_by``, ``operator`` (the first generating
            operator), ``crossover_type`` and ``island``.

        Raises:
            KeyError: If there is no such run.
            ValueError: If ``y_key`` is not a valid key.
        """

        with GenomeArchive.open_readonly(self.run(index).archive_path) as reader:
            return reader.points(y_key)

    def genealogy_payload(self, index: int, y_key: str) -> dict[str, Any]:
        """Returns every genome and every parent link of a run.

        Args:
            index: The run's index.
            y_key: The fitness key plotted on the y axis.

        Returns:
            ``points`` (as :meth:`points_payload`) and ``links`` (parallel
            ``child`` and ``parent`` arrays).

        Raises:
            KeyError: If there is no such run.
            ValueError: If ``y_key`` is not a valid key.
        """

        with GenomeArchive.open_readonly(self.run(index).archive_path) as reader:
            return {"points": reader.points(y_key), "links": reader.parent_links()}

    def history_payload(self, index: int) -> dict[str, Any]:
        """Returns a run's search-progress history.

        Args:
            index: The run's index.

        Returns:
            ``columns``: each history CSV column's values, keyed by column name
            (empty when the run recorded no history).

        Raises:
            KeyError: If there is no such run.
        """

        path = os.path.join(self.run(index).directory, HISTORY_FILENAME)
        if not os.path.isfile(path):
            return {"columns": {}}
        rows = load_history_csv(path)
        return {
            "columns": {
                name: [row.get(name) for row in rows] for name in history_columns(path)
            }
        }

    def operators_payload(self, index: int) -> dict[str, Any]:
        """Counts, per generating operator, how the genomes it made were inserted.

        Args:
            index: The run's index.

        Returns:
            ``operators``: insert-type counts keyed by operator.

        Raises:
            KeyError: If there is no such run.
        """

        with GenomeArchive.open_readonly(self.run(index).archive_path) as reader:
            return {"operators": reader.operator_counts()}

    def genome_payload(self, index: int, genome_number: int) -> dict[str, Any]:
        """Returns everything the detail panel shows about one genome.

        Args:
            index: The run's index.
            genome_number: The genome.

        Returns:
            ``summary``, the full serialized ``genome``, its ``children`` and
            ready-to-run ``commands``.

        Raises:
            KeyError: If there is no such run or genome.
        """

        run = self.run(index)
        with GenomeArchive.open_readonly(run.archive_path) as reader:
            genome = reader.get_genome_dict(genome_number)
            return {
                "summary": reader.get_summary(genome_number),
                "genome": genome,
                "children": reader.children(genome_number),
                "commands": genome_commands(run, genome),
            }

    def genome_json(self, index: int, genome_number: int) -> bytes:
        """Returns a genome's JSON exactly as a genome file would hold it.

        Args:
            index: The run's index.
            genome_number: The genome.

        Returns:
            The JSON bytes.

        Raises:
            KeyError: If there is no such run or genome.
        """

        with GenomeArchive.open_readonly(self.run(index).archive_path) as reader:
            return json.dumps(
                reader.get_genome_dict(genome_number), ensure_ascii=False, indent=4
            ).encode("utf-8")

    def image(self, index: int, genome_number: int, kind: str) -> bytes | None:
        """Returns one of a genome's rendered images.

        Args:
            index: The run's index.
            genome_number: The genome.
            kind: ``"diagram"`` or ``"training"``.

        Returns:
            The PNG bytes, or ``None`` if the image could not be drawn.

        Raises:
            KeyError: If there is no such run or genome.
            ValueError: If ``kind`` is not an image kind.
        """

        return self.renderer.render(self.run(index).archive_path, genome_number, kind)

    def ancestry_payload(
        self, index: int, genome_number: int, depth: int
    ) -> dict[str, Any]:
        """Returns a genome's ancestors, back a number of generations.

        Args:
            index: The run's index.
            genome_number: The genome.
            depth: How many generations back to go.

        Returns:
            ``nodes`` (each ancestor's number, generation and summary) and
            ``edges`` (``child``/``parent`` pairs).

        Raises:
            KeyError: If there is no such run or genome.
        """

        with GenomeArchive.open_readonly(self.run(index).archive_path) as reader:
            reader.get_summary(genome_number)
            return reader.ancestors(genome_number, depth)

    def compare_payload(
        self, index: int, genome_a: int, genome_b: int
    ) -> dict[str, Any]:
        """Compares two genomes of a run.

        Args:
            index: The run's index.
            genome_a: The first genome.
            genome_b: The second genome.

        Returns:
            ``a`` and ``b`` (each genome's summary and serialized genome), the
            ``gates`` diff, and key-by-key ``fitness`` and ``hyperparameters``.

        Raises:
            KeyError: If there is no such run or genome.
        """

        with GenomeArchive.open_readonly(self.run(index).archive_path) as reader:
            serialized = {
                number: reader.get_genome_dict(number)
                for number in (genome_a, genome_b)
            }
            summaries = {
                number: reader.get_summary(number) for number in (genome_a, genome_b)
            }

        first, second = serialized[genome_a], serialized[genome_b]
        return {
            "a": {"summary": summaries[genome_a], "genome": first},
            "b": {"summary": summaries[genome_b], "genome": second},
            "gates": compare_gates(first.get("gates") or [], second.get("gates") or []),
            "fitness": compare_values(first.get("fitness"), second.get("fitness")),
            "hyperparameters": compare_values(
                first.get("hyperparameters"), second.get("hyperparameters")
            ),
        }

    def groups_payload(self, metric: str = "best", conf: str = "std") -> dict[str, Any]:
        """Compares groups of runs.

        Each ``--groups`` group is compared, and so is every run that belongs to
        no group, on its own.

        Args:
            metric: The history column whose mean and band are charted.
            conf: The band: ``"std"`` or ``"95ci"``.

        Returns:
            ``metric``, ``conf``, the history ``metrics`` available, and per group
            its ``runs``, aggregated ``history`` (or ``history_error``), summary
            statistics of each run's best ``loss`` and ``target_metric``, and the
            summed ``operators`` counts.

        Raises:
            ValueError: If ``conf`` is not ``"std"`` or ``"95ci"``.
        """

        if conf not in ("std", "95ci"):
            raise ValueError("conf must be 'std' or '95ci'.")

        grouped = {index for members in self.groups.values() for index in members}
        entries = list(self.groups.items()) + [
            (run.name, [run.index]) for run in self.runs if run.index not in grouped
        ]

        available_metrics: set[str] = set()
        groups = []
        for name, members in entries:
            runs = [self.runs[index] for index in members]
            csv_paths = [
                path
                for path in (
                    os.path.join(run.directory, HISTORY_FILENAME) for run in runs
                )
                if os.path.isfile(path)
            ]
            for path in csv_paths:
                available_metrics.update(
                    column
                    for column in history_columns(path)
                    if column not in _HISTORY_INDEX_COLUMNS
                )

            history = None
            history_error = None
            if csv_paths:
                try:
                    steps, mean, low, high = aggregate_history(
                        csv_paths, metric=metric, conf=conf
                    )
                    history = {
                        "step": steps,
                        "mean": mean,
                        "low": low,
                        "high": high,
                        "n_runs": len(csv_paths),
                    }
                except RuntimeError as error:
                    history_error = str(error)
            else:
                history_error = "No search history was recorded."

            best_losses: list[float] = []
            best_targets: list[float] = []
            operators: dict[str, dict[str, int]] = {}
            for run in runs:
                try:
                    with GenomeArchive.open_readonly(run.archive_path) as reader:
                        best_loss = reader.best_value("loss", higher_is_better=False)
                        best_target = reader.best_value(
                            "target_metric", higher_is_better=True
                        )
                        run_operators = reader.operator_counts()
                except sqlite3.DatabaseError as error:
                    logger.warning("Could not read {}: {}", run.archive_path, error)
                    continue
                if best_loss is not None:
                    best_losses.append(best_loss["value"])
                if best_target is not None:
                    best_targets.append(best_target["value"])
                for operator, counts in run_operators.items():
                    totals = operators.setdefault(operator, {})
                    for insert_type, count in counts.items():
                        totals[insert_type] = totals.get(insert_type, 0) + count

            groups.append(
                {
                    "name": name,
                    "runs": [{"index": run.index, "name": run.name} for run in runs],
                    "history": history,
                    "history_error": history_error,
                    "best_loss": summary_statistics(best_losses),
                    "best_target_metric": summary_statistics(best_targets),
                    "operators": operators,
                }
            )

        return {
            "metric": metric,
            "conf": conf,
            "metrics": sorted(available_metrics),
            "groups": groups,
        }


#: API routes: a path pattern and the handler method serving it.
_ROUTES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/api/runs"), "_api_runs"),
    (re.compile(r"/api/groups"), "_api_groups"),
    (re.compile(r"/api/runs/(\d+)"), "_api_run"),
    (re.compile(r"/api/runs/(\d+)/genomes"), "_api_genomes"),
    (re.compile(r"/api/runs/(\d+)/points"), "_api_points"),
    (re.compile(r"/api/runs/(\d+)/genealogy"), "_api_genealogy"),
    (re.compile(r"/api/runs/(\d+)/history"), "_api_history"),
    (re.compile(r"/api/runs/(\d+)/operators"), "_api_operators"),
    (re.compile(r"/api/runs/(\d+)/compare"), "_api_compare"),
    (re.compile(r"/api/runs/(\d+)/genomes/(\d+)"), "_api_genome"),
    (re.compile(r"/api/runs/(\d+)/genomes/(\d+)\.json"), "_api_genome_json"),
    (
        re.compile(r"/api/runs/(\d+)/genomes/(\d+)/(diagram|training)\.png"),
        "_api_genome_image",
    ),
    (re.compile(r"/api/runs/(\d+)/genomes/(\d+)/ancestry"), "_api_ancestry"),
]


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Serves the viewer's static files and JSON API (GET requests only)."""

    server: ArtifactViewerServer
    server_version = "EXAQCArtifacts/1"

    def do_GET(self) -> None:
        """Handles a GET request, turning lookup and parameter errors into 404/400.

        Returns:
            None. Writes the response.
        """

        parts = urlsplit(self.path)
        query = {name: values[-1] for name, values in parse_qs(parts.query).items()}

        try:
            self._route(parts.path, query)
        except KeyError as error:
            self._send_error(
                HTTPStatus.NOT_FOUND, str(error.args[0]) if error.args else "Not found."
            )
        except ValueError as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            logger.exception("Error serving {}", self.path)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def log_message(self, format: str, *args: Any) -> None:
        """Sends the server's access log to the debug log instead of stderr.

        Args:
            format: The log message format string.
            *args: Values for ``format``.

        Returns:
            None.
        """

        logger.debug("{} - {}", self.address_string(), format % args)

    def _route(self, path: str, query: dict[str, str]) -> None:
        """Dispatches a request path to the static files or an API handler.

        Args:
            path: The request path.
            query: The request's query parameters.

        Returns:
            None. Writes the response.

        Raises:
            KeyError: If nothing is served at ``path``.
        """

        if path in ("/", "/index.html"):
            self._send_static("index.html")
            return
        if path.startswith("/static/"):
            self._send_static(path[len("/static/") :])
            return

        for pattern, handler_name in _ROUTES:
            match = pattern.fullmatch(path)
            if match:
                getattr(self, handler_name)(match, query)
                return

        raise KeyError(f"Nothing is served at {path}.")

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------

    def _api_runs(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves the run list.

        Args:
            match: The matched route.
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(self.server.viewer.runs_payload())

    def _api_groups(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves the run-group comparison (``metric`` and ``conf`` parameters).

        Args:
            match: The matched route.
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(
            self.server.viewer.groups_payload(
                query.get("metric") or "best", query.get("conf") or "std"
            )
        )

    def _api_run(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves one run's description.

        Args:
            match: The matched route (the run index).
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(self.server.viewer.run_payload(int(match.group(1))))

    def _api_genomes(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a page of a run's genomes.

        Args:
            match: The matched route (the run index).
            query: Sort, paging and filter parameters.

        Returns:
            None.
        """

        self._send_json(self.server.viewer.genomes_payload(int(match.group(1)), query))

    def _api_points(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a run's progress-chart points (``y`` parameter).

        Args:
            match: The matched route (the run index).
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(
            self.server.viewer.points_payload(
                int(match.group(1)), query.get("y") or "loss"
            )
        )

    def _api_genealogy(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a run's points and parent links (``y`` parameter).

        Args:
            match: The matched route (the run index).
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(
            self.server.viewer.genealogy_payload(
                int(match.group(1)), query.get("y") or "loss"
            )
        )

    def _api_history(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a run's search-progress history.

        Args:
            match: The matched route (the run index).
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(self.server.viewer.history_payload(int(match.group(1))))

    def _api_operators(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a run's operator insert-type counts.

        Args:
            match: The matched route (the run index).
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(self.server.viewer.operators_payload(int(match.group(1))))

    def _api_compare(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a comparison of genomes ``a`` and ``b``.

        Args:
            match: The matched route (the run index).
            query: The query parameters, which must include ``a`` and ``b``.

        Returns:
            None.

        Raises:
            ValueError: If ``a`` or ``b`` is missing.
        """

        if not query.get("a") or not query.get("b"):
            raise ValueError(
                "Give the two genomes to compare as ?a=<number>&b=<number>."
            )
        genome_a = _query_int(query, "a", 0, minimum=0)
        genome_b = _query_int(query, "b", 0, minimum=0)
        self._send_json(
            self.server.viewer.compare_payload(int(match.group(1)), genome_a, genome_b)
        )

    def _api_genome(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves one genome's details.

        Args:
            match: The matched route (run index and genome number).
            query: The query parameters.

        Returns:
            None.
        """

        self._send_json(
            self.server.viewer.genome_payload(int(match.group(1)), int(match.group(2)))
        )

    def _api_genome_json(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a genome's JSON as a file download.

        Args:
            match: The matched route (run index and genome number).
            query: The query parameters.

        Returns:
            None.
        """

        genome_number = int(match.group(2))
        body = self.server.viewer.genome_json(int(match.group(1)), genome_number)
        self._send_bytes(
            body, "application/json", filename=f"genome_{genome_number}.json"
        )

    def _api_genome_image(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a genome's rendered diagram or training plot.

        Args:
            match: The matched route (run index, genome number and image kind).
            query: The query parameters.

        Returns:
            None.

        Raises:
            KeyError: If the image could not be drawn.
        """

        kind = match.group(3)
        image = self.server.viewer.image(int(match.group(1)), int(match.group(2)), kind)
        if image is None:
            if kind == "training":
                raise KeyError("This genome recorded no training metrics to plot.")
            raise KeyError("This genome's diagram could not be drawn.")
        self._send_bytes(image, "image/png", cache=True)

    def _api_ancestry(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves a genome's ancestry graph (``depth`` parameter).

        Args:
            match: The matched route (run index and genome number).
            query: The query parameters.

        Returns:
            None.
        """

        depth = _query_int(
            query,
            "depth",
            DEFAULT_ANCESTRY_DEPTH,
            minimum=1,
            maximum=MAX_ANCESTRY_DEPTH,
        )
        self._send_json(
            self.server.viewer.ancestry_payload(
                int(match.group(1)), int(match.group(2)), depth
            )
        )

    # ------------------------------------------------------------------
    # Responses
    # ------------------------------------------------------------------

    def _send_static(self, name: str) -> None:
        """Serves a file from the static directory.

        Args:
            name: The file name (no directories).

        Returns:
            None.

        Raises:
            KeyError: If there is no such static file.
        """

        path = STATIC_DIRECTORY / name
        if not _STATIC_NAME.fullmatch(name) or not path.is_file():
            raise KeyError(f"There is no static file {name!r}.")
        self._send_bytes(
            path.read_bytes(),
            _CONTENT_TYPES.get(path.suffix, "text/plain; charset=utf-8"),
        )

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Sends a JSON response.

        Args:
            payload: The value to encode (see :func:`json_safe`).
            status: The HTTP status.

        Returns:
            None.
        """

        body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
        self._send_bytes(body, "application/json", status=status)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        """Sends an error, as JSON for API requests and plain text otherwise.

        Args:
            status: The HTTP status.
            message: What went wrong.

        Returns:
            None.
        """

        try:
            if self.path.startswith("/api/"):
                self._send_json({"error": message}, status=status)
            else:
                self._send_bytes(
                    message.encode("utf-8"), "text/plain; charset=utf-8", status=status
                )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        filename: str | None = None,
        cache: bool = False,
    ) -> None:
        """Sends a response body.

        Args:
            body: The response bytes.
            content_type: The ``Content-Type`` header.
            status: The HTTP status.
            filename: When given, the response is offered as a download with this
                file name.
            cache: Whether the browser may cache the response.

        Returns:
            None.
        """

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        if filename is not None:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.end_headers()
        self.wfile.write(body)


class ArtifactViewerServer(ThreadingHTTPServer):
    """The viewer's HTTP server, carrying the :class:`ArtifactViewer` it serves.

    Attributes:
        viewer: The data served by the API.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], viewer: ArtifactViewer) -> None:
        """Binds the server.

        Args:
            address: The ``(host, port)`` to listen on; port ``0`` picks a free
                port.
            viewer: The data to serve.
        """

        super().__init__(address, ViewerRequestHandler)
        self.viewer = viewer


def serve(
    runs: list[str],
    groups: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
    render_processes: int = 1,
) -> None:
    """Serves the viewer until interrupted.

    Args:
        runs: Run directories, directories of runs, or archive files.
        groups: Substrings grouping runs for comparison.
        host: The address to listen on.
        port: The port to listen on (``0`` picks a free port).
        open_browser: Whether to open the viewer in a web browser.
        render_processes: Worker processes rendering images.

    Returns:
        None. Runs until interrupted with Ctrl+C.

    Raises:
        FileNotFoundError: If a path does not exist or no runs are found.
        OSError: If the server cannot listen on ``host:port``.
    """

    discovered = discover_runs(runs)
    if not discovered:
        raise FileNotFoundError(
            f"No {ARCHIVE_FILENAME} archives were found in {', '.join(runs)}."
        )
    group_members = assign_groups(discovered, groups)

    renderer = RenderService(processes=render_processes)
    server = ArtifactViewerServer(
        (host, port), ArtifactViewer(discovered, group_members, renderer)
    )
    url = f"http://{host}:{server.server_address[1]}/"

    logger.info(
        "Serving {} run(s) at {} -- press Ctrl+C to stop.", len(discovered), url
    )
    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping the viewer.")
    finally:
        server.server_close()
        renderer.close()
