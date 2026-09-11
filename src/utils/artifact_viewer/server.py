"""Local web server behind ``python3 -m src.examples.exaqc_dashboard``.

It serves a single-page app (the files in ``static/``) and a small read-only
JSON API over EXAQC runs: either given run directories, or every run found below
a watched directory, re-scanned so that runs started later join in. Every
request reads a run straight from its ``genomes.sqlar`` archive with short
read-only queries, so a run can be browsed while its search is still writing to
it. Architecture diagrams and training plots are rendered on demand in a
background worker process -- which keeps matplotlib and the quantum frameworks
out of the request threads -- and cached in memory.
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
import time
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

#: The least time between scans for newly written run archives, in seconds.
#: Pages poll every few seconds, so this bounds how often the disk is searched.
RESCAN_INTERVAL_SECONDS = 5.0

#: Directories never searched for runs: a legacy run's per-genome JSON files,
#: which can number in the tens of thousands.
_SKIPPED_DIRECTORIES = frozenset({"all_genomes"})

#: The insert types each operator's insertion rates are broken down into, in the
#: order ``src.analysis.analyze_genome_generation`` tabulates them.
INSERTION_OUTCOMES = ("global_best", "local_best", "inserted", "discarded")

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
        name: A short display name: its path relative to the watched directory,
            or to the given run directories' common parent.
        directory: The run's output directory.
        archive_path: The path of its ``genomes.sqlar``.
        groups: The ``--groups`` substrings found in its path.
    """

    index: int
    name: str
    directory: str
    archive_path: str
    groups: list[str] = field(default_factory=list)


def find_archives(directory: str) -> list[str]:
    """Finds every run archive below a directory, at any depth.

    A directory holding an archive is a run, so its subdirectories are not
    searched further, and neither are legacy ``all_genomes`` directories; this
    keeps re-scanning a large experiment directory cheap.

    Args:
        directory: The directory to search.

    Returns:
        The absolute paths of the archives found, sorted.
    """

    archives = []
    for root, directories, files in os.walk(directory):
        if ARCHIVE_FILENAME in files:
            archives.append(os.path.abspath(os.path.join(root, ARCHIVE_FILENAME)))
            directories.clear()
        else:
            directories[:] = sorted(
                name for name in directories if name not in _SKIPPED_DIRECTORIES
            )
    return sorted(archives)


def _run_name(directory: str, base: str) -> str:
    """Names a run by its directory's path relative to a base directory.

    Args:
        directory: The run's output directory.
        base: The directory names are relative to, or ``""`` to name the run by
            its directory's own name.

    Returns:
        The display name.
    """

    name = os.path.relpath(directory, base) if base else os.path.basename(directory)
    if name in ("", "."):
        name = os.path.basename(directory) or directory
    return name


def assign_groups(runs: list[Run], groups: list[str] | None) -> dict[str, list[int]]:
    """Groups runs by substrings of their paths, as the analysis scripts do.

    A run joins every group whose substring appears in its directory path.

    Args:
        runs: The runs being served; each run's ``groups`` list is set.
        groups: The group substrings, or ``None`` for no groups.

    Returns:
        The indexes of each group's runs, keyed by group, in the order given.
    """

    members: dict[str, list[int]] = {group: [] for group in groups or []}
    for run in runs:
        run.groups = [group for group in members if group in run.directory]
        for group in run.groups:
            members[group].append(run.index)
    return members


class RunRegistry:
    """The runs the dashboard serves, picked up as their archives are written.

    Runs come either from a list of run output directories or from a watched
    directory, where every archive below it is a run -- including runs started
    after the dashboard. Both are re-scanned (at most every ``rescan_interval``
    seconds), so a run whose search has not written its archive yet -- or has
    not even started -- appears once it has. A run keeps the index (used in the
    page's URLs) and the name it was first found with: new runs are appended,
    and a run whose archive later
    disappears stays listed, reported as unreadable.

    Attributes:
        watch_directory: The watched directory, or ``None`` when serving given
            run directories.
        group_names: The ``--groups`` substrings runs are grouped by.
        rescan_interval: The least time between scans, in seconds.
    """

    def __init__(
        self,
        run_directories: list[str] | None = None,
        watch_directory: str | None = None,
        groups: list[str] | None = None,
        rescan_interval: float = RESCAN_INTERVAL_SECONDS,
    ) -> None:
        """Checks the run sources and finds the runs already written.

        Args:
            run_directories: Run output directories, or their archive files, to
                serve. They need not exist yet: a run that has not been started
                is waited for like one that has not written its archive.
            watch_directory: A directory whose runs, at any depth, are served.
            groups: Substrings grouping runs for comparison.
            rescan_interval: The least time between scans, in seconds.

        Raises:
            ValueError: If both or neither of ``run_directories`` and
                ``watch_directory`` are given.
            FileNotFoundError: If the watched directory does not exist.
            NotADirectoryError: If the watched path is not a directory.
        """

        if (run_directories is None) == (watch_directory is None):
            raise ValueError(
                "Give either run directories or a directory to watch, but not both."
            )

        self.group_names = list(groups or [])
        self.rescan_interval = rescan_interval
        self.watch_directory: str | None = None
        # given runs, written or not: (archive path, run directory, name)
        self._given: list[tuple[str, str, str]] = []

        if watch_directory is not None:
            if not os.path.exists(watch_directory):
                raise FileNotFoundError(f"{watch_directory!r} does not exist.")
            if not os.path.isdir(watch_directory):
                raise NotADirectoryError(f"{watch_directory!r} is not a directory.")
            self.watch_directory = os.path.abspath(watch_directory)
        else:
            sources: dict[str, str] = {}
            for path in run_directories or []:
                if os.path.isfile(path) or (
                    not os.path.exists(path) and path.endswith(".sqlar")
                ):
                    archive = os.path.abspath(path)
                    sources.setdefault(archive, os.path.dirname(archive))
                else:
                    # a run directory, which need not exist yet if its search has not started
                    directory = os.path.abspath(path)
                    sources.setdefault(
                        os.path.join(directory, ARCHIVE_FILENAME), directory
                    )
            directories = list(sources.values())
            common = os.path.commonpath(directories) if len(directories) > 1 else ""
            self._given = [
                (archive, directory, _run_name(directory, common))
                for archive, directory in sources.items()
            ]

        self._lock = threading.Lock()
        self._runs: list[Run] = []
        self._groups: dict[str, list[int]] = assign_groups([], self.group_names)
        self._last_scan: float | None = None
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> bool:
        """Adds the runs whose archives have been written since the last scan.

        Pages poll often, so unless forced a scan happens at most once every
        ``rescan_interval`` seconds.

        Args:
            force: Scan even if the last scan was recent.

        Returns:
            Whether any run was added.
        """

        with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._last_scan is not None
                and now - self._last_scan < self.rescan_interval
            ):
                return False
            self._last_scan = now

            if self.watch_directory is not None:
                found = [
                    (
                        archive,
                        os.path.dirname(archive),
                        _run_name(os.path.dirname(archive), self.watch_directory),
                    )
                    for archive in find_archives(self.watch_directory)
                ]
            else:
                found = [given for given in self._given if os.path.isfile(given[0])]

            known = {run.archive_path for run in self._runs}
            added = [entry for entry in found if entry[0] not in known]
            for archive, directory, name in added:
                self._runs.append(
                    Run(
                        index=len(self._runs),
                        name=name,
                        directory=directory,
                        archive_path=archive,
                    )
                )
                logger.info("Found run {} ({}).", name, directory)
            if added:
                self._groups = assign_groups(self._runs, self.group_names)
            return bool(added)

    @property
    def runs(self) -> list[Run]:
        """The runs found so far, in index order.

        Returns:
            A copy of the run list.
        """

        with self._lock:
            return list(self._runs)

    @property
    def groups(self) -> dict[str, list[int]]:
        """The indexes of each group's runs.

        Returns:
            A copy of the member indexes, keyed by group in ``--groups`` order.
        """

        with self._lock:
            return {group: list(members) for group, members in self._groups.items()}

    def run(self, index: int) -> Run:
        """Looks up a run, re-scanning (when due) if the index is not known yet.

        Args:
            index: The run's index.

        Returns:
            The run.

        Raises:
            KeyError: If there is no run with that index.
        """

        for attempt in range(2):
            with self._lock:
                if 0 <= index < len(self._runs):
                    return self._runs[index]
            if attempt == 0 and not self.refresh():
                break
        raise KeyError(f"There is no run {index}.")

    def source(self) -> dict[str, Any]:
        """Describes where the runs come from, for the run list.

        Returns:
            ``directory`` (the watched directory, or ``None``) and ``waiting``
            (the given run directories whose archive has not been written yet).
        """

        return {
            "directory": self.watch_directory,
            "waiting": [
                directory
                for archive, directory, _name in self._given
                if not os.path.isfile(archive)
            ],
        }


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


def insertion_rates_latex(
    columns: list[tuple[str, dict[str, dict[str, int]]]],
) -> str:
    """Writes insertion rates as the LaTeX table ``analyze_genome_generation`` prints.

    Each operator gets a block of rows -- the share of its genomes that became a
    global best, a local best, were inserted and were discarded, to three
    decimals -- with a column per group or run, and ``-`` where a column has no
    genomes from that operator. The text matches the script's output line for
    line (its spacing included), so a table can stand in for one it printed.

    Args:
        columns: Each column's label and its insert-type counts keyed by
            operator, each operator's counts including a ``total``.

    Returns:
        The ``tabular``'s LaTeX source, ending in a newline.
    """

    lines = [
        "\\begin{tabular}{lp{2cm}" + "p{1.5cm}" * len(columns) + "}",
        "\\toprule",
        " &",
        "".join(" & {\\bf " + label.replace("_", "\\_") + " }" for label, _ in columns)
        + "\\\\",
        "\\midrule",
    ]

    operators = sorted({operator for _, counts in columns for operator in counts})
    for operator in operators:
        cells: dict[str, str] = {outcome: "" for outcome in INSERTION_OUTCOMES}
        for _, counts in columns:
            operator_counts = counts.get(operator)
            for outcome in INSERTION_OUTCOMES:
                if operator_counts:
                    share = operator_counts.get(outcome, 0) / operator_counts["total"]
                    cells[outcome] += f" & {share:.3f}"
                else:
                    cells[outcome] += " & -"

        cleaned = operator.replace("n_ary", "n-ary").replace("_", "\\\\")
        lines += [
            "\\multirowcell{4}{"
            + cleaned
            + "} & global best"
            + cells["global_best"]
            + " \\\\",
            "& local best" + cells["local_best"] + " \\\\",
            "& inserted" + cells["inserted"] + "\\\\",
            "& discarded" + cells["discarded"] + " \\\\",
            "\\hline",
        ]

    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


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
    """The data behind the dashboard's JSON API.

    Attributes:
        registry: The runs being served.
        renderer: Renders genome images.
    """

    def __init__(self, registry: RunRegistry, renderer: RenderService) -> None:
        """Creates the viewer.

        Args:
            registry: The runs to serve.
            renderer: The image render service.
        """

        self.registry = registry
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

        return self.registry.run(index)

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
        """Lists every served run, first picking up any newly written runs.

        Returns:
            ``runs`` (each run's summary), ``groups`` (the group names) and
            ``source`` (see :meth:`RunRegistry.source`).
        """

        self.registry.refresh()
        return {
            "runs": [self._summary(run) for run in self.registry.runs],
            "groups": list(self.registry.groups),
            "source": self.registry.source(),
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

    def _comparison_entries(self) -> list[tuple[str, str, list[Run]]]:
        """Lists what run comparisons compare: each group, then each ungrouped run.

        Returns:
            ``(name, kind, runs)`` for every ``--groups`` group (``kind``
            ``"group"``, in order, possibly with no runs), followed by every run
            that belongs to no group, on its own (``kind`` ``"run"``).
        """

        runs = self.registry.runs
        groups = self.registry.groups
        grouped = {index for members in groups.values() for index in members}
        entries: list[tuple[str, str, list[Run]]] = [
            (name, "group", [runs[index] for index in members])
            for name, members in groups.items()
        ]
        entries += [
            (run.name, "run", [run]) for run in runs if run.index not in grouped
        ]
        return entries

    def groups_payload(self, metric: str = "best", conf: str = "std") -> dict[str, Any]:
        """Compares groups of runs.

        Each ``--groups`` group is compared, and so is every run that belongs to
        no group, on its own.

        Args:
            metric: The history column whose mean and band are charted.
            conf: The band: ``"std"`` or ``"95ci"``.

        Returns:
            ``metric``, ``conf``, the history ``metrics`` available, and per group
            its ``name``, ``kind`` (``"group"``, or ``"run"`` for a run in no
            group), ``runs``, aggregated ``history`` (or ``history_error``) and
            summary statistics of each run's best ``loss`` and
            ``target_metric``.

        Raises:
            ValueError: If ``conf`` is not ``"std"`` or ``"95ci"``.
        """

        if conf not in ("std", "95ci"):
            raise ValueError("conf must be 'std' or '95ci'.")

        self.registry.refresh()
        available_metrics: set[str] = set()
        groups = []
        for name, kind, runs in self._comparison_entries():
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
            for run in runs:
                try:
                    with GenomeArchive.open_readonly(run.archive_path) as reader:
                        best_loss = reader.best_value("loss", higher_is_better=False)
                        best_target = reader.best_value(
                            "target_metric", higher_is_better=True
                        )
                except sqlite3.DatabaseError as error:
                    logger.warning("Could not read {}: {}", run.archive_path, error)
                    continue
                if best_loss is not None:
                    best_losses.append(best_loss["value"])
                if best_target is not None:
                    best_targets.append(best_target["value"])

            groups.append(
                {
                    "name": name,
                    "kind": kind,
                    "runs": [{"index": run.index, "name": run.name} for run in runs],
                    "history": history,
                    "history_error": history_error,
                    "best_loss": summary_statistics(best_losses),
                    "best_target_metric": summary_statistics(best_targets),
                }
            )

        return {
            "metric": metric,
            "conf": conf,
            "metrics": sorted(available_metrics),
            "groups": groups,
        }

    def insertion_rates_payload(
        self, run_index: int | None = None, group: str | None = None
    ) -> dict[str, Any]:
        """Tabulates how the genomes each operator generated were inserted.

        Counting follows ``src.analysis.analyze_genome_generation``: a genome
        counts once, under its insert type, for every operator that generated
        it, and an operator's rates are shares of the genomes it generated. The
        columns depend on what is asked for: one run; one group (summed over
        its runs) followed by each of its runs; or, by default, every group and
        then every run that belongs to no group, as the script's ``--groups``
        table does.

        Args:
            run_index: The run to tabulate, if any.
            group: The group to tabulate, if any.

        Returns:
            ``scope`` (``kind`` ``"run"`` with the run's ``index`` and ``name``,
            ``"group"`` with its ``name``, or ``"all"``), ``outcomes`` (the
            standard insert types, then any others recorded), ``operators``
            (sorted), ``columns`` (each with a ``label``, ``kind``, its ``runs``,
            their ``genomes`` count and the insert-type ``counts`` keyed by
            operator, each including a ``total``), ``latex`` (see
            :func:`insertion_rates_latex`; for every group it has only the group
            columns, as the script's ``--groups`` table does, unless there are
            no groups) and ``errors`` (runs whose archive could not be read).

        Raises:
            ValueError: If both a run and a group are given.
            KeyError: If there is no such run or group.
        """

        if run_index is not None and group is not None:
            raise ValueError("Give a run or a group, not both.")

        scope: dict[str, Any]
        specs: list[tuple[str, str, list[Run]]]
        if run_index is not None:
            run = self.run(run_index)
            scope = {"kind": "run", "index": run.index, "name": run.name}
            specs = [(run.name, "run", [run])]
        elif group is not None:
            self.registry.refresh()
            groups = self.registry.groups
            if group not in groups:
                raise KeyError(f"There is no group {group!r}.")
            all_runs = self.registry.runs
            members = [all_runs[index] for index in groups[group]]
            scope = {"kind": "group", "name": group}
            specs = [(group, "group", members)]
            specs += [(member.name, "run", [member]) for member in members]
        else:
            self.registry.refresh()
            scope = {"kind": "all"}
            specs = self._comparison_entries()

        # each run is read once, even when it appears in more than one column
        read: dict[int, tuple[int, dict[str, dict[str, int]]] | None] = {}
        errors: list[dict[str, Any]] = []
        columns: list[dict[str, Any]] = []
        for label, kind, runs in specs:
            genomes = 0
            counts: dict[str, dict[str, int]] = {}
            for run in runs:
                if run.index not in read:
                    try:
                        with GenomeArchive.open_readonly(run.archive_path) as reader:
                            read[run.index] = (reader.count(), reader.operator_counts())
                    except sqlite3.DatabaseError as error:
                        logger.warning("Could not read {}: {}", run.archive_path, error)
                        errors.append(
                            {"index": run.index, "name": run.name, "error": str(error)}
                        )
                        read[run.index] = None
                entry = read[run.index]
                if entry is None:
                    continue
                genomes += entry[0]
                for operator, outcomes in entry[1].items():
                    totals = counts.setdefault(operator, {})
                    for outcome, count in outcomes.items():
                        totals[outcome] = totals.get(outcome, 0) + count
            for totals in counts.values():
                totals["total"] = sum(totals.values())
            columns.append(
                {
                    "label": label,
                    "kind": kind,
                    "runs": [{"index": run.index, "name": run.name} for run in runs],
                    "genomes": genomes,
                    "counts": counts,
                }
            )

        recorded = {
            outcome
            for column in columns
            for outcomes in column["counts"].values()
            for outcome in outcomes
        }
        # like the script's --groups table, a table of every group leaves out runs in no group
        latex_columns = [
            column
            for column in columns
            if scope["kind"] != "all"
            or column["kind"] == "group"
            or not self.registry.group_names
        ]
        return {
            "scope": scope,
            "outcomes": list(INSERTION_OUTCOMES)
            + sorted(recorded - set(INSERTION_OUTCOMES) - {"total"}),
            "operators": sorted(
                {operator for column in columns for operator in column["counts"]}
            ),
            "columns": columns,
            "latex": insertion_rates_latex(
                [(column["label"], column["counts"]) for column in latex_columns]
            ),
            "errors": errors,
        }


#: API routes: a path pattern and the handler method serving it.
_ROUTES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/api/runs"), "_api_runs"),
    (re.compile(r"/api/groups"), "_api_groups"),
    (re.compile(r"/api/insertion_rates"), "_api_insertion_rates"),
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
    server_version = "EXAQCMonitor/1"

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

    def _api_insertion_rates(self, match: re.Match[str], query: dict[str, str]) -> None:
        """Serves insertion rates for a ``run`` (index), a ``group`` (name) or every group.

        Args:
            match: The matched route.
            query: The query parameters.

        Returns:
            None.
        """

        run_index = (
            _query_int(query, "run", -1, minimum=0) if query.get("run") else None
        )
        self._send_json(
            self.server.viewer.insertion_rates_payload(
                run_index=run_index, group=query.get("group") or None
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
    runs: list[str] | None = None,
    directory: str | None = None,
    groups: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
    render_processes: int = 1,
    rescan_interval: float = RESCAN_INTERVAL_SECONDS,
) -> None:
    """Serves the dashboard until interrupted.

    Args:
        runs: Run output directories (or their archive files) to serve; a run
            whose archive has not been written yet (even one whose directory
            does not exist yet) appears once it is.
        directory: A directory to watch instead: every run below it is served,
            including runs started while the dashboard is running.
        groups: Substrings grouping runs for comparison.
        host: The address to listen on.
        port: The port to listen on (``0`` picks a free port).
        open_browser: Whether to open the dashboard in a web browser.
        render_processes: Worker processes rendering images.
        rescan_interval: The least time between scans for new runs, in seconds.

    Returns:
        None. Runs until interrupted with Ctrl+C.

    Raises:
        ValueError: If both or neither of ``runs`` and ``directory`` are given.
        FileNotFoundError: If the watched directory does not exist.
        NotADirectoryError: If the watched path is not a directory.
        OSError: If the server cannot listen on ``host:port``.
    """

    registry = RunRegistry(
        run_directories=runs,
        watch_directory=directory,
        groups=groups,
        rescan_interval=rescan_interval,
    )

    renderer = RenderService(processes=render_processes)
    try:
        server = ArtifactViewerServer((host, port), ArtifactViewer(registry, renderer))
    except OSError:
        renderer.close()
        raise
    url = f"http://{host}:{server.server_address[1]}/"

    if registry.watch_directory is not None:
        logger.info(
            "Watching {} for runs ({} found so far), serving at {} -- press Ctrl+C to stop.",
            registry.watch_directory,
            len(registry.runs),
            url,
        )
    else:
        logger.info(
            "Serving {} run(s) at {} -- press Ctrl+C to stop.", len(registry.runs), url
        )
        waiting = registry.source()["waiting"]
        if waiting:
            logger.info(
                "Waiting for {} to be written in: {}",
                ARCHIVE_FILENAME,
                ", ".join(waiting),
            )
    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping the dashboard.")
    finally:
        server.server_close()
        renderer.close()
