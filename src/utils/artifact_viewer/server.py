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
import shlex
import sqlite3
import statistics
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.utils.annotations import AnnotationStore
from src.utils.genome_archive import ARCHIVE_FILENAME, GenomeArchive

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

#: The most points a run comparison charts. Runs record a step whenever their
#: population changed, so comparing long runs would otherwise return a point per
#: insertion per run.
MAX_COMPARISON_POINTS = 500


def _aggregate_series(
    series: list[dict[str, list[Any]]], conf: str
) -> dict[str, Any] | None:
    """Summarizes one metric across several runs, insertion by insertion.

    A run records a step only when its population changed, so two runs of the
    same length still record different steps and intersecting them would leave
    almost nothing to plot. Instead every recorded step is kept and each run's
    last recorded value is carried forward across it. That is exact rather than
    interpolated: a run with no row at a step is a run whose population did not
    change there, so its statistics are unchanged too.

    A run contributes only across its own lifetime: it joins the average when it
    reaches a step and drops out past the last step it recorded. Carrying a value
    forward within a run is exact, but carrying it beyond the run's end is not --
    a search that has only reached its three hundredth insertion has not levelled
    off at its thousandth, and averaging it in there would say that it had.

    Args:
        series: Each run's population series, as
            :meth:`~src.utils.genome_archive.GenomeArchive.population_series`
            returns them.
        conf: The band around the mean: ``"std"`` for one standard deviation, or
            ``"95ci"`` for 1.96 standard errors.

    Returns:
        The ``step`` values with the ``mean``, ``low`` and ``high`` at each, how
        many runs were averaged at each (``runs_at_step``) and how many are
        being compared (``n_runs``), or ``None`` if no run recorded a value.
    """

    steps = sorted({step for run in series for step in run["step"]})
    if not steps:
        return None
    if len(steps) > MAX_COMPARISON_POINTS:
        stride = len(steps) / MAX_COMPARISON_POINTS
        steps = [
            steps[min(int(index * stride), len(steps) - 1)]
            for index in range(MAX_COMPARISON_POINTS)
        ]

    recorded = [
        [
            (step, value)
            for step, value in zip(run["step"], run["best"])
            if value is not None
        ]
        for run in series
    ]
    cursors = [0] * len(recorded)
    carried: list[float | None] = [None] * len(recorded)
    last_steps = [pairs[-1][0] if pairs else None for pairs in recorded]

    kept: list[int] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    counts: list[int] = []
    for step in steps:
        for index, pairs in enumerate(recorded):
            while cursors[index] < len(pairs) and pairs[cursors[index]][0] <= step:
                carried[index] = pairs[cursors[index]][1]
                cursors[index] += 1
            if last_steps[index] is None or step > last_steps[index]:
                carried[index] = None

        values = [value for value in carried if value is not None]
        if not values:
            continue

        mean = statistics.fmean(values)
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        spread = (
            deviation
            if conf == "std"
            else 1.96 * deviation / math.sqrt(max(len(values), 1))
        )
        kept.append(step)
        means.append(mean)
        lows.append(mean - spread)
        highs.append(mean + spread)
        counts.append(len(values))

    if not kept:
        return None

    return {
        "step": kept,
        "mean": means,
        "low": lows,
        "high": highs,
        "runs_at_step": counts,
        "n_runs": len(series),
    }


def default_comparison_metric(available: list[str]) -> str | None:
    """Chooses the metric a run comparison charts when the caller names none.

    ``target_metric`` is what a search is ultimately judged on, so it is
    preferred; ``loss`` is the fallback, since every task's population is ranked
    by it. Anything the runs recorded will do rather than charting nothing.

    Args:
        available: Every metric the compared runs recorded.

    Returns:
        The metric to chart, or ``None`` when the runs recorded nothing.
    """

    for preferred in ("target_metric", "loss"):
        if preferred in available:
            return preferred
    return available[0] if available else None


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


def query_int(
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
        allow_annotations: Whether notes and tags may be written; they can always
            be read.
    """

    def __init__(
        self,
        registry: RunRegistry,
        renderer: RenderService,
        allow_annotations: bool = False,
    ) -> None:
        """Creates the viewer.

        Args:
            registry: The runs to serve.
            renderer: The image render service.
            allow_annotations: Whether notes and tags may be written. Reading them
                is always allowed, and writing them changes only each run's
                ``annotations.sqlite`` -- never its archive.
        """

        self.registry = registry
        self.renderer = renderer
        self.allow_annotations = allow_annotations

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
            The run's summary plus its fitness keys, everything its genomes can
            be charted by (with the shorter ``primary_metrics`` to offer first),
            the values they can be filtered by, the ``unarchived_parents``
            (parents of stored genomes that are not stored themselves, i.e. the
            seed genome), the ``island_topology`` an island search recorded
            (``None`` otherwise), the ``speciation`` config a speciation search
            recorded (``None`` otherwise), and whether it recorded a search
            history. An archive that cannot be read fully is reported under
            ``error``, with those fields empty.

        Raises:
            KeyError: If there is no such run.
        """

        run = self.run(index)
        payload = self._summary(run)
        payload.update(
            {
                "fitness_keys": [],
                "metrics": [],
                "primary_metrics": [],
                "filter_options": {},
                "unarchived_parents": [],
                "island_topology": None,
                "speciation": None,
            }
        )
        try:
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                payload["fitness_keys"] = reader.fitness_keys()
                payload["metrics"] = reader.series_metrics()
                payload["primary_metrics"] = reader.primary_series_metrics()
                payload["filter_options"] = reader.filter_options()
                payload["unarchived_parents"] = reader.unarchived_parents()
                info = reader.run_info()
                payload["island_topology"] = info.get("island_topology")
                payload["speciation"] = info.get("speciation")
        except sqlite3.DatabaseError as error:
            # An archive the viewer cannot read in full still lists and browses:
            # the run page falls back to what its summary holds rather than
            # failing outright, as the run list and progress chart already do.
            logger.warning("Could not read {}: {}", run.archive_path, error)
            payload["error"] = str(error)
        payload["has_history"] = self._recorded_steps(run) > 0
        payload["annotations_enabled"] = self.allow_annotations
        return payload

    def _recorded_steps(self, run: Run) -> int:
        """Counts the population changes a run recorded.

        Args:
            run: The run to read.

        Returns:
            How many steps its archive holds, and zero when the archive cannot
            be read.
        """

        try:
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                return int(
                    reader.connection.execute(
                        "SELECT count(*) FROM population_events"
                    ).fetchone()[0]
                )
        except sqlite3.DatabaseError as error:
            logger.warning("Could not read {}'s population events: {}", run.name, error)
            return 0

    def genomes_payload(self, index: int, query: dict[str, str]) -> dict[str, Any]:
        """Lists a page of a run's genomes.

        Args:
            index: The run's index.
            query: ``sort`` (a fitness key or summary column, default ``loss``),
                ``desc`` (``1`` to sort descending), ``offset``, ``limit``, the
                filters ``insert_type``, ``generated_by``, ``crossover_type``,
                ``island`` and ``species``, and ``max_genome``: list only genomes
                numbered at most this. A client paging through a live run passes
                the first page's ``max_genome_number`` back, so genomes saved in
                between don't shift later pages (repeating or skipping rows).

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
        offset = query_int(query, "offset", 0, minimum=0)
        limit = query_int(query, "limit", 50, minimum=1, maximum=MAX_PAGE_SIZE)
        max_genome = (
            query_int(query, "max_genome", -1, minimum=0)
            if query.get("max_genome")
            else None
        )
        filters: dict[str, Any] = {
            "insert_type": query.get("insert_type") or None,
            "generated_by": query.get("generated_by") or None,
            "crossover_type": query.get("crossover_type") or None,
            "island": (
                query_int(query, "island", -1, minimum=0)
                if query.get("island")
                else None
            ),
            "species": (
                query_int(query, "species", -1, minimum=0)
                if query.get("species")
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
            operator), ``crossover_type``, ``island`` and ``species``.

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

    def history_payload(self, index: int, metric: str | None = None) -> dict[str, Any]:
        """Returns a run's search progress, recomputed from its archive.

        The statistics are not stored: the population's membership is replayed
        from the recorded changes and summarized over whichever genomes were
        alive at each step. That is what lets any metric a run's genomes recorded
        be charted -- a loss, a return, a fidelity, a gate count -- rather than
        only a fixed set decided while the search was running.

        Args:
            index: The run's index.
            metric: The metric to summarize; ``loss`` when the run recorded it,
                otherwise the first available, when not given.

        Returns:
            ``columns`` (``step``, ``population_size``, ``best``, ``mean`` and
            ``worst``, empty when the run recorded no population changes),
            ``metric`` (the one summarized, ``None`` when there is nothing to
            summarize) and ``metrics`` (everything that could be charted).

        Raises:
            KeyError: If there is no such run.
            ValueError: If ``metric`` was not recorded by the run's genomes.
        """

        run = self.run(index)
        empty: dict[str, Any] = {
            "columns": {},
            "metric": None,
            "metrics": [],
            "primary_metrics": [],
        }

        try:
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                available = reader.series_metrics()
                primary = reader.primary_series_metrics()
                if not available:
                    return empty
                chosen = metric or ("loss" if "loss" in available else available[0])
                if chosen not in available:
                    raise ValueError(
                        f"{chosen!r} was not recorded; available: {', '.join(available)}."
                    )
                columns = reader.population_series(
                    chosen, higher_is_better=higher_is_better(chosen)
                )
        except sqlite3.DatabaseError as error:
            logger.warning("Could not read {}'s search progress: {}", run.name, error)
            return empty

        return {
            "columns": columns,
            "metric": chosen,
            "metrics": available,
            "primary_metrics": primary,
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
            ``summary``, the full serialized ``genome``, its ``children``, the
            ``parent_islands`` (each parent's island, in ``summary["parents"]``
            order), ``parent_species`` (each parent's species, same order) and
            ready-to-run ``commands``.

        Raises:
            KeyError: If there is no such run or genome.
        """

        run = self.run(index)
        with GenomeArchive.open_readonly(run.archive_path) as reader:
            genome = reader.get_genome_dict(genome_number)
            summary = reader.get_summary(genome_number)
            return {
                "summary": summary,
                "genome": genome,
                "children": reader.children(genome_number),
                # beside the parents rather than in the summary, which every
                # genome listing shares
                "parent_islands": reader.islands_of(summary["parents"]),
                "parent_species": reader.species_of(summary["parents"]),
                "commands": genome_commands(run, genome),
            }

    def require_annotations(self) -> None:
        """Checks that notes and tags may be written here.

        Returns:
            None.

        Raises:
            PermissionError: If annotations were not allowed when the server was
                started.
        """

        if not self.allow_annotations:
            raise PermissionError(
                "Annotations are read-only here: start the dashboard (or exaqc_mcp) "
                "with --allow_annotations to write notes and tags."
            )

    def _writable_run(self, index: int, genome_number: int | None) -> Run:
        """Checks that an annotation may be written, and that what it is about exists.

        Args:
            index: The run's index.
            genome_number: The genome the annotation is about, or ``None`` for the
                run as a whole.

        Returns:
            The run.

        Raises:
            PermissionError: If annotations may not be written here.
            KeyError: If there is no such run, or no such genome in it.
        """

        self.require_annotations()
        run = self.run(index)
        if genome_number is not None:
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                found = reader.connection.execute(
                    "SELECT 1 FROM genomes WHERE genome_number = ?",
                    (int(genome_number),),
                ).fetchone()
            if found is None:
                raise KeyError(f"{run.name} has no genome {int(genome_number)}.")
        return run

    def annotations_payload(
        self,
        index: int,
        genome_number: int | None = None,
        tag: str | None = None,
        include_removed: bool = False,
    ) -> dict[str, Any]:
        """Returns the notes and tags recorded for a run, or for one of its genomes.

        Args:
            index: The run's index.
            genome_number: Only return this genome's notes and tags; every note
                and tag in the run when not given.
            tag: Only return tags with this name.
            include_removed: Also return tags that were removed, with when and by
                whom.

        Returns:
            ``enabled`` (whether annotations may be written here),
            ``genome_number``, ``notes`` (a note about the run as a whole has no
            genome number) and ``tags``.

        Raises:
            KeyError: If there is no such run.
        """

        store = AnnotationStore.beside(self.run(index).archive_path)
        return {
            "enabled": self.allow_annotations,
            "genome_number": genome_number,
            "notes": store.notes(genome_number=genome_number),
            "tags": store.tags(
                genome_number=genome_number, tag=tag, include_removed=include_removed
            ),
        }

    def add_note(
        self,
        index: int,
        text: str,
        source: str,
        author: str | None = None,
        genome_number: int | None = None,
    ) -> dict[str, Any]:
        """Records a note about a run, or about one of its genomes.

        Args:
            index: The run's index.
            text: What to note.
            source: Where the note is written from (``mcp`` or ``dashboard``).
            author: The writer's name, if they gave one.
            genome_number: The genome the note is about, or ``None`` for the run.

        Returns:
            The note as recorded.

        Raises:
            PermissionError: If annotations may not be written here.
            KeyError: If there is no such run or genome.
            ValueError: If the note is empty or too long, or the source or author
                is invalid.
        """

        run = self._writable_run(index, genome_number)
        return AnnotationStore.beside(run.archive_path).add_note(
            text, source, author, genome_number
        )

    def add_tag(
        self,
        index: int,
        genome_number: int,
        tag: str,
        source: str,
        author: str | None = None,
    ) -> dict[str, Any]:
        """Tags a genome, leaving it as it is if it already carries the tag.

        Args:
            index: The run's index.
            genome_number: The genome to tag.
            tag: The tag.
            source: Where the tag is written from (``mcp`` or ``dashboard``).
            author: The writer's name, if they gave one.

        Returns:
            The tag that now applies, with ``created`` saying whether it is new.

        Raises:
            PermissionError: If annotations may not be written here.
            KeyError: If there is no such run or genome.
            ValueError: If the tag, source or author is invalid.
        """

        run = self._writable_run(index, genome_number)
        return AnnotationStore.beside(run.archive_path).add_tag(
            genome_number, tag, source, author
        )

    def remove_tag(
        self,
        index: int,
        genome_number: int,
        tag: str,
        source: str,
        author: str | None = None,
    ) -> dict[str, Any]:
        """Removes a tag from a genome, keeping the record that it applied.

        Args:
            index: The run's index.
            genome_number: The genome to untag.
            tag: The tag to remove.
            source: Where the removal is made from (``mcp`` or ``dashboard``).
            author: The remover's name, if they gave one.

        Returns:
            The tag, stamped with its removal.

        Raises:
            PermissionError: If annotations may not be written here.
            KeyError: If there is no such run or genome, or the genome does not
                carry the tag.
            ValueError: If the tag, source or author is invalid.
        """

        run = self._writable_run(index, genome_number)
        return AnnotationStore.beside(run.archive_path).remove_tag(
            genome_number, tag, source, author
        )

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

    def groups_payload(
        self, metric: str | None = None, conf: str = "std"
    ) -> dict[str, Any]:
        """Compares groups of runs.

        Each ``--groups`` group is compared, and so is every run that belongs to
        no group, on its own. Every group's curve is recomputed from its runs'
        archives, so any metric the runs' genomes recorded can be compared.

        Args:
            metric: The metric whose mean and band are charted, as
                :meth:`~src.utils.genome_archive.GenomeArchive.series_metrics`
                lists them; chosen by :func:`default_comparison_metric` when not
                given, so the comparison charts something rather than nothing.
            conf: The band: ``"std"`` or ``"95ci"``.

        Returns:
            ``metric`` (the one charted), ``conf``, the ``metrics`` available
            across the runs with the shorter ``primary_metrics`` to offer first,
            and per group its ``name``, ``kind`` (``"group"``, or ``"run"`` for a
            run in no group), ``runs``, aggregated ``history`` (or
            ``history_error``) and summary statistics of each run's best ``loss``
            and ``target_metric``.

        Raises:
            ValueError: If ``conf`` is not ``"std"`` or ``"95ci"``.
        """

        if conf not in ("std", "95ci"):
            raise ValueError("conf must be 'std' or '95ci'.")

        self.registry.refresh()
        entries = self._comparison_entries()

        # Which metric to chart is chosen from everything the compared runs
        # recorded, so it cannot be decided while walking the groups: what each
        # run has is read first, and kept so the groups need not ask again.
        recorded: dict[str, list[str]] = {}
        offered: dict[str, list[str]] = {}
        for _, _, runs in entries:
            for run in runs:
                if run.archive_path in recorded:
                    continue
                try:
                    with GenomeArchive.open_readonly(run.archive_path) as reader:
                        recorded[run.archive_path] = reader.series_metrics()
                        offered[run.archive_path] = reader.primary_series_metrics()
                except sqlite3.DatabaseError as error:
                    logger.warning("Could not read {}: {}", run.archive_path, error)
                    recorded[run.archive_path] = []
                    offered[run.archive_path] = []

        available_metrics = sorted(
            {name for names in recorded.values() for name in names}
        )
        primary_metrics = sorted({name for names in offered.values() for name in names})
        charted = metric or default_comparison_metric(available_metrics)

        groups = []
        for name, kind, runs in entries:
            series: list[dict[str, list[Any]]] = []
            for run in runs:
                if charted is None or charted not in recorded.get(run.archive_path, []):
                    continue
                try:
                    with GenomeArchive.open_readonly(run.archive_path) as reader:
                        series.append(
                            reader.population_series(
                                charted, higher_is_better=higher_is_better(charted)
                            )
                        )
                except sqlite3.DatabaseError as error:
                    logger.warning("Could not read {}: {}", run.archive_path, error)

            history = None
            history_error = None
            if series:
                history = _aggregate_series(series, conf)
                if history is None:
                    history_error = f"No run recorded a value for {charted!r}."
            elif charted is None:
                history_error = "These runs recorded no search progress."
            else:
                history_error = f"No run recorded {charted!r}."

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
            "metric": charted,
            "conf": conf,
            "metrics": available_metrics,
            "primary_metrics": primary_metrics,
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
