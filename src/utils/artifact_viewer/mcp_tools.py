"""Read-only analysis tools an agent calls to interrogate EXAQC runs.

These are the implementations behind the MCP interface
(:mod:`src.utils.artifact_viewer.mcp_app`); they are plain methods returning
JSON-safe dicts, so they can be tested and benchmarked without a transport.

Every tool answers from the same data layer the dashboard uses
(:class:`~src.utils.artifact_viewer.server.ArtifactViewer`), and every result
carries a ``dashboard_url`` pointing at the page showing the same thing.

Two constraints shape the design, both measured on a 20k-genome archive:

* a whole-run point series serializes to about 1.8 MiB (roughly 461k tokens),
  so nothing returns a raw per-genome series -- series are downsampled and
  listings are capped and paged;
* queries spanning several runs cannot ``ATTACH`` the archives, because SQLite
  allows at most 10 attached databases and that ceiling is fixed at compile
  time. Cross-run SQL instead copies the per-run summary rows into one
  in-memory database, which costs about 5.5 ms per run.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import time
from typing import Any
from urllib.parse import quote

from src.utils.artifact_viewer.server import (
    ArtifactViewer,
    RenderService,
    Run,
    RunRegistry,
    higher_is_better,
)
from src.utils.genome_archive import GenomeArchive

#: Rows a listing returns when the caller does not say, and the most it may ask
#: for: 50 rows is roughly 4k tokens, 200 roughly 16k.
DEFAULT_ROWS = 50
MAX_ROWS = 200

#: The most points any series returns; longer series are strided down to this.
MAX_SERIES_POINTS = 500

#: Soft ceiling on a tool's serialized result. Rows are dropped past this and
#: the result says so, rather than returning a reply too large to read.
MAX_RESPONSE_BYTES = 64 * 1024

#: The most runs one cross-run query may roll up.
MAX_RUNS_PER_QUERY = 64

#: How long a ``query_sql`` statement may run before it is interrupted.
QUERY_TIMEOUT_SECONDS = 10.0

#: Genomes sampled when summarizing gate usage, which needs the stored JSON.
GATE_SAMPLE_SIZE = 200

#: SQLite authorizer actions a read-only query may perform. Everything else --
#: writes, ``ATTACH``, ``PRAGMA``, transactions -- is denied by the authorizer
#: rather than by inspecting the SQL text.
_ALLOWED_SQL_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)

#: The columns a rolled-up cross-run ``genomes`` table exposes to SQL.
_ROLLUP_COLUMNS = (
    "run",
    "genome_number",
    "insertion",
    "insert_type",
    "generated_by",
    "crossover_type",
    "island",
    "n_gates",
    "n_enabled_gates",
    "n_parameters",
    "fitness",
    "loss",
    "target_metric",
)


#: Metadata entries holding a per-epoch or per-episode metric series. Which of
#: these a genome has depends on its task, so they are discovered, not assumed.
_METRIC_SERIES = re.compile(r"_(epoch|episode)_metrics$")


def _flatten_metrics(record: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flattens one epoch's or episode's metrics into ``path -> number`` pairs.

    A metric is recorded as a bare number (``loss``), as a wrapper around a mean
    (``fidelity: {"mean": ...}``), or as a nested breakdown (a per-class accuracy
    holding ``acc``, ``correct`` and ``total`` per class as well as a ``mean``),
    so the record is walked to whatever depth it has.

    Args:
        record: One epoch's or episode's metrics.
        prefix: The dotted path this record sits under, when recursing.

    Returns:
        The numeric values, keyed by dotted path.
    """

    flattened: dict[str, float] = {}
    for name, value in (record or {}).items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[path] = value
    return flattened


def _markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    """Renders rows as a Markdown table, so an agent can quote a result directly.

    Args:
        columns: Column headings.
        rows: One list of cell values per row.

    Returns:
        The table, or a note when there are no rows.
    """

    if not rows:
        return "_no rows_"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append(
            "| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |"
        )
    return "\n".join(lines)


def _downsample(values: list[Any], limit: int = MAX_SERIES_POINTS) -> list[Any]:
    """Strides a series down to at most ``limit`` points, keeping the last one.

    Args:
        values: The series to shorten.
        limit: The most points to keep.

    Returns:
        The series itself when it is short enough, otherwise an evenly strided
        sample of it ending at the final value.
    """

    if len(values) <= limit:
        return values
    stride = len(values) / limit
    sampled = [values[int(index * stride)] for index in range(limit - 1)]
    sampled.append(values[-1])
    return sampled


def _statistics(values: list[float]) -> dict[str, Any]:
    """Summarizes numbers, including the quartiles a distribution is read from.

    Args:
        values: The numbers to summarize; non-numeric entries are ignored.

    Returns:
        ``n``, ``min``, ``p25``, ``median``, ``p75``, ``max``, ``mean`` and
        ``std``, or ``n`` alone when nothing numeric was given.
    """

    numbers = sorted(
        float(value) for value in values if isinstance(value, (int, float))
    )
    if not numbers:
        return {"n": 0}

    def percentile(fraction: float) -> float:
        """Returns the value at a fraction through the sorted numbers."""
        position = min(len(numbers) - 1, max(0, round(fraction * (len(numbers) - 1))))
        return numbers[position]

    return {
        "n": len(numbers),
        "min": numbers[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": numbers[-1],
        "mean": statistics.fmean(numbers),
        "std": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
    }


class DashboardTools:
    """The read-only tools exposed over MCP, backed by the dashboard's data layer.

    Attributes:
        registry: The runs being served.
        viewer: The data layer answering run, genome and comparison questions.
        base_url: The dashboard's base URL, used to build deep links.
    """

    def __init__(
        self,
        registry: RunRegistry,
        renderer: RenderService | None = None,
        base_url: str = "http://127.0.0.1:8000",
    ) -> None:
        """Creates the tool set.

        Args:
            registry: The runs to serve.
            renderer: The image renderer the viewer uses; tools never render, so
                an inline renderer is created when none is given.
            base_url: The dashboard's base URL for deep links.
        """

        self.registry = registry
        self.viewer = ArtifactViewer(registry, renderer or RenderService(processes=0))
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(self, run: int | str) -> Run:
        """Finds a run by index or by name.

        Args:
            run: A run index, or a name (or unique part of one).

        Returns:
            The run.

        Raises:
            KeyError: If no run, or more than one run, matches.
        """

        if isinstance(run, int) or (isinstance(run, str) and run.lstrip("-").isdigit()):
            return self.registry.run(int(run))

        runs = self.registry.runs
        exact = [candidate for candidate in runs if candidate.name == run]
        if exact:
            return exact[0]
        partial = [candidate for candidate in runs if run in candidate.name]
        if len(partial) == 1:
            return partial[0]
        if not partial:
            raise KeyError(
                f"There is no run {run!r}. Served runs: "
                + ", ".join(candidate.name for candidate in runs[:20])
            )
        raise KeyError(
            f"{run!r} matches several runs: "
            + ", ".join(candidate.name for candidate in partial[:20])
        )

    def _url(self, fragment: str) -> str:
        """Builds a dashboard deep link.

        Args:
            fragment: The page's hash route, e.g. ``/run/3``.

        Returns:
            The absolute URL.
        """

        return f"{self.base_url}/#{fragment}"

    def _fit(self, payload: dict[str, Any], rows_key: str) -> dict[str, Any]:
        """Trims a result's rows until it fits the response budget.

        Args:
            payload: The result, which must hold a list under ``rows_key``.
            rows_key: The key holding the rows that may be dropped.

        Returns:
            The payload, with rows dropped and ``truncated`` set when it was too
            large to return whole.
        """

        rows = payload.get(rows_key) or []
        while rows and len(json.dumps(payload, default=str)) > MAX_RESPONSE_BYTES:
            del rows[len(rows) // 2 :]
            payload[rows_key] = rows
            payload["truncated"] = True
            payload["hint"] = (
                f"Result was too large; {len(rows)} of the rows are shown. "
                "Narrow it with filters, a smaller limit, or an aggregate query."
            )
        return payload

    def _summary_rows(self, run: Run) -> list[tuple[Any, ...]]:
        """Reads one run's per-genome summary rows for a cross-run roll-up.

        Args:
            run: The run to read.

        Returns:
            One tuple per genome, in :data:`_ROLLUP_COLUMNS` order after the run
            name.
        """

        with GenomeArchive.open_readonly(run.archive_path) as reader:
            return reader.connection.execute(
                "SELECT genome_number, insertion, insert_type, generated_by, "
                "crossover_type, island, n_gates, n_enabled_gates, n_parameters, "
                "fitness, json_extract(fitness, '$.loss'), "
                "json_extract(fitness, '$.target_metric') FROM genomes"
            ).fetchall()

    def _rollup(self, runs: list[Run]) -> sqlite3.Connection:
        """Copies several runs' summary rows into one in-memory database.

        This is how cross-run SQL is served: SQLite allows at most 10 attached
        databases, a compile-time ceiling that cannot be raised at runtime, so
        the rows are gathered instead of the files being attached.

        Args:
            runs: The runs to roll up.

        Returns:
            A connection holding ``runs`` and ``genomes`` tables, the latter with
            a ``run`` column naming each row's run.

        Raises:
            ValueError: If more than :data:`MAX_RUNS_PER_QUERY` runs are given.
        """

        if len(runs) > MAX_RUNS_PER_QUERY:
            raise ValueError(
                f"{len(runs)} runs is more than the {MAX_RUNS_PER_QUERY} a single "
                "query may roll up; select fewer runs."
            )

        memory = sqlite3.connect(":memory:")
        memory.execute(
            "CREATE TABLE runs(run TEXT, run_index INTEGER, task TEXT, "
            "task_target TEXT, strategy TEXT, genomes INTEGER)"
        )
        memory.execute(
            "CREATE TABLE genomes(run TEXT, genome_number INTEGER, insertion INTEGER, "
            "insert_type TEXT, generated_by TEXT, crossover_type TEXT, island INTEGER, "
            "n_gates INTEGER, n_enabled_gates INTEGER, n_parameters INTEGER, "
            "fitness TEXT, loss REAL, target_metric REAL)"
        )
        for run in runs:
            rows = self._summary_rows(run)
            memory.executemany(
                "INSERT INTO genomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(run.name, *row) for row in rows],
            )
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                info = reader.run_info()
            memory.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.name,
                    run.index,
                    info.get("task"),
                    info.get("task_target"),
                    info.get("population_strategy"),
                    len(rows),
                ),
            )
        memory.commit()
        return memory

    def _selected_runs(
        self, runs: list[int | str] | None, group: str | None
    ) -> list[Run]:
        """Resolves a run selection given as indexes/names, a group, or neither.

        Args:
            runs: Run indexes or names, if any.
            group: A ``--groups`` group name, if any.

        Returns:
            The selected runs; every served run when neither is given.

        Raises:
            KeyError: If a run or the group is unknown.
        """

        if runs:
            return [self._resolve(run) for run in runs]
        if group:
            members = self.registry.groups.get(group)
            if members is None:
                raise KeyError(
                    f"There is no group {group!r}. Groups: "
                    + (", ".join(self.registry.groups) or "none")
                )
            all_runs = self.registry.runs
            return [all_runs[index] for index in members]
        return self.registry.runs

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_runs(self, name_contains: str | None = None) -> dict[str, Any]:
        """Lists the runs being served, newest information first.

        Args:
            name_contains: Only list runs whose name contains this.

        Returns:
            ``rows`` (one summary per run), ``total``, a Markdown ``table`` and
            the dashboard ``dashboard_url``.
        """

        payload = self.viewer.runs_payload()
        rows = [
            run
            for run in payload["runs"]
            if not name_contains or name_contains in run["name"]
        ]
        table = _markdown_table(
            ["run", "task", "genomes", "best loss", "best target_metric", "groups"],
            [
                [
                    run["name"],
                    f"{run.get('task')}/{run.get('task_target')}",
                    run.get("genomes"),
                    (run.get("best_loss") or {}).get("value"),
                    (run.get("best_target_metric") or {}).get("value"),
                    ", ".join(run.get("groups") or []) or "-",
                ]
                for run in rows
            ],
        )
        return self._fit(
            {
                "rows": rows,
                "total": len(rows),
                "groups": payload["groups"],
                "table": table,
                "dashboard_url": self._url("/"),
            },
            "rows",
        )

    def describe_run(self, run: int | str) -> dict[str, Any]:
        """Describes one run: what it was, how big it is, and what can be filtered.

        Args:
            run: A run index or name.

        Returns:
            The run's summary and provenance, its fitness keys, the values its
            genomes can be filtered by, and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run.
        """

        resolved = self._resolve(run)
        payload = self.viewer.run_payload(resolved.index)
        payload["dashboard_url"] = self._url(f"/run/{resolved.index}")
        return payload

    # ------------------------------------------------------------------
    # Genomes
    # ------------------------------------------------------------------

    def list_genomes(
        self,
        run: int | str,
        sort: str = "loss",
        descending: bool = False,
        insert_type: str | None = None,
        generated_by: str | None = None,
        island: int | None = None,
        limit: int = DEFAULT_ROWS,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Lists a page of a run's genomes, sorted and filtered.

        Args:
            run: A run index or name.
            sort: A fitness key or summary column to sort by.
            descending: Whether to sort largest first.
            insert_type: Only list genomes inserted this way.
            generated_by: Only list genomes generated by this operator.
            island: Only list genomes from this island.
            limit: Rows to return, at most :data:`MAX_ROWS`.
            offset: Rows to skip.

        Returns:
            ``rows``, ``total``, a Markdown ``table`` and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run.
            ValueError: If the sort key or a filter is not valid.
        """

        resolved = self._resolve(run)
        query = {
            "sort": sort,
            "desc": "1" if descending else "0",
            "limit": str(min(int(limit), MAX_ROWS)),
            "offset": str(int(offset)),
        }
        for name, value in (
            ("insert_type", insert_type),
            ("generated_by", generated_by),
            ("island", island),
        ):
            if value is not None:
                query[name] = str(value)

        payload = self.viewer.genomes_payload(resolved.index, query)
        rows = payload["rows"]
        keys = sorted({key for row in rows for key in (row.get("fitness") or {})})
        table = _markdown_table(
            ["#", *keys, "insert type", "generated by", "gates"],
            [
                [
                    row["genome_number"],
                    *[(row.get("fitness") or {}).get(key) for key in keys],
                    row.get("insert_type"),
                    ", ".join(row.get("generated_by") or []),
                    row.get("n_gates"),
                ]
                for row in rows
            ],
        )
        payload["table"] = table
        payload["dashboard_url"] = self._url(f"/run/{resolved.index}")
        return self._fit(payload, "rows")

    def get_genome(self, run: int | str, genome_number: int) -> dict[str, Any]:
        """Returns one genome in full, with its family and ready-to-run commands.

        Args:
            run: A run index or name.
            genome_number: The genome to fetch.

        Returns:
            Its ``summary``, the serialized ``genome``, its ``children``,
            copy-ready ``commands`` and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run or genome.
        """

        resolved = self._resolve(run)
        payload = self.viewer.genome_payload(resolved.index, int(genome_number))
        payload["dashboard_url"] = self._url(
            f"/run/{resolved.index}/genome/{int(genome_number)}"
        )
        return payload

    def genome_metrics(
        self, run: int | str, genome_number: int, series: str | None = None
    ) -> dict[str, Any]:
        """Returns the per-epoch or per-episode metrics a genome recorded.

        What a genome records depends on its task, so nothing is assumed: every
        metadata entry named ``*_epoch_metrics`` or ``*_episode_metrics`` is
        returned as its own series, keyed by the ``epoch`` or ``episode`` column
        it carries. Reinforcement-learning genomes record their training and
        evaluation series at different cadences, so the series are kept separate
        rather than merged. Nested values (a per-class accuracy breakdown, or a
        metric wrapped in a ``mean``) are flattened to dotted paths.

        Args:
            run: A run index or name.
            genome_number: The genome whose training history is read.
            series: Only return this series, e.g. ``"validation_epoch_metrics"``.

        Returns:
            ``series`` (each with its ``step`` column, ``metrics`` and ``records``),
            the ``available`` series names, and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run or genome.
            ValueError: If the genome recorded no such series.
        """

        resolved = self._resolve(run)
        with GenomeArchive.open_readonly(resolved.archive_path) as reader:
            metadata = reader.get_genome_dict(int(genome_number)).get("metadata") or {}

        available = [
            name
            for name, value in metadata.items()
            if _METRIC_SERIES.search(name) and isinstance(value, list) and value
        ]
        if series is not None and series not in available:
            raise ValueError(
                f"{series!r} was not recorded; available: {', '.join(available) or 'none'}."
            )

        recorded = []
        for name in available if series is None else [series]:
            records = [_flatten_metrics(record) for record in metadata[name]]
            step = "episode" if name.endswith("_episode_metrics") else "epoch"
            metrics = sorted(
                {key for record in records for key in record if key != step}
            )
            recorded.append(
                {
                    "name": name,
                    "step": step,
                    "metrics": metrics,
                    "records": records,
                }
            )

        return self._fit(
            {
                "run": resolved.name,
                "genome_number": int(genome_number),
                "available": available,
                "series": recorded,
                "dashboard_url": self._url(
                    f"/run/{resolved.index}/genome/{int(genome_number)}"
                ),
            },
            "series",
        )

    def compare_genomes(self, run: int | str, a: int, b: int) -> dict[str, Any]:
        """Compares two genomes of a run gate by gate and value by value.

        Args:
            run: A run index or name.
            a: The first genome's number.
            b: The second genome's number.

        Returns:
            The ``gates`` diff (matched by innovation number), key-by-key
            ``fitness`` and ``hyperparameters``, and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run or genome.
        """

        resolved = self._resolve(run)
        payload = self.viewer.compare_payload(resolved.index, int(a), int(b))
        payload["dashboard_url"] = self._url(
            f"/run/{resolved.index}/compare/{int(a)}/{int(b)}"
        )
        return payload

    def genome_lineage(
        self, run: int | str, genome_number: int, depth: int = 5
    ) -> dict[str, Any]:
        """Traces a genome's ancestry back through the operators that made it.

        Args:
            run: A run index or name.
            genome_number: The genome to trace.
            depth: How many generations back to walk.

        Returns:
            ``nodes`` (each ancestor with its generation), ``edges``,
            the genome's ``children`` and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run or genome.
        """

        resolved = self._resolve(run)
        payload = self.viewer.ancestry_payload(
            resolved.index, int(genome_number), int(depth)
        )
        with GenomeArchive.open_readonly(resolved.archive_path) as reader:
            payload["children"] = reader.children(int(genome_number))
        payload["dashboard_url"] = self._url(
            f"/run/{resolved.index}/genome/{int(genome_number)}"
        )
        return self._fit(payload, "nodes")

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def fitness_summary(
        self,
        run: int | str,
        key: str = "loss",
        group_by: str | None = None,
    ) -> dict[str, Any]:
        """Summarizes the distribution of a fitness key across a run's genomes.

        Args:
            run: A run index or name.
            key: The fitness key to summarize.
            group_by: Optionally ``insert_type``, ``island`` or ``generated_by``,
                to summarize each group separately.

        Returns:
            ``overall`` statistics, per-group ``rows`` when grouping, the
            ``best`` genome, a Markdown ``table`` and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run.
            ValueError: If ``group_by`` is not a groupable column.
        """

        resolved = self._resolve(run)
        if group_by is not None and group_by not in (
            "insert_type",
            "island",
            "generated_by",
        ):
            raise ValueError("group_by must be insert_type, island or generated_by.")

        with GenomeArchive.open_readonly(resolved.archive_path) as reader:
            if group_by == "generated_by":
                rows = reader.connection.execute(
                    "SELECT json_each.value, json_extract(fitness, ?) "
                    "FROM genomes, json_each(genomes.generated_by)",
                    (f"$.{key}",),
                ).fetchall()
            elif group_by:
                rows = reader.connection.execute(
                    f"SELECT {group_by}, json_extract(fitness, ?) FROM genomes",
                    (f"$.{key}",),
                ).fetchall()
            else:
                rows = [
                    (None, value)
                    for (value,) in reader.connection.execute(
                        "SELECT json_extract(fitness, ?) FROM genomes", (f"$.{key}",)
                    )
                ]
            best = reader.best_value(key, higher_is_better=higher_is_better(key))

        grouped: dict[Any, list[float]] = {}
        for group, value in rows:
            if value is not None:
                grouped.setdefault(group, []).append(value)

        overall = _statistics(
            [value for values in grouped.values() for value in values]
        )
        per_group = [
            {"group": group, **_statistics(values)}
            for group, values in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
        table = _markdown_table(
            ["group", "n", "min", "median", "mean", "max", "std"],
            (
                [
                    [
                        row["group"],
                        row["n"],
                        row.get("min"),
                        row.get("median"),
                        row.get("mean"),
                        row.get("max"),
                        row.get("std"),
                    ]
                    for row in per_group
                ]
                if group_by
                else [
                    [
                        "all",
                        overall["n"],
                        overall.get("min"),
                        overall.get("median"),
                        overall.get("mean"),
                        overall.get("max"),
                        overall.get("std"),
                    ]
                ]
            ),
        )
        return {
            "run": resolved.name,
            "key": key,
            "overall": overall,
            "rows": per_group if group_by else [],
            "best": best,
            "table": table,
            "dashboard_url": self._url(f"/run/{resolved.index}"),
        }

    def operator_insertion_rates(
        self, run: int | str | None = None, group: str | None = None
    ) -> dict[str, Any]:
        """Reports how the genomes each operator generated were inserted.

        Counting matches ``src.analysis.analyze_genome_generation``: a genome
        counts once for every operator that generated it.

        Args:
            run: A run index or name, for one run's rates.
            group: A group name, for a group's rates.

        Returns:
            The insertion-rate ``columns``, the ``operators`` covered, the
            ``latex`` table the analysis script prints, and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run or group.
            ValueError: If both a run and a group are given.
        """

        index = self._resolve(run).index if run is not None else None
        payload = self.viewer.insertion_rates_payload(run_index=index, group=group)
        scope = payload["scope"]
        if scope["kind"] == "run":
            payload["dashboard_url"] = self._url(f"/insertions/run/{scope['index']}")
        elif scope["kind"] == "group":
            payload["dashboard_url"] = self._url(
                f"/insertions/group/{quote(scope['name'], safe='')}"
            )
        else:
            payload["dashboard_url"] = self._url("/insertions")
        return payload

    def progress_series(
        self, run: int | str, metric: str = "best", max_points: int = 200
    ) -> dict[str, Any]:
        """Returns a run's search progress over time, downsampled.

        Args:
            run: A run index or name.
            metric: The series to return, one of ``best``, ``mean``, ``worst`` or
                ``population_size``.
            max_points: The most points to return, at most
                :data:`MAX_SERIES_POINTS`.

        Returns:
            ``step`` and ``value`` arrays, the ``metrics`` available, how many
            points the run recorded, and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run.
            ValueError: If the run recorded no such metric.
        """

        resolved = self._resolve(run)
        columns = self.viewer.history_payload(resolved.index)["columns"]
        if not columns:
            return {
                "run": resolved.name,
                "step": [],
                "value": [],
                "metrics": [],
                "note": "This run recorded no search progress.",
                "dashboard_url": self._url(f"/run/{resolved.index}"),
            }
        if metric not in columns:
            raise ValueError(
                f"{metric!r} was not recorded; available: {', '.join(sorted(columns))}."
            )

        limit = min(int(max_points), MAX_SERIES_POINTS)
        steps = columns.get("step") or list(range(len(columns[metric])))
        pairs = list(zip(steps, columns[metric]))
        sampled = _downsample(pairs, limit)
        return {
            "run": resolved.name,
            "metric": metric,
            "step": [step for step, _ in sampled],
            "value": [value for _, value in sampled],
            "metrics": sorted(columns),
            "recorded_points": len(pairs),
            "returned_points": len(sampled),
            "dashboard_url": self._url(f"/run/{resolved.index}"),
        }

    def gate_statistics(
        self, run: int | str, sample: int = GATE_SAMPLE_SIZE
    ) -> dict[str, Any]:
        """Summarizes genome size and which gate methods a run's genomes use.

        Size statistics cover every genome, since the archive stores them as
        columns. Gate-method usage needs the stored JSON, so it is taken from a
        sample of the best genomes rather than the whole run.

        Args:
            run: A run index or name.
            sample: How many genomes to read for gate-method usage.

        Returns:
            ``sizes`` (gate, enabled-gate and parameter statistics), ``methods``
            (usage counts over the sample), ``sampled`` and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run.
        """

        resolved = self._resolve(run)
        sample_size = max(1, min(int(sample), MAX_ROWS * 5))
        with GenomeArchive.open_readonly(resolved.archive_path) as reader:
            columns = {
                name: [
                    value
                    for (value,) in reader.connection.execute(
                        f"SELECT {name} FROM genomes"  # nosec: name is a literal below
                    )
                    if value is not None
                ]
                for name in ("n_gates", "n_enabled_gates", "n_parameters")
            }
            best = reader.list_genomes(sort_key="loss", limit=sample_size)
            methods: dict[str, int] = {}
            qubit_counts: dict[int, int] = {}
            for row in best:
                genome = reader.get_genome_dict(row["genome_number"])
                for gate in genome.get("gates") or []:
                    methods[gate.get("method_name", "?")] = (
                        methods.get(gate.get("method_name", "?"), 0) + 1
                    )
                    count = len(gate.get("qubits") or [])
                    qubit_counts[count] = qubit_counts.get(count, 0) + 1

        ordered = sorted(methods.items(), key=lambda item: -item[1])
        return {
            "run": resolved.name,
            "sizes": {name: _statistics(values) for name, values in columns.items()},
            "methods": dict(ordered),
            "qubits_per_gate": dict(sorted(qubit_counts.items())),
            "sampled": len(best),
            "table": _markdown_table(
                ["gate method", "uses"], [[name, count] for name, count in ordered[:30]]
            ),
            "dashboard_url": self._url(f"/run/{resolved.index}"),
        }

    def compare_runs(
        self,
        runs: list[int | str] | None = None,
        group: str | None = None,
        key: str = "loss",
    ) -> dict[str, Any]:
        """Compares runs by their best genome and their size.

        Args:
            runs: Run indexes or names to compare; every served run by default.
            group: A group name to compare instead.
            key: The fitness key compared.

        Returns:
            ``rows`` (one per run), ``summary`` statistics across the runs, a
            Markdown ``table`` and a ``dashboard_url``.

        Raises:
            KeyError: If a run or the group is unknown.
            ValueError: If too many runs are selected.
        """

        selected = self._selected_runs(runs, group)
        if len(selected) > MAX_RUNS_PER_QUERY:
            raise ValueError(
                f"{len(selected)} runs is more than the {MAX_RUNS_PER_QUERY} that can "
                "be compared at once; select fewer."
            )

        better_is_higher = higher_is_better(key)
        rows = []
        for run in selected:
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                best = reader.best_value(key, higher_is_better=better_is_higher)
                rows.append(
                    {
                        "run": run.name,
                        "index": run.index,
                        "genomes": reader.count(),
                        "best": (best or {}).get("value"),
                        "best_genome": (best or {}).get("genome_number"),
                        "groups": run.groups,
                    }
                )

        values = [row["best"] for row in rows if row["best"] is not None]
        return self._fit(
            {
                "key": key,
                "rows": rows,
                "summary": _statistics(values),
                "table": _markdown_table(
                    ["run", "genomes", f"best {key}", "genome"],
                    [
                        [row["run"], row["genomes"], row["best"], row["best_genome"]]
                        for row in rows
                    ],
                ),
                "dashboard_url": self._url("/groups"),
            },
            "rows",
        )

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------

    def describe_schema(self) -> dict[str, Any]:
        """Describes the tables ``query_sql`` can read, with worked examples.

        Returns:
            The single-run and cross-run schemas, the limits queries run under,
            and example statements.
        """

        return {
            "single_run": {
                "genomes": (
                    "genome_number, insertion, saved_at, insert_type, generated_by "
                    "(JSON array), crossover_type, island, n_gates, n_enabled_gates, "
                    "n_parameters, fitness (JSON object)"
                ),
                "genome_parents": "child, parent",
                "run_info": "key, value",
                "note": (
                    "Fitness keys are read with json_extract(fitness, '$.loss'); "
                    "operators with json_each(genomes.generated_by)."
                ),
            },
            "cross_run": {
                "genomes": ", ".join(_ROLLUP_COLUMNS),
                "runs": "run, run_index, task, task_target, strategy, genomes",
                "note": (
                    "Selecting more than one run rolls their summary rows into one "
                    "database, where loss and target_metric are plain columns."
                ),
            },
            "limits": {
                "statements": "one read-only SELECT or WITH per call",
                "rows": MAX_ROWS,
                "timeout_seconds": QUERY_TIMEOUT_SECONDS,
                "runs_per_query": MAX_RUNS_PER_QUERY,
            },
            "examples": [
                "SELECT insert_type, COUNT(*) FROM genomes GROUP BY 1 ORDER BY 2 DESC",
                "SELECT genome_number, json_extract(fitness, '$.loss') AS loss "
                "FROM genomes ORDER BY loss LIMIT 10",
                "SELECT run, MIN(loss) FROM genomes GROUP BY run ORDER BY 2",
                "SELECT json_each.value AS operator, COUNT(*) FROM genomes, "
                "json_each(genomes.generated_by) GROUP BY 1 ORDER BY 2 DESC",
            ],
        }

    def query_sql(
        self,
        sql: str,
        run: int | str | None = None,
        runs: list[int | str] | None = None,
        group: str | None = None,
        limit: int = MAX_ROWS,
    ) -> dict[str, Any]:
        """Runs one read-only SELECT over a run's archive or a roll-up of runs.

        The connection is opened read-only and an authorizer denies everything
        but reads, so writes, ``ATTACH`` and ``PRAGMA`` fail whatever the SQL
        says. Statements are wrapped in an enforced ``LIMIT`` and interrupted
        after :data:`QUERY_TIMEOUT_SECONDS`.

        Args:
            sql: The statement to run.
            run: A single run to query (its own tables).
            runs: Several runs to roll up and query together.
            group: A group of runs to roll up and query together.
            limit: The most rows to return, at most :data:`MAX_ROWS`.

        Returns:
            ``columns``, ``rows``, ``row_count``, the ``scope`` queried and a
            Markdown ``table``.

        Raises:
            KeyError: If a run or group is unknown.
            ValueError: If the statement is not a single read-only query, or it
                fails, or too many runs are selected.
        """

        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise ValueError("Give a SQL statement to run.")
        if ";" in statement:
            raise ValueError("Run one statement at a time.")
        if statement.split(None, 1)[0].upper() not in ("SELECT", "WITH"):
            raise ValueError("Only SELECT (or WITH ... SELECT) queries are allowed.")

        if run is not None and (runs or group):
            raise ValueError("Query a single run, or several runs, not both.")

        connection: sqlite3.Connection
        if run is not None:
            resolved = self._resolve(run)
            scope = {"kind": "run", "run": resolved.name}
            connection = sqlite3.connect(
                f"file:{resolved.archive_path}?mode=ro", uri=True
            )
        else:
            selected = self._selected_runs(runs, group)
            scope = {"kind": "rollup", "runs": [item.name for item in selected]}
            connection = self._rollup(selected)

        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, 10_000
        )
        connection.set_authorizer(
            lambda action, *_: (
                sqlite3.SQLITE_OK
                if action in _ALLOWED_SQL_ACTIONS
                else sqlite3.SQLITE_DENY
            )
        )

        rows_limit = min(int(limit), MAX_ROWS)
        try:
            cursor = connection.execute(
                f"SELECT * FROM ({statement}) LIMIT ?", (rows_limit,)
            )
            columns = [description[0] for description in cursor.description or []]
            rows = [list(row) for row in cursor.fetchall()]
        except sqlite3.Error as error:
            raise ValueError(f"Query failed: {error}") from error
        finally:
            connection.close()

        return self._fit(
            {
                "scope": scope,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "row_limit": rows_limit,
                "table": _markdown_table(columns, rows[:50]),
            },
            "rows",
        )
