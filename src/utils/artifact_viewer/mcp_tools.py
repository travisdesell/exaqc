"""Analysis tools an agent calls to interrogate EXAQC runs.

These are the implementations behind the MCP interface
(:mod:`src.utils.artifact_viewer.mcp_app`); they are plain methods returning
JSON-safe dicts, so they can be tested and benchmarked without a transport.

Every tool reads runs without changing them. The one exception is annotations:
when a server allows it, notes and tags can be recorded, and they are kept in
each run's ``annotations.sqlite`` beside its archive, never in the archive.

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

import csv
import io
import json
import re
import sqlite3
import statistics
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

from src.utils.annotations import NOTE_COLUMNS, TAG_COLUMNS, AnnotationStore
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

#: Rows one ``export_query`` page holds when the caller does not say, and the
#: most it may ask for. Export pages are read by a program rather than an agent,
#: so they are far larger than a listing.
DEFAULT_EXPORT_ROWS = 5_000
MAX_EXPORT_ROWS = 50_000

#: The most encoded bytes of rows one ``export_query`` page carries; a page ends
#: early rather than exceed it, and says where the next one starts.
MAX_EXPORT_BYTES = 4 * 1024 * 1024

#: How long an ``export_query`` statement may run: longer than an interactive
#: query, since a page may scan every genome of many runs.
EXPORT_TIMEOUT_SECONDS = 60.0

#: The encodings ``export_query`` returns a page in.
EXPORT_FORMATS = ("csv", "json")

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

#: The per-genome columns a cross-run roll-up holds: each column's name, its SQL
#: type, and the expression selecting it from an archive. Kept as one table so
#: the database the roll-up creates, the rows it copies into it and the schema it
#: advertises cannot disagree -- columns added to the archive but to only one of
#: those three are how the roll-up came to be missing some.
_ROLLUP_SELECT: tuple[tuple[str, str, str], ...] = (
    ("genome_number", "INTEGER", "genome_number"),
    ("insertion", "INTEGER", "insertion"),
    ("insert_type", "TEXT", "insert_type"),
    ("generated_by", "TEXT", "generated_by"),
    ("crossover_type", "TEXT", "crossover_type"),
    ("island", "INTEGER", "island"),
    ("n_gates", "INTEGER", "n_gates"),
    ("n_enabled_gates", "INTEGER", "n_enabled_gates"),
    ("n_parameters", "INTEGER", "n_parameters"),
    ("n_cnot", "INTEGER", "n_cnot"),
    ("n_rot", "INTEGER", "n_rot"),
    ("max_innovation_number", "INTEGER", "max_innovation_number"),
    ("generated_at_insertion", "INTEGER", "generated_at_insertion"),
    ("evaluation_seconds", "REAL", "evaluation_seconds"),
    ("evaluated_host", "TEXT", "evaluated_host"),
    ("evaluated_rank", "INTEGER", "evaluated_rank"),
    ("discard_reason", "TEXT", "discard_reason"),
    ("final_metrics", "TEXT", "final_metrics"),
    ("fitness", "TEXT", "fitness"),
    ("loss", "REAL", "json_extract(fitness, '$.loss')"),
    ("target_metric", "REAL", "json_extract(fitness, '$.target_metric')"),
)

#: Every column of the cross-run ``genomes`` table, the run's name first.
_ROLLUP_COLUMNS = ("run", *(name for name, _, _ in _ROLLUP_SELECT))

#: Archive tables copied whole into a cross-run roll-up, each with the columns
#: it keeps; the roll-up puts a ``run`` column naming each row's run in front.
_ROLLUP_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("genome_parents", ("child", "parent")),
    ("population_events", ("step", "recorded_at", "added", "removed")),
    ("run_info", ("key", "value")),
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


def _genome_operators_view(with_run: bool) -> str:
    """Builds the statement creating the ``genome_operators`` view.

    ``generated_by`` stores a genome's operators as a JSON array, which every
    per-operator question would otherwise unnest with ``json_each``. The view
    does that once: one row per operator applied, with its position and how many
    operators the genome had, so a genome made by several operators can be told
    apart from one made by a single operator.

    Args:
        with_run: Whether the view carries the roll-up's ``run`` column.

    Returns:
        The ``CREATE TEMP VIEW`` statement. A temporary view lives in the
        connection's own memory, so creating one never writes to an archive.
    """

    run = "genomes.run, " if with_run else ""
    return (
        "CREATE TEMP VIEW genome_operators AS SELECT "
        f"{run}genomes.genome_number, CAST(json_each.key AS INTEGER) AS position, "
        "json_each.value AS operator, "
        "json_array_length(genomes.generated_by) AS n_operators "
        "FROM genomes, json_each(genomes.generated_by)"
    )


def _add_annotation_tables(
    connection: sqlite3.Connection, runs: list[Run], with_run: bool
) -> None:
    """Gives a query connection ``notes`` and ``genome_tags`` tables to read.

    A run's annotations live in their own file beside its archive. Rather than
    attach that file -- which the query's authorizer forbids, and which would
    need a second mechanism for roll-ups -- its rows are copied in, the same way
    a roll-up gathers archives. Annotations are small, so this is cheap, and a run
    that has not been annotated still gets the tables, empty, so a query naming
    them does not fail.

    Args:
        connection: The connection the query will run on, before its authorizer
            is installed.
        runs: The runs whose annotations are copied.
        with_run: Whether the tables carry a roll-up's ``run`` column. A single
            run's tables are temporary instead, so creating them never writes to
            its archive.

    Returns:
        None. Creates and fills the two tables on ``connection``.
    """

    temporary = "" if with_run else "TEMP "
    for table, columns in (("notes", NOTE_COLUMNS), ("genome_tags", TAG_COLUMNS)):
        names = ("run", *columns) if with_run else columns
        connection.execute(f"CREATE {temporary}TABLE {table}({', '.join(names)})")
        placeholders = ", ".join(["?"] * len(names))
        for run in runs:
            rows = AnnotationStore.beside(run.archive_path).table_rows(table)
            connection.executemany(
                f"INSERT INTO {table} VALUES ({placeholders})",
                [(run.name, *row) if with_run else row for row in rows],
            )


def _single_statement(sql: str) -> str:
    """Checks that SQL is one read-only query and returns it without a terminator.

    Only a semicolon SQLite itself would treat as ending a statement is refused;
    one inside a string literal or a comment -- the separator of
    ``group_concat(name, ';')``, say -- is part of the query. This is the text
    half of the guard; the authorizer set on every query connection is the other.

    Args:
        sql: The statement as given.

    Returns:
        The statement, stripped of surrounding whitespace and trailing semicolons.

    Raises:
        ValueError: If there is no statement, there is more than one, or it does
            not start with ``SELECT`` or ``WITH``.
    """

    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise ValueError("Give a SQL statement to run.")
    for position, character in enumerate(statement):
        if character == ";" and sqlite3.complete_statement(statement[: position + 1]):
            raise ValueError("Run one statement at a time.")
    if statement.split(None, 1)[0].upper() not in ("SELECT", "WITH"):
        raise ValueError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    return statement


def _bounded(statement: str) -> str:
    """Wraps a statement so a row limit and an offset can be bound to it.

    The closing parenthesis goes on its own line, because a statement ending in
    a ``--`` comment would otherwise comment it out.

    Args:
        statement: A statement already checked by :func:`_single_statement`.

    Returns:
        A statement taking ``LIMIT`` and ``OFFSET`` parameters, in that order.
    """

    return f"SELECT * FROM ({statement}\n) LIMIT ? OFFSET ?"


def _csv_line(values: Sequence[Any]) -> str:
    """Encodes one row as a line of CSV.

    Args:
        values: The row's cells; ``None`` is written as an empty cell.

    Returns:
        The line, ending in a newline.
    """

    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue()


class DashboardTools:
    """The tools exposed over MCP, backed by the dashboard's data layer.

    Every tool reads, except the annotation writes -- notes and tags kept beside a
    run, never in its archive -- which a server may allow.

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
        allow_annotations: bool = False,
    ) -> None:
        """Creates the tool set.

        Args:
            registry: The runs to serve.
            renderer: The image renderer the viewer uses; tools never render, so
                an inline renderer is created when none is given.
            base_url: The dashboard's base URL for deep links.
            allow_annotations: Whether notes and tags may be written.
        """

        self.registry = registry
        self.viewer = ArtifactViewer(
            registry,
            renderer or RenderService(processes=0),
            allow_annotations=allow_annotations,
        )
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
        """Trims a result until it fits the response budget.

        A Markdown ``table`` only repeats rows the result already holds, so it is
        replaced by a note before any row is dropped.

        Args:
            payload: The result, which must hold a list under ``rows_key``.
            rows_key: The key holding the rows that may be dropped.

        Returns:
            The payload, with its table omitted when that makes it fit, and
            otherwise also rows dropped and ``truncated`` set.
        """

        rows = payload.get(rows_key) or []
        if (
            "table" in payload
            and len(json.dumps(payload, default=str)) > MAX_RESPONSE_BYTES
        ):
            payload["table"] = "_table omitted: the rows alone fill the response_"
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

        Raises:
            ValueError: If the run's archive predates a column the roll-up holds,
                which would otherwise fail as a bare SQL error naming neither the
                run nor what to do about it.
        """

        expressions = ", ".join(expression for _, _, expression in _ROLLUP_SELECT)
        required = {
            expression
            for _, _, expression in _ROLLUP_SELECT
            if expression.isidentifier()
        }

        with GenomeArchive.open_readonly(run.archive_path) as reader:
            present = {
                row[1]
                for row in reader.connection.execute("PRAGMA table_xinfo(genomes)")
            }
            missing = sorted(required - present)
            if missing:
                raise ValueError(
                    f"{run.name} was written before its archive recorded "
                    f"{', '.join(missing)}, so it cannot be rolled up with other "
                    "runs. Query it on its own, or re-run it to record them."
                )
            return reader.connection.execute(
                f"SELECT {expressions} FROM genomes"
            ).fetchall()

    def _rollup(self, runs: list[Run]) -> sqlite3.Connection:
        """Copies several runs' summary rows into one in-memory database.

        This is how cross-run SQL is served: SQLite allows at most 10 attached
        databases, a compile-time ceiling that cannot be raised at runtime, so
        the rows are gathered instead of the files being attached.

        Args:
            runs: The runs to roll up.

        Returns:
            A connection holding a ``runs`` table, the ``genomes`` summary rows
            and the :data:`_ROLLUP_TABLES` -- each with a ``run`` column naming
            each row's run -- and the ``genome_operators`` view over them.

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
        declarations = ", ".join(
            f"{name} {sql_type}" for name, sql_type, _ in _ROLLUP_SELECT
        )
        memory.execute(f"CREATE TABLE genomes(run TEXT, {declarations})")
        for table, columns in _ROLLUP_TABLES:
            memory.execute(f"CREATE TABLE {table}(run TEXT, {', '.join(columns)})")
        placeholders = ", ".join(["?"] * len(_ROLLUP_COLUMNS))
        for run in runs:
            rows = self._summary_rows(run)
            memory.executemany(
                f"INSERT INTO genomes VALUES ({placeholders})",
                [(run.name, *row) for row in rows],
            )
            with GenomeArchive.open_readonly(run.archive_path) as reader:
                info = reader.run_info()
                present = {
                    name
                    for (name,) in reader.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                for table, columns in _ROLLUP_TABLES:
                    if table not in present:
                        # an archive older than the table contributes no rows to it
                        continue
                    memory.executemany(
                        f"INSERT INTO {table} VALUES "
                        f"(?, {', '.join(['?'] * len(columns))})",
                        [
                            (run.name, *row)
                            for row in reader.connection.execute(
                                f"SELECT {', '.join(columns)} FROM {table}"
                            )
                        ],
                    )
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
        _add_annotation_tables(memory, runs, with_run=True)
        memory.execute(_genome_operators_view(with_run=True))
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

    def list_runs(
        self, name_contains: str | None = None, include_command_line: bool = False
    ) -> dict[str, Any]:
        """Lists the runs being served, newest information first.

        Args:
            name_contains: Only list runs whose name contains this.
            include_command_line: Whether each row keeps the run's command line.
                It is long and nearly identical across a group's runs, so it is
                left out unless asked for; ``describe_run`` always includes it.

        Returns:
            ``rows`` (one summary per run), ``total``, a Markdown ``table`` and
            the dashboard ``dashboard_url``.
        """

        payload = self.viewer.runs_payload()
        rows = [
            (
                run
                if include_command_line
                else {key: value for key, value in run.items() if key != "command_line"}
            )
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
            Its ``summary``, the serialized ``genome``, its ``children``, each
            parent's island (``parent_islands``), copy-ready ``commands`` and a
            ``dashboard_url``.

        Raises:
            KeyError: If there is no such run or genome.
        """

        resolved = self._resolve(run)
        payload = self.viewer.genome_payload(resolved.index, int(genome_number))
        payload["dashboard_url"] = self._url(
            f"/run/{resolved.index}/genome/{int(genome_number)}"
        )
        return payload

    def list_annotations(
        self,
        run: int | str,
        genome_number: int | None = None,
        tag: str | None = None,
        include_removed: bool = False,
    ) -> dict[str, Any]:
        """Lists the notes and tags recorded for a run, or for one of its genomes.

        Args:
            run: A run index or name.
            genome_number: Only list this genome's notes and tags.
            tag: Only list tags with this name, e.g. to find every candidate.
            include_removed: Also list tags that were removed, with when and by
                whom.

        Returns:
            ``notes`` and ``tags``, whether annotations may be written here
            (``enabled``), the ``run`` and a ``dashboard_url``.

        Raises:
            KeyError: If there is no such run.
        """

        resolved = self._resolve(run)
        number = None if genome_number is None else int(genome_number)
        payload = self.viewer.annotations_payload(
            resolved.index, number, tag, include_removed
        )
        payload["run"] = resolved.name
        payload["dashboard_url"] = self._url(
            f"/run/{resolved.index}" + ("" if number is None else f"/genome/{number}")
        )
        return self._fit(payload, "notes")

    def add_note(
        self,
        run: int | str,
        text: str,
        genome_number: int | None = None,
        author: str | None = None,
    ) -> dict[str, Any]:
        """Records a note about a run, or about one of its genomes.

        Notes are never edited or deleted, so this always adds one.

        Args:
            run: A run index or name.
            text: What to note.
            genome_number: The genome the note is about; the run when not given.
            author: A name to record with the note.

        Returns:
            The recorded ``note``, the ``run`` and a ``dashboard_url``.

        Raises:
            PermissionError: If annotations may not be written here.
            KeyError: If there is no such run or genome.
            ValueError: If the note is empty or too long, or the author invalid.
        """

        resolved = self._resolve(run)
        number = None if genome_number is None else int(genome_number)
        note = self.viewer.add_note(resolved.index, text, "mcp", author, number)
        return {
            "run": resolved.name,
            "note": note,
            "dashboard_url": self._url(
                f"/run/{resolved.index}"
                + ("" if number is None else f"/genome/{number}")
            ),
        }

    def tag_genome(
        self,
        run: int | str,
        genome_number: int,
        tag: str,
        author: str | None = None,
    ) -> dict[str, Any]:
        """Tags a genome, leaving it as it is if it already carries the tag.

        Args:
            run: A run index or name.
            genome_number: The genome to tag.
            tag: A short label such as ``candidate`` or ``needs:rerun``.
            author: A name to record with the tag.

        Returns:
            The ``tag`` that now applies (its ``created`` says whether it is new),
            the ``run`` and a ``dashboard_url``.

        Raises:
            PermissionError: If annotations may not be written here.
            KeyError: If there is no such run or genome.
            ValueError: If the tag or author is invalid.
        """

        resolved = self._resolve(run)
        applied = self.viewer.add_tag(
            resolved.index, int(genome_number), tag, "mcp", author
        )
        return {
            "run": resolved.name,
            "tag": applied,
            "dashboard_url": self._url(
                f"/run/{resolved.index}/genome/{int(genome_number)}"
            ),
        }

    def untag_genome(
        self,
        run: int | str,
        genome_number: int,
        tag: str,
        author: str | None = None,
    ) -> dict[str, Any]:
        """Removes a tag from a genome, keeping the record that it applied.

        Args:
            run: A run index or name.
            genome_number: The genome to untag.
            tag: The tag to remove.
            author: A name to record with the removal.

        Returns:
            The ``tag``, stamped with its removal, the ``run`` and a
            ``dashboard_url``.

        Raises:
            PermissionError: If annotations may not be written here.
            KeyError: If there is no such run or genome, or it lacks the tag.
            ValueError: If the tag or author is invalid.
        """

        resolved = self._resolve(run)
        removed = self.viewer.remove_tag(
            resolved.index, int(genome_number), tag, "mcp", author
        )
        return {
            "run": resolved.name,
            "tag": removed,
            "dashboard_url": self._url(
                f"/run/{resolved.index}/genome/{int(genome_number)}"
            ),
        }

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
        self,
        run: int | str,
        metric: str | None = None,
        statistic: str = "best",
        max_points: int = 200,
    ) -> dict[str, Any]:
        """Returns a run's search progress over time, downsampled.

        Two things are chosen separately: *which* metric to summarize over the
        population -- a fitness key, a circuit size, or anything the run's task
        recorded while training -- and *which* statistic of it to return. The
        statistics are recomputed from the genomes alive at each step, so a run
        can be followed by validation accuracy or episode return just as easily
        as by loss.

        Args:
            run: A run index or name.
            metric: The metric to summarize; the run's default (``loss`` when it
                recorded one) when not given. ``describe_run`` lists the choices.
            statistic: Which series of it to return: ``best``, ``mean``,
                ``worst`` or ``population_size``.
            max_points: The most points to return, at most
                :data:`MAX_SERIES_POINTS`.

        Returns:
            ``step`` and ``value`` arrays, the ``metric`` and ``statistic`` they
            describe, the ``metrics`` worth charting and the ``statistics``
            available, how many points the run recorded, and a
            ``dashboard_url``.

        Raises:
            KeyError: If there is no such run.
            ValueError: If the run recorded no such metric, or ``statistic`` is
                not one of the series returned.
        """

        resolved = self._resolve(run)
        payload = self.viewer.history_payload(resolved.index, metric)
        columns = payload["columns"]
        if not columns:
            return {
                "run": resolved.name,
                "step": [],
                "value": [],
                "metrics": [],
                "statistics": [],
                "note": "This run recorded no search progress.",
                "dashboard_url": self._url(f"/run/{resolved.index}"),
            }

        statistics = [name for name in columns if name != "step"]
        if statistic not in statistics:
            raise ValueError(
                f"{statistic!r} is not a statistic; choose one of: "
                f"{', '.join(statistics)}."
            )

        limit = min(int(max_points), MAX_SERIES_POINTS)
        steps = columns.get("step") or list(range(len(columns[statistic])))
        pairs = list(zip(steps, columns[statistic]))
        sampled = _downsample(pairs, limit)
        # the curated list rather than every key: a per-class breakdown can run to
        # dozens of entries, and the rest stay reachable through describe_run
        offered = payload.get("primary_metrics") or payload.get("metrics") or []
        return {
            "run": resolved.name,
            "metric": payload.get("metric"),
            "statistic": statistic,
            "step": [step for step, _ in sampled],
            "value": [value for _, value in sampled],
            "metrics": offered,
            "recorded_metrics": len(payload.get("metrics") or []),
            "statistics": statistics,
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

    def _query_connection(
        self,
        run: int | str | None,
        runs: list[int | str] | None,
        group: str | None,
        timeout_seconds: float,
    ) -> tuple[sqlite3.Connection, dict[str, Any]]:
        """Opens the guarded, read-only connection a SQL tool runs a statement on.

        A single run is read from its archive opened ``mode=ro``; several are
        rolled into memory. Either way the connection gains the
        ``genome_operators`` view, is interrupted past its deadline, and has an
        authorizer denying everything but reads, so writes, ``ATTACH`` and
        ``PRAGMA`` fail whatever the SQL says.

        Args:
            run: A single run to query (its own tables).
            runs: Several runs to roll up and query together.
            group: A group of runs to roll up and query together.
            timeout_seconds: How long a statement may run before it is interrupted.

        Returns:
            The connection, which the caller closes, and the ``scope`` it covers.

        Raises:
            KeyError: If a run or group is unknown.
            ValueError: If both a single run and several are given, or too many
                runs are selected.
        """

        if run is not None and (runs or group):
            raise ValueError("Query a single run, or several runs, not both.")

        connection: sqlite3.Connection
        if run is not None:
            resolved = self._resolve(run)
            scope: dict[str, Any] = {"kind": "run", "run": resolved.name}
            connection = sqlite3.connect(
                f"file:{resolved.archive_path}?mode=ro", uri=True
            )
            connection.execute(_genome_operators_view(with_run=False))
            _add_annotation_tables(connection, [resolved], with_run=False)
        else:
            selected = self._selected_runs(runs, group)
            scope = {"kind": "rollup", "runs": [item.name for item in selected]}
            connection = self._rollup(selected)

        deadline = time.monotonic() + timeout_seconds
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
        return connection, scope

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
                    "n_parameters, n_cnot, n_rot, max_innovation_number, "
                    "generated_at_insertion (genomes "
                    "inserted when it was generated; insertion - generated_at_insertion is "
                    "how many insertions it waited), evaluation_seconds, evaluated_host, "
                    "evaluated_rank, discard_reason (worse_than_population, "
                    "duplicate_of_better or generated_before_repopulation), "
                    "final_metrics (JSON object), "
                    "fitness (JSON object)"
                ),
                "genome_operators": (
                    "genome_number, position, operator, n_operators -- a view with one "
                    "row per operator that generated a genome, so a genome made by "
                    "several operators has several rows and n_operators says how many"
                ),
                "genome_parents": "child, parent",
                "population_events": "step, recorded_at, added (JSON), removed (JSON)",
                "notes": (
                    "note_id, genome_number (NULL for a note about the run as a whole), "
                    "text, source ('mcp' or 'dashboard'), author, created_at -- notes "
                    "people and agents recorded; never edited or deleted"
                ),
                "genome_tags": (
                    "tag_id, genome_number, tag, source, author, added_at, removed_at "
                    "(NULL while the tag still applies), removed_source, removed_author"
                ),
                "run_info": (
                    "key, value (JSON) -- the run's task, provenance and command line; "
                    "runs started since it was recorded also hold operator_selection: "
                    "mutation_weights (relative integer weights), crossover_rates "
                    "(fraction of post-initialization children per crossover), "
                    "mutation_strategy and parent_strategy"
                ),
                "note": (
                    "Fitness keys are read with json_extract(fitness, '$.loss'); "
                    "operators with the genome_operators view. final_metrics "
                    "holds the last value of every metric a genome recorded while "
                    "training, keyed '<series>.<metric>' -- the dots are part of the "
                    "key, so the path is quoted: "
                    "json_extract(final_metrics, '$.\"validation_epoch_metrics.loss\"')."
                ),
            },
            "cross_run": {
                "genomes": ", ".join(_ROLLUP_COLUMNS),
                "genome_operators": "run, genome_number, position, operator, n_operators",
                **{
                    table: ", ".join(("run", *columns))
                    for table, columns in _ROLLUP_TABLES
                },
                "notes": ", ".join(("run", *NOTE_COLUMNS)),
                "genome_tags": ", ".join(("run", *TAG_COLUMNS)),
                "runs": "run, run_index, task, task_target, strategy, genomes",
                "note": (
                    "Selecting more than one run rolls their rows into one database, "
                    "where every table gains a run column naming each row's run (it "
                    "exists only here, not when querying a single run) and loss and "
                    "target_metric are plain genome columns. Genome numbers repeat "
                    "across runs, so join on run as well as the genome number."
                ),
            },
            "limits": {
                "statements": "one read-only SELECT or WITH per call",
                "rows": MAX_ROWS,
                "timeout_seconds": QUERY_TIMEOUT_SECONDS,
                "runs_per_query": MAX_RUNS_PER_QUERY,
                "export": (
                    f"export_query runs the same statements without the {MAX_ROWS}-row "
                    f"cap, returning pages of up to {MAX_EXPORT_ROWS} rows as CSV or "
                    "JSON, each with the next_offset to continue from. ORDER BY the "
                    "query so pages are stable: a run still searching gains genomes "
                    "between pages."
                ),
            },
            "examples": [
                "SELECT insert_type, COUNT(*) FROM genomes GROUP BY 1 ORDER BY 2 DESC",
                "SELECT genome_number, json_extract(fitness, '$.loss') AS loss "
                "FROM genomes ORDER BY loss LIMIT 10",
                "SELECT run, MIN(loss) FROM genomes GROUP BY run ORDER BY 2",
                "SELECT json_each.value AS operator, COUNT(*) FROM genomes, "
                "json_each(genomes.generated_by) GROUP BY 1 ORDER BY 2 DESC",
                "SELECT run, genome_number, json_extract(final_metrics, "
                "'$.\"validation_epoch_metrics.loss\"') AS validation_loss "
                "FROM genomes WHERE validation_loss IS NOT NULL "
                "ORDER BY validation_loss LIMIT 5",
                "SELECT o.operator, AVG(g.insert_type != 'discarded') AS kept "
                "FROM genome_operators o JOIN genomes g "
                "ON g.run = o.run AND g.genome_number = o.genome_number "
                "WHERE o.n_operators = 1 GROUP BY 1 ORDER BY 2 DESC",
                "SELECT p.run, AVG(child.loss < parent.loss) AS beat_parent "
                "FROM genome_parents p "
                "JOIN genomes child ON child.run = p.run AND child.genome_number = p.child "
                "JOIN genomes parent ON parent.run = p.run "
                "AND parent.genome_number = p.parent GROUP BY p.run",
                "SELECT t.run, t.tag, COUNT(*) AS genomes, AVG(g.loss) AS mean_loss "
                "FROM genome_tags t JOIN genomes g "
                "ON g.run = t.run AND g.genome_number = t.genome_number "
                "WHERE t.removed_at IS NULL GROUP BY 1, 2",
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
        says (see :meth:`_query_connection`). Statements are wrapped in an
        enforced ``LIMIT`` and interrupted after :data:`QUERY_TIMEOUT_SECONDS`.
        To collect more rows than the cap, page through :meth:`export_query`.

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

        statement = _single_statement(sql)
        connection, scope = self._query_connection(
            run, runs, group, QUERY_TIMEOUT_SECONDS
        )

        rows_limit = min(int(limit), MAX_ROWS)
        try:
            cursor = connection.execute(_bounded(statement), (rows_limit, 0))
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

    def export_query(
        self,
        sql: str,
        run: int | str | None = None,
        runs: list[int | str] | None = None,
        group: str | None = None,
        format: str = "csv",
        limit: int = DEFAULT_EXPORT_ROWS,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Returns one page of a read-only query's full result, for a client to collect.

        ``query_sql`` is sized for an agent reading its answer; this is sized for
        a program gathering data to analyze itself, such as per-genome rows for a
        statistical test. It accepts the same statements under the same guards,
        but returns pages far larger than the row cap, encoded compactly and with
        no Markdown table, each saying where the next page starts. Pages are cut
        with ``LIMIT``/``OFFSET`` over a fresh read, so an ordered query pages
        stably; an unordered one, or a run still searching, can shift rows
        between pages.

        Args:
            sql: The statement to run.
            run: A single run to query (its own tables).
            runs: Several runs to roll up and query together.
            group: A group of runs to roll up and query together.
            format: ``csv`` for a CSV document with a header line, or ``json``
                for the rows as lists.
            limit: The most rows in this page, at most :data:`MAX_EXPORT_ROWS`.
                A page also ends before its rows pass :data:`MAX_EXPORT_BYTES`,
                though it always holds at least one row.
            offset: How many rows of the result to skip: the previous page's
                ``next_offset``.

        Returns:
            The ``columns``, the page as ``csv`` or ``rows``, its ``offset``,
            ``row_count`` and ``row_limit``, the ``next_offset`` to continue from
            (``None`` on the last page), ``complete`` and the ``scope`` queried.

        Raises:
            KeyError: If a run or group is unknown.
            ValueError: If the format is unknown, the statement is not a single
                read-only query or it fails, or too many runs are selected.
        """

        if format not in EXPORT_FORMATS:
            raise ValueError(f"format must be one of: {', '.join(EXPORT_FORMATS)}.")
        statement = _single_statement(sql)
        page_limit = max(1, min(int(limit), MAX_EXPORT_ROWS))
        start = max(0, int(offset))
        connection, scope = self._query_connection(
            run, runs, group, EXPORT_TIMEOUT_SECONDS
        )

        page: list[Any] = []
        size = 0
        more = False
        try:
            # one row past the page is read, to learn whether another page follows
            cursor = connection.execute(_bounded(statement), (page_limit + 1, start))
            columns = [description[0] for description in cursor.description or []]
            for row in cursor:
                if len(page) == page_limit:
                    more = True
                    break
                entry: Any = _csv_line(row) if format == "csv" else list(row)
                encoded = entry if format == "csv" else json.dumps(entry, default=str)
                row_bytes = len(encoded.encode("utf-8"))
                if page and size + row_bytes > MAX_EXPORT_BYTES:
                    more = True
                    break
                page.append(entry)
                size += row_bytes
        except sqlite3.Error as error:
            raise ValueError(f"Query failed: {error}") from error
        finally:
            connection.close()

        payload: dict[str, Any] = {
            "scope": scope,
            "format": format,
            "columns": columns,
            "offset": start,
            "row_count": len(page),
            "row_limit": page_limit,
            "next_offset": start + len(page) if more else None,
            "complete": not more,
        }
        if format == "csv":
            payload["csv"] = _csv_line(columns) + "".join(page)
        else:
            payload["rows"] = page
        return payload
