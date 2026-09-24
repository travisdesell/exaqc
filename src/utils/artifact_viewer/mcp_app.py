"""The MCP server exposing the dashboard's analysis tools to agents.

This wraps :class:`~src.utils.artifact_viewer.mcp_tools.DashboardTools` in the
MCP Python SDK's :class:`~mcp.server.mcpserver.MCPServer`, which turns each
function's type hints into a tool schema. The same server object is served two
ways: mounted at ``/mcp`` inside the dashboard's application
(:func:`~src.utils.artifact_viewer.app.create_app`) and run over stdio by
``python3 -m src.examples.exaqc_mcp``.

Run identifiers are strings everywhere so the tool schemas stay simple: a tool
accepts either a run's index (``"3"``) or its name (``"iris_1"``).
"""

from __future__ import annotations

import functools
import sqlite3
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.utils.artifact_viewer.mcp_tools import (
    DEFAULT_EXPORT_ROWS,
    DEFAULT_ROWS,
    GATE_SAMPLE_SIZE,
    MAX_EXPORT_ROWS,
    MAX_ROWS,
    DashboardTools,
)
from src.utils.artifact_viewer.server import RenderService, RunRegistry

#: Shown to the agent when it connects, so it knows what it is looking at and
#: which tool to reach for first.
INSTRUCTIONS = """
These tools read EXAQC neuroevolution runs: every genome a search evaluated, its
fitness, the operators that generated it, its ancestry, and the search's
progress over time. Runs are stored as SQLite archives and are read-only here.

Start with list_runs, then describe_run for the run you care about: it reports
the fitness keys and the values genomes can be filtered by. Use the typed tools
for common questions, and query_sql (with describe_schema) for anything else.

Results are capped: listings return at most a few hundred rows and series are
downsampled, so prefer aggregates over fetching every genome. To collect rows
for your own analysis instead, page through export_query, which runs the same
SQL without the row cap. Most results carry a dashboard_url that opens the same
view in a browser.

Notes and tags people or agents have recorded are listed by list_annotations,
and queryable as the notes and genome_tags tables. When this server allows it,
add_note, tag_genome and untag_genome record them -- beside a run, never in its
archive.
"""


class _AnticipatedFailures:
    """Presents :class:`DashboardTools` with its expected failures as tool errors.

    The tool layer refuses bad arguments the way ordinary Python does, with
    ``ValueError`` and ``KeyError``: an unknown run, a metric no genome recorded,
    SQL naming a column that is not there. To the MCP SDK any exception other
    than :class:`~mcp.server.mcpserver.exceptions.ToolError` is a crash -- the
    caller is told only ``Error executing tool <name>`` and the message is logged
    as a server traceback instead. That leaves an agent unable to correct a call
    it could have fixed had it been told what was wrong, and fills the log with
    tracebacks for ordinary mistakes.

    Translating here rather than in the tool layer keeps
    :mod:`src.utils.artifact_viewer.mcp_tools` free of SDK imports, so the
    transport stays swappable, and covers every tool -- including any added later
    -- rather than each registration having to remember.

    Attributes:
        tools: The tool layer being wrapped.
    """

    def __init__(self, tools: DashboardTools) -> None:
        """Wraps a tool layer.

        Args:
            tools: The tools whose anticipated failures are translated.
        """

        self.tools = tools

    def __getattr__(self, name: str) -> Any:
        """Looks up a tool, wrapping it to translate its expected failures.

        Args:
            name: The attribute being read, which for a tool is its method name.

        Returns:
            The attribute: wrapped when it is callable, unchanged otherwise.
        """

        attribute = getattr(self.tools, name)
        if not callable(attribute):
            return attribute

        @functools.wraps(attribute)
        def call(*arguments: Any, **keywords: Any) -> Any:
            """Calls the tool, reporting an expected failure as a tool error."""

            try:
                return attribute(*arguments, **keywords)
            except KeyError as error:
                # KeyError renders its argument quoted ("'no such run'"), so the
                # message is taken from the argument rather than from str().
                message = str(error.args[0]) if error.args else str(error)
                raise ToolError(message) from error
            except ValueError as error:
                raise ToolError(str(error)) from error
            except PermissionError as error:
                # an annotation write on a server that does not allow them
                raise ToolError(str(error)) from error
            except sqlite3.DatabaseError as error:
                # A query's own failures are reported by query_sql, but reading
                # an archive to build a roll-up happens before that, so a
                # database error can still reach here -- and it is the caller's
                # selection of runs that provoked it.
                raise ToolError(f"Could not read a run's archive: {error}") from error

        return call


def build_mcp_server(
    registry: RunRegistry,
    renderer: RenderService | None = None,
    base_url: str = "http://127.0.0.1:8000",
    allow_annotations: bool = False,
) -> MCPServer:
    """Builds the MCP server exposing one set of runs.

    Args:
        registry: The runs to serve.
        renderer: The dashboard's image renderer, shared so tools and the web UI
            use one cache; an inline renderer is created when none is given.
        base_url: The dashboard's base URL, used for the deep links in results.
        allow_annotations: Whether to offer the tools that write notes and tags.
            Without it they are not registered at all, so an agent is never
            shown a tool it cannot use; ``list_annotations`` is offered either way.

    Returns:
        The configured server, ready to mount over HTTP or run over stdio.
    """

    tools = _AnticipatedFailures(
        DashboardTools(registry, renderer, base_url, allow_annotations)
    )
    server = MCPServer(
        name="exaqc-dashboard",
        title="EXAQC run analysis",
        instructions=INSTRUCTIONS.strip(),
        version="1",
    )

    @server.tool(
        description=(
            "List the EXAQC runs being served, with their task, size and best fitness. "
            "Command lines are left out unless include_command_line is set."
        )
    )
    def list_runs(
        name_contains: str | None = None, include_command_line: bool = False
    ) -> dict[str, Any]:
        """Lists the served runs.

        Args:
            name_contains: Only list runs whose name contains this.
            include_command_line: Whether each row keeps the run's command line.

        Returns:
            One summary row per run, plus the groups they belong to.
        """

        return tools.list_runs(name_contains, include_command_line)

    @server.tool(
        description=(
            "Describe one run: what it was searching for, its size, the fitness keys it "
            "recorded and the values its genomes can be filtered by."
        )
    )
    def describe_run(run: str) -> dict[str, Any]:
        """Describes a run.

        Args:
            run: A run index or name.

        Returns:
            The run's summary, fitness keys and filter options.
        """

        return tools.describe_run(run)

    @server.tool(
        description=(
            "List a page of a run's genomes, sorted by a fitness key or column and "
            f"optionally filtered. Returns at most {MAX_ROWS} rows."
        )
    )
    def list_genomes(
        run: str,
        sort: str = "loss",
        descending: bool = False,
        insert_type: str | None = None,
        generated_by: str | None = None,
        island: int | None = None,
        species: int | None = None,
        limit: int = DEFAULT_ROWS,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Lists genomes of a run.

        Args:
            run: A run index or name.
            sort: A fitness key or summary column to sort by.
            descending: Whether to sort largest first.
            insert_type: Only list genomes inserted this way.
            generated_by: Only list genomes generated by this operator.
            island: Only list genomes from this island.
            species: Only list genomes from this species.
            limit: Rows to return.
            offset: Rows to skip.

        Returns:
            The matching page of genome summaries.
        """

        return tools.list_genomes(
            run,
            sort,
            descending,
            insert_type,
            generated_by,
            island,
            species,
            limit,
            offset,
        )

    @server.tool(
        description="Fetch one genome in full: its fitness, gates, parents, children and commands."
    )
    def get_genome(run: str, genome_number: int) -> dict[str, Any]:
        """Fetches one genome.

        Args:
            run: A run index or name.
            genome_number: The genome to fetch.

        Returns:
            The genome and its family.
        """

        return tools.get_genome(run, genome_number)

    @server.tool(
        description=(
            "List the notes and tags recorded for a run, or for one genome: conclusions "
            "people or agents wrote down, and labels such as 'candidate' marking genomes. "
            "Pass tag to find every genome carrying it."
        )
    )
    def list_annotations(
        run: str,
        genome_number: int | None = None,
        tag: str | None = None,
        include_removed: bool = False,
    ) -> dict[str, Any]:
        """Lists notes and tags.

        Args:
            run: A run index or name.
            genome_number: Only list this genome's notes and tags.
            tag: Only list tags with this name.
            include_removed: Also list tags that were removed.

        Returns:
            The notes and tags, and whether annotations may be written here.
        """

        return tools.list_annotations(run, genome_number, tag, include_removed)

    if allow_annotations:

        @server.tool(
            description=(
                "Record a note about a run, or about one genome when genome_number is "
                "given. Notes are kept beside the run, never in its archive, and are "
                "never edited or deleted, so write what was concluded and why."
            )
        )
        def add_note(
            run: str,
            text: str,
            genome_number: int | None = None,
            author: str | None = None,
        ) -> dict[str, Any]:
            """Records a note.

            Args:
                run: A run index or name.
                text: What to note.
                genome_number: The genome the note is about; the run otherwise.
                author: A name to record with the note.

            Returns:
                The recorded note.
            """

            return tools.add_note(run, text, genome_number, author)

        @server.tool(
            description=(
                "Tag a genome with a short label (letters, digits and _ . : -, no spaces), "
                "such as 'candidate' or 'needs:rerun'. Tagging a genome that already "
                "carries the tag changes nothing."
            )
        )
        def tag_genome(
            run: str, genome_number: int, tag: str, author: str | None = None
        ) -> dict[str, Any]:
            """Tags a genome.

            Args:
                run: A run index or name.
                genome_number: The genome to tag.
                tag: The tag.
                author: A name to record with the tag.

            Returns:
                The tag that now applies.
            """

            return tools.tag_genome(run, genome_number, tag, author)

        @server.tool(
            description=(
                "Remove a tag from a genome. The tag's history is kept: it is marked "
                "removed, with when and by whom, rather than deleted."
            )
        )
        def untag_genome(
            run: str, genome_number: int, tag: str, author: str | None = None
        ) -> dict[str, Any]:
            """Removes a tag from a genome.

            Args:
                run: A run index or name.
                genome_number: The genome to untag.
                tag: The tag to remove.
                author: A name to record with the removal.

            Returns:
                The tag, marked removed.
            """

            return tools.untag_genome(run, genome_number, tag, author)

    @server.tool(
        description=(
            "Return the per-epoch or per-episode metrics a genome recorded during "
            "training (loss, accuracies, returns, fidelities -- whatever its task "
            "records), each series keyed by its own epoch or episode column."
        )
    )
    def genome_metrics(
        run: str, genome_number: int, series: str | None = None
    ) -> dict[str, Any]:
        """Returns a genome's training history.

        Args:
            run: A run index or name.
            genome_number: The genome whose history is read.
            series: Only return this series, e.g. "validation_epoch_metrics".

        Returns:
            Each recorded series, with its step column, metrics and records.
        """

        return tools.genome_metrics(run, genome_number, series)

    @server.tool(
        description="Compare two genomes of a run: their gates by innovation number, fitness and hyperparameters."
    )
    def compare_genomes(run: str, a: int, b: int) -> dict[str, Any]:
        """Compares two genomes.

        Args:
            run: A run index or name.
            a: The first genome's number.
            b: The second genome's number.

        Returns:
            The differences between them.
        """

        return tools.compare_genomes(run, a, b)

    @server.tool(
        description="Trace a genome's ancestry back through the operators that produced it."
    )
    def genome_lineage(run: str, genome_number: int, depth: int = 5) -> dict[str, Any]:
        """Traces a genome's lineage.

        Args:
            run: A run index or name.
            genome_number: The genome to trace.
            depth: Generations to walk back.

        Returns:
            The ancestry graph and the genome's children.
        """

        return tools.genome_lineage(run, genome_number, depth)

    @server.tool(
        description=(
            "Summarize the distribution of a fitness key across a run's genomes, "
            "optionally grouped by insert_type, island, species or generated_by."
        )
    )
    def fitness_summary(
        run: str, key: str = "loss", group_by: str | None = None
    ) -> dict[str, Any]:
        """Summarizes a fitness key.

        Args:
            run: A run index or name.
            key: The fitness key to summarize.
            group_by: Optional grouping column.

        Returns:
            Overall and per-group statistics, and the best genome.
        """

        return tools.fitness_summary(run, key, group_by)

    @server.tool(
        description=(
            "Report how the genomes each operator generated were inserted (global best, "
            "local best, inserted, discarded), for a run or a group of runs."
        )
    )
    def operator_insertion_rates(
        run: str | None = None, group: str | None = None
    ) -> dict[str, Any]:
        """Reports operator insertion rates.

        Args:
            run: A run index or name.
            group: A group name instead.

        Returns:
            The insertion-rate table and its LaTeX form.
        """

        return tools.operator_insertion_rates(run, group)

    @server.tool(
        description=(
            "Return a run's search progress over time, downsampled. Choose which metric "
            "to summarize over the population (a fitness key, a circuit size, or a metric "
            "the task recorded while training) and which statistic of it to return."
        )
    )
    def progress_series(
        run: str,
        metric: str | None = None,
        statistic: str = "best",
        max_points: int = 200,
    ) -> dict[str, Any]:
        """Returns a run's progress series.

        Args:
            run: A run index or name.
            metric: The metric summarized over the population; the run's default
                when not given.
            statistic: Which series of it to return: ``best``, ``mean``,
                ``worst`` or ``population_size``.
            max_points: The most points to return.

        Returns:
            The downsampled series, the metrics worth charting and the
            statistics available.
        """

        return tools.progress_series(run, metric, statistic, max_points)

    @server.tool(
        description="Summarize genome sizes across a run and which gate methods its best genomes use."
    )
    def gate_statistics(run: str, sample: int = GATE_SAMPLE_SIZE) -> dict[str, Any]:
        """Summarizes gate usage and genome size.

        Args:
            run: A run index or name.
            sample: How many genomes to read for gate-method usage.

        Returns:
            Size statistics and gate-method counts.
        """

        return tools.gate_statistics(run, sample)

    @server.tool(
        description="Compare several runs (or a group) by their best genome and size."
    )
    def compare_runs(
        runs: list[str] | None = None, group: str | None = None, key: str = "loss"
    ) -> dict[str, Any]:
        """Compares runs.

        Args:
            runs: Run indexes or names; every run by default.
            group: A group name instead.
            key: The fitness key compared.

        Returns:
            Per-run bests and statistics across them.
        """

        return tools.compare_runs(runs, group, key)

    @server.tool(
        description="Describe the tables, columns and limits that query_sql works with, with examples."
    )
    def describe_schema() -> dict[str, Any]:
        """Describes the SQL schema.

        Returns:
            The single-run and cross-run schemas, limits and example queries.
        """

        return tools.describe_schema()

    @server.tool(
        description=(
            "Run one read-only SELECT over a run's archive, or over several runs rolled "
            "into one table with a run column. Call describe_schema first."
        )
    )
    def query_sql(
        sql: str,
        run: str | None = None,
        runs: list[str] | None = None,
        group: str | None = None,
        limit: int = MAX_ROWS,
    ) -> dict[str, Any]:
        """Runs a read-only query.

        Args:
            sql: The statement to run.
            run: A single run to query.
            runs: Several runs to roll up and query together.
            group: A group of runs to roll up and query together.
            limit: The most rows to return.

        Returns:
            The result columns and rows.
        """

        return tools.query_sql(sql, run, runs, group, limit)

    @server.tool(
        description=(
            "Page through the full result of a read-only SELECT (the statements query_sql "
            f"accepts) as compact CSV or JSON, up to {MAX_EXPORT_ROWS} rows a page, for a "
            "client collecting data to analyze itself. Pass each page's next_offset back "
            "as offset until complete is true, and ORDER BY the query so pages are stable."
        )
    )
    def export_query(
        sql: str,
        run: str | None = None,
        runs: list[str] | None = None,
        group: str | None = None,
        format: str = "csv",
        limit: int = DEFAULT_EXPORT_ROWS,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Returns one page of a query's full result.

        Args:
            sql: The statement to run.
            run: A single run to query.
            runs: Several runs to roll up and query together.
            group: A group of runs to roll up and query together.
            format: ``csv`` or ``json``.
            limit: The most rows in the page.
            offset: Rows of the result to skip: the previous page's next_offset.

        Returns:
            The page, its columns, and the offset the next page starts at.
        """

        return tools.export_query(sql, run, runs, group, format, limit, offset)

    return server
