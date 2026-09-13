"""Tests for the MCP interface (``python3 -m src.examples.exaqc_mcp``).

The dashboard's analysis tools are exposed to agents two ways: mounted at
``/mcp`` inside the dashboard's application, and over stdio by the entry point.
Both serve the same :class:`~src.utils.artifact_viewer.mcp_tools.DashboardTools`,
so these tests check that tool layer directly -- its answers, its response caps
and the guards on ``query_sql`` -- and then check that a real MCP client can
complete a handshake and call a tool, in process and over HTTP.

The small archives are built with the helpers the dashboard's own tests use.
"""

from __future__ import annotations

import asyncio
import hashlib
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn
from mcp import Client

from src.examples import exaqc_mcp
from src.utils.artifact_viewer.app import create_app
from src.utils.artifact_viewer.mcp_app import build_mcp_server
from src.utils.artifact_viewer.mcp_tools import (
    MAX_ROWS,
    MAX_SERIES_POINTS,
    DashboardTools,
)
from src.utils.artifact_viewer.server import ArtifactViewer, RenderService, RunRegistry
from tests.test_exaqc_dashboard import build_run, standard_genomes

#: The tools the interface is expected to expose.
EXPECTED_TOOLS = {
    "list_runs",
    "describe_run",
    "list_genomes",
    "get_genome",
    "genome_metrics",
    "compare_genomes",
    "genome_lineage",
    "fitness_summary",
    "operator_insertion_rates",
    "progress_series",
    "gate_statistics",
    "compare_runs",
    "describe_schema",
    "query_sql",
}


@pytest.fixture
def tools(tmp_path: Path) -> DashboardTools:
    """Builds two small runs and the tool layer serving them.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).

    Returns:
        The tools, serving runs named ``iris_1`` and ``iris_2``.
    """

    build_run(tmp_path / "runs" / "iris_1", standard_genomes())
    build_run(tmp_path / "runs" / "iris_2", standard_genomes()[:2])
    registry = RunRegistry(
        run_directories=[
            str(tmp_path / "runs" / "iris_1"),
            str(tmp_path / "runs" / "iris_2"),
        ],
        groups=["iris"],
    )
    return DashboardTools(registry, RenderService(processes=0), "http://dash.test:8000")


@contextmanager
def _serving(tools: DashboardTools) -> Iterator[str]:
    """Serves the dashboard and its mounted MCP endpoint on a free port.

    Args:
        tools: The tool layer whose runs are served.

    Yields:
        The base URL, e.g. ``http://127.0.0.1:54321``.
    """

    mcp_server = build_mcp_server(tools.registry)
    app = create_app(
        ArtifactViewer(tools.registry, RenderService(processes=0)), mcp_server
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(
        target=running.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not running.started and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        running.should_exit = True
        thread.join(timeout=10)
        listener.close()


def test_parser_defaults() -> None:
    """The MCP entry point's arguments and defaults match the documentation."""

    args = exaqc_mcp.build_parser().parse_args(["--runs", "runs/iris_1"])

    assert args.runs == ["runs/iris_1"]
    assert args.directory is None
    assert args.groups is None
    assert args.dashboard_url == "http://127.0.0.1:8000"
    assert args.logging_level == "WARNING"

    watching = exaqc_mcp.build_parser().parse_args(["--directory", "runs"])
    assert watching.directory == "runs"

    # one of --runs and --directory is required, but not both
    for arguments in ([], ["--runs", "a", "--directory", "b"]):
        with pytest.raises(SystemExit):
            exaqc_mcp.build_parser().parse_args(arguments)


def test_every_tool_is_registered_with_a_schema(tools: DashboardTools) -> None:
    """The server exposes the documented tools, each with a described schema.

    Args:
        tools: The tool layer fixture.
    """

    server = build_mcp_server(tools.registry)
    registered = asyncio.run(server.list_tools())

    assert {tool.name for tool in registered} == EXPECTED_TOOLS
    by_name = {tool.name: tool for tool in registered}
    assert set(by_name["query_sql"].input_schema["properties"]) == {
        "sql",
        "run",
        "runs",
        "group",
        "limit",
    }
    assert by_name["query_sql"].input_schema["required"] == ["sql"]
    assert all(tool.description for tool in registered)


def test_runs_and_genomes_are_listed_with_deep_links(tools: DashboardTools) -> None:
    """Listing tools report the runs, their genomes and where to see them.

    Args:
        tools: The tool layer fixture.
    """

    runs = tools.list_runs()
    assert runs["total"] == 2
    assert [row["name"] for row in runs["rows"]] == ["iris_1", "iris_2"]
    assert runs["dashboard_url"] == "http://dash.test:8000/#/"
    assert "| run |" in runs["table"]

    described = tools.describe_run("iris_1")
    assert described["fitness_keys"] == ["loss", "target_metric"]
    assert described["dashboard_url"].endswith("#/run/0")

    genomes = tools.list_genomes("iris_1", sort="loss")
    assert [row["genome_number"] for row in genomes["rows"]] == [3, 2, 1, 4]
    assert genomes["total"] == 4

    # runs are addressable by name or by index, and an unknown one is reported
    assert tools.describe_run("0")["name"] == "iris_1"
    with pytest.raises(KeyError, match="no run"):
        tools.list_genomes("nope")


def test_genome_detail_comparison_and_lineage(tools: DashboardTools) -> None:
    """A genome can be fetched, compared and traced back through its ancestry.

    Args:
        tools: The tool layer fixture.
    """

    genome = tools.get_genome("iris_1", 3)
    assert genome["summary"]["parents"] == [1, 2]
    assert genome["children"] == [4]
    assert "--archive" in genome["commands"]["refine_genome"]

    comparison = tools.compare_genomes("iris_1", 1, 2)
    assert [gate["innovation_number"] for gate in comparison["gates"]["only_a"]] == [2]
    assert comparison["dashboard_url"].endswith("#/run/0/compare/1/2")

    lineage = tools.genome_lineage("iris_1", 4, depth=5)
    generations = {
        node["genome_number"]: node["generation"] for node in lineage["nodes"]
    }
    assert generations == {4: 0, 3: 1, 1: 2, 2: 2, 0: 3}
    assert lineage["children"] == []


def test_genome_metrics_keeps_each_recorded_series_separate(tmp_path: Path) -> None:
    """Training histories are returned per series, flattened, keyed by their own step.

    What a genome records depends on its task: classification and teacher
    genomes record per-epoch series, reinforcement learning records per-episode
    ones whose training and evaluation halves run at different cadences and
    carry different metrics. Nothing may be merged or assumed.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = standard_genomes()
    genomes[0].serialized["metadata"]["training_epoch_metrics"] = [
        {
            "epoch": epoch,
            "loss": 1.0 / (epoch + 1),
            "mean_class_accuracy": {"0": {"acc": 0.9, "correct": 9}, "mean": 0.95},
        }
        for epoch in range(3)
    ]
    genomes[0].serialized["metadata"]["validation_epoch_metrics"] = [
        {"epoch": epoch, "loss": 1.1 / (epoch + 1)} for epoch in range(3)
    ]
    genomes[1].serialized["metadata"]["training_episode_metrics"] = [
        {"episode": episode, "return": float(episode), "loss": 0.5}
        for episode in range(6)
    ]
    genomes[1].serialized["metadata"]["evaluation_episode_metrics"] = [
        {"episode": 0, "return_mean": 1.0, "return_std": 0.1}
    ]
    build_run(tmp_path / "metrics_run", genomes)
    tools = DashboardTools(
        RunRegistry(run_directories=[str(tmp_path / "metrics_run")]),
        RenderService(processes=0),
    )

    epochs = tools.genome_metrics(0, 1)
    assert epochs["available"] == [
        "training_epoch_metrics",
        "validation_epoch_metrics",
    ]
    training = next(
        entry for entry in epochs["series"] if entry["name"] == "training_epoch_metrics"
    )
    assert training["step"] == "epoch"
    # the nested per-class breakdown is flattened rather than dropped or stringified
    assert "mean_class_accuracy.0.acc" in training["metrics"]
    assert "mean_class_accuracy.mean" in training["metrics"]
    assert training["records"][0]["mean_class_accuracy.mean"] == 0.95

    episodes = tools.genome_metrics(0, 2)
    by_name = {entry["name"]: entry for entry in episodes["series"]}
    assert set(by_name) == {"training_episode_metrics", "evaluation_episode_metrics"}
    assert by_name["training_episode_metrics"]["step"] == "episode"
    # the two series are recorded at different cadences, so they stay separate
    assert len(by_name["training_episode_metrics"]["records"]) == 6
    assert len(by_name["evaluation_episode_metrics"]["records"]) == 1
    assert by_name["evaluation_episode_metrics"]["metrics"] == [
        "return_mean",
        "return_std",
    ]

    one = tools.genome_metrics(0, 2, series="evaluation_episode_metrics")
    assert [entry["name"] for entry in one["series"]] == ["evaluation_episode_metrics"]
    with pytest.raises(ValueError, match="not recorded"):
        tools.genome_metrics(0, 2, series="nope")

    # a genome that recorded nothing says so rather than failing
    assert tools.genome_metrics(0, 4)["series"] == []


def test_aggregates_summarize_fitness_and_operators(tools: DashboardTools) -> None:
    """Aggregate tools answer without returning every genome.

    Args:
        tools: The tool layer fixture.
    """

    summary = tools.fitness_summary("iris_1", key="loss")
    assert summary["overall"]["n"] == 4
    assert summary["overall"]["min"] == pytest.approx(0.3)
    assert summary["best"]["genome_number"] == 3

    grouped = tools.fitness_summary("iris_1", key="loss", group_by="insert_type")
    assert {row["group"] for row in grouped["rows"]} == {
        "inserted",
        "global_best",
        "discarded",
    }

    rates = tools.operator_insertion_rates(run="iris_1")
    assert rates["columns"][0]["counts"]["add_gate"]["total"] == 2
    assert "\\begin{tabular}" in rates["latex"]

    compared = tools.compare_runs(group="iris")
    assert [row["run"] for row in compared["rows"]] == ["iris_1", "iris_2"]
    assert compared["summary"]["n"] == 2


def test_series_are_downsampled_rather_than_returned_whole(
    tools: DashboardTools,
) -> None:
    """Progress series are capped, so a long run cannot flood the reply.

    Args:
        tools: The tool layer fixture.
    """

    series = tools.progress_series("iris_1", metric="best", max_points=2)
    assert series["returned_points"] <= 2
    assert series["recorded_points"] >= series["returned_points"]
    assert len(series["step"]) == len(series["value"])
    assert "best" in series["metrics"]

    with pytest.raises(ValueError, match="not recorded"):
        tools.progress_series("iris_1", metric="nonexistent")

    capped = tools.progress_series("iris_1", max_points=10_000)
    assert capped["returned_points"] <= MAX_SERIES_POINTS


def test_listings_are_capped(tools: DashboardTools) -> None:
    """A caller cannot ask for more rows than the interface will return.

    Args:
        tools: The tool layer fixture.
    """

    listing = tools.list_genomes("iris_1", limit=10_000)
    assert listing["limit"] == MAX_ROWS


def test_query_sql_reads_one_run_and_several(tools: DashboardTools) -> None:
    """SQL runs against one archive, or a roll-up of runs with a run column.

    Args:
        tools: The tool layer fixture.
    """

    single = tools.query_sql(
        "select insert_type, count(*) from genomes group by 1 order by 1", run="iris_1"
    )
    assert single["columns"][0] == "insert_type"
    assert single["rows"] == [["discarded", 1], ["global_best", 2], ["inserted", 1]]
    assert single["scope"] == {"kind": "run", "run": "iris_1"}

    rolled = tools.query_sql(
        "select run, count(*) as genomes, min(loss) as best from genomes "
        "group by run order by run",
        runs=["iris_1", "iris_2"],
    )
    assert rolled["rows"] == [["iris_1", 4, 0.3], ["iris_2", 2, 0.4]]
    assert rolled["scope"]["kind"] == "rollup"

    grouped = tools.query_sql("select count(*) from genomes", group="iris")
    assert grouped["rows"] == [[6]]

    # the roll-up also carries the run metadata table
    assert tools.query_sql("select count(*) from runs", group="iris")["rows"] == [[2]]


def test_query_sql_enforces_read_only_access(tools: DashboardTools) -> None:
    """Only single read-only statements run: the text check and the authorizer.

    The first-token check rejects obvious writes, and SQLite's authorizer rejects
    what that check cannot see, such as PRAGMA table-valued functions.

    Args:
        tools: The tool layer fixture.
    """

    for statement in (
        "delete from genomes",
        "update genomes set island = 1",
        "attach database 'x' as y",
        "pragma table_info(genomes)",
        "select 1; select 2",
        "",
    ):
        with pytest.raises(ValueError):
            tools.query_sql(statement, run="iris_1")

    # these pass the text check, so the authorizer has to stop them
    for statement in (
        "select * from pragma_table_info('genomes')",
        "select load_extension('/tmp/evil.so')",
    ):
        with pytest.raises(ValueError, match="not authorized|Query failed"):
            tools.query_sql(statement, run="iris_1")

    # and a legitimate common table expression still works
    allowed = tools.query_sql(
        "with best as (select genome_number, json_extract(fitness, '$.loss') as loss "
        "from genomes order by loss limit 2) select * from best",
        run="iris_1",
    )
    assert [row[0] for row in allowed["rows"]] == [3, 2]

    # the archive is untouched by a query
    assert tools.list_genomes("iris_1")["total"] == 4


def test_query_sql_limits_rows(tools: DashboardTools) -> None:
    """Every query runs under an enforced row limit.

    Args:
        tools: The tool layer fixture.
    """

    limited = tools.query_sql(
        "select genome_number from genomes", run="iris_1", limit=2
    )
    assert limited["row_count"] == 2
    assert limited["row_limit"] == 2

    clamped = tools.query_sql(
        "select genome_number from genomes", run="iris_1", limit=10_000
    )
    assert clamped["row_limit"] == MAX_ROWS


def test_schema_description_lists_what_can_be_queried(tools: DashboardTools) -> None:
    """describe_schema tells an agent the tables, limits and example queries.

    Args:
        tools: The tool layer fixture.
    """

    schema = tools.describe_schema()
    assert "genomes" in schema["single_run"]
    assert "run" in schema["cross_run"]["genomes"]
    assert schema["limits"]["rows"] == MAX_ROWS
    assert any("json_extract" in example for example in schema["examples"])


def test_a_client_can_handshake_in_process(tools: DashboardTools) -> None:
    """An MCP client connects to the server object and calls a tool.

    Args:
        tools: The tool layer fixture.
    """

    server = build_mcp_server(tools.registry)

    async def exchange() -> tuple[int, str]:
        """Connects in process, lists the tools and calls one."""
        async with Client(server) as client:
            listed = await client.list_tools()
            names = listed.tools if hasattr(listed, "tools") else listed
            result = await client.call_tool(
                "query_sql", {"sql": "select count(*) from genomes", "run": "iris_1"}
            )
            return len(names), result.content[0].text

    count, text = asyncio.run(exchange())
    assert count == len(EXPECTED_TOOLS)
    assert "4" in text


def test_the_dashboard_serves_mcp_on_the_same_port(tools: DashboardTools) -> None:
    """The mounted endpoint answers at /mcp while the dashboard keeps serving.

    Args:
        tools: The tool layer fixture.
    """

    with _serving(tools) as base_url:

        async def exchange() -> list[str]:
            """Handshakes over HTTP and lists the tool names."""
            async with Client(f"{base_url}/mcp") as client:
                listed = await client.list_tools()
                names = listed.tools if hasattr(listed, "tools") else listed
                return [tool.name for tool in names]

        assert set(asyncio.run(exchange())) == EXPECTED_TOOLS

        with urllib.request.urlopen(f"{base_url}/api/runs") as response:
            assert response.status == 200


def test_tools_never_modify_an_archive(tools: DashboardTools) -> None:
    """Reading through every tool leaves the archives byte for byte unchanged.

    The tools open archives read-only -- a ``query_only`` connection, or a
    ``mode=ro`` URI for SQL -- so rather than asserting how a connection was
    opened, this checks the property those choices exist to guarantee, and that
    no journal or write-ahead log is left behind.

    Args:
        tools: The tool layer fixture.
    """

    def fingerprint() -> dict[Path, tuple[int, str]]:
        """Returns each archive's size and content hash."""
        return {
            path: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (Path(run.archive_path) for run in tools.registry.runs)
        }

    before = fingerprint()
    assert before, "expected the fixture to serve archives"

    tools.list_runs()
    tools.describe_run("iris_1")
    tools.list_genomes("iris_1")
    tools.get_genome("iris_1", 3)
    tools.genome_lineage("iris_1", 4)
    tools.fitness_summary("iris_1", group_by="insert_type")
    tools.operator_insertion_rates(run="iris_1")
    tools.gate_statistics("iris_1")
    tools.compare_runs(group="iris")
    tools.progress_series("iris_1")
    tools.query_sql("select count(*) from genomes", run="iris_1")
    tools.query_sql(
        "select run, count(*) from genomes group by run", runs=["iris_1", "iris_2"]
    )

    assert fingerprint() == before
    for path in before:
        assert not list(path.parent.glob("genomes.sqlar-*"))
