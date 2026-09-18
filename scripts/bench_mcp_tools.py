"""Measures what each MCP tool costs an agent: how long it takes and how big it is.

The interface is aggregate-first because a reply has to fit in a model's context
as well as return quickly, so both are measured here: the median wall-clock time
of a call and the size of its serialized result, in KiB and in an estimate of
tokens. Run it against any archive::

    python3 -m scripts.bench_mcp_tools --runs ./artifacts/iris --repeat 5
    python3 -m scripts.bench_mcp_tools --directory ~/Data/2026_ppsn_exaqc --csv tools.csv

The token estimate is bytes / 4, the usual rule of thumb for English-and-JSON
text; it is reported to compare tools with each other, not as an exact count.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from typing import Any, Callable

from loguru import logger

from src.utils.artifact_viewer.mcp_tools import DashboardTools
from src.utils.artifact_viewer.server import RunRegistry

#: Bytes per token, the rule of thumb used to report result sizes.
BYTES_PER_TOKEN = 4


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the benchmark.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description="Measure the latency and response size of each MCP tool."
    )

    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help="Run output directories (or their genomes.sqlar files) to measure against.",
    )
    sources.add_argument(
        "--directory",
        type=str,
        default=None,
        help="A directory to search for runs instead.",
    )

    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Which run the per-run tools are measured on; the largest one by default.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="How many times each tool is called; the median is reported.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write the measurements to this CSV file as well as printing them.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="WARNING",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    return parser


def measure(call: Callable[[], Any], repeat: int) -> tuple[float, int]:
    """Times a tool call and measures the size of what it returns.

    Args:
        call: Invokes the tool with no arguments.
        repeat: How many times to call it.

    Returns:
        The median latency in milliseconds and the serialized size in bytes.
    """

    timings = []
    payload: Any = None
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        payload = call()
        timings.append(1000 * (time.perf_counter() - start))
    return statistics.median(timings), len(json.dumps(payload, default=str))


def benchmark(tools: DashboardTools, run: str, repeat: int) -> list[dict[str, Any]]:
    """Measures every tool against one run, and the cross-run tools against all.

    Args:
        tools: The tool layer to measure.
        run: The run the per-run tools are called on.
        repeat: How many times each tool is called.

    Returns:
        One row per tool: its name, median latency, size in KiB and estimated
        tokens.
    """

    best = tools.list_genomes(run, limit=2)["rows"]
    genome = best[0]["genome_number"] if best else 1
    other = best[1]["genome_number"] if len(best) > 1 else genome

    calls: list[tuple[str, Callable[[], Any]]] = [
        ("list_runs", lambda: tools.list_runs()),
        ("describe_run", lambda: tools.describe_run(run)),
        ("list_genomes (50)", lambda: tools.list_genomes(run, limit=50)),
        ("list_genomes (200)", lambda: tools.list_genomes(run, limit=200)),
        ("get_genome", lambda: tools.get_genome(run, genome)),
        ("compare_genomes", lambda: tools.compare_genomes(run, genome, other)),
        ("genome_lineage", lambda: tools.genome_lineage(run, genome, depth=5)),
        ("fitness_summary", lambda: tools.fitness_summary(run)),
        (
            "fitness_summary (grouped)",
            lambda: tools.fitness_summary(run, group_by="generated_by"),
        ),
        ("operator_insertion_rates", lambda: tools.operator_insertion_rates(run=run)),
        ("progress_series", lambda: tools.progress_series(run)),
        ("gate_statistics", lambda: tools.gate_statistics(run)),
        ("compare_runs", lambda: tools.compare_runs()),
        ("describe_schema", lambda: tools.describe_schema()),
        (
            "query_sql (aggregate)",
            lambda: tools.query_sql(
                "select insert_type, count(*) from genomes group by 1", run=run
            ),
        ),
        (
            "query_sql (top-k)",
            lambda: tools.query_sql(
                "select genome_number, json_extract(fitness, '$.loss') as loss "
                "from genomes order by loss limit 20",
                run=run,
            ),
        ),
    ]

    rows = []
    for name, call in calls:
        latency, size = measure(call, repeat)
        rows.append(
            {
                "tool": name,
                "median_ms": round(latency, 2),
                "kib": round(size / 1024, 1),
                "tokens": size // BYTES_PER_TOKEN,
            }
        )
        logger.debug("measured {}", name)
    return rows


def main() -> None:
    """Runs the benchmark and prints a table of the results.

    Returns:
        None. Writes the CSV as well when ``--csv`` is given.
    """

    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    try:
        registry = RunRegistry(
            run_directories=args.runs, watch_directory=args.directory
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if not registry.runs:
        parser.error("No runs were found to measure.")

    tools = DashboardTools(registry)
    run = args.run
    if run is None:
        counts = [
            (row["genomes"] or 0, row["name"]) for row in tools.list_runs()["rows"]
        ]
        run = max(counts)[1]

    genomes = tools.describe_run(run)["genomes"]
    print(
        f"Measuring the MCP tools on {run} ({genomes} genomes), {args.repeat} calls each.\n"
    )

    rows = benchmark(tools, run, args.repeat)
    width = max(len(row["tool"]) for row in rows)
    print(f"{'tool'.ljust(width)}  {'median ms':>10}  {'KiB':>8}  {'~tokens':>9}")
    for row in rows:
        print(
            f"{row['tool'].ljust(width)}  {row['median_ms']:>10.2f}  "
            f"{row['kib']:>8.1f}  {row['tokens']:>9,}"
        )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
