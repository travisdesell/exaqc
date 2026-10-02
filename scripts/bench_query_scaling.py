"""Measures how archive queries scale with run size, and across many runs.

Two questions decide whether the analysis interface holds up on real searches:

* **Within a run**: do the queries an agent asks stay fast as a run grows from a
  thousand to a hundred thousand genomes, and does indexing the generated
  ``loss`` column (format version 2) actually help? The same ranking query is
  timed against the indexed column and against ``json_extract`` of the stored
  fitness JSON, which is what a format-version-1 archive must use.

* **Across runs**: cross-run queries cannot attach one database per run, because
  SQLite allows at most ten attached databases and that ceiling is set when the
  library is compiled -- the probe below reports what this machine permits. They
  instead copy each run's summary rows into one in-memory database, so the cost
  of that roll-up is measured as the number of runs grows.

Synthetic archives are generated so the benchmark is self-contained::

    python3 -m scripts.bench_query_scaling --sizes 1000 20000 100000
    python3 -m scripts.bench_query_scaling --sizes 5000 --runs 1 2 4 8 --csv scaling.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from src.utils.artifact_viewer.mcp_tools import DashboardTools
from src.utils.artifact_viewer.server import RunRegistry
from src.utils.genome_archive import GenomeArchive

#: Operators the synthetic genomes are attributed to, so grouping queries have
#: something realistic to group by.
_OPERATORS = (
    ["add_gate"],
    ["remove_gate"],
    ["modify_parameters"],
    ["binary_crossover"],
    ["n_ary_crossover"],
)


class _SyntheticGenome:
    """A stand-in genome that serializes like a real one, for generating archives.

    Attributes:
        genome_number: The genome's number.
    """

    def __init__(self, genome_number: int, parents: list[int], loss: float) -> None:
        """Builds one synthetic genome.

        Args:
            genome_number: The genome's number.
            parents: The genomes it was generated from.
            loss: Its loss, which drives every ranking query.
        """

        self.genome_number = genome_number
        self._serialized: dict[str, Any] = {
            "genome_number": genome_number,
            "task": "classification",
            "task_target": "synthetic",
            "target": "pennylane",
            "fitness": {"loss": loss, "target_metric": max(0.0, 1.5 - loss)},
            "hyperparameters": {"epochs": 10},
            "metadata": {
                "parent_genomes": parents,
                "generated_by": random.choice(_OPERATORS),
                "insert_type": random.choice(["inserted", "inserted", "discarded"]),
            },
            "gates": [
                {
                    "innovation_number": index,
                    "method_name": "rx",
                    "qubits": [["input", 0]],
                    "parameters": {"theta": 0.1},
                    "depth": index / 10,
                    "enabled": True,
                    "target": "pennylane",
                }
                for index in range(random.randint(1, 12))
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializes the genome.

        Returns:
            The serialized genome.
        """

        return self._serialized


def build_archive(directory: Path, genomes: int, seed: int = 11) -> Path:
    """Generates a synthetic run archive of a given size.

    Args:
        directory: The run directory to create.
        genomes: How many genomes to store.
        seed: Seed for the random operators, parents and losses.

    Returns:
        The run directory.
    """

    random.seed(seed)
    best = 1.4
    with GenomeArchive.create(str(directory)) as archive:
        archive.set_run_info(task="classification", task_target="synthetic")
        for number in range(2, genomes + 2):
            parents = [1] if number < 12 else random.sample(range(2, number), 1)
            best = min(best, best + random.uniform(-0.001, 0.3))
            archive.add_genome(
                _SyntheticGenome(number, parents, best + random.uniform(0, 0.3)),
                insertion=number - 1,
            )
    return directory


def _median_ms(call: Callable[[], Any], repeat: int) -> float:
    """Times a call and returns its median duration in milliseconds.

    Args:
        call: The call to time.
        repeat: How many times to run it.

    Returns:
        The median duration in milliseconds.
    """

    timings = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        call()
        timings.append(1000 * (time.perf_counter() - start))
    return statistics.median(timings)


def attach_limit() -> int:
    """Reports how many databases this SQLite build allows to be attached.

    The runtime limit can only be lowered below the compile-time maximum, so a
    build compiled with the default of 10 cannot be raised to SQLite's documented
    ceiling of 125.

    Returns:
        The number of databases that could actually be attached.
    """

    connection = sqlite3.connect(":memory:")
    try:
        connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 125)
    except (AttributeError, sqlite3.Error):
        pass

    attached = 0
    try:
        for index in range(130):
            connection.execute(f"ATTACH ':memory:' AS probe{index}")
            attached += 1
    except sqlite3.Error:
        pass
    connection.close()
    return attached


def within_run(directory: Path, genomes: int, repeat: int) -> list[dict[str, Any]]:
    """Times the queries an agent asks of a single run.

    Args:
        directory: The run directory to query.
        genomes: How many genomes it holds, reported alongside each timing.
        repeat: How many times each query runs.

    Returns:
        One row per query: the run size, the query and its median latency.
    """

    tools = DashboardTools(RunRegistry(run_directories=[str(directory)]))
    with GenomeArchive.open_readonly(str(directory)) as reader:
        connection = reader.connection
        queries: list[tuple[str, Callable[[], Any]]] = [
            (
                "filtered count",
                lambda: connection.execute(
                    "SELECT COUNT(*) FROM genomes WHERE insert_type = 'inserted'"
                ).fetchall(),
            ),
            (
                "top-10 by loss (indexed column)",
                lambda: connection.execute(
                    "SELECT genome_number, loss FROM genomes ORDER BY loss LIMIT 10"
                ).fetchall(),
            ),
            (
                "top-10 by loss (json_extract)",
                lambda: connection.execute(
                    "SELECT genome_number, json_extract(fitness, '$.loss') AS loss "
                    "FROM genomes ORDER BY loss LIMIT 10"
                ).fetchall(),
            ),
            (
                "operator insertion rates",
                lambda: connection.execute(
                    "SELECT json_each.value, genomes.insert_type, COUNT(*) FROM genomes, "
                    "json_each(genomes.generated_by) GROUP BY 1, 2"
                ).fetchall(),
            ),
            (
                "median loss",
                lambda: connection.execute(
                    "SELECT loss FROM genomes ORDER BY loss "
                    "LIMIT 1 OFFSET (SELECT COUNT(*) / 2 FROM genomes)"
                ).fetchall(),
            ),
            ("lineage, depth 5", lambda: reader.ancestors(genomes, 5)),
            ("fitness_summary tool", lambda: tools.fitness_summary(0)),
            ("list_genomes tool (50)", lambda: tools.list_genomes(0, limit=50)),
        ]

        rows = []
        for name, call in queries:
            rows.append(
                {
                    "genomes": genomes,
                    "query": name,
                    "median_ms": round(_median_ms(call, repeat), 3),
                }
            )
    return rows


def across_runs(
    directory: Path, counts: list[int], repeat: int, workspace: Path
) -> list[dict[str, Any]]:
    """Times a cross-run query as the number of runs grows.

    Args:
        directory: A run archive copied to stand in for each run.
        counts: The run counts to measure.
        repeat: How many times each query runs.
        workspace: Where the copies are made.

    Returns:
        One row per run count: the count, the median latency and the genomes
        rolled up.
    """

    copies = []
    rows = []
    for count in sorted(counts):
        while len(copies) < count:
            target = workspace / f"run_{len(copies):03d}"
            shutil.copytree(directory, target)
            copies.append(str(target))

        tools = DashboardTools(RunRegistry(run_directories=copies[:count]))
        statement = "SELECT run, COUNT(*), MIN(loss) FROM genomes GROUP BY run"
        latency = _median_ms(
            lambda: tools.query_sql(statement, runs=list(range(count))), repeat
        )
        rolled = tools.query_sql(statement, runs=list(range(count)))
        rows.append(
            {
                "runs": count,
                "median_ms": round(latency, 1),
                "genomes": sum(row[1] for row in rolled["rows"]),
            }
        )
        logger.debug("rolled up {} runs", count)
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the benchmark.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description="Measure archive query latency by run size and by run count."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 20000, 100000],
        help="Genome counts of the synthetic archives to generate and query.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Run counts to measure a cross-run roll-up query over.",
    )
    parser.add_argument(
        "--rollup_size",
        type=int,
        default=1000,
        help="Genomes per run in the cross-run measurement.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="How many times each query runs; the median is reported.",
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


def main() -> None:
    """Generates the archives, runs both measurements and prints them.

    Returns:
        None. Writes the CSV as well when ``--csv`` is given.
    """

    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    workspace = Path(tempfile.mkdtemp(prefix="exaqc_scaling_"))
    rows: list[dict[str, Any]] = []
    try:
        print(
            f"SQLite {sqlite3.sqlite_version}: at most {attach_limit()} databases "
            "can be attached at once, so cross-run queries roll rows up instead.\n"
        )

        for size in args.sizes:
            directory = build_archive(workspace / f"size_{size}", size)
            rows.extend(within_run(directory, size, args.repeat))

        print(f"{'genomes':>8}  {'query':38}  {'median ms':>10}")
        for row in rows:
            print(f"{row['genomes']:>8,}  {row['query']:38}  {row['median_ms']:>10.3f}")

        source = build_archive(workspace / "rollup_source", args.rollup_size)
        rollup = across_runs(source, args.runs, args.repeat, workspace / "copies")
        print(f"\n{'runs':>5}  {'genomes':>9}  {'roll-up ms':>11}")
        for row in rollup:
            print(f"{row['runs']:>5}  {row['genomes']:>9,}  {row['median_ms']:>11.1f}")

        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["measurement", "size", "query", "median_ms"])
                for row in rows:
                    writer.writerow(
                        ["within_run", row["genomes"], row["query"], row["median_ms"]]
                    )
                for row in rollup:
                    writer.writerow(
                        ["across_runs", row["runs"], "roll-up query", row["median_ms"]]
                    )
            print(f"\nWrote {args.csv}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
