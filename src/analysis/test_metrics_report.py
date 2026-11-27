"""Reports held-out test metrics for the genomes a search would hand back.

A classification run started with ``--test_fraction`` records every genome's
``test_loss`` and ``test_accuracy`` (its best weights scored once on a test set
the search never saw). This prints, per run:

- a single-objective run: its best genome by ``loss`` (the value selection
  used);
- a multi-objective run (``classification_moo``): every genome on the final
  population's Pareto front over the run's recorded objectives, plus the
  front's best by the first objective.

and, across runs, the mean and standard deviation of the test accuracy and
loss of each run's best genome. ``--csv`` also writes every reported row.

Example::

    python3 -m src.analysis.test_metrics_report --runs artifacts/genome_archives/* --csv test.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sqlite3
import statistics
import sys
from typing import Any

from src.utils.artifact_viewer.server import pareto_front_mask
from src.utils.genome_archive import ARCHIVE_FILENAME

#: Fitness keys a run records when it has a test set.
TEST_KEYS: tuple[str, str] = ("test_loss", "test_accuracy")


def build_parser() -> argparse.ArgumentParser:
    """Creates the report's command-line parser.

    Returns:
        The parser.
    """

    parser = argparse.ArgumentParser(
        description="Report held-out test metrics for each run's best genome or Pareto front."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=f"Run directories (holding {ARCHIVE_FILENAME}) or archive files; shell globs work.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Also write every reported genome to this CSV file.",
    )
    return parser


def archive_path(run: str) -> str:
    """Finds a run's archive from a directory or file path.

    Args:
        run: A run directory or an archive file.

    Returns:
        The archive file's path.

    Raises:
        FileNotFoundError: If no archive is there.
    """

    path = os.path.join(run, ARCHIVE_FILENAME) if os.path.isdir(run) else run
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {ARCHIVE_FILENAME} at {run}")
    return path


def final_population(connection: sqlite3.Connection) -> set[int]:
    """Replays a run's population events to find its final population.

    Args:
        connection: An open connection to the run's archive.

    Returns:
        The genome numbers the population held after the last insertion.
    """

    members: set[int] = set()
    for added, removed in connection.execute(
        "SELECT added, removed FROM population_events ORDER BY step"
    ):
        members.difference_update(json.loads(removed or "[]"))
        members.update(json.loads(added or "[]"))
    return members


def report_run(path: str) -> list[dict[str, Any]]:
    """Selects a run's reported genomes, as described in the module docstring.

    Args:
        path: The run's archive.

    Returns:
        One row per reported genome: ``run``, ``genome``, ``role`` (``best`` or
        ``front``), the run's objective values, ``loss``, ``target_metric``,
        ``test_loss`` and ``test_accuracy``.

    Raises:
        ValueError: If the run recorded no test metrics (it was not started
            with ``--test_fraction``) or holds no genomes.
    """

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    info = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM run_info")
    }
    genomes = {
        number: json.loads(fitness)
        for number, fitness in connection.execute(
            "SELECT genome_number, fitness FROM genomes"
        )
    }
    connection_population = final_population(connection)
    connection.close()
    if not genomes:
        raise ValueError(f"{path} holds no genomes.")
    if not any(TEST_KEYS[0] in fitness for fitness in genomes.values()):
        raise ValueError(
            f"{path} recorded no test metrics; start the run with --test_fraction."
        )

    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    objectives = (info.get("multi_objective") or {}).get("objectives") or []

    def row(number: int, role: str) -> dict[str, Any]:
        """Builds one report row.

        Args:
            number: The genome.
            role: ``best`` or ``front``.

        Returns:
            The row.
        """
        fitness = genomes[number]
        values = {o["name"]: fitness.get(o["name"]) for o in objectives}
        return {
            "run": run,
            "genome": number,
            "role": role,
            **values,
            "loss": fitness.get("loss"),
            "target_metric": fitness.get("target_metric"),
            "test_loss": fitness.get("test_loss"),
            "test_accuracy": fitness.get("test_accuracy"),
        }

    if not objectives:
        best = min(genomes, key=lambda n: (genomes[n].get("loss", float("inf")), n))
        return [row(best, "best")]

    population = sorted(n for n in connection_population if n in genomes) or sorted(
        genomes
    )
    first, second = objectives[0], objectives[1]
    xs = [genomes[n].get(first["name"]) for n in population]
    ys = [genomes[n].get(second["name"]) for n in population]
    mask = pareto_front_mask(
        xs,
        ys,
        first["direction"] == "maximize",
        second["direction"] == "maximize",
    )
    front = [n for n, on in zip(population, mask) if on]
    sign = -1.0 if first["direction"] == "maximize" else 1.0
    best = min(front, key=lambda n: (sign * genomes[n][first["name"]], n))
    return [row(best, "best")] + [row(n, "front") for n in front if n != best]


def main() -> None:
    """Prints the report (and writes the CSV when asked).

    Returns:
        None. Exits with an error message if no run could be reported.
    """

    args = build_parser().parse_args()
    paths = sorted(
        {
            archive_path(p)
            for pattern in args.runs
            for p in (glob.glob(pattern) or [pattern])
        }
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            rows.extend(report_run(path))
        except ValueError as error:
            print(f"skipping: {error}", file=sys.stderr)
    if not rows:
        sys.exit("no run had test metrics to report.")

    for r in rows:
        extra = "  ".join(
            f"{k} {r[k]:.4g}"
            for k in r
            if k not in ("run", "genome", "role", "loss", "target_metric", *TEST_KEYS)
            and isinstance(r[k], (int, float))
        )
        print(
            f"{r['run']:60s} {r['role']:5s} genome {r['genome']:5d}  "
            f"fitness loss {r['loss']:.4f} acc {r['target_metric']:.3f}  |  "
            f"TEST loss {r['test_loss']:.4f} acc {r['test_accuracy']:.3f}  {extra}"
        )

    best = [r for r in rows if r["role"] == "best"]
    accuracy = [r["test_accuracy"] for r in best]
    loss = [r["test_loss"] for r in best]

    def spread(values: list[float]) -> float:
        """The sample standard deviation, or 0 for a single run.

        Args:
            values: One value per run.

        Returns:
            The standard deviation.
        """
        return statistics.stdev(values) if len(values) > 1 else 0.0

    print(
        f"\n{len(best)} runs, best genome per run: test accuracy "
        f"{statistics.mean(accuracy):.4f} ± {spread(accuracy):.4f}, test loss "
        f"{statistics.mean(loss):.4f} ± {spread(loss):.4f}"
    )

    if args.csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
