"""Reads the search-progress history CSVs an EXAQC run writes.

Every run records a row in ``exaqc_history.csv`` after each genome is inserted
(see :class:`~src.utils.profiler.EXAQCProfiler`). This module only reads those
files, and deliberately imports nothing from the quantum stack, so lightweight
tools such as the EXAQC dashboard can load and aggregate histories without
paying for the profiler's dependencies.
"""

from __future__ import annotations

import csv
import math

import numpy as np

#: File name of a run's search-progress history inside its output directory.
HISTORY_FILENAME = "exaqc_history.csv"


def _to_float(value: str | None) -> float:
    """Parses one CSV cell as a float.

    Args:
        value: The cell's text.

    Returns:
        The number, or NaN if the cell is empty or not a number.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def history_columns(path: str) -> list[str]:
    """Reads the column names of a history CSV.

    Args:
        path: Path to a history CSV.

    Returns:
        The header's column names, in file order (empty for an empty file).
    """

    with open(path, "r", newline="") as history_file:
        return next(csv.reader(history_file), [])


def load_history_csv(path: str) -> list[dict[str, float]]:
    """Loads a history CSV into float-valued rows.

    Args:
        path: Path to a CSV file produced by ``EXAQCProfiler``.

    Returns:
        One dict per row, mapping column names to float values. Values that do
        not parse become NaN.
    """

    with open(path, "r", newline="") as history_file:
        return [
            {key: _to_float(value) for key, value in row.items()}
            for row in csv.DictReader(history_file)
        ]


def aggregate_history(
    csv_paths: list[str],
    metric: str = "top5_mean",
    conf: str = "std",
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Aligns several runs' histories and summarizes one metric across them.

    The runs are aligned on the step values they all share, and the metric's
    mean and a confidence band around it are computed at each of those steps.

    Args:
        csv_paths: History CSVs produced by ``EXAQCProfiler``, one per run.
        metric: Name of the metric column to summarize; a run without it
            contributes NaN.
        conf: Confidence band style: ``"std"`` for mean ± 1 standard deviation,
            or ``"95ci"`` for mean ± 1.96 * std / sqrt(number of runs).

    Returns:
        The common steps, followed by the metric's mean, lower bound and upper
        bound at each of them.

    Raises:
        FileNotFoundError: If ``csv_paths`` is empty.
        RuntimeError: If the runs have no step values in common.
    """

    if not csv_paths:
        raise FileNotFoundError("No history CSVs were given to aggregate.")

    runs = [load_history_csv(path) for path in csv_paths]

    step_sets = [
        {
            int(row["step"])
            for row in run
            if math.isfinite(row.get("step", float("nan")))
        }
        for run in runs
    ]
    common_steps = sorted(set.intersection(*step_sets))
    if not common_steps:
        raise RuntimeError(
            "No common steps across runs. Try using the same run length."
        )

    values = []
    for run in runs:
        step_to_value = {
            int(row["step"]): row.get(metric, float("nan"))
            for row in run
            if math.isfinite(row.get("step", float("nan")))
        }
        values.append([step_to_value[step] for step in common_steps])

    values_array = np.array(values, dtype=np.float32)
    # a step where every run is NaN would warn about an empty slice; its mean is
    # NaN either way
    with np.errstate(invalid="ignore"):
        mean = (
            np.nanmean(values_array, axis=0)
            if np.isfinite(values_array).any()
            else np.full(len(common_steps), np.nan)
        )
        std = (
            np.nanstd(values_array, axis=0)
            if np.isfinite(values_array).any()
            else np.full(len(common_steps), np.nan)
        )

    if conf.lower() == "std":
        return common_steps, mean, mean - std, mean + std

    sem = std / max(math.sqrt(len(runs)), 1.0)
    return common_steps, mean, mean - 1.96 * sem, mean + 1.96 * sem
