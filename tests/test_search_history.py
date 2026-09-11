"""Tests for :mod:`src.utils.search_history`.

Runs record their progress in ``exaqc_history.csv``. The profiler's multi-run
plots and the EXAQC dashboard's run comparison both aggregate those files with
:func:`~src.utils.search_history.aggregate_history`, so these tests pin its
alignment and statistics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils.search_history import (
    aggregate_history,
    history_columns,
    load_history_csv,
)


def write_history(path, rows: list[tuple[int, float]]) -> str:
    """Writes a minimal history CSV.

    Args:
        path: The file to write.
        rows: ``(step, best)`` pairs.

    Returns:
        The path written, as a string.
    """

    lines = ["step,best,top5_mean"] + [
        f"{step},{best},{best * 2}" for step, best in rows
    ]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_load_history_csv_parses_floats_and_blanks(tmp_path) -> None:
    """Cells become floats, and unparsable cells become NaN.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = tmp_path / "history.csv"
    path.write_text("step,best\n1,0.5\n2,\n")

    rows = load_history_csv(str(path))

    assert history_columns(str(path)) == ["step", "best"]
    assert rows[0] == {"step": 1.0, "best": 0.5}
    assert math.isnan(rows[1]["best"])


def test_aggregate_history_aligns_on_common_steps(tmp_path) -> None:
    """Only steps every run reached are aggregated.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    first = write_history(tmp_path / "a.csv", [(1, 1.0), (2, 2.0), (3, 3.0)])
    second = write_history(tmp_path / "b.csv", [(1, 3.0), (2, 4.0)])

    steps, mean, low, high = aggregate_history(
        [first, second], metric="best", conf="std"
    )

    assert steps == [1, 2]
    np.testing.assert_allclose(mean, [2.0, 3.0])
    np.testing.assert_allclose(low, [1.0, 2.0])
    np.testing.assert_allclose(high, [3.0, 4.0])

    _, ci_mean, ci_low, ci_high = aggregate_history(
        [first, second], metric="top5_mean", conf="95ci"
    )
    half_width = 1.96 * 2.0 / math.sqrt(2)
    np.testing.assert_allclose(ci_mean, [4.0, 6.0])
    np.testing.assert_allclose(ci_high - ci_mean, [half_width, half_width], rtol=1e-5)
    np.testing.assert_allclose(ci_mean - ci_low, [half_width, half_width], rtol=1e-5)


def test_aggregate_history_errors(tmp_path) -> None:
    """No files, or runs sharing no steps, are reported.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    with pytest.raises(FileNotFoundError):
        aggregate_history([])

    first = write_history(tmp_path / "a.csv", [(1, 1.0)])
    second = write_history(tmp_path / "b.csv", [(2, 1.0)])
    with pytest.raises(RuntimeError):
        aggregate_history([first, second], metric="best")


def test_a_missing_metric_aggregates_to_nan(tmp_path) -> None:
    """Asking for a column the runs lack yields NaN rather than an error.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    path = write_history(tmp_path / "a.csv", [(1, 1.0), (2, 2.0)])

    steps, mean, _, _ = aggregate_history([path], metric="absent")

    assert steps == [1, 2]
    assert np.isnan(mean).all()
