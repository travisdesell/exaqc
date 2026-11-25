"""Many dates of the same chip's calibration, for the drift study.

A drift series is simply the archive folder of one chip,
``data/calibration/<chip>/``, holding one raw calibration file per date as
``src.examples.collect_calibration`` writes them (``<chip>_<YYYY-MM-DD>.json``).
Each file is the same single-source file a noise run uses; the drift study
reads several of them and follows a fixed set of physical qubits across dates.
IBM serves past calibrations at least a year back (checked Nov 2026), each as
the calibration that was current at the end of the requested day.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from src.noise.calibration.backends import (
    fetch_raw_calibration,
    raw_calibration_path,
    save_raw_calibration,
)

if TYPE_CHECKING:
    from qiskit_ibm_runtime import IBMBackend


def date_range(first: date, last: date) -> list[date]:
    """Every calendar day from ``first`` to ``last`` inclusive.

    Args:
        first: The first day.
        last: The last day.

    Returns:
        The days in order; empty if ``last`` is before ``first``.
    """
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def collect_series(
    backend: IBMBackend, dates: list[date], out_dir: str | Path
) -> list[Path]:
    """Fetches and stores one raw calibration per date for one chip.

    Loops ``fetch_raw_calibration`` / ``save_raw_calibration`` over
    ``dates``. A date whose file already exists is skipped without a fetch,
    so the loop can be re-run to fill gaps or extend the series. A date IBM
    returns nothing for (before the chip went online) is logged and skipped.

    Args:
        backend: A backend from ``load_backend``.
        dates: The calendar days to fetch (end-of-day calibration each).
        out_dir: The archive root, such as ``data/calibration``.

    Returns:
        The paths written or already present, in date order.
    """
    paths: list[Path] = []
    for day in dates:
        expected = raw_calibration_path(out_dir, backend.name, day.isoformat())
        if expected.exists():
            paths.append(expected)
            continue
        try:
            raw = fetch_raw_calibration(backend, day)
        except ValueError as error:
            logger.warning("{}: {}", backend.name, error)
            continue
        path = save_raw_calibration(raw, out_dir)
        logger.info(
            "Saved {} calibration for {} (stamped {}) to {}",
            backend.name,
            day,
            raw["calibration_date"],
            path,
        )
        paths.append(path)
    return paths
