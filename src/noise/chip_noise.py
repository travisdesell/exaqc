"""Noise settings taken from one stored IBM calibration file.

One file is the single source for a run: a raw calibration written by
``src.examples.collect_calibration`` (``data/calibration/<chip>/<chip>_<date>.json``),
holding the chip's calibration exactly as IBM reported it on that date. A run
names it with ``--noise_calibration``, either as the file's path or as
``<chip>:<YYYY-MM-DD>`` (:func:`resolve_calibration` then finds the stored
file, or fetches that day from IBM once if it is missing), and the physical
qubits its circuit wires map to with ``--noise_qubits``. Nothing is averaged:
circuit wire ``i`` gets the listed ``i``-th physical qubit's own T1, T2,
readout error and gate errors, and each consecutive pair its own two-qubit
error (``PennyLaneNoiseModel.from_calibration``).

Every genome records the file name, the qubits and the values it was scored
with, so jobs that start at different times on the cluster use identical noise
on identical qubits.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

CALIBRATION_DIR = "data/calibration"
"""Where calibration files live: the collector's ``--out_dir`` default."""

_CHIP_AND_DAY = re.compile(r"^(?P<chip>[A-Za-z0-9_-]+):(?P<day>\d{4}-\d{2}-\d{2})$")
"""The ``<chip>:<YYYY-MM-DD>`` form of ``--noise_calibration``, e.g. ``ibm_fez:2026-11-06``."""


def resolve_calibration(spec: str | Path, out_dir: str | Path | None = None) -> Path:
    """Turns ``--noise_calibration`` into the file to read, fetching it once if missing.

    Two forms are accepted. A path, such as
    ``data/calibration/ibm_fez/ibm_fez_2026-11-06.json``, must exist.
    ``<chip>:<YYYY-MM-DD>``, such as ``ibm_fez:2026-11-06``, names the file
    the collector would write for that chip and day under ``out_dir``: if it
    is there it is used as is, without touching the network; if not, that
    day's calibration is fetched from IBM with the account saved once by
    ``QiskitRuntimeService.save_account`` and stored under that name, exactly
    as ``collect_calibration --chip <chip> --date <day>`` would, so later runs
    find it. The day is always given; there is no default to today.

    Args:
        spec: The flag's value: a path, or ``<chip>:<YYYY-MM-DD>``.
        out_dir: The archive root the chip-and-day form looks in and saves
            to; ``None`` means :data:`CALIBRATION_DIR`.

    Returns:
        The calibration file, which exists when this returns.

    Raises:
        ValueError: If a path does not exist; if the day is not a calendar
            day or lies in the future; or if the file is missing and fetching
            it failed (no saved account, no network, unknown chip, no
            calibration served for that day), naming the collect command to
            run by hand.
    """
    text = str(spec)
    match = _CHIP_AND_DAY.match(text)
    if match is None:
        path = Path(spec)
        if not path.is_file():
            raise ValueError(f"--noise_calibration file not found: {spec}")
        return path

    chip, day_text = match.group("chip"), match.group("day")
    try:
        day = date.fromisoformat(day_text)
    except ValueError:
        raise ValueError(
            f"--noise_calibration {text!r}: {day_text!r} is not a calendar day (YYYY-MM-DD)."
        )
    if day > date.today():
        raise ValueError(f"--noise_calibration {text!r}: {day_text} is in the future.")

    # imported here: base_noise_model imports this module
    from src.noise.calibration.backends import raw_calibration_path

    if out_dir is None:
        out_dir = CALIBRATION_DIR
    path = raw_calibration_path(out_dir, chip, day_text)
    if path.is_file():
        return path

    logger.info(
        "{} is not stored; fetching the {} calibration of {} from IBM",
        path,
        chip,
        day_text,
    )
    by_hand = (
        f"python3 -m src.examples.collect_calibration --chip {chip} --date {day_text} "
        f"--out_dir {out_dir} --api_key_file <your IBM Cloud key file>"
    )
    try:
        from src.noise.calibration.backends import (
            connect,
            fetch_raw_calibration,
            load_backend,
            save_raw_calibration,
        )

        service = connect()
        raw = fetch_raw_calibration(load_backend(service, chip), day)
        saved = save_raw_calibration(raw, out_dir)
    except Exception as exc:  # whatever stopped the fetch, the fix is the same command
        raise ValueError(
            f"--noise_calibration {text!r}: {path} is not stored and fetching it failed "
            f"({exc}). Save your IBM account once with QiskitRuntimeService.save_account, "
            f"or run: {by_hand}"
        ) from exc
    logger.info(
        "Saved the {} calibration dated {} to {}", chip, raw["calibration_date"], saved
    )
    return saved


def noise_hyperparameters_from_calibration(
    path: str | Path,
    physical_qubits: list[int],
    noise_type: str = "ibm_backend",
) -> dict[str, Any]:
    """Turns a calibration file and the chosen qubits into genome noise settings.

    Called once at run start by ``from_args`` when ``--noise_calibration`` and
    ``--noise_qubits`` are given. Finds the file (:func:`resolve_calibration`,
    fetching it once if a ``<chip>:<day>`` form names a day not stored yet),
    reads it, takes the chosen qubits' own values (``from_calibration``) and
    returns them as the ``noise_*`` settings every genome carries, plus the
    file's path so each genome records where its noise came from.

    Args:
        path: The calibration file, e.g.
            ``data/calibration/ibm_fez/ibm_fez_2026-11-06.json``, or
            ``<chip>:<YYYY-MM-DD>``, e.g. ``ibm_fez:2026-11-06``.
        physical_qubits: The chip's qubits the circuit sits on, in wire order.
        noise_type: The noise channel these values drive.

    Returns:
        The ``noise_*`` settings, including ``noise_per_qubit``,
        ``noise_chip``, ``noise_calibration_date`` and ``noise_calibration``
        (the resolved file's path, never the ``<chip>:<day>`` form).

    Raises:
        ValueError: From :func:`resolve_calibration` (missing file, bad day,
            failed fetch); if the file cannot be read as a calibration; or
            from ``from_calibration`` (a qubit not on the chip, unwired
            neighbours, a missing or unusable value).
    """
    # imported here: base_noise_model imports this module
    from src.noise.calibration.backends import load_raw_calibration
    from src.noise.pennylane_noise_model import PennyLaneNoiseModel

    path = resolve_calibration(path)
    raw = load_raw_calibration(path)
    model = PennyLaneNoiseModel.from_calibration(raw, physical_qubits, noise_type)
    hp = model.to_hyperparameters()
    hp["noise_calibration"] = str(path)
    return hp
