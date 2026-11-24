"""Tests for resolving ``--noise_calibration`` into a stored calibration file.

The flag takes a file path or ``<chip>:<YYYY-MM-DD>``. These tests check that
``resolve_calibration``:

* returns an existing path as is and refuses a missing one;
* with ``<chip>:<day>``, uses the stored file without touching the network;
* with ``<chip>:<day>`` and no stored file, fetches that day with the
  collector's own functions and stores it under the collector's name;
* when the fetch fails, stops with the ``collect_calibration`` command to run;
* refuses a day that is not a calendar day or lies in the future;

and that ``noise_hyperparameters_from_calibration`` records the resolved path,
never the ``<chip>:<day>`` text. No test talks to IBM: the collector's
functions are replaced with fakes.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

import src.noise.calibration.backends as backends
from src.noise import chip_noise
from src.noise.calibration.backends import raw_calibration_path

CHIP = "ibm_fez"
DAY = "2026-11-06"


def _store(out_dir: Path, chip: str, day: str, content: dict[str, Any]) -> Path:
    """Writes a stand-in calibration file where the collector would put it.

    Args:
        out_dir: The archive root.
        chip: The chip name.
        day: The archive day, ``YYYY-MM-DD``.
        content: What to write, as JSON.

    Returns:
        The path written.
    """
    path = raw_calibration_path(out_dir, chip, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes any attempt to connect to IBM fail the test.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        None. Replaces ``backends.connect``.
    """

    def refuse(*args: Any, **kwargs: Any) -> None:
        """Fails the test: the resolver must not connect when the file is stored.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            None. Never returns normally.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("connect() must not be called when the file is stored")

    monkeypatch.setattr(backends, "connect", refuse)


def test_path_form_returns_existing_file(tmp_path: Path) -> None:
    """A path that exists comes back unchanged."""
    path = tmp_path / "any_name.json"
    path.write_text("{}", encoding="utf-8")
    assert chip_noise.resolve_calibration(path) == path
    assert chip_noise.resolve_calibration(str(path)) == path


def test_path_form_missing_file_is_refused(tmp_path: Path) -> None:
    """A path that does not exist stops with the file name."""
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="file not found"):
        chip_noise.resolve_calibration(missing)


def test_chip_day_form_uses_stored_file_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``<chip>:<day>`` finds the collector's file and never connects."""
    stored = _store(tmp_path, CHIP, DAY, {"format": 1})
    _forbid_network(monkeypatch)
    assert chip_noise.resolve_calibration(f"{CHIP}:{DAY}", tmp_path) == stored


def test_chip_day_form_fetches_and_stores_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``<chip>:<day>`` with no stored file fetches that day and stores it."""
    raw = {
        "format": 1,
        "backend": {
            "name": CHIP,
            "n_qubits": 156,
            "basis_gates": ["cz", "rz", "sx", "x"],
            "coupling_map": [[0, 1]],
        },
        "requested_date": DAY,
        "fetched_at": "2026-11-16T12:00:00+00:00",
        "calibration_date": "2026-11-06T11:18:28-05:00",
        "properties": {},
    }
    seen: dict[str, Any] = {}

    def fake_connect(*args: Any, **kwargs: Any) -> str:
        """Stands in for the IBM service.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            A token the other fakes recognise.
        """
        seen["connected"] = True
        return "service"

    def fake_load_backend(service: str, chip: str) -> str:
        """Records which chip was asked for.

        Args:
            service: The fake service.
            chip: The chip name.

        Returns:
            The chip name, standing in for the backend.
        """
        seen["chip"] = chip
        return chip

    def fake_fetch(backend: str, day: date | None = None) -> dict[str, Any]:
        """Records which day was asked for and returns the stand-in calibration.

        Args:
            backend: The fake backend.
            day: The day asked for.

        Returns:
            The stand-in raw calibration.
        """
        seen["day"] = day
        return raw

    monkeypatch.setattr(backends, "connect", fake_connect)
    monkeypatch.setattr(backends, "load_backend", fake_load_backend)
    monkeypatch.setattr(backends, "fetch_raw_calibration", fake_fetch)

    path = chip_noise.resolve_calibration(f"{CHIP}:{DAY}", tmp_path)

    assert path == raw_calibration_path(tmp_path, CHIP, DAY)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == raw
    assert seen == {"connected": True, "chip": CHIP, "day": date(2026, 11, 6)}


def test_chip_day_form_failed_fetch_names_the_collect_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed fetch stops with the reason and the command to run by hand."""

    def no_account(*args: Any, **kwargs: Any) -> None:
        """Stands in for a connect with no saved account.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            None. Never returns normally.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("no saved account")

    monkeypatch.setattr(backends, "connect", no_account)
    with pytest.raises(ValueError) as excinfo:
        chip_noise.resolve_calibration(f"{CHIP}:{DAY}", tmp_path)
    message = str(excinfo.value)
    assert "no saved account" in message
    assert f"collect_calibration --chip {CHIP} --date {DAY}" in message
    assert str(tmp_path) in message
    assert not raw_calibration_path(tmp_path, CHIP, DAY).exists()


def test_chip_day_form_rejects_bad_and_future_days(tmp_path: Path) -> None:
    """A day that is not a calendar day, or lies ahead, is refused before any lookup."""
    with pytest.raises(ValueError, match="not a calendar day"):
        chip_noise.resolve_calibration(f"{CHIP}:2026-13-40", tmp_path)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="in the future"):
        chip_noise.resolve_calibration(f"{CHIP}:{tomorrow}", tmp_path)


def test_hyperparameters_record_the_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The genome setting holds the file's path, not the ``<chip>:<day>`` text."""
    stored = _store(tmp_path, CHIP, DAY, {"format": 1})
    _forbid_network(monkeypatch)
    monkeypatch.setattr(chip_noise, "CALIBRATION_DIR", str(tmp_path))

    class FakeModel:
        """Stands in for the noise model built from a calibration."""

        def to_hyperparameters(self) -> dict[str, Any]:
            """Returns a minimal settings dict.

            Returns:
                One recognisable key.
            """
            return {"noise_type": "ibm_backend"}

    import src.noise.pennylane_noise_model as pennylane_noise_model

    monkeypatch.setattr(
        pennylane_noise_model.PennyLaneNoiseModel,
        "from_calibration",
        classmethod(lambda cls, raw, qubits, noise_type: FakeModel()),
    )

    # resolve_calibration takes its default archive root at call time
    hp = chip_noise.noise_hyperparameters_from_calibration(f"{CHIP}:{DAY}", [0, 1])
    assert hp["noise_calibration"] == str(stored)
    assert hp["noise_type"] == "ibm_backend"
