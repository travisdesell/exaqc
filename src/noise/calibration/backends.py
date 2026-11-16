"""Fetching a real IBM backend's calibration and storing it raw.

Everything comes from IBM's own service: a ``QiskitRuntimeService`` backend
whose ``properties()`` is the current calibration and whose
``properties(datetime=...)`` is the calibration as it stood at a past moment.
The service needs the user's IBM Cloud API key, read from a JSON file the user
downloaded from IBM Cloud (its ``"apikey"`` field) and passed to the service in
memory, or from the account the user saved once with Qiskit's
``QiskitRuntimeService.save_account``. This module never writes, logs or
returns the key.

The raw calibration is stored as one JSON file per chip and day (the day
asked for; the served calibration's own stamp is inside), holding
``BackendProperties.to_dict()`` unchanged plus the chip's fixed layout, so that
``properties_from_raw`` rebuilds the same ``BackendProperties`` object offline
and every later step (chip-averaged model, per-qubit snapshot, drift study)
reads the file instead of the network.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService
    from qiskit_ibm_runtime.models import BackendProperties

RAW_CALIBRATION_FORMAT = 1
"""Bumped when the raw file layout changes."""


def connect(
    api_key_file: str | Path | None = None, instance: str | None = None
) -> QiskitRuntimeService:
    """Opens the IBM Quantum service with the user's credentials.

    Args:
        api_key_file: A JSON file downloaded from IBM Cloud holding an
            ``"apikey"`` field (and optionally a ``"crn"`` field naming the
            instance). ``None`` uses the account saved with
            ``QiskitRuntimeService.save_account``.
        instance: The instance CRN to use; overrides a ``"crn"`` in the file.
            ``None`` lets the service pick the account's only instance.

    Returns:
        The connected service.

    Raises:
        FileNotFoundError: If ``api_key_file`` does not exist.
        KeyError: If the file has no ``"apikey"`` field.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    if api_key_file is None:
        return (
            QiskitRuntimeService(instance=instance)
            if instance
            else QiskitRuntimeService()
        )

    with open(api_key_file, encoding="utf-8") as handle:
        key_file = json.load(handle)
    token = key_file["apikey"]
    crn = instance or key_file.get("crn")
    kwargs: dict[str, Any] = {"channel": "ibm_quantum_platform", "token": token}
    if crn:
        kwargs["instance"] = crn
    return QiskitRuntimeService(**kwargs)


def available_backends(service: QiskitRuntimeService) -> list[str]:
    """Names of the IBM backends the connected account can see.

    Args:
        service: A service from :func:`connect`.

    Returns:
        Backend names such as ``"ibm_fez"``, sorted.
    """
    return sorted(backend.name for backend in service.backends())


def load_backend(service: QiskitRuntimeService, backend_name: str) -> IBMBackend:
    """Fetches one IBM backend.

    Args:
        service: A service from :func:`connect`.
        backend_name: The IBM backend name, such as ``"ibm_fez"``.

    Returns:
        The backend object, with ``properties()`` and ``configuration()``.

    Raises:
        qiskit_ibm_runtime.exceptions.QiskitBackendNotFoundError: If the
            account cannot see a backend of that name.
    """
    return service.backend(backend_name)


def _end_of_day_utc(day: date) -> datetime:
    """The last moment of a calendar day, in UTC.

    Args:
        day: The calendar day.

    Returns:
        ``23:59:59`` UTC on that day, so a lookup returns the calibration that
        was current at the end of the day.
    """
    return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)


def properties_on_date(
    backend: IBMBackend, day: date | None = None
) -> BackendProperties:
    """The backend's calibration as it stood at the end of a given day.

    Args:
        backend: A backend from :func:`load_backend`.
        day: The calendar day to look up, or ``None`` for the current
            calibration.

    Returns:
        A ``BackendProperties`` object; its ``last_update_date`` says exactly
        which calibration was served.
    """
    if day is None:
        return backend.properties()
    return backend.properties(datetime=_end_of_day_utc(day))


def fetch_raw_calibration(
    backend: IBMBackend, day: date | None = None
) -> dict[str, Any]:
    """Reads a backend's calibration and layout into one JSON-safe dict.

    Args:
        backend: A backend from :func:`load_backend`.
        day: The calendar day to look up, or ``None`` for the current
            calibration.

    Returns:
        ``format``, ``backend`` (name, number of qubits, basis gates, coupling
        map), ``requested_date`` (ISO day or ``None``), ``fetched_at`` (ISO
        UTC time), ``calibration_date`` (the served calibration's
        ``last_update_date``, ISO) and ``properties``
        (``BackendProperties.to_dict()`` with datetimes as ISO strings).

    Raises:
        ValueError: If the backend returned no calibration for that day.
    """
    props = properties_on_date(backend, day)
    if props is None:
        raise ValueError(f"{backend.name} returned no calibration for {day}.")
    config = backend.configuration()
    coupling_map = getattr(config, "coupling_map", None)
    return {
        "format": RAW_CALIBRATION_FORMAT,
        "backend": {
            "name": backend.name,
            "n_qubits": int(config.n_qubits),
            "basis_gates": list(config.basis_gates),
            "coupling_map": (
                [list(pair) for pair in coupling_map] if coupling_map else None
            ),
        },
        "requested_date": day.isoformat() if day else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration_date": props.last_update_date.isoformat(),
        "properties": _jsonable(props.to_dict()),
    }


def _jsonable(value: Any) -> Any:
    """Converts datetimes inside a nested dict/list to ISO strings.

    Args:
        value: A dict, list, datetime or scalar.

    Returns:
        The same structure with every ``datetime`` replaced by its ISO string.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def archive_day(raw: dict[str, Any]) -> str:
    """The calendar day a raw calibration is filed under.

    The requested day when one was given (so a day-by-day series has exactly
    one file per day, even on days the chip was not recalibrated and IBM
    served an earlier calibration), else the served calibration's own day.
    The served stamp is always inside the file as ``calibration_date``.

    Args:
        raw: Output of :func:`fetch_raw_calibration`.

    Returns:
        ``YYYY-MM-DD``.
    """
    return (raw.get("requested_date") or raw["calibration_date"])[:10]


def raw_calibration_path(out_dir: str | Path, backend_name: str, day: str) -> Path:
    """The file a raw calibration is stored under.

    Args:
        out_dir: The archive root, such as ``data/calibration``.
        backend_name: The chip, such as ``"ibm_fez"``.
        day: The archive day, ``YYYY-MM-DD`` (see :func:`archive_day`).

    Returns:
        ``<out_dir>/<chip>/<chip>_<YYYY-MM-DD>.json``.
    """
    return Path(out_dir) / backend_name / f"{backend_name}_{day[:10]}.json"


def save_raw_calibration(raw: dict[str, Any], out_dir: str | Path) -> Path:
    """Writes a raw calibration to the archive, one file per chip and day.

    Args:
        raw: Output of :func:`fetch_raw_calibration`.
        out_dir: The archive root, such as ``data/calibration``.

    Returns:
        The path written (parent folders are created), named by
        :func:`archive_day`. An existing file for the same chip and day is
        overwritten.
    """
    path = raw_calibration_path(out_dir, raw["backend"]["name"], archive_day(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path


def load_raw_calibration(path: str | Path) -> dict[str, Any]:
    """Reads a raw calibration file back.

    Args:
        path: A file written by :func:`save_raw_calibration`.

    Returns:
        The dict :func:`fetch_raw_calibration` produced.

    Raises:
        ValueError: If the file's ``format`` is newer than this code knows.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("format", 0) > RAW_CALIBRATION_FORMAT:
        raise ValueError(
            f"{path}: raw calibration format {raw['format']} is newer than {RAW_CALIBRATION_FORMAT}."
        )
    return raw


def properties_from_raw(raw: dict[str, Any]) -> BackendProperties:
    """Rebuilds the ``BackendProperties`` object from a raw calibration, offline.

    Args:
        raw: Output of :func:`load_raw_calibration`.

    Returns:
        A ``BackendProperties`` equal to what the backend served, so code
        written against ``props.t1(q)``, ``props.gate_error(...)`` and
        ``props.gates`` works unchanged on stored data.
    """
    from qiskit_ibm_runtime.models import BackendProperties

    return BackendProperties.from_dict(raw["properties"])
