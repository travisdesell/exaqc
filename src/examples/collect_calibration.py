"""Save a real IBM chip's calibration, raw, one file per chip and day.

The only step of the noise work that talks to IBM. Everything downstream (the
chip-averaged noise model, the per-qubit snapshot made by
``make_noise_snapshot``, the drift study) reads the files this writes.

    python3 -m src.examples.collect_calibration --chip ibm_fez ibm_kingston \\
        --api_key_file ~/.ibm/quantum_apikey.json

writes ``data/calibration/<chip>/<chip>_<YYYY-MM-DD>.json`` per chip (named by
the day asked for; the served calibration's own stamp is inside), holding the
calibration as IBM reports it (``BackendProperties.to_dict()``) plus the chip's
layout. ``--date`` fetches the calibration as it stood at the end of a past
day instead, and ``--from_date`` / ``--to_date`` fetch a whole range of days
(skipping days already stored), which is how the drift series is filled in;
IBM serves at least a year back. The API key is read
from the file's ``"apikey"`` field and used in memory only; it is never
written to any output.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from loguru import logger

from src.noise.calibration.backends import (
    available_backends,
    connect,
    fetch_raw_calibration,
    load_backend,
    save_raw_calibration,
)
from src.noise.calibration.drift import collect_series, date_range


def build_parser() -> argparse.ArgumentParser:
    """Creates the command-line parser.

    Returns:
        The parser with ``--chip``, ``--api_key_file``, ``--instance``,
        ``--date``, ``--from_date``, ``--to_date``, ``--out_dir`` and
        ``--list_chips``.
    """
    parser = argparse.ArgumentParser(
        description="Save a real IBM chip's calibration as a raw JSON file."
    )
    parser.add_argument(
        "--chip",
        type=str,
        nargs="+",
        default=None,
        help="IBM backend name(s), e.g. ibm_fez ibm_kingston. Required unless --list_chips is given.",
    )
    parser.add_argument(
        "--api_key_file",
        type=str,
        default=None,
        help=(
            "JSON file downloaded from IBM Cloud with an 'apikey' field (optionally a 'crn' field). "
            "Default: the account saved with QiskitRuntimeService.save_account."
        ),
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance CRN; default: the 'crn' in the key file, else the account's only instance.",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Past day (YYYY-MM-DD) whose end-of-day calibration to fetch; default: the current one.",
    )
    parser.add_argument(
        "--from_date",
        type=date.fromisoformat,
        default=None,
        help="With --to_date: fetch every day from this one (YYYY-MM-DD) inclusive, skipping days already stored.",
    )
    parser.add_argument(
        "--to_date",
        type=date.fromisoformat,
        default=None,
        help="Last day of the --from_date range (YYYY-MM-DD, inclusive); default: today.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/calibration",
        help="Archive root; files go to <out_dir>/<chip>/<chip>_<date>.json.",
    )
    parser.add_argument(
        "--list_chips",
        action="store_true",
        help="Print the backends the account can see and exit.",
    )
    return parser


def main() -> None:
    """Connects, fetches each named chip's calibration and writes it to the archive.

    Returns:
        None. Writes one file per chip and logs each path and calibration date.
    """
    parser = build_parser()
    args = parser.parse_args()

    service = connect(args.api_key_file, args.instance)
    if args.list_chips:
        for name in available_backends(service):
            print(name)
        return
    if args.chip is None:
        parser.error("--chip is required (or use --list_chips).")

    if args.from_date is not None:
        if args.date is not None:
            parser.error("--from_date cannot be combined with --date.")
        days = date_range(args.from_date, args.to_date or date.today())
        if not days:
            parser.error("--to_date is before --from_date.")
        for chip in args.chip:
            paths = collect_series(
                load_backend(service, chip), days, Path(args.out_dir)
            )
            logger.info("{}: {} of {} days stored", chip, len(paths), len(days))
        return

    for chip in args.chip:
        backend = load_backend(service, chip)
        raw = fetch_raw_calibration(backend, args.date)
        path = save_raw_calibration(raw, Path(args.out_dir))
        logger.info(
            "Saved {} calibration dated {} ({} qubits, {} gate entries) to {}",
            raw["backend"]["name"],
            raw["calibration_date"],
            raw["backend"]["n_qubits"],
            len(raw["properties"].get("gates", [])),
            path,
        )


if __name__ == "__main__":
    main()
