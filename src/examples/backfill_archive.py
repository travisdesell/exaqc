"""Bring run archives written before archive format 7 up to it.

Fills the training-metric projections (``initial_metrics``, ``best_metrics``,
``last_metrics``) and the inherited weights' fitness (``initial_fitness``, and
with it ``target_gain``) for the genomes an older archive already holds, from
each genome's stored JSON (see :mod:`src.utils.archive_backfill`)::

    python3 -m src.examples.backfill_archive ./artifacts/walker2d_1 ./artifacts/walker2d_2 \\
        --output_dir ./backfilled

Writing is always explicit: ``--output_dir`` copies each run's archive there and
backfills the copy, leaving the original untouched, while ``--in_place``
backfills the archives themselves. An archive whose latest genome was saved
within ``--idle_minutes`` is refused, since a search may still be writing it.
"""

from __future__ import annotations

import argparse
import os
import sys

from loguru import logger

from src.utils.archive_backfill import (
    DEFAULT_IDLE_MINUTES,
    ArchiveInUseError,
    BackfillResult,
    backfill_archive,
    copy_archive,
)
from src.utils.genome_archive import resolve_archive_path


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the archive backfill.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fill the archive format 7 training-metric and initial-fitness columns for the "
            "genomes older run archives already hold."
        )
    )

    parser.add_argument(
        "archives",
        type=str,
        nargs="+",
        help="Run output directories, or their genomes.sqlar files, to backfill.",
    )

    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Copy each run's archive into <output_dir>/<run directory name>/ and backfill the "
            "copy, leaving the original untouched. An existing copy is never overwritten."
        ),
    )
    destination.add_argument(
        "--in_place",
        action="store_true",
        default=False,
        help="Backfill the given archives themselves.",
    )

    parser.add_argument(
        "--idle_minutes",
        type=float,
        default=DEFAULT_IDLE_MINUTES,
        help=(
            "Refuse an archive whose latest genome was saved within this many minutes, since a "
            "search may still be writing to it; 0 skips the check."
        ),
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels. Pick DEBUG to show everything.""",
    )

    return parser


def _remove_copy(archive_path: str) -> None:
    """Removes a copied archive, its journal files and its directory if left empty.

    Args:
        archive_path: The copied archive.

    Returns:
        None. Deletes the files this backfill created for the copy.
    """

    for suffix in ("", "-journal", "-wal", "-shm"):
        if os.path.exists(archive_path + suffix):
            os.remove(archive_path + suffix)
    directory = os.path.dirname(archive_path)
    if not os.listdir(directory):
        os.rmdir(directory)


def backfill_one(
    path: str, output_dir: str | None, idle_minutes: float
) -> BackfillResult:
    """Backfills one run's archive, in place or as a copy.

    Args:
        path: A run output directory or its ``genomes.sqlar``.
        output_dir: The directory to copy the run into before backfilling it;
            ``None`` to backfill the archive in place.
        idle_minutes: Refuse the archive if its latest genome was saved within
            this many minutes.

    Returns:
        What the backfill found and changed.

    Raises:
        FileNotFoundError: If there is no archive at ``path``.
        FileExistsError: If ``output_dir`` already holds a copy of this run.
        ArchiveInUseError: If a search appears to still be writing to it; a
            copy made for it is removed again.
    """

    source = resolve_archive_path(path)
    if output_dir is None:
        logger.warning(
            "backfilling {} in place; make sure no search is still writing to it.",
            source,
        )
        return backfill_archive(source, idle_minutes=idle_minutes)

    # the source is only ever copied: even opening it could recover (and so
    # rewrite) a run that was stopped mid-write
    copy = copy_archive(source, output_dir)
    logger.info("copied {} to {}.", source, copy)
    try:
        return backfill_archive(copy, idle_minutes=idle_minutes, source_path=source)
    except ArchiveInUseError:
        _remove_copy(copy)
        raise


def main() -> None:
    """Backfills every archive given on the command line.

    Returns:
        None. Exits with status 1 if any archive could not be backfilled; the
        others are still backfilled.
    """

    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.logging_level)

    failed = 0
    for path in args.archives:
        try:
            result = backfill_one(
                path,
                None if args.in_place else args.output_dir,
                args.idle_minutes,
            )
        except (OSError, ValueError, RuntimeError) as error:
            logger.error("did not backfill {}: {}", path, error)
            failed += 1
            continue
        print(
            f"{result.archive_path}: {result.genomes} genomes, {result.updated} updated; "
            f"initial_fitness recorded {result.initial_fitness_recorded}, derived "
            f"{result.initial_fitness_derived}, unavailable "
            f"{result.initial_fitness_unavailable} ({result.seconds:.1f}s)"
        )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
