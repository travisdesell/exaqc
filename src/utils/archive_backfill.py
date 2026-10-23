"""Brings run archives written before archive format 7 up to it.

Archive format 7 added summary columns that runs written earlier hold only as
``NULL``: the training-metric projections (``initial_metrics``, ``best_metrics``
and ``last_metrics``) and the inherited weights' fitness (``initial_fitness``,
from which ``initial_loss``, ``initial_target_metric`` and ``target_gain`` are
generated). Every genome's full JSON is stored in the archive, so they can be
filled in afterwards:

* the projections are re-derived from each genome's ``*_epoch_metrics`` and
  ``*_episode_metrics`` series exactly as a format-7 search records them;
* ``initial_fitness`` is taken from ``metadata["initial_fitness"]`` when the
  genome recorded it. A reinforcement-learning genome that did not is scored
  from its episode-0 evaluation (the inherited weights, before training) with
  the fitness formula the search uses. Classification and teacher genomes never
  stored their epoch-0 *training* metrics, so theirs is left ``NULL`` rather
  than guessed.

Nothing else a run recorded is rewritten -- not its fitness, its
``format_version`` or its old ``final_metrics`` column. What the backfill did is
recorded in ``run_info`` under :data:`BACKFILL_RUN_INFO_KEY`, and running it
again changes nothing.

The command-line entry point is :mod:`src.examples.backfill_archive`.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.objectives.reinforcement_learning_objective import (
    reinforcement_learning_fitness,
)
from src.utils.genome_archive import (
    ARCHIVE_FILENAME,
    GenomeArchive,
    _chunks,
    _finite_or_none,
    _metric_projections,
    _provenance,
    resolve_archive_path,
)

#: The ``run_info`` key a backfill records itself under.
BACKFILL_RUN_INFO_KEY = "backfilled_format_7"

#: How many genomes are read, and their changed rows written, per transaction.
BACKFILL_BATCH_SIZE = 500

#: The default for how recently (in minutes) an archive's latest genome may have
#: been saved before the archive is taken to be still written by a search.
DEFAULT_IDLE_MINUTES = 10.0

#: The training-return weight a reinforcement-learning run used when its
#: ``run_info`` arguments do not say (the ``--train_vs_validation_bias``
#: default).
DEFAULT_TRAIN_VS_VALIDATION_BIAS = 0.1

#: Files SQLite keeps beside a database that can hold its committed or in-flight
#: state (a rollback journal or a write-ahead log). They are copied with the
#: archive, so the copy opens to the same state the original would; ``-shm`` is
#: rebuilt from the log and is not needed.
_SIDECAR_SUFFIXES = ("-journal", "-wal")

#: The summary columns a backfill fills, in the order its updates bind them.
_BACKFILLED_COLUMNS = (
    "initial_metrics",
    "best_metrics",
    "last_metrics",
    "initial_fitness",
)


class ArchiveInUseError(RuntimeError):
    """Raised for an archive a search appears to still be writing to."""


@dataclass
class BackfillResult:
    """What backfilling one archive found and changed.

    Attributes:
        source_path: The archive that was read.
        archive_path: The archive that was written: ``source_path`` itself when
            backfilling in place, otherwise its copy.
        genomes: How many genomes the archive holds.
        updated: How many genomes' summary rows changed; 0 when the archive was
            already up to date.
        initial_fitness_recorded: Genomes whose ``initial_fitness`` came from
            their own ``metadata["initial_fitness"]``.
        initial_fitness_derived: Reinforcement-learning genomes whose
            ``initial_fitness`` was scored from their episode-0 evaluation.
        initial_fitness_unavailable: Genomes left with no ``initial_fitness``.
        seconds: How long the backfill took, excluding any copy.
    """

    source_path: str
    archive_path: str
    genomes: int = 0
    updated: int = 0
    initial_fitness_recorded: int = 0
    initial_fitness_derived: int = 0
    initial_fitness_unavailable: int = 0
    seconds: float = 0.0


def check_idle(
    archive_path: str, idle_minutes: float, now: float | None = None
) -> None:
    """Refuses an archive whose latest genome was saved too recently.

    A search writes no marker when it finishes, so the only sign that one is
    still running is how recently it stored a genome. A run that stores genomes
    less often than ``idle_minutes`` cannot be told apart from a finished one.

    Args:
        archive_path: The archive to check.
        idle_minutes: How long ago, at least, its latest genome must have been
            saved; 0 skips the check.
        now: The current Unix time; defaults to the clock.

    Returns:
        None.

    Raises:
        ArchiveInUseError: If the latest genome was saved within
            ``idle_minutes``.
    """

    if idle_minutes <= 0:
        return
    with GenomeArchive.open_readonly(archive_path) as reader:
        last_saved_at = reader.last_saved_at()
    if last_saved_at is None:
        return
    idle_seconds = (time.time() if now is None else now) - last_saved_at
    if idle_seconds < idle_minutes * 60.0:
        raise ArchiveInUseError(
            f"{archive_path} stored a genome {idle_seconds / 60.0:.1f} minutes ago, so a "
            f"search may still be writing to it; wait until it has been idle for "
            f"{idle_minutes:g} minutes (or lower --idle_minutes once it has stopped)."
        )


def copy_archive(source_path: str, output_dir: str) -> str:
    """Copies a run's archive into a run directory of the same name under ``output_dir``.

    The archive's rollback journal or write-ahead log, when present, is copied
    with it, so a run that was stopped mid-write opens (and is recovered) as the
    copy rather than the original.

    Args:
        source_path: The archive to copy.
        output_dir: The directory to copy the run into; the copy is
            ``<output_dir>/<run directory name>/genomes.sqlar``.

    Returns:
        The path of the copied archive.

    Raises:
        FileExistsError: If the copy's destination already holds an archive,
            which is never overwritten.
    """

    run_name = os.path.basename(os.path.dirname(os.path.abspath(source_path)))
    destination_dir = os.path.join(output_dir, run_name)
    destination = os.path.join(destination_dir, ARCHIVE_FILENAME)
    if os.path.exists(destination):
        raise FileExistsError(
            f"{destination} already exists; give an --output_dir that does not "
            "already hold this run."
        )

    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_path, destination)
    for suffix in _SIDECAR_SUFFIXES:
        if os.path.exists(source_path + suffix):
            shutil.copy2(source_path + suffix, destination + suffix)
    return destination


def initial_fitness_for(
    genome: dict[str, Any], task: str | None, train_vs_validation_bias: float
) -> tuple[dict[str, Any] | None, str]:
    """Finds the fitness a genome's inherited weights had, before training.

    Args:
        genome: The serialized genome.
        task: The run's task, used when the genome does not record its own.
        train_vs_validation_bias: The training-return weight the run scored
            reinforcement-learning fitness with.

    Returns:
        The inherited weights' fitness (``None`` when it cannot be known) and
        where it came from: ``"recorded"`` (the genome's
        ``metadata["initial_fitness"]``), ``"derived"`` (scored from a
        reinforcement-learning genome's episode-0 evaluation) or
        ``"unavailable"``.
    """

    metadata = genome.get("metadata") or {}
    recorded = metadata.get("initial_fitness")
    if recorded is not None:
        return recorded, "recorded"

    if (genome.get("task") or task) != "reinforcement_learning":
        return None, "unavailable"

    # episode 0 evaluates the inherited weights before anything is trained; runs
    # written before it was recorded start at episode 1 and cannot be scored
    records = metadata.get("evaluation_episode_metrics") or []
    initial = records[0] if records else None
    if not isinstance(initial, dict) or initial.get("episode") != 0:
        return None, "unavailable"

    # with nothing trained, the training side holds the evaluation's returns, as
    # the trainer records for the inherited weights
    training = {
        "return_mean": initial["return_mean"],
        "best_episode_return": initial["best_episode_return"],
    }
    fitness = genome.get("fitness") or {}
    return (
        reinforcement_learning_fitness(
            training,
            initial,
            train_vs_validation_bias=train_vs_validation_bias,
            env_id=fitness.get("env_id"),
            eval_policy=metadata.get("eval_policy", "greedy"),
        ),
        "derived",
    )


def _summary_values(
    genome: dict[str, Any], initial_fitness: dict[str, Any] | None
) -> tuple[str, str, str, str | None]:
    """Encodes a genome's backfilled columns as ``add_genome`` stores them.

    Args:
        genome: The serialized genome.
        initial_fitness: Its inherited weights' fitness, if known.

    Returns:
        The ``initial_metrics``, ``best_metrics``, ``last_metrics`` and
        ``initial_fitness`` column values, as JSON text (``initial_fitness`` is
        ``None`` when unknown).
    """

    initial, best, last = _metric_projections(genome.get("metadata") or {})
    return (
        json.dumps(_finite_or_none(initial)),
        json.dumps(_finite_or_none(best)),
        json.dumps(_finite_or_none(last)),
        (
            json.dumps(_finite_or_none(initial_fitness))
            if initial_fitness is not None
            else None
        ),
    )


def backfill_archive(
    archive_path: str,
    idle_minutes: float = DEFAULT_IDLE_MINUTES,
    source_path: str | None = None,
) -> BackfillResult:
    """Fills an archive's format-7 summary columns for the genomes it holds.

    Only rows whose values change are written, in transactions of
    :data:`BACKFILL_BATCH_SIZE` genomes, so running it again on an up-to-date
    archive writes nothing.

    Args:
        archive_path: The ``genomes.sqlar`` to backfill (or its run directory);
            it is written in place.
        idle_minutes: Refuse the archive if its latest genome was saved within
            this many minutes (see :func:`check_idle`); 0 skips the check.
        source_path: The archive ``archive_path`` was copied from, recorded in
            the result; defaults to ``archive_path``.

    Returns:
        What the backfill found and changed. When any row changed, it is also
        recorded in the archive's ``run_info`` under
        :data:`BACKFILL_RUN_INFO_KEY`.

    Raises:
        FileNotFoundError: If there is no archive at ``archive_path``.
        ValueError: If the archive file is not named ``genomes.sqlar`` (an
            archive is reopened for writing through its run directory).
        ArchiveInUseError: If a search appears to still be writing to it.
        RuntimeError: If a batch of updates could not be written.
    """

    archive_path = resolve_archive_path(archive_path)
    if os.path.basename(archive_path) != ARCHIVE_FILENAME:
        raise ValueError(
            f"{archive_path} is not named {ARCHIVE_FILENAME}; only a run's own archive "
            "can be backfilled."
        )
    check_idle(archive_path, idle_minutes)

    started = time.perf_counter()
    result = BackfillResult(
        source_path=source_path or archive_path, archive_path=archive_path
    )

    with GenomeArchive.open_readonly(archive_path) as reader:
        run_info = reader.run_info()
    arguments = run_info.get("arguments") or {}
    task = run_info.get("task")
    bias = float(
        arguments.get("train_vs_validation_bias", DEFAULT_TRAIN_VS_VALIDATION_BIAS)
    )

    # reopening for writing adds the format-7 columns an older archive lacks
    with GenomeArchive.create(
        os.path.dirname(archive_path),
        shared_file_system=bool(arguments.get("shared_file_system", False)),
        expect_existing=True,
    ) as archive:
        batch: list[tuple[int, tuple[str, str, str, str | None]]] = []

        def flush() -> None:
            """Writes the batch's changed rows and empties it.

            Returns:
                None. Updates the archive's rows whose backfilled columns differ,
                adds them to ``result.updated`` and clears ``batch``.

            Raises:
                RuntimeError: If the archive stayed locked through every retry.
            """

            if not batch:
                return
            current: dict[int, tuple[Any, ...]] = {}
            numbers = [genome_number for genome_number, _ in batch]
            for chunk in _chunks(numbers):
                rows = archive.connection.execute(
                    f"SELECT genome_number, {', '.join(_BACKFILLED_COLUMNS)} FROM genomes "
                    f"WHERE genome_number IN ({', '.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
                current.update((row[0], tuple(row[1:])) for row in rows)
            changed = [
                (*values, genome_number)
                for genome_number, values in batch
                if current.get(genome_number) != values
            ]
            batch.clear()
            if not changed:
                return

            def store(connection: sqlite3.Connection) -> None:
                """Updates the changed rows' backfilled columns."""

                connection.executemany(
                    "UPDATE genomes SET "
                    + ", ".join(f"{column} = ?" for column in _BACKFILLED_COLUMNS)
                    + " WHERE genome_number = ?",
                    changed,
                )

            if not archive._write(f"{len(changed)} backfilled genomes", store):
                raise RuntimeError(
                    f"Could not write backfilled genomes to {archive_path}; it stayed "
                    "locked. Rows written before this batch are kept, and running the "
                    "backfill again finishes it."
                )
            result.updated += len(changed)

        for genome_number, genome in archive.iter_genome_dicts():
            initial_fitness, source = initial_fitness_for(genome, task, bias)
            if source == "recorded":
                result.initial_fitness_recorded += 1
            elif source == "derived":
                result.initial_fitness_derived += 1
            else:
                result.initial_fitness_unavailable += 1
            result.genomes += 1
            batch.append((genome_number, _summary_values(genome, initial_fitness)))
            if len(batch) >= BACKFILL_BATCH_SIZE:
                flush()
            if result.genomes % 5000 == 0:
                logger.info("{}: read {} genomes.", archive_path, result.genomes)
        flush()

        result.seconds = time.perf_counter() - started
        if result.updated:
            archive.set_run_info(
                **{
                    BACKFILL_RUN_INFO_KEY: {
                        "backfilled_at": time.time(),
                        "git_commit": _provenance().get("git_commit"),
                        "genomes": result.genomes,
                        "updated": result.updated,
                        "initial_fitness": {
                            "recorded": result.initial_fitness_recorded,
                            "derived": result.initial_fitness_derived,
                            "unavailable": result.initial_fitness_unavailable,
                        },
                    }
                }
            )

    logger.info(
        "{}: {} genomes, {} updated; initial fitness recorded for {}, derived for {}, "
        "unavailable for {} ({:.1f}s).",
        archive_path,
        result.genomes,
        result.updated,
        result.initial_fitness_recorded,
        result.initial_fitness_derived,
        result.initial_fitness_unavailable,
        result.seconds,
    )
    return result
