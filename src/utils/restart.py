"""Restarting a stopped, canceled or crashed EXAQC search from its archive.

A run's archive records everything needed to continue it: the arguments the
search was started with, every genome it evaluated, which of them the population
held after each insertion, and the highest gate innovation number handed out.
This module reads that back -- a population strategy holding the same genomes,
the counters the search stopped at, and an innovation number generator that will
not hand out a number twice -- so a run can carry on where it left off.

Two things are deliberately not restored, because nothing records them: the
random number generators' state, and any genome that was still being evaluated
when the run stopped (its number is simply never used again). A restarted run
therefore *continues* the search rather than reproducing the run that would have
happened had it never stopped.

Archives written before runs recorded their arguments and innovation numbers
cannot be restarted; they are still listed, browsed and charted as before.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.innovation import innovation_number_generator
from src.evolution.population_strategy import PopulationStrategy
from src.utils.genome_archive import (
    ARCHIVE_FORMAT_VERSION,
    GenomeArchive,
    resolve_archive_path,
)

if TYPE_CHECKING:
    from src.evolution.exaqc import EXAQC

#: Arguments a restart may be given values for that differ from the run it
#: continues: how many genomes the run should end up evaluating, where its
#: outputs go, and how this process runs. Every other argument must match what
#: the archive recorded, so a restart continues the same search rather than
#: quietly becoming a different one.
OVERRIDABLE_ARGUMENTS = frozenset(
    {
        "number_genomes",
        "out_dir",
        "shared_file_system",
        "device",
        "logging_level",
        "restart",
        "overwrite_archive",
        "force_restart",
    }
)


@dataclass
class RestartState:
    """What a stopped run left in its archive, ready to restore into a search.

    Attributes:
        archive_path: The ``genomes.sqlar`` the state was read from.
        arguments: The arguments the original run was started with.
        run_info: Everything the archive records about the run.
        population: The genomes the population held when the run stopped. Each
            carries the metadata the search wrote, including the island it was
            inserted into.
        best_genome: The best genome the run ever inserted by the population's
            own ranking, which need not still be in the population.
        target_metric_best: The genome with the highest ``target_metric``, so a
            restarted run only rewrites that best-genome file on a genuine
            improvement.
        next_genome_number: The number the next generated genome should take.
        inserted_genomes: How many genomes the run had inserted.
        max_innovation_number: The highest gate innovation number handed out, so
            a restart never reuses one for a different gate.
    """

    archive_path: str
    arguments: dict[str, Any]
    run_info: dict[str, Any]
    population: list[CircuitGenome] = field(default_factory=list)
    best_genome: CircuitGenome | None = None
    target_metric_best: CircuitGenome | None = None
    next_genome_number: int = 1
    inserted_genomes: int = 0
    max_innovation_number: int = 0

    @property
    def island_neighbors(self) -> list[list[int]] | None:
        """The islands each island drew parents from, as the run recorded them.

        A ``random`` topology is drawn randomly when the islands are built, so a
        restart must re-apply what the run actually used rather than drawing a
        new graph.

        Returns:
            Each island's neighbor ids, by island id, or ``None`` when the run
            recorded no island topology (it used a single population).
        """

        topology = self.run_info.get("island_topology") or {}
        neighbors = topology.get("neighbors")
        return list(neighbors) if neighbors else None


def archive_exists(out_dir: str) -> bool:
    """Reports whether a run directory already holds an archive.

    Args:
        out_dir: A run's output directory, or a ``genomes.sqlar`` file.

    Returns:
        True if an archive is there to be restarted or overwritten.
    """

    try:
        resolve_archive_path(out_dir)
    except FileNotFoundError:
        return False
    return True


def load(out_dir: str) -> RestartState:
    """Reads a stopped run's archive into the state needed to continue it.

    Args:
        out_dir: The run's output directory, or its ``genomes.sqlar``.

    Returns:
        The run's :class:`RestartState`.

    Raises:
        FileNotFoundError: If there is no archive at ``out_dir``.
        ValueError: If the archive holds no genomes, or was written before runs
            recorded what a restart needs.
    """

    archive_path = resolve_archive_path(out_dir)

    with GenomeArchive.open_readonly(archive_path) as reader:
        run_info = reader.run_info()
        version = run_info.get("format_version")

        if "arguments" not in run_info:
            raise ValueError(
                f"{archive_path} was written before runs recorded the arguments they "
                f"were started with (archive format {version}, restarting needs "
                f"{ARCHIVE_FORMAT_VERSION}), so it cannot be restarted. Start a new run "
                "instead, or point --out_dir at a directory of its own."
            )

        stored, max_genome, max_insertion, max_innovation = reader.connection.execute(
            "SELECT COUNT(*), MAX(genome_number), MAX(insertion), "
            "MAX(max_innovation_number) FROM genomes"
        ).fetchone()

        if not stored:
            raise ValueError(
                f"{archive_path} holds no evaluated genomes, so there is nothing to "
                "restart from."
            )

        if max_innovation is None:
            raise ValueError(
                f"{archive_path} records no gate innovation numbers (archive format "
                f"{version}, restarting needs {ARCHIVE_FORMAT_VERSION}), so a restart "
                "could reuse one for a different gate. Start a new run instead."
            )

        population = [
            CircuitGenome.from_dict(reader.get_genome_dict(number))
            for number in reader.population_at()
        ]
        best_genome = _best_by(reader, "loss", "ASC")
        target_metric_best = _best_by(reader, "target_metric", "DESC")

    logger.info(
        "restarting {} from {} inserted genomes ({} in the population)",
        archive_path,
        max_insertion,
        len(population),
    )

    return RestartState(
        archive_path=archive_path,
        arguments=dict(run_info["arguments"]),
        run_info=run_info,
        population=population,
        best_genome=best_genome,
        target_metric_best=target_metric_best,
        next_genome_number=int(max_genome) + 1,
        inserted_genomes=int(max_insertion or 0),
        max_innovation_number=int(max_innovation),
    )


def _best_by(
    reader: GenomeArchive, column: str, direction: str
) -> CircuitGenome | None:
    """Loads the genome holding the best value of a generated fitness column.

    Args:
        reader: The archive to read, opened for reading.
        column: ``"loss"`` or ``"target_metric"``.
        direction: ``"ASC"`` when lower is better, ``"DESC"`` when higher is.

    Returns:
        The best genome, or None when no genome recorded that key.
    """

    row = reader.connection.execute(
        f"SELECT genome_number FROM genomes WHERE {column} IS NOT NULL "
        f"ORDER BY {column} {direction} LIMIT 1"
    ).fetchone()
    return (
        None if row is None else CircuitGenome.from_dict(reader.get_genome_dict(row[0]))
    )


def restart_arguments(
    state: RestartState, args: argparse.Namespace
) -> argparse.Namespace:
    """Builds the arguments a restarted run uses: the original run's, plus overrides.

    The archive is authoritative, so the search continues exactly as configured;
    only :data:`OVERRIDABLE_ARGUMENTS` are taken from this invocation, and any
    argument new since the run started keeps the value given here.

    Args:
        state: The stopped run's state.
        args: The arguments this invocation was given.

    Returns:
        A namespace holding the run's arguments with the allowed overrides
        applied.
    """

    values = dict(state.arguments)
    given = vars(args)
    for name, value in given.items():
        if name in OVERRIDABLE_ARGUMENTS or name not in values:
            values[name] = value
    return argparse.Namespace(**values)


def differing_arguments(
    state: RestartState, args: argparse.Namespace
) -> dict[str, tuple[Any, Any]]:
    """Lists the arguments this invocation would change about the stopped run.

    :data:`OVERRIDABLE_ARGUMENTS` are left out, as are arguments that did not
    exist when the run started.

    Args:
        state: The stopped run's state.
        args: The arguments this invocation was given.

    Returns:
        ``(recorded value, given value)`` for each differing argument, keyed by
        argument name.
    """

    given = vars(args)
    return {
        name: (recorded, given[name])
        for name, recorded in state.arguments.items()
        if name not in OVERRIDABLE_ARGUMENTS
        and name in given
        and given[name] != recorded
    }


def restored_strategy(
    state: RestartState,
    args: argparse.Namespace,
    compare: Callable[[CircuitGenome, CircuitGenome], int],
) -> PopulationStrategy:
    """Rebuilds the run's population strategy, holding what it held when it stopped.

    Args:
        state: The stopped run's state.
        args: The arguments the restarted run uses (see
            :func:`restart_arguments`).
        compare: Genome comparison used to order the population.

    Returns:
        The strategy, restored.
    """

    population = PopulationStrategy.from_args(args, compare)
    population.restore(state)
    return population


def prepare(
    args: argparse.Namespace, fail: Callable[[str], Any]
) -> tuple[argparse.Namespace, RestartState | None, int]:
    """Decides whether this run continues the one in ``--out_dir``, and how far.

    Called by every entry point straight after parsing, on every rank, so a
    restarted run rebuilds its objective from the same configuration as its
    search. Nothing is written here: an overwritten run is discarded by
    :meth:`~src.utils.genome_archive.GenomeArchive.create`, which only the
    serial run and the MPI master reach.

    Args:
        args: The arguments this invocation was given.
        fail: Called with an explanation when the run cannot start, and expected
            not to return (``parser.error``).

    Returns:
        The arguments to run with (the continued run's, for a restart), the
        state being continued (``None`` for a new run), and how many genomes to
        evaluate now -- for a restart, however many of ``--number_genomes`` the
        run has left to evaluate.
    """

    wanted = getattr(args, "restart", "auto")
    overwriting = getattr(args, "overwrite_archive", False)

    # Discarding the run and insisting on continuing one are contradictory; the
    # default 'auto' only asks for a restart when there is a run to continue, so
    # discarding takes precedence over it.
    if overwriting and wanted == "require":
        fail(
            f"--overwrite_archive discards the run in {args.out_dir}, so it cannot be "
            "combined with --restart require."
        )

    if overwriting or not archive_exists(args.out_dir):
        if wanted == "require":
            fail(
                f"--restart require was given, but {args.out_dir} holds no run to continue."
            )
        return args, None, args.number_genomes

    if wanted == "never":
        fail(
            f"{args.out_dir} already holds a run. Pass --restart auto to continue it, "
            "--overwrite_archive to discard it and start a new one, or give another "
            "--out_dir."
        )

    try:
        state = load(args.out_dir)
    except ValueError as error:
        fail(str(error))

    differences = differing_arguments(state, args)
    if differences and not getattr(args, "force_restart", False):
        listed = "; ".join(
            f"{name}: the run has {recorded!r}, this command gives {given!r}"
            for name, (recorded, given) in sorted(differences.items())
        )
        fail(
            f"this command's arguments differ from the run in {args.out_dir} ({listed}). "
            "Restart it with the arguments it was started with, pass --force_restart to "
            "continue it as it was configured, or start a new run elsewhere."
        )

    if differences:
        logger.warning(
            "continuing the run as it was configured, ignoring {} changed argument(s): {}",
            len(differences),
            ", ".join(sorted(differences)),
        )

    arguments = restart_arguments(state, args)
    remaining = max(0, int(arguments.number_genomes) - state.inserted_genomes)

    if remaining:
        logger.info(
            "continuing {}: {} of {} genomes evaluated, {} to go.",
            args.out_dir,
            state.inserted_genomes,
            arguments.number_genomes,
            remaining,
        )
    else:
        logger.warning(
            "{} has already evaluated {} genomes, which is at least the {} asked for; "
            "raise --number_genomes to evaluate more.",
            args.out_dir,
            state.inserted_genomes,
            arguments.number_genomes,
        )

    return arguments, state, remaining


def resume(
    search: EXAQC, state: RestartState, args: argparse.Namespace | None = None
) -> None:
    """Points a freshly built search at where the stopped run left off.

    The search's population is expected to have been restored already (see
    :func:`restored_strategy`); this restores the counters around it and the
    innovation numbering, so the genomes it generates continue the run's
    numbering rather than colliding with it.

    Args:
        search: The search built from the run's arguments.
        state: The stopped run's state.
        args: The arguments the restart was given. When given, and the search
            writes to an archive, the restart is recorded there -- unless the run
            has already evaluated every genome asked for, so that requeueing a
            finished run does not fill its record with restarts that did nothing.

    Returns:
        None. Sets the search's ``genome_number``, ``inserted_genomes`` and
        ``target_metric_best_genome``, advances the shared innovation number
        generator past every number the run handed out, and records the restart.
    """

    innovation_number_generator.current_innovation_number = max(
        innovation_number_generator.current_innovation_number,
        state.max_innovation_number,
    )
    search.genome_number = max(search.genome_number, state.next_genome_number - 1)
    search.inserted_genomes = state.inserted_genomes
    search.target_metric_best_genome = state.target_metric_best

    wanted = getattr(args, "number_genomes", None) if args is not None else None
    nothing_to_do = wanted is not None and state.inserted_genomes >= int(wanted)

    if args is not None and search.archive is not None and not nothing_to_do:
        record_restart(search.archive, state, args)


def record_restart(
    archive: GenomeArchive, state: RestartState, args: argparse.Namespace
) -> None:
    """Records that a run was restarted, without disturbing what it already recorded.

    A run's ``command_line`` and ``start_time`` describe the run that created the
    archive, so each restart is appended to a ``restarts`` list instead of
    replacing them.

    Args:
        archive: The run's archive, opened for writing.
        state: The state the restart continues from.
        args: The arguments the restart was given.

    Returns:
        None. Appends an entry to the archive's ``restarts`` run info.
    """

    restarts = list(archive.run_info().get("restarts") or [])
    restarts.append(
        {
            "restarted_at": time.time(),
            "command_line": " ".join(sys.argv),
            "from_insertion": state.inserted_genomes,
            "number_genomes": getattr(args, "number_genomes", None),
        }
    )
    archive.set_run_info(restarts=restarts)
