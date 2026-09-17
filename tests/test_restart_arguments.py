"""Tests for the restart flags every search entry point exposes.

Restarting is decided before anything is built, by
:func:`src.evolution.restart.prepare`, from the ``--restart``,
``--overwrite_archive`` and ``--force_restart`` flags that
:meth:`src.utils.genome_archive.GenomeArchive.initialize_parser` adds. These
tests pin the flags each entry point offers -- a search whose parser lost them
could not be restarted at all -- and the decisions ``prepare`` makes from them,
without running a search.
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend before anything imports pyplot, so
# importing the entry points stays headless.
import matplotlib

matplotlib.use("Agg")

import importlib  # noqa: E402
from argparse import Namespace  # noqa: E402
from typing import Any, NoReturn  # noqa: E402

import pytest  # noqa: E402

from src.evolution import restart  # noqa: E402
from src.evolution.steady_state_population import SteadyStatePopulation  # noqa: E402
from src.utils.genome_archive import GenomeArchive  # noqa: E402
from tests.test_restart import (  # noqa: E402
    STEADY_STATE_ARGUMENTS,
    compare,
    run_a_search,
)

#: The entry points a run can be started (and restarted) with.
ENTRY_POINTS: tuple[str, ...] = (
    "classification",
    "teacher",
    "reinforcement_learning",
)


class Refused(Exception):
    """Raised instead of exiting when ``prepare`` refuses to start a run."""


def refuse(message: str) -> NoReturn:
    """Stands in for ``parser.error``, which exits rather than returning.

    Args:
        message: Why the run cannot start.

    Raises:
        Refused: Always, carrying ``message``.
    """

    raise Refused(message)


def arguments(out_dir: Any, **overrides: Any) -> Namespace:
    """Builds the arguments an invocation would carry.

    Args:
        out_dir: The run's output directory.
        **overrides: Values replacing the defaults.

    Returns:
        The namespace ``prepare`` reads.
    """

    values: dict[str, Any] = {
        "out_dir": str(out_dir),
        "shared_file_system": False,
        "number_genomes": 10,
        "restart": "auto",
        "overwrite_archive": False,
        "force_restart": False,
    }
    values.update(overrides)
    return Namespace(**values)


def a_stopped_run(run_dir: Any, genomes: int = 6) -> None:
    """Leaves a finished run in ``run_dir`` for a restart to continue.

    Args:
        run_dir: The run's output directory.
        genomes: How many genomes it evaluated.

    Returns:
        None. Writes the run's archive.
    """

    run_a_search(
        run_dir,
        SteadyStatePopulation(4, compare),
        {**STEADY_STATE_ARGUMENTS, "number_genomes": genomes},
        genomes=genomes,
    )


@pytest.mark.parametrize("module_name", ENTRY_POINTS)
def test_every_entry_point_offers_the_restart_flags(module_name: str) -> None:
    """Each search entry point exposes the restart flags with the same defaults.

    Args:
        module_name: The entry point under ``src.examples`` to check.
    """

    module = importlib.import_module(f"src.examples.{module_name}")
    actions = {action.dest: action for action in module.build_parser()._actions}

    assert actions["restart"].default == "auto"
    assert set(actions["restart"].choices) == {"never", "auto", "require"}
    assert actions["overwrite_archive"].default is False
    assert actions["force_restart"].default is False


def test_a_new_run_starts_when_nothing_is_there(tmp_path) -> None:
    """With no archive to continue, the run starts as given.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    args, state, run_for = restart.prepare(arguments(tmp_path / "new"), refuse)

    assert state is None
    assert run_for == 10
    assert args.number_genomes == 10


def test_starting_over_a_run_that_is_already_there_is_refused(tmp_path) -> None:
    """Asking never to restart will not write into a directory holding a run.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "run"
    a_stopped_run(run_dir)

    with pytest.raises(Refused, match="already holds a run"):
        restart.prepare(arguments(run_dir, restart="never"), refuse)


def test_restarting_continues_the_run_for_the_genomes_it_has_left(tmp_path) -> None:
    """``--number_genomes`` is the run's total, so a restart evaluates the rest.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "run"
    a_stopped_run(run_dir, genomes=6)

    args, state, run_for = restart.prepare(
        arguments(run_dir, restart="auto", number_genomes=10), refuse
    )

    assert state is not None
    assert state.inserted_genomes == 6
    assert run_for == 4
    # the run's own configuration is what continues
    assert args.max_population_size == STEADY_STATE_ARGUMENTS["max_population_size"]

    # asking for no more than it already has leaves nothing to evaluate
    _, _, nothing_to_do = restart.prepare(
        arguments(run_dir, restart="auto", number_genomes=6), refuse
    )
    assert nothing_to_do == 0


def test_requiring_a_restart_fails_when_there_is_nothing_to_continue(tmp_path) -> None:
    """``--restart require`` catches a mistyped output directory.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    with pytest.raises(Refused, match="holds no run to continue"):
        restart.prepare(arguments(tmp_path / "typo", restart="require"), refuse)


def test_a_restart_that_would_change_the_search_is_refused(tmp_path) -> None:
    """Arguments that differ from the run stop the restart, unless forced.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "run"
    a_stopped_run(run_dir)

    changed = arguments(
        run_dir, restart="auto", number_genomes=20, max_population_size=99
    )
    with pytest.raises(Refused, match="max_population_size"):
        restart.prepare(changed, refuse)

    changed.force_restart = True
    args, state, run_for = restart.prepare(changed, refuse)

    assert state is not None
    # forcing continues the run as it was configured, not as this command asks
    assert args.max_population_size == STEADY_STATE_ARGUMENTS["max_population_size"]
    assert run_for == 14


def test_overwriting_wins_over_restarting_when_a_restart_is_only_offered(
    tmp_path,
) -> None:
    """Discarding a run beats the default offer to continue one, but not a demand.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "run"
    a_stopped_run(run_dir)

    # the default 'auto' asks to continue a run only when one is being kept
    _, state, run_for = restart.prepare(
        arguments(run_dir, restart="auto", overwrite_archive=True), refuse
    )
    assert state is None
    assert run_for == 10

    with pytest.raises(Refused, match="cannot be combined"):
        restart.prepare(
            arguments(run_dir, restart="require", overwrite_archive=True), refuse
        )


def test_overwriting_starts_a_new_run_in_place(tmp_path) -> None:
    """``--overwrite_archive`` discards the run and its best-genome files.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    run_dir = tmp_path / "run"
    a_stopped_run(run_dir)
    (run_dir / "best_fitness.json").write_text("{}")

    args, state, run_for = restart.prepare(
        arguments(run_dir, overwrite_archive=True), refuse
    )
    assert state is None
    assert run_for == 10

    # nothing is removed until the archive is opened for writing, which only the
    # serial run and the MPI master do
    archive = GenomeArchive.from_args(args)
    try:
        assert archive.count() == 0
    finally:
        archive.close()

    assert not (run_dir / "best_fitness.json").exists()
