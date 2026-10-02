"""Checks ``scripts/exaqc_rl_job.sh`` against the entry point it submits.

The job script builds a command line for
:mod:`src.examples.reinforcement_learning`, so it duplicates two things the
Python side owns: which environments accept ``--healthy_reward``, and the flags
themselves. These tests keep the two from drifting apart -- a mismatch would
only surface as a Slurm job that dies on startup, hours after submission.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import subprocess
import sys

from pathlib import Path

import pytest

from src.examples.reinforcement_learning import (
    ENV_CHOICES,
    ENV_IDS,
    build_parser,
    environment_knob_kwargs,
    supported_env_knobs,
)

JOB_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "exaqc_rl_job.sh"
SUBMIT_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "submit_exaqc_rl_jobs.sh"
)

#: Environments to build the job command for. Covers both sides of the
#: ``--healthy_reward`` split, including HalfCheetah -- MuJoCo locomotion that
#: nonetheless has no alive bonus, and so the easiest one to get wrong.
_SAMPLE_ENVIRONMENTS: tuple[str, ...] = (
    "walker2d",
    "hopper",
    "ant",
    "humanoid",
    "halfcheetah",
    "cartpole",
    "frozenlake",
    "mountaincar_continuous",
)


def _job_command(environment: str, healthy_reward: str | None = None) -> list[str]:
    """Runs the job script in dry-run mode and returns the arguments it builds.

    Args:
        environment: The ``--env`` value to build the command for.
        healthy_reward: Value for the ``HEALTHY_REWARD`` environment variable,
            or None to leave it unset (the default, which passes no
            ``--healthy_reward`` at all).

    Returns:
        The arguments the script would pass to the reinforcement-learning entry
        point, with the interpreter and module name stripped.

    Raises:
        AssertionError: If the script exits non-zero.
    """

    result = subprocess.run(
        [
            "bash",
            str(JOB_SCRIPT),
            environment,
            "6",
            "6",
            "exaqc_testtag_tree_2_" + environment + "_i20_1",
            "20",
            "5",
            "tree",
            "2",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            **({} if healthy_reward is None else {"HEALTHY_REWARD": healthy_reward}),
        },
    )
    assert result.returncode == 0, result.stderr

    argv = shlex.split(result.stdout)
    return argv[argv.index("src.examples.reinforcement_learning") + 1 :]


def test_healthy_reward_environment_list_matches_the_entry_point() -> None:
    """The script's environment list matches what Gymnasium actually accepts.

    The script cannot import the package -- its dry-run check deliberately runs
    before the venv is activated -- so it carries its own list. This asserts
    that list is exactly the set of environments whose Gymnasium constructor
    takes ``healthy_reward``.
    """

    text = JOB_SCRIPT.read_text()
    match = re.search(r'^HEALTHY_REWARD_ENVIRONMENTS="([^"]*)"', text, re.M)
    assert match, "the job script no longer defines HEALTHY_REWARD_ENVIRONMENTS"

    listed = set(match.group(1).split())
    accepted = {
        name
        for name in ENV_CHOICES
        if "healthy_reward" in supported_env_knobs(ENV_IDS[name])
    }

    assert listed == accepted, (
        f"the job script lists {sorted(listed)} as accepting --healthy_reward, "
        f"but the environments that actually accept it are {sorted(accepted)}"
    )


@pytest.mark.parametrize("environment", _SAMPLE_ENVIRONMENTS)
def test_job_command_parses_and_only_sets_supported_knobs(environment: str) -> None:
    """The command the job builds parses, and its knobs suit the environment.

    Args:
        environment: The ``--env`` the command is built for.
    """

    argv = _job_command(environment)
    assert "" not in argv, "an empty argument leaked into the command"

    with contextlib.redirect_stderr(io.StringIO()):
        args = build_parser().parse_args(argv)

    assert args.env == environment

    # HEALTHY_REWARD is unset here, so the environment's own value stands and
    # no reward knob is forwarded at all
    assert environment_knob_kwargs(args, environment) == {}
    assert args.healthy_reward is None


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the job script needs a POSIX shell"
)
def test_job_script_is_valid_shell() -> None:
    """The job script parses as shell, so a syntax error cannot reach the queue."""

    result = subprocess.run(
        ["bash", "-n", str(JOB_SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_job_name_and_archive_directory_are_the_same() -> None:
    """A queued job and the archive it writes carry the same name.

    The submitting script owns the name and passes it down, so this asserts the
    two scripts agree rather than that they happen to build matching strings.
    The island count is part of the name because two island counts of the same
    experiment must not share an archive: ``--restart`` defaults to ``auto``, so
    a shared directory would silently resume the wrong run.
    """

    environment = {**os.environ, "DRY_RUN": "1", "N_ISLANDS": "30"}
    submitted = subprocess.run(
        [
            "sh",
            str(SUBMIT_SCRIPT),
            "3",
            "5",
            "phase1",
            "walker2d",
            "6",
            "6",
            "tree",
            "2",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert submitted.returncode == 0, submitted.stderr

    lines = submitted.stdout.strip().splitlines()
    assert len(lines) == 3, f"expected runs 3..5 inclusive, got {len(lines)}"

    seen: set[str] = set()
    for line in lines:
        argv = shlex.split(line)
        name = argv[argv.index("-J") + 1]
        assert "i30" in name, f"the island count is missing from {name!r}"

        # the submitter spells the job script relative to its own location, so
        # it is found by suffix rather than by an exact path
        position = next(
            index
            for index, value in enumerate(argv)
            if value.endswith("exaqc_rl_job.sh")
        )
        job_arguments = argv[position + 1 :]
        built = subprocess.run(
            ["bash", str(JOB_SCRIPT), *job_arguments],
            capture_output=True,
            text=True,
            env=environment,
        )
        assert built.returncode == 0, built.stderr

        command = shlex.split(built.stdout)
        out_dir = command[command.index("--out_dir") + 1]
        assert out_dir.rsplit("/", 1)[-1] == name
        seen.add(name)

    assert len(seen) == 3, "each run must get its own archive directory"


def test_healthy_reward_is_passed_only_when_the_variable_asks_for_it() -> None:
    """``HEALTHY_REWARD`` opts a run into a non-default upright bonus.

    Left unset, nothing is forwarded and Gymnasium's own value applies, which
    keeps a run comparable with published baselines.
    """

    for environment in ("walker2d", "hopper", "ant", "humanoid"):
        arguments = _job_command(environment, healthy_reward="0.2")
        with contextlib.redirect_stderr(io.StringIO()):
            parsed = build_parser().parse_args(arguments)
        assert environment_knob_kwargs(parsed, environment) == {"healthy_reward": 0.2}


@pytest.mark.parametrize(
    "environment,value,expected",
    [
        ("halfcheetah", "0.2", "no healthy (alive) bonus"),
        ("cartpole", "0.2", "no healthy (alive) bonus"),
        ("walker2d", "abc", "must be a number"),
        ("walker2d", "1.2.3", "must be a number"),
    ],
)
def test_healthy_reward_is_rejected_when_it_cannot_apply(
    environment: str, value: str, expected: str
) -> None:
    """A ``HEALTHY_REWARD`` that cannot be honoured fails before the job runs.

    Dropping it silently would leave a run tagged as one reward setting while
    having trained under another, so the script refuses -- mirroring
    ``environment_knob_kwargs``, but without spending a scheduling slot first.

    Args:
        environment: The ``--env`` to build the command for.
        value: The ``HEALTHY_REWARD`` value to offer.
        expected: Text the refusal must mention.
    """

    result = subprocess.run(
        [
            "bash",
            str(JOB_SCRIPT),
            environment,
            "6",
            "6",
            f"exaqc_t_tree_2_{environment}_i20_1",
            "20",
            "5",
            "tree",
            "2",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "1", "HEALTHY_REWARD": value},
    )

    assert result.returncode != 0, "the job script accepted an impossible setting"
    assert expected in result.stderr
