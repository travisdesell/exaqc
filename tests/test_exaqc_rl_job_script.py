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


def _job_command(environment: str) -> list[str]:
    """Runs the job script in dry-run mode and returns the arguments it builds.

    Args:
        environment: The ``--env`` value to build the command for.

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
            "1",
            "20",
            "5",
            "tree_2",
            "tree",
            "2",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "1"},
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

    # never raises: the script only passes knobs the environment supports
    knobs = environment_knob_kwargs(args, environment)

    if "healthy_reward" in supported_env_knobs(ENV_IDS[environment]):
        assert knobs["healthy_reward"] == pytest.approx(0.2)
    else:
        assert "healthy_reward" not in knobs


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the job script needs a POSIX shell"
)
def test_job_script_is_valid_shell() -> None:
    """The job script parses as shell, so a syntax error cannot reach the queue."""

    result = subprocess.run(
        ["bash", "-n", str(JOB_SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
