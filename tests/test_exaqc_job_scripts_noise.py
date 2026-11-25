"""Checks the NOISE_* settings of the run and Slurm scripts.

``scripts/run_classification.sh``, ``scripts/run_rl.sh`` and the Slurm pair
``scripts/submit_exaqc_rl_jobs.sh`` / ``scripts/exaqc_rl_job.sh`` all take the
same ``NOISE_*`` environment variables. These tests run them in ``DRY_RUN``
mode, which prints the commands instead of running or submitting anything, and
check that the variables reach the entry point's command exactly when they
should and that the Slurm run name carries the noise setting.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from src.examples.classification import build_parser as build_classification_parser
from src.examples.reinforcement_learning import build_parser as build_rl_parser

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUN_CLASSIFICATION = SCRIPTS / "run_classification.sh"
RUN_RL = SCRIPTS / "run_rl.sh"
RL_JOB = SCRIPTS / "exaqc_rl_job.sh"
RL_SUBMIT = SCRIPTS / "submit_exaqc_rl_jobs.sh"

#: Environment variables the scripts read as settings, removed from the
#: inherited environment so a test only sees the ones it sets.
_SETTING_VARIABLES: tuple[str, ...] = (
    "NOISE_CALIBRATION",
    "NOISE_QUBITS",
    "NOISE_TYPE",
    "NOISE_P_1Q",
    "NOISE_P_2Q",
    "NUMBER_GENOMES",
    "MAX_POPULATION_SIZE",
    "ACCOUNT",
    "LOG_DIR",
    "ARCHIVE_DIR",
    "VENV",
    "TIME_LIMIT",
    "N_ISLANDS",
    "MAX_ISLAND_SIZE",
    "HEALTHY_REWARD",
    "EPISODES",
    "IMPROVEMENT_CUTOFF",
    "LOG_EVERY",
    "EVAL_SEED",
    "USE_ONLY",
    "EPOCHS",
    "BATCH_SIZE",
    "QIM",
    "QOM",
    "ENCODING",
    "DECODING",
    "DATASET",
    "ENVIRONMENT",
    "POPULATION",
    "TOPOLOGY",
    "TAG",
    "OUT_DIR",
    "BASE_DIR",
    "N_PROCESSES",
)

#: Chip noise for an 8-wire circuit (breast_cancer: 8 input, 1 output qubits).
CHIP_8 = {"NOISE_CALIBRATION": "ibm_fez:2026-11-06", "NOISE_QUBITS": "0,1,2,3,4,5,6,7"}

#: Chip noise for a 6-wire circuit (the RL examples: 6 input, 6 output qubits).
CHIP_6 = {"NOISE_CALIBRATION": "ibm_fez:2026-11-06", "NOISE_QUBITS": "0,1,2,3,4,5"}

#: Synthetic depolarizing noise at the strengths the noise study uses.
SYNTHETIC = {"NOISE_TYPE": "depolarizing", "NOISE_P_1Q": "0.001", "NOISE_P_2Q": "0.01"}

#: The RL submit script's arguments for one halfcheetah run on a ring.
RL_SUBMIT_ARGUMENTS = ["1", "1", "tag", "halfcheetah", "6", "6", "ring"]

#: The RL job script's arguments for the same run.
RL_JOB_ARGUMENTS = ["halfcheetah", "6", "6", "name", "20", "5", "ring"]


def _environment(**settings: str) -> dict[str, str]:
    """Builds the environment a script runs under in a test.

    Args:
        settings: Setting variables to set for the script.

    Returns:
        The inherited environment without any setting variable, plus
        ``DRY_RUN`` and ``settings``.
    """
    inherited = {k: v for k, v in os.environ.items() if k not in _SETTING_VARIABLES}
    return {**inherited, "DRY_RUN": "1", **settings}


def _run(
    script: Path, arguments: list[str], **settings: str
) -> subprocess.CompletedProcess[str]:
    """Runs a script in ``DRY_RUN`` mode.

    Args:
        script: The script to run; ``sh`` runs a submit script and ``bash`` the
            others, as they are invoked for real.
        arguments: Its positional arguments.
        settings: Setting variables to set for it.

    Returns:
        The completed process, with its stdout and stderr captured.
    """
    shell = "sh" if script.name.startswith("submit_") else "bash"
    return subprocess.run(
        [shell, str(script), *arguments],
        capture_output=True,
        text=True,
        env=_environment(**settings),
    )


def _entry_argv(
    script: Path, module: str, arguments: list[str], **settings: str
) -> list[str]:
    """Returns the entry-point arguments a run or job script would pass.

    Args:
        script: The run or job script.
        module: The entry-point module name the command names, e.g.
            ``src.examples.classification``.
        arguments: The script's positional arguments.
        settings: Setting variables to set for it.

    Returns:
        The arguments after the module name, split as a shell would.

    Raises:
        AssertionError: If the script exits non-zero.
    """
    result = _run(script, arguments, **settings)
    assert result.returncode == 0, result.stderr
    argv = shlex.split(result.stdout)
    return argv[argv.index(module) + 1 :]


def _submitted(arguments: list[str], **settings: str) -> list[str]:
    """Returns the one sbatch command the RL submit script would run.

    Args:
        arguments: The submit script's positional arguments, for one run.
        settings: Setting variables to set for it.

    Returns:
        The printed ``sbatch`` command, split as a shell would.

    Raises:
        AssertionError: If the script exits non-zero or prints other than one
            command.
    """
    result = _run(RL_SUBMIT, arguments, **settings)
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(commands) == 1, result.stdout
    return shlex.split(commands[0])


@pytest.mark.parametrize("script", [RUN_CLASSIFICATION, RUN_RL, RL_JOB, RL_SUBMIT])
def test_scripts_are_valid_shell(script: Path) -> None:
    """Each script passes its shell's syntax check."""
    shell = "sh" if script.name.startswith("submit_") else "bash"
    result = subprocess.run([shell, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_local_classification_script_takes_the_noise_settings() -> None:
    """run_classification.sh passes the NOISE_* variables through unchanged."""
    chip = {
        **CHIP_8,
        "DATASET": "breast_cancer",
        "INPUT_QUBITS": "8",
        "OUTPUT_QUBITS": "1",
    }
    args = build_classification_parser().parse_args(
        _entry_argv(RUN_CLASSIFICATION, "src.examples.classification", [], **chip)
    )
    assert args.noise_calibration == "ibm_fez:2026-11-06"
    assert args.noise_qubits == "0,1,2,3,4,5,6,7"

    args = build_classification_parser().parse_args(
        _entry_argv(RUN_CLASSIFICATION, "src.examples.classification", [], **SYNTHETIC)
    )
    assert args.noise_type == "depolarizing"
    assert args.noise_p_1q == pytest.approx(0.001)
    assert args.noise_p_2q == pytest.approx(0.01)

    args = build_classification_parser().parse_args(
        _entry_argv(RUN_CLASSIFICATION, "src.examples.classification", [])
    )
    assert args.noise_type == "none" and args.noise_calibration is None


def test_local_rl_script_takes_the_noise_settings() -> None:
    """run_rl.sh passes the NOISE_* variables through unchanged."""
    chip = {**CHIP_6, "INPUT_QUBITS": "6", "OUTPUT_QUBITS": "6"}
    args = build_rl_parser().parse_args(
        _entry_argv(RUN_RL, "src.examples.reinforcement_learning", [], **chip)
    )
    assert args.noise_calibration == "ibm_fez:2026-11-06"
    assert args.noise_qubits == "0,1,2,3,4,5"

    args = build_rl_parser().parse_args(
        _entry_argv(RUN_RL, "src.examples.reinforcement_learning", [], **SYNTHETIC)
    )
    assert args.noise_type == "depolarizing"

    args = build_rl_parser().parse_args(
        _entry_argv(RUN_RL, "src.examples.reinforcement_learning", [])
    )
    assert args.noise_type == "none" and args.noise_calibration is None


def test_chip_noise_reaches_the_rl_job_and_its_name() -> None:
    """Chip noise becomes the two --noise flags and is part of the run name.

    A noiseless submission keeps the name it had before the noise settings
    existed, so earlier archives are still found by their names.
    """
    noisy = _submitted(RL_SUBMIT_ARGUMENTS, **CHIP_6)
    name = noisy[noisy.index("-J") + 1]
    assert name == "exaqc_tag_ring_halfcheetah_ibm_fez_2026-11-06_q0-1-2-3-4-5_i20_1"

    clean = _submitted(RL_SUBMIT_ARGUMENTS)
    assert clean[clean.index("-J") + 1] == "exaqc_tag_ring_halfcheetah_i20_1"

    job_arguments = noisy[noisy.index(str(RL_JOB)) + 1 :]
    args = build_rl_parser().parse_args(
        _entry_argv(
            RL_JOB, "src.examples.reinforcement_learning", job_arguments, **CHIP_6
        )
    )
    assert args.env == "halfcheetah"
    assert args.noise_calibration == "ibm_fez:2026-11-06"
    assert args.noise_qubits == "0,1,2,3,4,5"
    assert args.out_dir.endswith("/" + name)


def test_synthetic_noise_reaches_the_rl_job_and_its_name() -> None:
    """Synthetic noise becomes the three --noise flags and is part of the name."""
    argv = _submitted(RL_SUBMIT_ARGUMENTS, **SYNTHETIC)
    name = argv[argv.index("-J") + 1]
    assert name == "exaqc_tag_ring_halfcheetah_depolarizing_p1q0.001_p2q0.01_i20_1"

    args = build_rl_parser().parse_args(
        _entry_argv(
            RL_JOB, "src.examples.reinforcement_learning", RL_JOB_ARGUMENTS, **SYNTHETIC
        )
    )
    assert args.noise_type == "depolarizing"
    assert args.noise_p_1q == pytest.approx(0.001)
    assert args.noise_p_2q == pytest.approx(0.01)
    assert args.noise_calibration is None


def test_account_and_paths_are_overridable() -> None:
    """ACCOUNT, LOG_DIR and ARCHIVE_DIR override the scripts' defaults."""
    argv = _submitted(RL_SUBMIT_ARGUMENTS, ACCOUNT="examm-rnn", LOG_DIR="/home/me/logs")
    assert argv[argv.index("-A") + 1] == "examm-rnn"
    assert argv[argv.index("-o") + 1].startswith("/home/me/logs/")

    args = build_rl_parser().parse_args(
        _entry_argv(
            RL_JOB,
            "src.examples.reinforcement_learning",
            RL_JOB_ARGUMENTS,
            ARCHIVE_DIR="/home/me/archives",
        )
    )
    assert args.out_dir == "/home/me/archives/name"
