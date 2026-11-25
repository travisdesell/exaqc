"""Checks ``scripts/exaqc_moo_job.sh`` and its submit script against the entry point.

The job script builds a command line for :mod:`src.examples.classification_moo`;
a flag it gets wrong would only surface as a Slurm job that dies on startup,
hours after submission. These tests build the command with ``DRY_RUN`` and
parse it with the entry point's own parser.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.examples.classification_moo import build_parser, parse_objectives

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
JOB_SCRIPT = SCRIPTS / "exaqc_moo_job.sh"
SUBMIT_SCRIPT = SCRIPTS / "submit_exaqc_moo_jobs.sh"

#: Settings the scripts read from the environment; removed from the inherited
#: environment so a test only sees the ones it sets.
_SETTINGS: tuple[str, ...] = (
    "DEVICE_CARD",
    "NOISE_IDLE",
    "NOISE_CALIBRATION",
    "NOISE_QUBITS",
    "OBJECTIVES",
    "ALGORITHM",
    "MAX_POPULATION_SIZE",
    "NUMBER_GENOMES",
    "EPOCHS",
    "IMPROVEMENT_CUTOFF",
    "BATCH_SIZE",
    "QIM",
    "QOM",
    "ENCODING",
    "DECODING",
    "TRAINING_SAMPLES",
    "VALIDATION_SAMPLES",
    "INPUT_QUBITS",
    "OUTPUT_QUBITS",
    "N_TASKS",
    "TIME_LIMIT",
    "ACCOUNT",
)


def _environment(**settings: str) -> dict[str, str]:
    """The environment to run a script in: dry run, plus the given settings.

    Args:
        **settings: Setting variables to set.

    Returns:
        The environment.
    """

    environment = {k: v for k, v in os.environ.items() if k not in _SETTINGS}
    environment.update(DRY_RUN="1", ARCHIVE_DIR="/archives", LOG_DIR="/logs")
    environment.update(settings)
    return environment


def _job_command(
    dataset: str = "breast_cancer", **settings: str
) -> subprocess.CompletedProcess[str]:
    """Runs the job script in dry-run mode.

    Args:
        dataset: The dataset argument.
        **settings: Setting variables to set.

    Returns:
        The finished process; on success its stdout is the search command.
    """

    return subprocess.run(
        ["bash", str(JOB_SCRIPT), dataset, "8", "1", "run_name"],
        capture_output=True,
        text=True,
        env=_environment(**settings),
        check=False,
    )


def _parsed(**settings: str) -> Any:
    """Parses the job script's command with the entry point's parser.

    Args:
        **settings: Setting variables to set.

    Returns:
        The parsed arguments.
    """

    result = _job_command(**settings)
    assert result.returncode == 0, result.stderr
    words = shlex.split(result.stdout)
    module = words.index("src.examples.classification_moo")
    return build_parser().parse_args(words[module + 1 :])


def test_scripts_are_valid_shell() -> None:
    """Both scripts pass their shells' syntax checks."""

    subprocess.run(["bash", "-n", str(JOB_SCRIPT)], check=True)
    subprocess.run(["sh", "-n", str(SUBMIT_SCRIPT)], check=True)


def test_default_job_is_noise_aware() -> None:
    """By default the run trains under a device card and minimizes loss and p_fail."""

    args = _parsed()

    assert args.dataset == "breast_cancer"
    assert (args.input_qubits, args.output_qubits) == (8, 1)
    assert args.quantum_input_mode == "ry"
    assert args.encoding == "linear"
    assert args.noise_device_card == "transmon_detuned_lindblad_markov"
    assert args.noise_idle
    assert [o.name for o in parse_objectives(args.objectives)] == ["loss", "p_fail"]
    assert args.population_strategy == "nsga2"
    assert args.out_dir == "/archives/run_name"


def test_noiseless_baseline_and_settings_reach_the_command() -> None:
    """DEVICE_CARD=none drops the noise, and the other settings are passed on."""

    baseline = _parsed(DEVICE_CARD="none")
    assert baseline.noise_device_card is None
    assert baseline.objectives == ["loss:1", "n_enabled_gates:1"]

    args = _parsed(
        ALGORITHM="nsga3",
        OBJECTIVES="loss:1 p_fail:1 total_time_ns:1",
        NOISE_IDLE="0",
        EPOCHS="5",
        TRAINING_SAMPLES="100",
        MAX_POPULATION_SIZE="12",
    )
    assert args.population_strategy == "nsga3"
    assert args.objectives == ["loss:1", "p_fail:1", "total_time_ns:1"]
    assert not args.noise_idle
    assert args.epochs == 5
    assert args.training_samples == 100
    assert args.max_population_size == 12


@pytest.mark.parametrize(
    "settings",
    [
        {"ALGORITHM": "nsga4"},
        {"NOISE_IDLE": "yes"},
        {"EPOCHS": "-1"},
        {"OBJECTIVES": "loss;rm"},
    ],
)
def test_bad_settings_are_rejected(settings: dict[str, str]) -> None:
    """Malformed settings stop the job script with a message.

    Args:
        settings: The bad setting.
    """

    result = _job_command(**settings)
    assert result.returncode != 0
    assert "error:" in result.stderr


def test_submit_names_jobs_and_archives_by_card_and_algorithm() -> None:
    """Each run gets one name for its job, logs and archive, with PPSN qubits."""

    result = subprocess.run(
        ["sh", str(SUBMIT_SCRIPT), "1", "2", "card", "wine"],
        capture_output=True,
        text=True,
        env=_environment(DEVICE_CARD="transmon_detuned_mixed"),
        check=True,
    )
    lines = result.stdout.splitlines()

    assert len(lines) == 2
    words = shlex.split(lines[0])
    name = "exaqc_moo_card_wine_transmon_detuned_mixed_nsga2_1"
    assert words[words.index("-J") + 1] == name
    assert words[-4:] == ["wine", "4", "2", name]

    bad = subprocess.run(
        ["sh", str(SUBMIT_SCRIPT), "1", "1", "card", "mnist"],
        capture_output=True,
        text=True,
        env=_environment(),
        check=False,
    )
    assert bad.returncode != 0
