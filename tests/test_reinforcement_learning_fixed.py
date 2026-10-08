"""Tests for the classical reinforcement-learning baseline entry point.

``src.examples.reinforcement_learning_fixed`` trains a fixed classical MLP with
the same RL trainers the quantum search uses. Its parser once hand-copied the
trainers' flags and drifted from them (it lost ``--eval_seed``), and its model
once lacked methods the trainers had started calling, so it crashed on every
run without any test noticing. These tests pin both: the parser shares the
trainers' flags and defaults with the quantum entry point, and ``main`` runs to
completion.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import src.examples.reinforcement_learning as quantum_entry_point
import src.examples.reinforcement_learning_fixed as fixed_entry_point


def defaults_of(build_parser: Callable[[], argparse.ArgumentParser]) -> dict[str, Any]:
    """Reads every argument's default off a parser factory.

    Args:
        build_parser: A zero-argument callable returning an ``ArgumentParser``.

    Returns:
        Each argument's default, keyed by destination.
    """

    return {action.dest: action.default for action in build_parser()._actions}


def test_trainer_flags_match_the_quantum_entry_point() -> None:
    """The baseline takes the trainers' flags with the quantum search's defaults."""

    fixed = defaults_of(fixed_entry_point.build_parser)
    quantum = defaults_of(quantum_entry_point.build_parser)

    # every flag the trainer classes register (all four are registered by both
    # entry points) must be present with the same default
    trainer_parser = argparse.ArgumentParser()
    for trainer_class in (
        fixed_entry_point.ReinforcementLearningTrainer,
        fixed_entry_point.ReinforceTrainer,
        fixed_entry_point.PPOTrainer,
        fixed_entry_point.QLearningTrainer,
    ):
        trainer_class.initialize_parser(trainer_parser)
    trainer_dests = {
        action.dest for action in trainer_parser._actions if action.dest != "help"
    }

    for dest in trainer_dests:
        assert dest in fixed, f"--{dest} is missing from the baseline's parser"
        assert fixed[dest] == quantum[dest]

    # the seeds are drawn per run unless given, as in the quantum search
    assert fixed["training_seed"] is None
    assert fixed["eval_seed"] is None


@pytest.mark.parametrize("algo", ["reinforce", "ppo", "q_learning"])
def test_main_trains_and_saves_a_gif(
    algo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``main`` trains the classical model and writes its rollout GIF.

    Args:
        algo: The RL algorithm to train with.
        monkeypatch: Used to set ``sys.argv``.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    gif = tmp_path / "rollout.gif"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reinforcement_learning_fixed.py",
            "--env",
            "cartpole",
            "--algo",
            algo,
            "--episodes",
            "3",
            "--eval_episodes",
            "1",
            "--log_every",
            "1",
            "--improvement_cutoff",
            "1",
            "--eval_seed",
            "1000000",
            "--rollout_steps",
            "64",
            "--max_steps",
            "50",
            "--visualize_episodes",
            "1",
            "--output_file",
            str(gif),
            "--out_dir",
            str(tmp_path / "out"),
            "--logging_level",
            "WARNING",
        ],
    )

    fixed_entry_point.main()

    assert gif.is_file() and gif.stat().st_size > 0
    assert (tmp_path / "out" / "run.log").is_file()
