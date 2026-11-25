"""Tests for reproducible stochastic evaluation, seeded by ``eval_seed``.

Evaluation episode ``i`` resets from ``eval_seed + i`` and, under a stochastic
policy, samples its actions from a generator seeded with the same value. So an
evaluation is determined by the genome and its ``eval_seed`` alone: evaluating
twice gives the same actions, genomes sharing an ``eval_seed`` face the same
sampling noise, and evaluating never draws from the global generator training
is seeded through (``--training_seed``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Normal

from src.circuits.circuit import CircuitGenome
from src.trainer.reinforcement_trainer import RLEnvironment, sample_from

from tests.reinforcement_trainer_test_utils import (
    build_rl_genome,
    build_trainer,
    make_continuous_test_environment,
    make_test_environment,
)


def recorded_actions(
    genome: CircuitGenome,
    environment: RLEnvironment,
    eval_seed: int,
    eval_policy: str = "stochastic",
) -> list[Any]:
    """Evaluates a genome and returns every action its evaluation took.

    Args:
        genome: The genome to evaluate (its model already initialized).
        environment: The environment to evaluate on.
        eval_seed: The evaluation seed to use.
        eval_policy: The action-selection regime to evaluate under.

    Returns:
        The actions passed to ``env.step``, in order, as plain Python values.
    """

    trainer = build_trainer("reinforce")
    genome.hyperparameters["eval_seed"] = eval_seed
    genome.hyperparameters["eval_policy"] = eval_policy
    hp = trainer.resolve_hyperparameters(genome)

    actions: list[Any] = []
    original_make = environment.make

    def recording_make() -> Any:
        """Builds the environment with its ``step`` recording each action.

        Returns:
            The environment the wrapped :meth:`RLEnvironment.make` returns.
        """

        env = original_make()
        step = env.step

        def recording_step(action: Any) -> Any:
            """Records an action, then steps the environment with it."""

            actions.append(np.asarray(action).tolist())
            return step(action)

        env.step = recording_step
        return env

    environment.make = recording_make
    try:
        trainer.evaluate(genome, environment, hp)
    finally:
        environment.make = original_make
    return actions


def build(continuous: bool) -> tuple[CircuitGenome, RLEnvironment]:
    """Builds a genome and the test environment it acts in.

    Args:
        continuous: Whether to use the continuous (``Box``) test environment.

    Returns:
        The genome, with its model initialized, and its environment.
    """

    genome, observation_features = build_rl_genome(
        genome_number=1,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=build_trainer("reinforce"),
        continuous=continuous,
    )
    genome.initialize_model()
    environment = (
        make_continuous_test_environment(observation_features)
        if continuous
        else make_test_environment(observation_features)
    )
    return genome, environment


def test_sample_from_draws_reproducibly_from_the_same_distribution() -> None:
    """A seeded draw repeats exactly and matches the distribution's own sampler."""

    normal = Normal(torch.tensor([0.5, -0.25]), torch.tensor([0.1, 2.0]))
    first = sample_from(normal, torch.Generator().manual_seed(7))
    second = sample_from(normal, torch.Generator().manual_seed(7))
    assert torch.equal(first, second)
    assert first.shape == normal.sample().shape

    categorical = Categorical(logits=torch.tensor([0.0, 1.0, 2.0]))
    draws = [
        int(sample_from(categorical, torch.Generator().manual_seed(seed)))
        for seed in range(200)
    ]
    assert set(draws) <= {0, 1, 2}
    # the most likely category should also be the most frequent draw
    assert max(set(draws), key=draws.count) == 2

    # many seeded Normal draws have the distribution's mean and spread
    samples = torch.stack(
        [
            sample_from(normal, torch.Generator().manual_seed(seed))
            for seed in range(2000)
        ]
    )
    assert samples.mean(0) == pytest.approx([0.5, -0.25], abs=0.15)
    assert samples.std(0) == pytest.approx([0.1, 2.0], rel=0.1)


@pytest.mark.parametrize("continuous", [False, True])
def test_stochastic_evaluation_repeats_exactly_for_the_same_eval_seed(
    continuous: bool,
) -> None:
    """The same genome and ``eval_seed`` take the same actions; another seed differs.

    Args:
        continuous: Whether to evaluate in the continuous test environment.
    """

    genome, environment = build(continuous)

    # the global generator is in a different state each time, which a seeded
    # evaluation must not depend on
    torch.manual_seed(1)
    first = recorded_actions(genome, environment, eval_seed=1000)
    torch.manual_seed(2)
    second = recorded_actions(genome, environment, eval_seed=1000)
    other = recorded_actions(genome, environment, eval_seed=5000)

    assert first and first == second
    assert other != first


def test_evaluation_never_draws_from_the_global_generator() -> None:
    """Training randomness is unaffected by however much evaluation samples."""

    genome, environment = build(continuous=True)

    torch.manual_seed(3)
    before = torch.get_rng_state()
    recorded_actions(genome, environment, eval_seed=1000)

    assert torch.equal(torch.get_rng_state(), before)


def test_greedy_evaluation_is_unchanged() -> None:
    """Greedy evaluation takes the policy mean, whatever the seed."""

    genome, environment = build(continuous=True)

    first = recorded_actions(genome, environment, eval_seed=1000, eval_policy="greedy")
    other = recorded_actions(genome, environment, eval_seed=5000, eval_policy="greedy")

    # the test environment's observations do not depend on the reset seed, so
    # a deterministic policy takes the same actions from either seed
    assert first and first == other
