"""Tests for the optional RL training stabilizers.

Covers the switches added for MuJoCo-style PPO training, all off by default:

* ``normalize_observations`` -- running mean/std observation normalization,
  whose statistics are saved in genome metadata and reused at evaluation;
* ``ppo_reward_scaling`` -- rewards divided by the running return std;
* ``lr_anneal`` -- linear learning-rate decay over the episodes;
* ``ppo_target_kl`` -- KL early stopping of a rollout's PPO epochs; and
* ``log_std_min`` / ``log_std_max`` -- the continuous policy's log-std clamp.
"""

from __future__ import annotations

from typing import Any

import argparse
import math

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.circuits.circuit import CircuitGenome
from src.trainer.ppo_trainer import PPOTrainer
from src.trainer.reinforce_trainer import ReinforceTrainer
from src.trainer.reinforcement_trainer import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    OBSERVATION_NORMALIZER_KEY,
    RL_HYPERPARAMETER_DEFAULTS,
    RLEnvironment,
    ReinforcementLearningTrainer,
    action_distribution,
    encode_observation,
    log_std_bounds,
    reset_observation_normalizer,
)
from src.trainer.running_normalizers import (
    ObservationNormalizer,
    RewardScaler,
    RunningMeanStd,
)
from tests.reinforcement_trainer_test_utils import (
    build_rl_genome,
    build_trainer,
    make_continuous_test_environment,
    prepare_single_update,
)


def _ppo_setup(
    **hyperparameters: object,
) -> tuple[PPOTrainer, CircuitGenome, RLEnvironment]:
    """Builds a PPO trainer, a continuous genome and its environment.

    Args:
        **hyperparameters: Overrides merged into the genome's hyperparameters.

    Returns:
        ``(trainer, genome, environment)``.
    """

    trainer = build_trainer("ppo")
    genome, n_features = build_rl_genome(
        0, "pennylane", "deep", "linear", "linear", trainer, continuous=True
    )
    genome.hyperparameters.update(hyperparameters)
    environment = make_continuous_test_environment(n_features)
    return trainer, genome, environment


# ---------------------------------------------------------------------------
# Running statistics
# ---------------------------------------------------------------------------


def test_running_mean_std_matches_numpy_over_batches() -> None:
    """Folding batches in gives the mean and variance of all the samples."""

    rng = np.random.default_rng(0)
    samples = rng.normal(3.0, 2.0, size=(200, 3))
    statistics = RunningMeanStd(shape=(3,), epsilon=1e-12)
    for start in range(0, 200, 7):
        statistics.update(samples[start : start + 7])

    np.testing.assert_allclose(statistics.mean, samples.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(statistics.var, samples.var(axis=0), rtol=1e-6)


def test_observation_normalizer_round_trips_and_standardizes() -> None:
    """Saved statistics rebuild an identical normalizer, and outputs are clipped."""

    rng = np.random.default_rng(1)
    normalizer = ObservationNormalizer(2, clip=5.0)
    normalizer.update(rng.normal([10.0, -4.0], [3.0, 0.5], size=(500, 2)))

    restored = ObservationNormalizer.from_dict(normalizer.to_dict())
    probe = np.array([[10.0, -4.0], [1000.0, -1000.0]])
    np.testing.assert_allclose(restored.normalize(probe), normalizer.normalize(probe))
    assert np.allclose(normalizer.normalize(probe[0]), 0.0, atol=0.2)
    np.testing.assert_allclose(normalizer.normalize(probe[1]), [5.0, -5.0])


def test_reward_scaler_preserves_sign_and_resets_return() -> None:
    """Scaling keeps each reward's sign and restarts the return at episode end."""

    scaler = RewardScaler(gamma=0.9)
    scaled = [scaler.scale(reward, done=False) for reward in (5.0, -3.0, 2.0)]
    assert [math.copysign(1, value) for value in scaled] == [1, -1, 1]
    assert scaler.running_return != 0.0

    scaler.scale(1.0, done=True)
    assert scaler.running_return == 0.0
    assert abs(scaler.scale(1e9, done=False)) <= scaler.clip


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------


def test_encode_observation_is_unchanged_when_off() -> None:
    """With the switch off, encoding is exactly the environment's encoder."""

    _, genome, environment = _ppo_setup()
    observation = np.arange(environment.n_observation_features, dtype=np.float32)

    assert torch.equal(
        encode_observation(genome, environment, observation, update=True),
        environment.encode(observation),
    )
    assert OBSERVATION_NORMALIZER_KEY not in genome.metadata


def test_encode_observation_updates_only_when_asked() -> None:
    """Training encodes update the statistics; evaluation encodes do not."""

    _, genome, environment = _ppo_setup(normalize_observations=True)
    observation = np.full(environment.n_observation_features, 4.0, np.float32)

    encode_observation(genome, environment, observation)
    frozen = genome._observation_normalizer.count
    encode_observation(genome, environment, observation, update=False)
    assert genome._observation_normalizer.count == frozen

    encode_observation(genome, environment, observation, update=True)
    assert genome._observation_normalizer.count == pytest.approx(frozen + 1)


def _improving_evaluations(trainer: Any) -> None:
    """Makes each of a trainer's evaluations score higher than the last.

    The test environment pays the same return every episode, so without this
    no evaluation would beat the pre-training one (episode 0) and training would
    keep the inherited weights -- and with them the pre-training statistics.

    Args:
        trainer: The trainer whose ``evaluate`` is wrapped, in place.

    Returns:
        None. Replaces ``trainer.evaluate``.
    """

    evaluate = trainer.evaluate
    calls = [0]

    def improving(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Calls the real evaluation and raises its return by the call count."""
        result = evaluate(*args, **kwargs)
        calls[0] += 1
        result["return_mean"] = float(result["return_mean"]) + calls[0]
        return result

    trainer.evaluate = improving


def test_training_saves_normalizer_and_evaluation_reuses_it() -> None:
    """Training writes the statistics to metadata, and a clone inherits them."""

    trainer, genome, environment = _ppo_setup(normalize_observations=True)
    _improving_evaluations(trainer)
    trainer.train(genome, environment)

    # a trained checkpoint was kept, so its statistics were collected in training
    assert genome.metadata["best_episode"] > 0
    saved = genome.metadata[OBSERVATION_NORMALIZER_KEY]
    assert len(saved["mean"]) == environment.n_observation_features
    assert saved["count"] > 1.0

    observation = np.full(environment.n_observation_features, 0.3, np.float32)
    expected = encode_observation(genome, environment, observation)

    child = genome.copy()
    reset_observation_normalizer(child)
    assert torch.allclose(encode_observation(child, environment, observation), expected)


def test_normalizer_is_restored_with_untrained_weights() -> None:
    """When training never beats the pre-training evaluation, its statistics go too.

    The weights restored are the inherited ones, so the statistics restored with
    them are the ones they were evaluated under, not those collected while
    training.
    """

    trainer, genome, environment = _ppo_setup(normalize_observations=True)
    # what the genome inherited (empty, or a parent's statistics)
    inherited = dict(genome.metadata.get(OBSERVATION_NORMALIZER_KEY) or {})
    trainer.train(genome, environment)

    assert genome.metadata["best_episode"] == 0
    restored = genome.metadata[OBSERVATION_NORMALIZER_KEY]
    # nothing collected while training survives: the count is the inherited one
    # (or a fresh normalizer's near-zero count when nothing was inherited)
    assert restored["count"] == inherited.get("count", restored["count"])
    assert restored["count"] < 1.0 or inherited


# ---------------------------------------------------------------------------
# Reward scaling
# ---------------------------------------------------------------------------


def test_ppo_reward_scaling_scales_targets_but_not_logged_returns() -> None:
    """Scaled rollouts change the stored rewards, not the reported returns."""

    trainer, genome, environment = _ppo_setup(ppo_reward_scaling=True)
    _, hp = prepare_single_update(trainer, genome)
    trainer.begin_training(hp)

    rollout = trainer._collect_rollout(genome, environment, 0, hp)

    # the test environment pays 1.0 per step
    assert not torch.allclose(rollout["rewards"], torch.ones_like(rollout["rewards"]))
    assert torch.all(rollout["rewards"] > 0)
    assert rollout["episode_returns"] == [float(hp.max_steps)] * len(
        rollout["episode_returns"]
    )


def test_begin_training_starts_a_fresh_reward_scaler() -> None:
    """Each training run gets its own scaler; with scaling off there is none."""

    trainer = PPOTrainer()
    hp = SimpleNamespace(ppo_reward_scaling=True, gamma=0.99)
    trainer.begin_training(hp)
    first = trainer._reward_scaler
    trainer.begin_training(hp)
    assert trainer._reward_scaler is not first

    trainer.begin_training(SimpleNamespace(ppo_reward_scaling=False, gamma=0.99))
    assert trainer._reward_scaler is None


# ---------------------------------------------------------------------------
# Learning-rate annealing
# ---------------------------------------------------------------------------


class _LearningRateRecorder(ReinforceTrainer):
    """A trainer whose update only records the optimizer's learning rate."""

    def __init__(self) -> None:
        """Starts with no recorded rates."""

        self.rates: list[float] = []

    def run_update(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        optimizer: torch.optim.Optimizer,
        episode_index: int,
        hp: SimpleNamespace,
    ) -> tuple[float, dict[str, float]]:
        """Records the current learning rate instead of training.

        Args:
            genome: Unused.
            environment: Unused.
            optimizer: The optimizer whose rate is recorded.
            episode_index: Unused.
            hp: Unused.

        Returns:
            ``(0.0, {})``.
        """

        self.rates.append(optimizer.param_groups[0]["lr"])
        return 0.0, {}


@pytest.mark.parametrize("schedule", ["none", "linear"])
def test_lr_anneal_schedules(schedule: str) -> None:
    """``linear`` decays the rate towards 0 over the episodes; ``none`` keeps it.

    Args:
        schedule: The ``lr_anneal`` value under test.
    """

    trainer = _LearningRateRecorder()
    genome, n_features = build_rl_genome(
        0, "pennylane", "deep", "linear", "linear", trainer, continuous=True
    )
    genome.hyperparameters.update(
        episodes=4, lr_anneal=schedule, learning_rate=0.1, log_every=100
    )
    trainer.train(genome, make_continuous_test_environment(n_features))

    if schedule == "linear":
        assert trainer.rates == pytest.approx([0.1, 0.075, 0.05, 0.025])
    else:
        assert trainer.rates == pytest.approx([0.1] * 4)


# ---------------------------------------------------------------------------
# KL early stopping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_kl", [0.0, 1e-12])
def test_ppo_target_kl_stops_epochs_early(target_kl: float) -> None:
    """A tiny KL target cuts the epochs short; 0 runs every minibatch.

    Args:
        target_kl: The ``ppo_target_kl`` value under test.
    """

    trainer, genome, environment = _ppo_setup(
        ppo_target_kl=target_kl, ppo_passes=3, ppo_minibatch=4, rollout_steps=10
    )
    torch.manual_seed(0)
    optimizer, hp = prepare_single_update(trainer, genome)
    _, info = trainer.run_update(genome, environment, optimizer, 0, hp)

    full = hp.ppo_passes * math.ceil(info["rollout_transitions"] / hp.ppo_minibatch)
    if target_kl == 0.0:
        assert info["ppo_epochs"] == full
    else:
        assert info["ppo_epochs"] <= 1
    assert info["approx_kl"] >= 0.0


# ---------------------------------------------------------------------------
# Log-std clamp
# ---------------------------------------------------------------------------


def test_action_distribution_uses_given_log_std_bounds() -> None:
    """The continuous policy's scale is clamped to the bounds passed in."""

    environment = make_continuous_test_environment(4)
    part = torch.tensor([0.0, 0.0, 10.0, -10.0])

    default = action_distribution(part, environment)
    assert torch.allclose(default.scale, torch.tensor([LOG_STD_MAX, LOG_STD_MIN]).exp())

    tight = action_distribution(part, environment, -2.0, 0.5)
    assert torch.allclose(tight.scale, torch.tensor([0.5, -2.0]).exp())


def test_log_std_bounds_reads_genome_and_rejects_inverted() -> None:
    """Bounds come from the genome's hyperparameters, and must be ordered."""

    _, genome, _ = _ppo_setup()
    assert log_std_bounds(genome) == (LOG_STD_MIN, LOG_STD_MAX)

    genome.hyperparameters.update(log_std_min=-5.0, log_std_max=0.5)
    assert log_std_bounds(genome) == (-5.0, 0.5)

    genome.hyperparameters["log_std_max"] = -6.0
    with pytest.raises(ValueError):
        log_std_bounds(genome)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_flag_defaults_match_hyperparameter_defaults() -> None:
    """Every new flag defaults to the value the trainers fall back to."""

    parser = argparse.ArgumentParser()
    ReinforcementLearningTrainer.initialize_parser(parser)
    PPOTrainer.initialize_parser(parser)
    args = parser.parse_args([])

    for name in (
        "normalize_observations",
        "lr_anneal",
        "log_std_min",
        "log_std_max",
        "ppo_target_kl",
        "ppo_reward_scaling",
    ):
        assert getattr(args, name) == RL_HYPERPARAMETER_DEFAULTS[name], name

    args = parser.parse_args(
        [
            "--normalize_observations",
            "--ppo_reward_scaling",
            "--lr_anneal",
            "linear",
            "--ppo_target_kl",
            "0.015",
            "--log_std_max",
            "0.5",
        ]
    )
    assert args.normalize_observations and args.ppo_reward_scaling
    assert (args.lr_anneal, args.ppo_target_kl, args.log_std_max) == (
        "linear",
        0.015,
        0.5,
    )
