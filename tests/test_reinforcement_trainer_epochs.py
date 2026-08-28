"""End-to-end ``ReinforcementLearningTrainer.train`` tests over one+ episodes.

These tests drive the full training loop (a fixed number of training
episodes) of each RL algorithm on the deterministic test environment and
check the trainer's bookkeeping:

* per-episode metrics are recorded in ``genome.metadata``;
* ``best_training_metrics`` / ``best_validation_metrics`` are populated with
  finite returns; and
* the run completes for every target / trainer / encoder / decoder
  combination.

They complement ``test_reinforcement_trainer_gradients.py`` (which verifies
gradient flow through the three genome stages) by covering the outer loop,
the evaluation/best-snapshot bookkeeping, and the encoder/decoder variety.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

import pytest

from src.circuits.circuit import CircuitGenome
from src.examples.reinforcement_learning import (
    CONTINUOUS_ENVS,
    ENV_IDS,
    make_environment,
)

from tests.reinforcement_trainer_test_utils import (
    CONTINUOUS_TRAINER_NAMES,
    ENCODER_DECODER_PAIRS,
    TRAINER_NAMES,
    build_rl_genome,
    build_trainer,
    make_continuous_test_environment,
    make_test_environment,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")

#: Continuous ``--env`` names whose Gymnasium ids are MuJoCo tasks (everything
#: except Pendulum, which is classic control). Instantiating these requires the
#: optional ``mujoco`` dependency, so the spec test skips them when it is
#: unavailable. Sorted for a deterministic parametrization order.
_MUJOCO_ENV_NAMES: tuple[str, ...] = tuple(
    sorted(name for name in CONTINUOUS_ENVS if name != "pendulum")
)


def _assert_return_metrics(metrics: dict[str, float]) -> None:
    """Asserts a return-metrics dict has finite ``return_mean`` and best return.

    Args:
        metrics: A ``best_training_metrics`` or ``best_validation_metrics``
            dict recorded by the trainer.
    """

    assert "return_mean" in metrics
    assert math.isfinite(metrics["return_mean"])
    assert "best_episode_return" in metrics
    assert math.isfinite(metrics["best_episode_return"])


@pytest.mark.parametrize("trainer_name", TRAINER_NAMES)
@pytest.mark.parametrize("target", TARGETS)
def test_train_records_per_episode_and_best_metrics(
    target: str, trainer_name: str
) -> None:
    """Training records per-episode metrics and finite best-metric summaries.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        trainer_name: The RL algorithm to exercise.
    """

    trainer = build_trainer(trainer_name)
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    episode_metrics = genome.metadata["training_episode_metrics"]
    assert len(episode_metrics) == genome.hyperparameters["episodes"]
    for entry in episode_metrics:
        assert "episode" in entry
        assert "return" in entry
        assert math.isfinite(entry["return"])

    _assert_return_metrics(genome.metadata["best_training_metrics"])
    _assert_return_metrics(genome.metadata["best_validation_metrics"])
    # the deterministic env yields a constant +1 per step, so returns are >= 0
    assert genome.metadata["best_validation_metrics"]["return_mean"] >= 0.0


@pytest.mark.parametrize("encoder_name,decoder_name", ENCODER_DECODER_PAIRS)
@pytest.mark.parametrize("target", TARGETS)
def test_train_runs_across_encoder_decoder_combinations(
    target: str, encoder_name: str, decoder_name: str
) -> None:
    """Training completes for both trainable and stateless coder pairs.

    Uses REINFORCE (which needs no value output) so the ``identity``/
    ``clipped`` pair -- where the decoder has no trainable parameters -- is a
    valid configuration.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        encoder_name: Either ``"identity"`` or ``"linear"``.
        decoder_name: Either ``"clipped"`` or ``"linear"``.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=2,
        target=target,
        complexity="shallow",
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    assert (
        len(genome.metadata["training_episode_metrics"])
        == genome.hyperparameters["episodes"]
    )
    _assert_return_metrics(genome.metadata["best_validation_metrics"])


@pytest.mark.parametrize("target", TARGETS)
def test_train_with_no_trainable_parameters_only_evaluates(target: str) -> None:
    """A parameter-free genome is evaluated rather than trained.

    With an ``IdentityEncoder``, a ``ClippedDecoder``, and no parametric
    gates, the genome's hybrid model has zero trainable parameters, so the
    trainer should take its evaluation-only path: no per-episode training
    metrics, but best-metric summaries still recorded.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=3,
        target=target,
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        trainer=trainer,
        include_parametric=False,
    )
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    assert genome.metadata["training_episode_metrics"] == []
    _assert_return_metrics(genome.metadata["best_training_metrics"])
    _assert_return_metrics(genome.metadata["best_validation_metrics"])


@pytest.mark.parametrize("trainer_name", TRAINER_NAMES)
def test_train_respects_episode_count_from_hyperparameters(trainer_name: str) -> None:
    """The number of recorded training episodes matches the hyperparameter.

    Args:
        trainer_name: The RL algorithm to exercise.
    """

    trainer = build_trainer(trainer_name)
    genome, observation_features = build_rl_genome(
        genome_number=4,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    genome.hyperparameters["episodes"] = 3
    environment = make_test_environment(observation_features)

    trainer.train(genome, environment)

    assert isinstance(genome, CircuitGenome)
    assert len(genome.metadata["training_episode_metrics"]) == 3


def test_training_return_mean_is_exponential_moving_average(monkeypatch) -> None:
    """``best_training_metrics['return_mean']`` is an EMA of episode returns.

    Drives ``train`` with a scripted sequence of episode returns (by patching
    ``run_update``) and asserts the reported training return mean equals the
    exponential moving average ``ema = alpha * return + (1 - alpha) * ema``
    (seeded with the first return), not a plain arithmetic mean. ``alpha`` is
    read from the genome's ``ema_alpha`` hyperparameter. ``best_episode_return``
    must still be the maximum raw return.

    Args:
        monkeypatch: pytest fixture used to script ``run_update``'s returns.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=7,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    scripted_returns = [10.0, 0.0, 4.0, 8.0]
    alpha = 0.5
    genome.hyperparameters["episodes"] = len(scripted_returns)
    genome.hyperparameters["ema_alpha"] = alpha
    environment = make_test_environment(observation_features)

    remaining = list(scripted_returns)

    def scripted_update(genome_, environment_, optimizer_, episode_index_, hp_):
        return remaining.pop(0), {}

    monkeypatch.setattr(trainer, "run_update", scripted_update)

    trainer.train(genome, environment)

    expected_ema = scripted_returns[0]
    for value in scripted_returns[1:]:
        expected_ema = alpha * value + (1.0 - alpha) * expected_ema

    metrics = genome.metadata["best_training_metrics"]
    assert metrics["return_mean"] == pytest.approx(expected_ema)
    # distinct from a plain mean, so this genuinely tests the EMA
    assert metrics["return_mean"] != pytest.approx(float(np.mean(scripted_returns)))
    assert metrics["best_episode_return"] == max(scripted_returns)


def test_frozenlake_is_flagged_deterministic_only_when_not_slippery() -> None:
    """FrozenLake is deterministic unless slippery; other envs are stochastic."""

    assert make_environment("frozenlake").deterministic is True
    assert make_environment("frozenlake", is_slippery=True).deterministic is False
    assert make_environment("cartpole").deterministic is False


@pytest.mark.parametrize("deterministic", [False, True])
def test_evaluate_runs_single_episode_for_deterministic_environment(
    deterministic: bool,
) -> None:
    """``evaluate`` runs one episode for a deterministic env, else eval_episodes.

    Greedy evaluation of a deterministic environment yields identical episodes,
    so only one is run; a stochastic environment runs the full
    ``eval_episodes`` count.

    Args:
        deterministic: Whether the environment is marked deterministic.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    genome.initialize_model()
    hp = trainer.resolve_hyperparameters(genome)
    assert hp.eval_episodes > 1  # so the two cases differ

    environment = dataclasses.replace(
        make_test_environment(observation_features), deterministic=deterministic
    )

    # count how many episodes evaluate() actually rolls (one env.make per episode)
    episode_count = 0
    original_make = environment.make

    def counting_make(*args, **kwargs):
        nonlocal episode_count
        episode_count += 1
        return original_make(*args, **kwargs)

    environment.make = counting_make
    trainer.evaluate(genome, environment, hp)

    assert episode_count == (1 if deterministic else hp.eval_episodes)


# ---------------------------------------------------------------------------
# Continuous (Box) action spaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trainer_name", CONTINUOUS_TRAINER_NAMES)
@pytest.mark.parametrize("target", TARGETS)
def test_train_records_metrics_on_continuous_environment(
    target: str, trainer_name: str
) -> None:
    """The policy-gradient trainers train end to end on a continuous env.

    Drives the full :meth:`train` loop against the ``Box``-action test
    environment (Gaussian policy) and checks the same bookkeeping the discrete
    case checks: per-episode metrics for every episode plus finite best-metric
    summaries. Only policy-gradient trainers are exercised (value-based methods
    reject continuous spaces; see
    :func:`test_value_based_trainer_rejects_continuous_environment`).

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        trainer_name: The (policy-gradient) RL algorithm to exercise.
    """

    trainer = build_trainer(trainer_name)
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
        continuous=True,
    )
    environment = make_continuous_test_environment(observation_features)

    trainer.train(genome, environment)

    episode_metrics = genome.metadata["training_episode_metrics"]
    assert len(episode_metrics) == genome.hyperparameters["episodes"]
    for entry in episode_metrics:
        assert "episode" in entry
        assert math.isfinite(entry["return"])

    _assert_return_metrics(genome.metadata["best_training_metrics"])
    _assert_return_metrics(genome.metadata["best_validation_metrics"])


@pytest.mark.parametrize("trainer_name", ["q_learning", "sarsa"])
def test_value_based_trainer_rejects_continuous_environment(
    trainer_name: str,
) -> None:
    """Value-based trainers raise a clear error on a continuous environment.

    Q-learning / SARSA select actions by argmax / epsilon-greedy over
    enumerable action values, so they cannot drive a continuous ``Box`` action
    space; :meth:`train` must fail fast with a descriptive ``ValueError``.

    Args:
        trainer_name: The value-based algorithm to exercise.
    """

    trainer = build_trainer(trainer_name)
    assert trainer.supports_continuous is False

    genome, observation_features = build_rl_genome(
        genome_number=1,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
        continuous=True,
    )
    environment = make_continuous_test_environment(observation_features)

    with pytest.raises(ValueError, match="continuous"):
        trainer.train(genome, environment)


def test_make_environment_builds_continuous_pendulum() -> None:
    """``make_environment('pendulum')`` yields a correct continuous spec.

    Pendulum is classic control (no MuJoCo needed), so its dimensions and
    action bounds can always be checked: a 3-dim observation, a single
    continuous action in ``[-2, 2]``, ``continuous=True``, and two policy
    outputs (a mean and a log-std for the one action dimension).
    """

    environment = make_environment("pendulum")

    assert environment.env_id == "Pendulum-v1"
    assert environment.continuous is True
    assert environment.n_observation_features == 3
    assert environment.n_actions == 1
    # a mean and a log-std per action dimension
    assert environment.n_policy_outputs == 2
    assert environment.action_low is not None and environment.action_high is not None
    assert environment.action_low.shape == (1,)
    assert np.allclose(environment.action_low, -2.0)
    assert np.allclose(environment.action_high, 2.0)


@pytest.mark.parametrize("env_name", _MUJOCO_ENV_NAMES)
def test_make_environment_builds_continuous_mujoco(env_name: str) -> None:
    """Each MuJoCo ``--env`` builds a continuous spec probed from the real env.

    Skips when the optional ``mujoco`` dependency is unavailable. Rather than
    hardcoding the (version-dependent) observation/action sizes, this asserts
    the spec is internally consistent with the environment Gymnasium actually
    constructs: matching observation and action dimensions, a value/high action
    bound per dimension, and ``n_policy_outputs == 2 * n_actions``.

    Args:
        env_name: A MuJoCo environment name from :data:`CONTINUOUS_ENVS`.
    """

    pytest.importorskip("mujoco")
    import gymnasium as gym

    environment = make_environment(env_name)
    assert environment.continuous is True
    assert environment.env_id == ENV_IDS[env_name]

    probe = gym.make(environment.env_id)
    try:
        expected_obs = int(np.prod(probe.observation_space.shape))
        expected_action_dim = int(np.prod(probe.action_space.shape))
    finally:
        probe.close()

    assert environment.n_observation_features == expected_obs
    assert environment.n_actions == expected_action_dim
    assert environment.n_policy_outputs == 2 * expected_action_dim
    assert environment.action_low.shape == (expected_action_dim,)
    assert environment.action_high.shape == (expected_action_dim,)
