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

import argparse
import dataclasses
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

import pytest

from src.circuits.circuit import CircuitGenome
from src.examples.reinforcement_learning import (
    CONTINUOUS_ENVS,
    ENV_IDS,
    MUJOCO_ENV_FLAGS,
    MUJOCO_ENV_KNOBS,
    environment_knob_kwargs,
    supported_env_knobs,
    ReinforcementLearningObjective,
    build_parser,
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
    # recorded the same way as supervised genomes: what the optimizer updated
    assert (
        genome.metadata["n_trainable_parameters"] == genome.count_trainable_parameters()
    )
    assert genome.metadata["n_trainable_parameters"] > 0
    # the deterministic env yields a constant +1 per step, so returns are >= 0
    assert genome.metadata["best_validation_metrics"]["return_mean"] >= 0.0


def test_genomes_train_on_random_seeds_unless_one_is_fixed() -> None:
    """Without a fixed seed each genome draws its own; a fixed seed is kept.

    Every genome training on the same seeded episodes would let the search
    select for those particular episodes, so seeds are random by default.
    """

    trainer = build_trainer("reinforce")
    seeds = []
    for genome_number, seed in ((1, None), (2, None), (3, 1234)):
        genome, observation_features = build_rl_genome(
            genome_number=genome_number,
            target="pennylane",
            complexity="shallow",
            encoder_name="linear",
            decoder_name="linear",
            trainer=trainer,
        )
        genome.hyperparameters["seed"] = seed
        trainer.train(genome, make_test_environment(observation_features))
        seeds.append(genome.metadata["training_seed"])
        # the drawn seed is recorded, not written back into the inherited hyperparameters
        assert genome.hyperparameters["seed"] == seed

    assert all(isinstance(seed, int) for seed in seeds)
    assert seeds[0] != seeds[1]
    assert seeds[2] == 1234


@pytest.mark.parametrize("trainer_name", ["reinforce", "actor_critic", "ppo"])
def test_evaluation_seeds_never_overlap_training_seeds(trainer_name: str) -> None:
    """A genome is never scored on an episode it trained on.

    Training and evaluation seeds were once derived from the same base, and
    PPO's ``seed + episode_index * SEED_BLOCK + episode`` put outer episode 1
    exactly on the evaluation seeds. The two ranges are now kept disjoint by
    construction, whatever the trainer.

    Args:
        trainer_name: The RL algorithm whose seed range is checked.
    """

    trainer = build_trainer(trainer_name)
    for _ in range(25):
        genome = SimpleNamespace(
            hyperparameters={
                "seed": None,
                "eval_seed": None,
                "episodes": 100,
                "rollout_steps": 2048,
                "eval_episodes": 10,
            }
        )
        hp = trainer.resolve_hyperparameters(genome)

        first, last = trainer.training_seed_span(hp)
        evaluation = range(hp.eval_seed, hp.eval_seed + hp.eval_episodes)
        assert not (evaluation.start < last and first < evaluation.stop)


def test_eval_seed_is_random_per_genome_unless_pinned() -> None:
    """Unpinned evaluation seeds differ per genome; a pinned one is shared.

    Leaving ``--eval_seed`` unset scores each genome on its own episodes.
    Pinning it evaluates every genome on the *same* episodes, so their
    fitnesses become directly comparable; the training seed moves out of the
    way instead, which is free because it was drawn at random anyway.
    """

    trainer = build_trainer("ppo")

    def resolve(eval_seed: int | None) -> SimpleNamespace:
        """Resolves hyperparameters for a fresh genome.

        Args:
            eval_seed: The ``eval_seed`` hyperparameter to resolve with.

        Returns:
            The resolved hyperparameters.
        """

        return trainer.resolve_hyperparameters(
            SimpleNamespace(
                hyperparameters={
                    "seed": None,
                    "eval_seed": eval_seed,
                    "episodes": 20,
                    "eval_episodes": 5,
                }
            )
        )

    drawn = {resolve(None).eval_seed for _ in range(20)}
    assert len(drawn) == 20

    pinned = [resolve(4242) for _ in range(20)]
    assert {hp.eval_seed for hp in pinned} == {4242}
    # the per-genome training seeds are still independent of one another
    assert len({hp.seed for hp in pinned}) == 20


def test_pinned_seed_and_eval_seed_that_overlap_are_rejected() -> None:
    """Pinning both seeds into the same range raises rather than silently leaking.

    Neither seed can be moved without breaking what was asked for, so this is
    a configuration error the caller has to resolve.
    """

    trainer = build_trainer("ppo")
    genome = SimpleNamespace(
        hyperparameters={
            "seed": 777,
            "eval_seed": 777,
            "episodes": 20,
            "eval_episodes": 5,
        }
    )

    with pytest.raises(ValueError, match="overlaps the training seeds"):
        trainer.resolve_hyperparameters(genome)


@pytest.mark.parametrize("bias", [0.1, 0.25])
def test_fitness_loss_weights_training_return_by_the_bias(bias: float) -> None:
    """The bias weights the training return and the evaluation return gets the rest.

    Args:
        bias: The ``train_vs_validation_bias`` to score with.
    """

    class FixedReturnsTrainer:
        """Stands in for a trainer, recording fixed training and evaluation returns."""

        def train(self, genome: SimpleNamespace, environment: SimpleNamespace) -> None:
            """Records a training return of 100 and an evaluation return of 300.

            Args:
                genome: The genome whose metadata is set.
                environment: Unused.
            """

            genome.metadata["best_training_metrics"] = {
                "return_mean": 100.0,
                "best_episode_return": 120.0,
            }
            genome.metadata["best_validation_metrics"] = {
                "return_mean": 300.0,
                "return_std": 5.0,
            }

    genome = SimpleNamespace(genome_number=1, metadata={}, fitness=None)
    objective = ReinforcementLearningObjective(
        environment=SimpleNamespace(env_id="Fake-v0"),
        trainer=FixedReturnsTrainer(),
        train_vs_validation_bias=bias,
    )

    objective(genome)

    assert genome.fitness["loss"] == pytest.approx(-(bias * 100.0 + (1 - bias) * 300.0))
    assert genome.fitness["train_return_mean"] == 100.0
    assert genome.fitness["eval_return_mean"] == 300.0


def test_parser_defaults_weight_evaluation_and_draw_random_seeds() -> None:
    """The documented defaults: a 0.1 training-return weight and random seeds."""

    defaults = {action.dest: action.default for action in build_parser()._actions}

    assert defaults["train_vs_validation_bias"] == 0.1
    assert defaults["seed"] is None
    assert ReinforcementLearningObjective.__init__.__defaults__ == (0.1,)


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
    assert genome.metadata["n_trainable_parameters"] == 0
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
@pytest.mark.parametrize("eval_policy", ["greedy", "stochastic"])
def test_evaluate_collapses_to_one_episode_only_when_greedy_and_deterministic(
    eval_policy: str,
    deterministic: bool,
) -> None:
    """``evaluate`` rolls one episode only for a *greedy* deterministic env.

    Greedy evaluation of a deterministic environment yields identical episodes,
    so only one is run. That reasoning does not extend to a stochastic policy:
    its own action sampling makes every episode differ even when the
    environment is deterministic, so the full ``eval_episodes`` count must
    still be rolled. Every other combination runs ``eval_episodes``.

    Args:
        eval_policy: The action-selection regime to evaluate under.
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
    genome.hyperparameters["eval_policy"] = eval_policy
    hp = trainer.resolve_hyperparameters(genome)
    assert hp.eval_episodes > 1  # so the two cases differ

    environment = dataclasses.replace(
        make_test_environment(observation_features), deterministic=deterministic
    )

    # count how many episodes evaluate() actually rolls (one env.make per episode)
    episode_count = 0
    original_make = environment.make

    def counting_make() -> Any:
        """Counts one rolled episode and builds the environment as usual.

        Returns:
            The environment the wrapped :meth:`RLEnvironment.make` returns.
        """

        nonlocal episode_count
        episode_count += 1
        return original_make()

    environment.make = counting_make
    trainer.evaluate(genome, environment, hp)

    collapses = eval_policy == "greedy" and deterministic
    assert episode_count == (1 if collapses else hp.eval_episodes)


def test_evaluate_match_resolves_per_trainer() -> None:
    """``eval_policy="match"`` picks each algorithm's own objective's regime.

    The policy-gradient trainers optimize the sampled policy, while the
    value-based trainers learn a greedy target policy, so ``"match"`` must not
    resolve to the same regime for both.
    """

    assert build_trainer("reinforce").natural_eval_policy == "stochastic"
    assert build_trainer("actor_critic").natural_eval_policy == "stochastic"
    assert build_trainer("ppo").natural_eval_policy == "stochastic"
    assert build_trainer("q_learning").natural_eval_policy == "greedy"
    assert build_trainer("sarsa").natural_eval_policy == "greedy"

    for algo, expected in (("ppo", "stochastic"), ("q_learning", "greedy")):
        trainer = build_trainer(algo)
        hp = SimpleNamespace(eval_policy="match")
        assert trainer.resolve_eval_policy(hp) == expected
        assert trainer.selected_eval_policy(hp) == expected

        # "both" scores under two regimes but reports the natural one as fitness
        hp = SimpleNamespace(eval_policy="both")
        assert trainer.resolve_eval_policy(hp) == "both"
        assert trainer.selected_eval_policy(hp) == expected

    with pytest.raises(ValueError, match="Unknown eval_policy"):
        build_trainer("ppo").resolve_eval_policy(SimpleNamespace(eval_policy="nope"))


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


def test_supported_env_knobs_is_read_from_each_environment() -> None:
    """Which reward knobs exist differs per environment, and is introspected.

    HalfCheetah cannot terminate, so it has no health concept and no
    ``healthy_reward``; only Ant and Humanoid have a contact cost. Reading this
    off each constructor rather than hardcoding it keeps it correct across
    Gymnasium versions.
    """

    walker = supported_env_knobs(ENV_IDS["walker2d"])
    assert {"forward_reward_weight", "ctrl_cost_weight", "healthy_reward"} <= walker
    assert "contact_cost_weight" not in walker

    cheetah = supported_env_knobs(ENV_IDS["halfcheetah"])
    assert "forward_reward_weight" in cheetah
    assert "healthy_reward" not in cheetah
    assert "terminate_when_unhealthy" not in cheetah

    assert "contact_cost_weight" in supported_env_knobs(ENV_IDS["ant"])
    assert "contact_cost_weight" in supported_env_knobs(ENV_IDS["humanoid"])

    # the classic-control tasks have no reward knobs at all
    assert supported_env_knobs(ENV_IDS["cartpole"]) == frozenset()
    assert supported_env_knobs(ENV_IDS["pendulum"]) == frozenset()


def _knob_namespace(**overrides: float | bool | None) -> argparse.Namespace:
    """Builds a parsed-argument namespace with every knob unset but the given ones.

    Args:
        **overrides: Knob values to set; everything else defaults to ``None``,
            meaning "leave the environment's own value alone".

    Returns:
        A namespace shaped like the entry points' parsed arguments.
    """

    values: dict[str, float | bool | None] = {
        name: None for name in (*MUJOCO_ENV_KNOBS, *MUJOCO_ENV_FLAGS)
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_environment_knob_kwargs_collects_only_what_was_set() -> None:
    """Unset knobs are not forwarded, so environment defaults stay authoritative."""

    assert environment_knob_kwargs(_knob_namespace(), "walker2d") == {}
    assert environment_knob_kwargs(_knob_namespace(healthy_reward=0.1), "walker2d") == {
        "healthy_reward": 0.1
    }
    assert environment_knob_kwargs(
        _knob_namespace(terminate_when_unhealthy=False), "walker2d"
    ) == {"terminate_when_unhealthy": False}


@pytest.mark.parametrize(
    "env_name,knob",
    [
        ("halfcheetah", "healthy_reward"),
        ("walker2d", "contact_cost_weight"),
        ("cartpole", "healthy_reward"),
    ],
)
def test_environment_knob_kwargs_rejects_an_unsupported_knob(
    env_name: str, knob: str
) -> None:
    """Asking for a knob an environment lacks fails rather than being ignored.

    Args:
        env_name: The environment the knob is requested for.
        knob: A knob that environment does not accept.
    """

    with pytest.raises(ValueError, match=f"does not accept --{knob}"):
        environment_knob_kwargs(_knob_namespace(**{knob: 0.1}), env_name)


def test_make_environment_rejects_knobs_for_a_non_mujoco_environment() -> None:
    """A discrete task takes no reward knobs, and says so rather than dropping them."""

    with pytest.raises(ValueError, match="takes no reward knobs"):
        make_environment("cartpole", env_kwargs={"healthy_reward": 0.1})


def test_environment_knobs_reach_the_built_environment() -> None:
    """A knob passed through ``make_environment`` changes the actual physics.

    Skips when the optional ``mujoco`` dependency is unavailable.
    """

    pytest.importorskip("mujoco")

    default = make_environment("walker2d").make().unwrapped
    assert default.healthy_reward == 1.0

    tuned = (
        make_environment(
            "walker2d",
            env_kwargs={"healthy_reward": 0.1, "forward_reward_weight": 5.0},
        )
        .make()
        .unwrapped
    )
    assert tuned.healthy_reward == 0.1
    assert tuned._forward_reward_weight == 5.0
