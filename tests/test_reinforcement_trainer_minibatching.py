"""Tests for the batched (minibatched) forward passes of the RL trainers.

REINFORCE and actor-critic evaluate a whole episode, and PPO each minibatch,
in one batched ``genome.forward`` call rather than one sample at a time. These
tests check that batching changes nothing but speed:

* the distribution helpers give the same values on a batch as on each sample
  stacked together -- in particular a batch of ``Categorical``
  log-probabilities keeps its batch dimension, while a ``Normal``'s action
  dimension is summed;
* ``genome.forward`` on a stacked batch matches per-sample forwards;
* the loss and every parameter's gradient of a batched update match a
  per-sample reference built in this module the way the trainers did before
  batching (one forward pass, log-probability and entropy per step); and
* a PPO update whose last minibatch holds a single transition still runs.

The references replay the trainer's rollout by reseeding torch's global RNG,
so the reference and the trainer sample the same actions from the same
weights on the deterministic test environments.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from torch import Tensor
from torch.distributions import Categorical, Normal

from src.circuits.circuit import CircuitGenome
from src.trainer.ppo_trainer import PPOTrainer
from src.trainer.reinforcement_trainer import (
    RLEnvironment,
    ReinforcementLearningTrainer,
    _normalize,
    action_distribution,
    discounted_returns,
    distribution_entropy,
    distribution_log_prob,
    gae_advantages,
    split_policy_value,
    to_env_action,
)

from tests.reinforcement_trainer_test_utils import (
    CONTINUOUS_TRAINER_NAMES,
    DEFAULT_OBSERVATION_FEATURES,
    build_classification_genome,
    build_trainer,
    make_continuous_test_environment,
    make_test_environment,
    prepare_single_update,
    rl_hyperparameters,
)

#: Quantum readout modes exercised (pennylane implements both).
OUTPUT_MODES: tuple[str, ...] = ("expval", "probs")

#: Episode length used by the trainer-equivalence tests.
MAX_STEPS: int = 20

#: Relative / absolute tolerances for batched-vs-per-sample comparisons.
#: Batching only reorders float32 sums, so the two agree to rounding.
RTOL: float = 1e-5
ATOL: float = 1e-6

#: Seed for torch's global RNG, reset before the reference and the trainer so
#: both sample the same actions.
TORCH_SEED: int = 1234


def _make_environment(continuous: bool, max_steps: int) -> RLEnvironment:
    """Builds the discrete or continuous deterministic test environment.

    Args:
        continuous: Whether to build the ``Box``-action environment.
        max_steps: Episode length.

    Returns:
        The configured :class:`RLEnvironment`.
    """

    if continuous:
        return make_continuous_test_environment(
            DEFAULT_OBSERVATION_FEATURES, max_steps=max_steps
        )
    return make_test_environment(DEFAULT_OBSERVATION_FEATURES, max_steps=max_steps)


def _build_genome(
    trainer: ReinforcementLearningTrainer,
    environment: RLEnvironment,
    output_mode: str,
    target: str = "pennylane",
    **hyperparameters: Any,
) -> CircuitGenome:
    """Builds a small RL genome with a chosen quantum readout mode.

    Mirrors :func:`tests.reinforcement_trainer_test_utils.build_rl_genome`, but
    passes the quantum output mode through so the decoder is sized for it
    (``expval`` reads one value per qubit, ``probs`` one per basis state).

    Args:
        trainer: The trainer that will consume the genome (its
            ``n_value_outputs`` sets the extra decoder outputs).
        environment: The environment the genome acts in (sizes the policy).
        output_mode: The genome's ``quantum_output_mode``.
        target: Either ``"pennylane"`` or ``"qiskit"``.
        **hyperparameters: Overrides merged on top of the tiny RL config.

    Returns:
        The genome, not yet initialized.
    """

    genome, _ = build_classification_genome(
        genome_number=0,
        target=target,
        complexity="deep",
        encoder_name="linear",
        decoder_name="linear",
        n_classes=environment.n_policy_outputs + trainer.n_value_outputs,
        n_features=DEFAULT_OBSERVATION_FEATURES,
        quantum_output_mode=output_mode,
    )
    genome.hyperparameters.update(rl_hyperparameters())
    genome.hyperparameters.update({"max_steps": MAX_STEPS, **hyperparameters})
    return genome


def _gradients(genome: CircuitGenome) -> list[Tensor | None]:
    """Clones every parameter's current gradient.

    Args:
        genome: The genome whose parameters are read.

    Returns:
        One detached clone per parameter, or ``None`` for one with no gradient.
    """

    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in genome.parameters()
    ]


def _assert_gradients_close(
    actual: list[Tensor | None], expected: list[Tensor | None]
) -> None:
    """Asserts two gradient lists match parameter by parameter.

    Args:
        actual: Gradients from the batched trainer update.
        expected: Gradients from the per-sample reference.

    Raises:
        AssertionError: If the lists differ in length, a gradient is present in
            one but not the other, or any gradient differs beyond tolerance.
    """

    assert len(actual) == len(expected)
    assert any(grad is not None for grad in expected), "reference had no gradients"
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert (got is None) == (want is None), f"gradient presence differs at {index}"
        if want is not None:
            torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def _loss_terms(
    log_probs: Tensor,
    entropies: Tensor,
    advantages: Tensor,
    hp: SimpleNamespace,
) -> Tensor:
    """Builds the shared policy-gradient plus entropy loss.

    Args:
        log_probs: Per-step log-probabilities of the taken actions.
        entropies: Per-step policy entropies.
        advantages: Per-step advantages (detached here).
        hp: Resolved hyperparameters (reads ``entropy_coef``).

    Returns:
        ``-(log_probs * advantages).mean()`` plus the entropy bonus.
    """

    loss = -(log_probs * advantages.detach()).mean()
    if hp.entropy_coef > 0:
        loss = loss - hp.entropy_coef * entropies.mean()
    return loss


def _per_sample_episode(
    genome: CircuitGenome,
    environment: RLEnvironment,
    hp: SimpleNamespace,
    with_value: bool,
) -> tuple[list[float], Tensor, Tensor, Tensor | None]:
    """Rolls one episode evaluating the policy one step at a time, with gradients.

    This is how REINFORCE and actor-critic built their loss before batching:
    each step's forward pass, log-probability, entropy and value are tracked
    as the episode is rolled.

    Args:
        genome: The genome being trained.
        environment: The environment to roll out in.
        hp: Resolved hyperparameters.
        with_value: Whether the genome carries a trailing value output.

    Returns:
        A tuple ``(rewards, log_probs, entropies, values)``; ``values`` is
        ``None`` when ``with_value`` is false.
    """

    env = environment.make()
    observation, _ = env.reset(seed=hp.seed)
    rewards: list[float] = []
    log_probs: list[Tensor] = []
    entropies: list[Tensor] = []
    values: list[Tensor] = []

    for _ in range(hp.max_steps):
        output = genome.forward(environment.encode(observation))
        if with_value:
            part, value = split_policy_value(output, environment)
            values.append(value)
        else:
            part = output[: environment.n_policy_outputs]
        distribution = action_distribution(part, environment)
        action = distribution.sample()
        log_probs.append(distribution_log_prob(distribution, action))
        entropies.append(distribution_entropy(distribution))

        observation, reward, terminated, truncated, _ = env.step(
            to_env_action(action, environment)
        )
        rewards.append(float(reward))
        if terminated or truncated:
            break

    env.close()
    return (
        rewards,
        torch.stack(log_probs),
        torch.stack(entropies),
        torch.stack(values) if with_value else None,
    )


def _reinforce_reference(
    genome: CircuitGenome, environment: RLEnvironment, hp: SimpleNamespace
) -> Tensor:
    """Builds the REINFORCE loss one step at a time.

    Args:
        genome: The genome being trained.
        environment: The environment to roll out in.
        hp: Resolved hyperparameters.

    Returns:
        The scalar REINFORCE loss.
    """

    rewards, log_probs, entropies, _ = _per_sample_episode(
        genome, environment, hp, with_value=False
    )
    returns = discounted_returns(rewards, hp.gamma)
    advantages = returns - returns.mean() if hp.baseline == "mean" else returns
    return _loss_terms(log_probs, entropies, advantages, hp)


def _actor_critic_reference(
    genome: CircuitGenome, environment: RLEnvironment, hp: SimpleNamespace
) -> Tensor:
    """Builds the actor-critic loss one step at a time.

    Args:
        genome: The genome being trained.
        environment: The environment to roll out in.
        hp: Resolved hyperparameters.

    Returns:
        The scalar actor-critic loss.
    """

    rewards, log_probs, entropies, values = _per_sample_episode(
        genome, environment, hp, with_value=True
    )
    assert values is not None
    returns = discounted_returns(rewards, hp.gamma)
    policy_and_entropy = _loss_terms(
        log_probs, entropies, returns - values.detach(), hp
    )
    value_loss = 0.5 * (returns - values).pow(2).mean()
    return policy_and_entropy + hp.value_coef * value_loss


def _ppo_reference(
    genome: CircuitGenome, environment: RLEnvironment, hp: SimpleNamespace
) -> Tensor:
    """Builds the loss of PPO's first minibatch one transition at a time.

    Collects the same rollout the trainer does (``_collect_rollout`` is
    unchanged by batching) and draws the same minibatch order, then evaluates
    the minibatch per transition.

    Args:
        genome: The genome being trained.
        environment: The environment to roll out in.
        hp: Resolved hyperparameters.

    Returns:
        The scalar clipped-surrogate loss of the first minibatch.
    """

    trainer = PPOTrainer()
    rollout = trainer._collect_rollout(genome, environment, 0, hp)
    advantages, returns = gae_advantages(
        rollout["rewards"],
        rollout["old_values"],
        rollout["dones"],
        gamma=hp.gamma,
        lam=hp.gae_lambda,
    )
    advantages = _normalize(advantages)

    n_transitions = len(rollout["observations"])
    order = torch.randperm(n_transitions)
    index = order[: min(hp.ppo_minibatch, n_transitions)]

    log_probs: list[Tensor] = []
    entropies: list[Tensor] = []
    values: list[Tensor] = []
    for i in index.tolist():
        part, value = split_policy_value(
            genome.forward(rollout["observations"][i]), environment
        )
        distribution = action_distribution(part, environment)
        log_probs.append(distribution_log_prob(distribution, rollout["actions"][i]))
        entropies.append(distribution_entropy(distribution))
        values.append(value)

    ratio = torch.exp(torch.stack(log_probs) - rollout["old_log_probs"][index])
    surrogate_1 = ratio * advantages[index]
    surrogate_2 = (
        torch.clamp(ratio, 1.0 - hp.ppo_clip, 1.0 + hp.ppo_clip) * advantages[index]
    )
    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
    value_loss = 0.5 * (returns[index] - torch.stack(values)).pow(2).mean()
    loss = policy_loss + hp.value_coef * value_loss
    if hp.entropy_coef > 0:
        loss = loss - hp.entropy_coef * torch.stack(entropies).mean()
    return loss


#: Per-sample reference loss builders, by trainer name.
REFERENCES: dict[
    str, Callable[[CircuitGenome, RLEnvironment, SimpleNamespace], Tensor]
] = {
    "reinforce": _reinforce_reference,
    "actor_critic": _actor_critic_reference,
    "ppo": _ppo_reference,
}


# ---------------------------------------------------------------------------
# (1) distribution helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("continuous", [False, True], ids=["discrete", "continuous"])
def test_helpers_on_batch_match_stacked_per_sample(continuous: bool) -> None:
    """Batched helper calls equal stacking the per-sample calls.

    Args:
        continuous: Whether to use the continuous (``Normal``) environment.
    """

    torch.manual_seed(0)
    environment = _make_environment(continuous, MAX_STEPS)
    batch_size = 5
    outputs = torch.randn(batch_size, environment.n_policy_outputs + 1)

    part, value = split_policy_value(outputs, environment)
    assert part.shape == (batch_size, environment.n_policy_outputs)
    assert value.shape == (batch_size,)

    distribution = action_distribution(part, environment)
    assert isinstance(distribution, Normal if continuous else Categorical)
    actions = distribution.sample()
    log_probs = distribution_log_prob(distribution, actions)
    entropies = distribution_entropy(distribution)
    assert log_probs.shape == (batch_size,)
    assert entropies.shape == (batch_size,)

    for i in range(batch_size):
        single_part, single_value = split_policy_value(outputs[i], environment)
        single = action_distribution(single_part, environment)
        torch.testing.assert_close(part[i], single_part)
        torch.testing.assert_close(value[i], single_value)
        torch.testing.assert_close(
            log_probs[i], distribution_log_prob(single, actions[i])
        )
        torch.testing.assert_close(entropies[i], distribution_entropy(single))


def test_discrete_log_prob_keeps_batch_dimension() -> None:
    """A batch of ``Categorical`` log-probabilities is not summed over the batch."""

    logits = torch.tensor([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]])
    distribution = Categorical(logits=logits)
    actions = torch.tensor([0, 1, 1])

    log_probs = distribution_log_prob(distribution, actions)

    torch.testing.assert_close(log_probs, distribution.log_prob(actions))
    torch.testing.assert_close(
        distribution_entropy(distribution), distribution.entropy()
    )


def test_continuous_log_prob_sums_action_dimension() -> None:
    """A batch of ``Normal`` log-probabilities sums only the action dimension."""

    mean = torch.tensor([[0.0, 1.0], [0.5, -0.5], [1.0, 0.0]])
    distribution = Normal(mean, torch.full_like(mean, 0.7))
    actions = torch.tensor([[0.1, 0.9], [0.0, 0.0], [1.5, -0.2]])

    log_probs = distribution_log_prob(distribution, actions)

    torch.testing.assert_close(log_probs, distribution.log_prob(actions).sum(-1))
    torch.testing.assert_close(
        distribution_entropy(distribution), distribution.entropy().sum(-1)
    )


# ---------------------------------------------------------------------------
# (2) batched genome forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,output_mode",
    [("pennylane", "expval"), ("pennylane", "probs"), ("qiskit", "probs")],
)
def test_genome_forward_on_batch_matches_per_sample(
    target: str, output_mode: str
) -> None:
    """``genome.forward`` on a stacked batch equals per-sample forwards.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        output_mode: The genome's ``quantum_output_mode``.
    """

    torch.manual_seed(0)
    environment = _make_environment(False, MAX_STEPS)
    trainer = build_trainer("actor_critic")
    genome = _build_genome(trainer, environment, output_mode, target=target)
    genome.initialize_model()
    observations = torch.rand(6, DEFAULT_OBSERVATION_FEATURES)

    batched = genome.forward(observations)
    per_sample = torch.stack([genome.forward(row) for row in observations])

    assert batched.shape == (6, environment.n_policy_outputs + 1)
    torch.testing.assert_close(batched, per_sample, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# (3) batched update == per-sample reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_mode", OUTPUT_MODES)
@pytest.mark.parametrize("continuous", [False, True], ids=["discrete", "continuous"])
@pytest.mark.parametrize("trainer_name", CONTINUOUS_TRAINER_NAMES)
def test_batched_update_matches_per_sample_reference(
    trainer_name: str, continuous: bool, output_mode: str
) -> None:
    """A batched update's loss and gradients match the per-sample reference.

    The reference loss is built and backpropagated first (without an
    optimizer step), then the trainer's batched ``run_update`` is run from the
    same weights and RNG state. For PPO the comparison is the first
    minibatch's loss and gradients, captured at its ``optimizer.step``.

    Args:
        trainer_name: ``"reinforce"``, ``"actor_critic"`` or ``"ppo"``.
        continuous: Whether to use the continuous (``Normal``) environment.
        output_mode: The genome's ``quantum_output_mode``.
    """

    environment = _make_environment(continuous, MAX_STEPS)
    trainer = build_trainer(trainer_name)
    genome = _build_genome(
        trainer,
        environment,
        output_mode,
        rollout_steps=32,
        ppo_minibatch=8,
        ppo_passes=1,
    )
    torch.manual_seed(0)
    optimizer, hp = prepare_single_update(trainer, genome)

    torch.manual_seed(TORCH_SEED)
    reference_loss = REFERENCES[trainer_name](genome, environment, hp)
    optimizer.zero_grad()
    reference_loss.backward()
    reference_gradients = _gradients(genome)
    optimizer.zero_grad()

    captured: dict[str, Any] = {}
    original_backward = Tensor.backward
    original_step = optimizer.step

    def record_backward(self: Tensor, *args: Any, **kwargs: Any) -> None:
        """Records the first loss backpropagated, then backpropagates it."""

        captured.setdefault("loss", float(self.item()))
        original_backward(self, *args, **kwargs)

    def record_step(*args: Any, **kwargs: Any) -> Any:
        """Records the gradients at the first optimizer step, then steps."""

        captured.setdefault("gradients", _gradients(genome))
        return original_step(*args, **kwargs)

    optimizer.step = record_step
    torch.manual_seed(TORCH_SEED)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Tensor, "backward", record_backward)
        trainer.run_update(genome, environment, optimizer, 0, hp)

    assert captured["loss"] == pytest.approx(
        float(reference_loss.item()), rel=RTOL, abs=ATOL
    )
    _assert_gradients_close(captured["gradients"], reference_gradients)


# ---------------------------------------------------------------------------
# (4) PPO with a final minibatch of one transition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("continuous", [False, True], ids=["discrete", "continuous"])
def test_ppo_runs_with_final_minibatch_of_one(continuous: bool) -> None:
    """A PPO update whose last minibatch holds one transition still runs.

    Five-step episodes and ``rollout_steps=9`` collect ten transitions, so a
    minibatch of three leaves a final minibatch of one.

    Args:
        continuous: Whether to use the continuous (``Normal``) environment.
    """

    environment = _make_environment(continuous, 5)
    trainer = build_trainer("ppo")
    genome = _build_genome(
        trainer,
        environment,
        "probs",
        max_steps=5,
        rollout_steps=9,
        ppo_minibatch=3,
        ppo_passes=2,
    )
    torch.manual_seed(0)
    optimizer, hp = prepare_single_update(trainer, genome)

    batch_sizes: list[int] = []
    original_forward = genome.forward

    def record_forward(x: Tensor) -> Tensor:
        """Records each batched forward's batch size, then runs the forward."""

        if x.dim() > 1:
            batch_sizes.append(int(x.shape[0]))
        return original_forward(x)

    genome.forward = record_forward
    _, info = trainer.run_update(genome, environment, optimizer, 0, hp)

    assert info["rollout_transitions"] == 10
    assert batch_sizes == [3, 3, 3, 1] * hp.ppo_passes
    assert torch.isfinite(torch.tensor(info["loss"]))
    for parameter in genome.parameters():
        if parameter.grad is not None:
            assert torch.all(torch.isfinite(parameter.grad))


@pytest.mark.parametrize("trainer_name", ["reinforce", "actor_critic"])
def test_episode_trainers_run_one_batched_forward(trainer_name: str) -> None:
    """REINFORCE and actor-critic evaluate the whole episode in one forward.

    Args:
        trainer_name: ``"reinforce"`` or ``"actor_critic"``.
    """

    environment = _make_environment(False, MAX_STEPS)
    trainer = build_trainer(trainer_name)
    genome = _build_genome(trainer, environment, "probs")
    torch.manual_seed(0)
    optimizer, hp = prepare_single_update(trainer, genome)

    batch_sizes: list[int] = []
    original_forward = genome.forward

    def record_forward(x: Tensor) -> Tensor:
        """Records each batched forward's batch size, then runs the forward."""

        if x.dim() > 1:
            batch_sizes.append(int(x.shape[0]))
        return original_forward(x)

    genome.forward = record_forward
    trainer.run_update(genome, environment, optimizer, 0, hp)

    assert batch_sizes == [MAX_STEPS]
