"""On-policy advantage actor-critic (A2C-style) trainer for circuit genomes.

See :mod:`src.trainer.reinforcement_trainer` for the shared training scaffold
and environment abstraction this trainer builds on.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from torch import Tensor

from src.circuits.circuit import CircuitGenome
from src.trainer.reinforcement_trainer import (
    RLEnvironment,
    ReinforcementLearningTrainer,
    action_distribution,
    discounted_returns,
    distribution_entropy,
    distribution_log_prob,
    split_policy_value,
    to_env_action,
)


class ActorCriticTrainer(ReinforcementLearningTrainer):
    """On-policy advantage actor-critic (A2C-style) trainer.

    One outer-loop episode runs a single environment episode and then performs
    a single epoch (weight update). It reads the state value from the genome's
    extra decoder output and jointly optimizes
    ``L = L_pi + value_coef * L_v - entropy_coef * H`` where the advantage is
    the Monte-Carlo return minus the value baseline. Because the value is a
    decoder output rather than a separate head, it is evolved and serialized
    alongside the policy.
    """

    #: Requires one extra decoder output for the scalar state value.
    n_value_outputs: int = 1

    def run_update(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        optimizer: torch.optim.Optimizer,
        episode_index: int,
        hp: SimpleNamespace,
    ) -> tuple[float, dict[str, float]]:
        """Runs one episode and performs one weight update (epoch).

        Rolls a single episode, then evaluates the episode's log-probabilities,
        entropies, and state values (from the decoder's extra output) in one
        batched forward pass and applies one combined policy + value + entropy
        gradient step using the Monte-Carlo return minus the value baseline as
        the advantage.

        Args:
            genome: The genome policy/value network being trained.
            environment: The environment to roll the episode in.
            optimizer: The optimizer over the genome's parameters.
            episode_index: Zero-based index of this episode (seeds the reset).
            hp: Resolved hyperparameters.

        Returns:
            A tuple ``(episode_return, info)`` with the episode's total reward
            and a dict of the scalar ``"loss"``, ``"policy_loss"`` and
            ``"value_loss"`` terms.
        """

        env = environment.make()
        observation, _ = env.reset(seed=hp.seed + episode_index)

        observations: list[Tensor] = []
        actions: list[Tensor] = []
        rewards: list[float] = []
        episode_return = 0.0

        # The rollout only samples actions, so it runs without gradients; the
        # log-probabilities, entropies and values are recomputed below in one
        # batched forward pass over the whole episode. The weights do not
        # change during the episode, so this gives the same loss as tracking
        # gradients step by step, without holding a graph per step.
        for _ in range(hp.max_steps):
            encoded = environment.encode(observation)
            with torch.no_grad():
                part, _ = split_policy_value(genome.forward(encoded), environment)
                action = action_distribution(part, environment).sample()

            observations.append(encoded)
            actions.append(action)

            observation, reward, terminated, truncated, _ = env.step(
                to_env_action(action, environment)
            )
            rewards.append(float(reward))
            episode_return += float(reward)
            if terminated or truncated:
                break

        env.close()

        if not rewards:
            return episode_return, {"loss": 0.0}

        output = genome.forward(torch.stack(observations))
        part, value_tensor = split_policy_value(output, environment)
        distribution = action_distribution(part, environment)
        log_prob_tensor = distribution_log_prob(distribution, torch.stack(actions))
        entropy_tensor = distribution_entropy(distribution)

        returns = discounted_returns(rewards, hp.gamma)
        advantages = returns - value_tensor.detach()

        policy_loss = -(log_prob_tensor * advantages.detach()).mean()
        value_loss = 0.5 * (returns.detach() - value_tensor).pow(2).mean()
        entropy_loss = (
            -hp.entropy_coef * entropy_tensor.mean() if hp.entropy_coef > 0 else 0.0
        )

        loss = (
            policy_loss
            + hp.value_coef * value_loss
            + (entropy_loss if isinstance(entropy_loss, Tensor) else 0.0)
        )

        optimizer.zero_grad()
        # A genome whose only parameterized gates are disabled (or transiently
        # dropped) produces outputs that are constant w.r.t. the circuit
        # weights, so the loss has no grad_fn and backward() would raise. Skip
        # the update for such a step.
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        return episode_return, {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }
