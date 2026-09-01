"""Proximal Policy Optimization (PPO) trainer for circuit genomes.

See :mod:`src.trainer.reinforcement_trainer` for the shared training scaffold
and environment abstraction this trainer builds on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torch import Tensor

from src.circuits.circuit import CircuitGenome
from src.trainer.reinforcement_trainer import (
    RLEnvironment,
    ReinforcementLearningTrainer,
    RLHyperparameters,
    _normalize,
    action_distribution,
    distribution_entropy,
    distribution_log_prob,
    gae_advantages,
    split_policy_value,
    to_env_action,
)


class PPOTrainer(ReinforcementLearningTrainer):
    """Proximal Policy Optimization trainer with GAE.

    PPO is the one trainer whose outer-loop episode spans *several*
    environment episodes: each outer-loop episode collects a rollout of at
    least ``rollout_steps`` transitions (spanning whole environment episodes),
    computes GAE advantages under the behavior policy, and then performs many
    epochs (weight updates) across ``ppo_passes`` passes of clipped-objective
    minibatch updates. The state value is read from the genome's extra decoder
    output, so it is evolved and serialized alongside the policy.
    """

    #: Requires one extra decoder output for the scalar state value.
    n_value_outputs: int = 1

    def _collect_rollout(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        episode_index: int,
        hp: RLHyperparameters,
    ) -> dict[str, Any]:
        """Collects a behavior-policy rollout spanning one or more episodes.

        Runs full environment episodes back-to-back (without gradients) until
        at least ``rollout_steps`` transitions have been gathered, recording
        the behavior-policy log-probabilities and value estimates needed for
        the clipped PPO objective.

        Args:
            genome: The genome policy/value network to roll out.
            environment: The environment to roll out in.
            episode_index: Zero-based index of the outer-loop episode, used
                (together with a per-episode counter) to seed each reset.
            hp: Resolved hyperparameters.

        Returns:
            A dict of stacked transition tensors (``observations``,
            ``actions``, ``old_log_probs``, ``old_values``, ``rewards``,
            ``dones``) plus ``episode_returns``, the list of per-episode
            returns collected during the rollout. ``actions`` is a list of
            per-step action tensors (scalar for discrete spaces, vectors for
            continuous ones).
        """

        observations: list[Tensor] = []
        actions: list[Tensor] = []
        old_log_probs: list[float] = []
        old_values: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        episode_returns: list[float] = []

        collected = 0
        episode = 0
        while collected < hp.rollout_steps:
            env = environment.make()
            observation, _ = env.reset(seed=hp.seed + episode_index * 10_000 + episode)
            episode += 1
            episode_return = 0.0

            for _ in range(hp.max_steps):
                encoded = environment.encode(observation)
                with torch.no_grad():
                    output = genome.forward(encoded)
                    part, value = split_policy_value(output, environment)
                    distribution = action_distribution(part, environment)
                    action = distribution.sample()

                observations.append(encoded.detach())
                actions.append(action.detach())
                old_log_probs.append(
                    float(distribution_log_prob(distribution, action).item())
                )
                old_values.append(float(value.item()))

                observation, reward, terminated, truncated, _ = env.step(
                    to_env_action(action, environment)
                )
                done = terminated or truncated
                rewards.append(float(reward))
                dones.append(1.0 if done else 0.0)
                episode_return += float(reward)
                collected += 1

                if done:
                    break

            env.close()
            episode_returns.append(episode_return)

        return {
            "observations": observations,
            "actions": actions,
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
            "old_values": torch.tensor(old_values, dtype=torch.float32),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
            "episode_returns": episode_returns,
        }

    def run_update(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        optimizer: torch.optim.Optimizer,
        episode_index: int,
        hp: RLHyperparameters,
    ) -> tuple[float, dict[str, float]]:
        """Collects a multi-episode rollout and performs the PPO epochs.

        Collects a behavior-policy rollout (several environment episodes),
        computes normalized GAE advantages, then runs ``ppo_passes`` passes
        over the rollout, each performing several minibatch epochs (weight
        updates) of the clipped surrogate plus value and entropy losses.

        Args:
            genome: The genome policy/value network being trained.
            environment: The environment to roll out in.
            optimizer: The optimizer over the genome's parameters.
            episode_index: Zero-based index of this outer-loop episode.
            hp: Resolved hyperparameters.

        Returns:
            A tuple ``(mean_episode_return, info)`` with the mean return over
            the rollout's episodes and a dict of the final ``"loss"`` and the
            number of ``"rollout_transitions"``.
        """

        rollout = self._collect_rollout(genome, environment, episode_index, hp)

        advantages, returns = gae_advantages(
            rollout["rewards"],
            rollout["old_values"],
            rollout["dones"],
            gamma=hp.gamma,
            lam=hp.gae_lambda,
        )
        advantages = _normalize(advantages)

        observations = rollout["observations"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_log_probs"]

        n_transitions = len(observations)
        minibatch = min(hp.ppo_minibatch, n_transitions)
        last_loss = 0.0

        for _ in range(hp.ppo_passes):
            order = torch.randperm(n_transitions)
            for start in range(0, n_transitions, minibatch):
                index = order[start : start + minibatch]

                new_log_probs: list[Tensor] = []
                new_values: list[Tensor] = []
                entropies: list[Tensor] = []

                for i in index.tolist():
                    output = genome.forward(observations[i])
                    part, value = split_policy_value(output, environment)
                    distribution = action_distribution(part, environment)
                    new_log_probs.append(
                        distribution_log_prob(distribution, actions[i])
                    )
                    entropies.append(distribution_entropy(distribution))
                    new_values.append(value)

                new_log_prob_tensor = torch.stack(new_log_probs)
                new_value_tensor = torch.stack(new_values)
                entropy_tensor = torch.stack(entropies)

                ratio = torch.exp(new_log_prob_tensor - old_log_probs[index])
                surrogate_1 = ratio * advantages[index]
                surrogate_2 = (
                    torch.clamp(ratio, 1.0 - hp.ppo_clip, 1.0 + hp.ppo_clip)
                    * advantages[index]
                )
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_loss = (
                    0.5 * (returns[index].detach() - new_value_tensor).pow(2).mean()
                )
                entropy_loss = (
                    -hp.entropy_coef * entropy_tensor.mean()
                    if hp.entropy_coef > 0
                    else 0.0
                )

                loss = (
                    policy_loss
                    + hp.value_coef * value_loss
                    + (entropy_loss if isinstance(entropy_loss, Tensor) else 0.0)
                )

                optimizer.zero_grad()
                # A genome whose only parameterized gates are disabled (or
                # transiently dropped) produces outputs that are constant w.r.t.
                # the circuit weights, so the loss has no grad_fn and backward()
                # would raise. Skip the update for such a minibatch.
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                last_loss = float(loss.item())

        mean_return = (
            float(np.mean(rollout["episode_returns"]))
            if rollout["episode_returns"]
            else 0.0
        )
        return mean_return, {
            "loss": last_loss,
            "rollout_transitions": float(n_transitions),
        }
