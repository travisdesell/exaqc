"""REINFORCE (Monte-Carlo policy gradient) trainer for circuit genomes.

See :mod:`src.trainer.reinforcement_trainer` for the shared training scaffold
and environment abstraction this trainer builds on.
"""

from __future__ import annotations

import argparse

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
    to_env_action,
)


class ReinforceTrainer(ReinforcementLearningTrainer):
    """Monte-Carlo policy-gradient (REINFORCE) trainer.

    One outer-loop episode runs a single environment episode and then performs
    a single epoch (weight update), applying the policy-gradient loss
    ``L = -E[log pi(a|s) * advantage] - entropy_coef * H[pi]`` where the
    advantage is either the return minus its mean (``baseline="mean"``) or
    the raw return.
    """

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds REINFORCE's own command-line argument.

        Args:
            parser: The parser to add the argument to.

        Returns:
            None. Mutates ``parser`` by adding ``--baseline``.
        """

        parser.add_argument(
            "--baseline",
            choices=["mean", "none"],
            default="mean",
            help="REINFORCE advantage baseline ('mean' subtracts the batch-mean return).",
        )

    def run_update(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        optimizer: torch.optim.Optimizer,
        episode_index: int,
        hp: SimpleNamespace,
    ) -> tuple[float, dict[str, float]]:
        """Runs one episode and performs one weight update (epoch).

        Rolls a single episode (sampling actions from the policy), computes
        discounted returns and the baseline-adjusted advantage, and applies
        one REINFORCE gradient step, evaluating the episode's log-probabilities
        in one batched forward pass.

        Args:
            genome: The genome policy being trained.
            environment: The environment to roll the episode in.
            optimizer: The optimizer over the genome's parameters.
            episode_index: Zero-based index of this episode (seeds the reset).
            hp: Resolved hyperparameters.

        Returns:
            A tuple ``(episode_return, info)`` with the episode's total reward
            and a dict holding the scalar ``"loss"``.
        """

        env = environment.make()
        observation, _ = env.reset(seed=hp.seed + episode_index)

        observations: list[Tensor] = []
        actions: list[Tensor] = []
        rewards: list[float] = []
        episode_return = 0.0

        # The rollout only samples actions, so it runs without gradients; the
        # log-probabilities and entropies are recomputed below in one batched
        # forward pass over the whole episode. The weights do not change during
        # the episode, so this gives the same loss as tracking gradients step
        # by step, without holding a graph per step.
        for _ in range(hp.max_steps):
            encoded = environment.encode(observation)
            with torch.no_grad():
                part = genome.forward(encoded)[..., : environment.n_policy_outputs]
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

        returns = discounted_returns(rewards, hp.gamma)
        advantages = returns - returns.mean() if hp.baseline == "mean" else returns

        output = genome.forward(torch.stack(observations))
        distribution = action_distribution(
            output[..., : environment.n_policy_outputs], environment
        )
        log_prob_tensor = distribution_log_prob(distribution, torch.stack(actions))
        entropy_tensor = distribution_entropy(distribution)

        policy_loss = -(log_prob_tensor * advantages.detach()).mean()
        entropy_loss = (
            -hp.entropy_coef * entropy_tensor.mean() if hp.entropy_coef > 0 else 0.0
        )
        loss = policy_loss + (entropy_loss if isinstance(entropy_loss, Tensor) else 0.0)

        optimizer.zero_grad()
        # A genome whose only parameterized gates are disabled (or transiently
        # dropped) produces a policy that is constant w.r.t. the circuit
        # weights, so the loss has no grad_fn and backward() would raise. Skip
        # the update for such a step.
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        return episode_return, {"loss": float(loss.item())}
