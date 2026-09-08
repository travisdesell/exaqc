"""Semi-gradient value-based (Q-learning / SARSA) trainer for circuit genomes.

See :mod:`src.trainer.reinforcement_trainer` for the shared training scaffold
and environment abstraction this trainer builds on.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from torch import Tensor

from src.circuits.circuit import CircuitGenome
from src.trainer.reinforcement_trainer import (
    RLEnvironment,
    ReinforcementLearningTrainer,
)


class QLearningTrainer(ReinforcementLearningTrainer):
    """Semi-gradient value-based trainer (Q-learning or SARSA).

    One outer-loop episode runs a single environment episode and performs one
    epoch (weight update) at every environment step -- an online,
    semi-gradient temporal-difference update rather than a single update per
    episode.

    The genome output vector is interpreted directly as the action-value
    function ``Q(s, .)``. Actions are chosen epsilon-greedily and the circuit
    is updated with a temporal-difference target bootstrapped from a detached
    forward pass (i.e. no separate target network):

    * Q-learning: ``y = r + gamma * (1 - done) * max_a' Q(s', a')``
    * SARSA:      ``y = r + gamma * (1 - done) * Q(s', a'_epsilon)``

    Because it enumerates actions (argmax / epsilon-greedy over ``Q(s, .)``),
    this trainer is discrete-only: it sets ``supports_continuous = False`` and
    :meth:`~src.trainer.reinforcement_trainer.ReinforcementLearningTrainer.train`
    raises if it is paired with a continuous environment.

    Args:
        sarsa: If True, use the on-policy SARSA target; otherwise use the
            off-policy Q-learning (max) target.
    """

    #: Value-based action selection is discrete-only (argmax / epsilon-greedy).
    supports_continuous: bool = False

    def __init__(self, *, sarsa: bool = False) -> None:
        """Initializes the value-based trainer.

        Args:
            sarsa: If True, use the on-policy SARSA target; otherwise use the
                off-policy Q-learning (max) target.
        """

        super().__init__()
        self.sarsa = sarsa

    def _epsilon_greedy(self, q_values: Tensor, epsilon: float) -> int:
        """Selects an action with epsilon-greedy exploration.

        Args:
            q_values: Action-value tensor of shape ``(n_actions,)``.
            epsilon: Probability of choosing a random action.

        Returns:
            The selected action index.
        """

        if torch.rand(()) < epsilon:
            return int(torch.randint(low=0, high=q_values.numel(), size=(1,)).item())
        return int(torch.argmax(q_values).item())

    def run_update(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        optimizer: torch.optim.Optimizer,
        episode_index: int,
        hp: SimpleNamespace,
    ) -> tuple[float, dict[str, float]]:
        """Runs one episode with a per-step temporal-difference update.

        Rolls a single episode, selecting actions epsilon-greedily (with
        epsilon decayed by episode index), and performs one epoch (weight
        update) at each environment step against the detached TD target.

        Args:
            genome: The genome Q-network being trained.
            environment: The environment to roll the episode in.
            optimizer: The optimizer over the genome's parameters.
            episode_index: Zero-based index of this episode (seeds the reset
                and sets the epsilon decay).
            hp: Resolved hyperparameters.

        Returns:
            A tuple ``(episode_return, info)`` with the episode's total reward
            and a dict of the mean per-step ``"loss"`` and the ``"epsilon"``
            used.
        """

        epsilon = max(hp.epsilon_min, hp.epsilon * (hp.epsilon_decay**episode_index))

        env = environment.make()
        observation, _ = env.reset(seed=hp.seed + episode_index)
        episode_return = 0.0
        losses: list[float] = []

        for _ in range(hp.max_steps):
            q_values = self.policy_logits(genome, environment, observation)
            action = self._epsilon_greedy(q_values.detach(), epsilon)

            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            with torch.no_grad():
                if done:
                    bootstrap = torch.tensor(0.0)
                else:
                    next_q = self.policy_logits(genome, environment, next_observation)
                    if self.sarsa:
                        next_action = self._epsilon_greedy(next_q, epsilon)
                        bootstrap = next_q[next_action]
                    else:
                        bootstrap = torch.max(next_q)
                target = float(reward) + hp.gamma * float(bootstrap.item())

            loss = (q_values[action] - target).pow(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            observation = next_observation
            episode_return += float(reward)
            if done:
                break

        env.close()

        return episode_return, {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon": float(epsilon),
        }
