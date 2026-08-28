"""Registry of reinforcement-learning trainers keyed by algorithm name.

This aggregation module imports the individual trainer classes (each defined
in its own module) and exposes them through :data:`TRAINER_REGISTRY` for
pluggable, config-driven construction from a command-line argument. Keeping
the registry here -- rather than in :mod:`src.trainer.reinforcement_trainer`
(which holds the base class the subclasses import) -- avoids a circular
import between the base module and the subclass modules.
"""

from __future__ import annotations

from src.trainer.reinforcement_trainer import ReinforcementLearningTrainer
from src.trainer.actor_critic_trainer import ActorCriticTrainer
from src.trainer.ppo_trainer import PPOTrainer
from src.trainer.q_learning_trainer import QLearningTrainer
from src.trainer.reinforce_trainer import ReinforceTrainer

#: Registry mapping algorithm names to trainer classes, for pluggable
#: construction from a command-line argument or config.
TRAINER_REGISTRY: dict[str, type[ReinforcementLearningTrainer]] = {
    "reinforce": ReinforceTrainer,
    "actor_critic": ActorCriticTrainer,
    "a2c": ActorCriticTrainer,
    "ppo": PPOTrainer,
    "q_learning": QLearningTrainer,
    "sarsa": QLearningTrainer,
}
