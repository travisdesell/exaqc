"""Shared test helpers for exercising ``src.trainer.reinforcement_trainer``.

This module is intentionally *not* named ``test_*.py`` so pytest will not
collect it directly; it is imported by the ``test_reinforcement_trainer_*``
modules instead (mirroring ``tests/supervised_trainer_test_utils.py``).

It centralizes:

* a tiny, deterministic Gymnasium environment (registered once on import)
  whose observation dimensionality and action count are configurable, so the
  same environment can satisfy an ``IdentityEncoder`` genome (whose raw
  observation size must equal the circuit's quantum-input size) as well as a
  ``LinearEncoder`` genome (any observation size);
* construction of RL-ready :class:`~src.circuits.circuit.CircuitGenome`
  instances of varying complexity, reusing
  :func:`tests.supervised_trainer_test_utils.build_classification_genome`;
* construction of the pluggable
  :class:`~src.trainer.reinforcement_trainer.RLEnvironment`; and
* small utilities for running a single trainer update and inspecting whether
  gradients reached the encoder, quantum circuit, and decoder.
"""

from __future__ import annotations

import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces

from src.circuits.circuit import CircuitGenome
from src.trainer.reinforcement_trainer import (
    RLEnvironment,
    RLHyperparameters,
    ReinforcementLearningTrainer,
    box_observation_encoder,
)
from src.trainer.rl_trainer_registry import TRAINER_REGISTRY

# Re-export helpers shared with the supervised suite so the RL tests have a
# single import surface.
from tests.supervised_trainer_test_utils import (  # noqa: F401
    COMPLEXITY_LEVELS,
    build_classification_genome,
    circuit_trainable_parameters,
    decoder_trainable_parameters,
    encoder_trainable_parameters,
    hybrid_named_parameters,
)

#: Gymnasium id under which the discrete deterministic test environment is
#: registered.
TEST_ENV_ID: str = "ExaqcTrainerTestEnv-v0"

#: Gymnasium id under which the continuous (``Box``-action) test environment
#: is registered.
CONTINUOUS_TEST_ENV_ID: str = "ExaqcTrainerContinuousTestEnv-v0"

#: Trainer algorithm names exercised across the suite (one representative per
#: distinct algorithm; ``"sarsa"`` shares :class:`QLearningTrainer` with
#: ``"q_learning"`` but uses the on-policy target).
TRAINER_NAMES: tuple[str, ...] = (
    "reinforce",
    "actor_critic",
    "ppo",
    "q_learning",
)

#: Trainer algorithm names that support continuous (``Box``) action spaces --
#: the policy-gradient trainers only. The value-based trainer
#: (``q_learning`` / ``sarsa``) enumerates discrete actions and is excluded.
CONTINUOUS_TRAINER_NAMES: tuple[str, ...] = (
    "reinforce",
    "actor_critic",
    "ppo",
)

#: encoder_name/decoder_name pairs used to exercise trainable vs. stateless
#: coders (``"linear"``/``"linear"`` are torch modules; ``"identity"``/
#: ``"clipped"`` carry no trainable parameters of their own).
ENCODER_DECODER_PAIRS: tuple[tuple[str, str], ...] = (
    ("linear", "linear"),
    ("identity", "clipped"),
)

#: Number of raw observation features used when the genome's encoder is a
#: ``LinearEncoder`` (an identity encoder overrides this to the quantum-input
#: size instead).
DEFAULT_OBSERVATION_FEATURES: int = 4

#: Number of discrete actions the test environment exposes.
DEFAULT_N_ACTIONS: int = 2

#: Action dimensionality the continuous test environment exposes (the number
#: of ``Box`` action components).
DEFAULT_ACTION_DIM: int = 2

#: A deliberately tiny hyperparameter configuration so the whole suite runs
#: quickly. Circuit forward/backward passes are the expensive part, so episode
#: lengths and counts are kept small.
_RL_CONFIG: dict[str, object] = {
    "episodes": 2,
    "eval_episodes": 2,
    "max_steps": 5,
    "learning_rate": 0.05,
    "gamma": 0.99,
    "log_every": 1,
    "seed": 0,
    # baseline="none" keeps single-step advantages non-zero, and a small
    # entropy bonus guarantees a gradient path through the policy logits even
    # if the advantage happens to vanish.
    "baseline": "none",
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "gae_lambda": 0.95,
    "rollout_steps": 10,
    "ppo_passes": 2,
    "ppo_minibatch": 8,
    "epsilon": 0.2,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9,
}


class _DeterministicTestEnv(gym.Env):
    """A tiny deterministic Gymnasium environment for trainer tests.

    The observation is a fixed-length, non-degenerate float vector that
    varies with the step counter (so the policy sees changing inputs), the
    reward is a constant ``1.0`` per step (so discounted returns differ across
    timesteps, keeping advantages non-zero), and the episode terminates after
    ``max_steps`` steps. Both the observation size and the action count are
    configurable via ``gym.make`` keyword arguments.

    Args:
        n_observation_features: Length of the observation vector.
        n_actions: Number of discrete actions.
        max_steps: Steps before the episode terminates.
    """

    def __init__(
        self,
        n_observation_features: int = DEFAULT_OBSERVATION_FEATURES,
        n_actions: int = DEFAULT_N_ACTIONS,
        max_steps: int = 5,
    ):
        super().__init__()
        self.n_observation_features = int(n_observation_features)
        self.n_actions = int(n_actions)
        self.max_steps = int(max_steps)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_observation_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_actions)
        self._step_index = 0

    def _observation(self) -> np.ndarray:
        """Builds the deterministic observation for the current step.

        Returns:
            A non-degenerate ``float32`` vector that varies per step and per
            feature.
        """

        base = 0.1 * (self._step_index + 1)
        offsets = np.arange(self.n_observation_features, dtype=np.float32) * 0.05
        return np.clip(base + offsets, -1.0, 1.0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to its initial observation.

        Args:
            seed: Optional seed (accepted for Gymnasium compatibility; the
                environment is deterministic regardless).
            options: Unused Gymnasium options dict.

        Returns:
            A tuple ``(observation, info)``.
        """

        super().reset(seed=seed)
        self._step_index = 0
        return self._observation(), {}

    def step(self, action: int):
        """Advances the environment by one step.

        Args:
            action: The (ignored) discrete action.

        Returns:
            A ``(observation, reward, terminated, truncated, info)`` tuple.
        """

        self._step_index += 1
        terminated = self._step_index >= self.max_steps
        return self._observation(), 1.0, terminated, False, {}


class _ContinuousTestEnv(_DeterministicTestEnv):
    """A tiny deterministic Gymnasium environment with a ``Box`` action space.

    Identical to :class:`_DeterministicTestEnv` (same deterministic
    observations, constant per-step reward, and fixed episode length) except
    the action space is a continuous ``Box`` of dimensionality ``action_dim``.
    The sampled action value is ignored, so the environment stays fully
    deterministic while still exercising the continuous (Gaussian) policy path
    end to end.

    Args:
        n_observation_features: Length of the observation vector.
        action_dim: Number of continuous action components.
        max_steps: Steps before the episode terminates.
    """

    def __init__(
        self,
        n_observation_features: int = DEFAULT_OBSERVATION_FEATURES,
        action_dim: int = DEFAULT_ACTION_DIM,
        max_steps: int = 5,
    ):
        super().__init__(
            n_observation_features=n_observation_features,
            n_actions=action_dim,
            max_steps=max_steps,
        )
        self.action_dim = int(action_dim)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )


def _register_test_environment() -> None:
    """Registers the discrete and continuous test environments exactly once."""

    if TEST_ENV_ID not in gym.registry:
        gym.register(id=TEST_ENV_ID, entry_point=_DeterministicTestEnv)
    if CONTINUOUS_TEST_ENV_ID not in gym.registry:
        gym.register(id=CONTINUOUS_TEST_ENV_ID, entry_point=_ContinuousTestEnv)


# Register on import so ``gym.make(TEST_ENV_ID, ...)`` works everywhere below.
_register_test_environment()


def rl_hyperparameters() -> dict[str, object]:
    """Returns a fresh copy of the tiny RL hyperparameter configuration.

    Returns:
        A new dict so callers can mutate it without affecting others.
    """

    return dict(_RL_CONFIG)


def build_trainer(name: str) -> ReinforcementLearningTrainer:
    """Constructs a trainer of the given algorithm with the tiny test config.

    Args:
        name: An algorithm name from
            :data:`src.trainer.rl_trainer_registry.TRAINER_REGISTRY`.

    Returns:
        A configured :class:`ReinforcementLearningTrainer` subclass instance.
    """

    trainer_class = TRAINER_REGISTRY[name]
    kwargs = dict(_RL_CONFIG)
    if name == "sarsa":
        kwargs["sarsa"] = True
    return trainer_class(**kwargs)


def build_rl_genome(
    genome_number: int,
    target: str,
    complexity: str,
    encoder_name: str,
    decoder_name: str,
    trainer: ReinforcementLearningTrainer,
    *,
    n_actions: int = DEFAULT_N_ACTIONS,
    continuous: bool = False,
    include_parametric: bool = True,
) -> tuple[CircuitGenome, int]:
    """Builds an RL-ready genome sized for the given trainer and environment.

    The decoder is sized to ``n_policy_outputs + trainer.n_value_outputs`` so
    that advantage methods (actor-critic, PPO) read their scalar state value
    from the extra decoder output. For a discrete action space the policy
    occupies ``n_actions`` outputs; for a continuous one it occupies
    ``2 * n_actions`` (a mean and a log-std per action dimension), mirroring
    :attr:`~src.trainer.reinforcement_trainer.RLEnvironment.n_policy_outputs`.
    RL hyperparameters are merged into ``genome.hyperparameters`` (on top of
    the quantum input/output modes the genome needs) so the trainer resolves
    them per genome.

    Args:
        genome_number: Unique genome identifier.
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: A complexity level from :data:`COMPLEXITY_LEVELS`.
        encoder_name: Either ``"identity"`` or ``"linear"``.
        decoder_name: Either ``"clipped"`` or ``"linear"``.
        trainer: The trainer that will consume the genome (its
            ``n_value_outputs`` sets the number of extra decoder outputs).
        n_actions: Number of discrete actions, or (when ``continuous``) the
            continuous action dimensionality.
        continuous: Whether the target action space is continuous (sizes the
            policy at ``2 * n_actions`` decoder outputs instead of
            ``n_actions``).
        include_parametric: Whether the circuit has trainable gates.

    Returns:
        A tuple ``(genome, observation_features)`` where
        ``observation_features`` is the encoded-observation length the
        environment must produce (i.e. the genome encoder's input size).
    """

    n_policy_outputs = 2 * n_actions if continuous else n_actions
    genome, observation_features = build_classification_genome(
        genome_number=genome_number,
        target=target,
        complexity=complexity,
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        include_parametric=include_parametric,
        n_classes=n_policy_outputs + trainer.n_value_outputs,
        n_features=DEFAULT_OBSERVATION_FEATURES,
    )
    genome.hyperparameters.update(rl_hyperparameters())
    return genome, observation_features


def make_test_environment(
    n_observation_features: int,
    *,
    n_actions: int = DEFAULT_N_ACTIONS,
    max_steps: int | None = None,
) -> RLEnvironment:
    """Builds an :class:`RLEnvironment` backed by the deterministic test env.

    Args:
        n_observation_features: Observation length the genome's encoder
            expects (as returned by :func:`build_rl_genome`).
        n_actions: Number of discrete actions.
        max_steps: Episode length; defaults to the tiny test configuration.

    Returns:
        A configured :class:`RLEnvironment`.
    """

    resolved_max_steps = int(
        max_steps if max_steps is not None else _RL_CONFIG["max_steps"]
    )
    return RLEnvironment(
        env_id=TEST_ENV_ID,
        n_actions=n_actions,
        n_observation_features=n_observation_features,
        obs_encoder=box_observation_encoder(),
        env_kwargs={
            "n_observation_features": n_observation_features,
            "n_actions": n_actions,
            "max_steps": resolved_max_steps,
        },
    )


def make_continuous_test_environment(
    n_observation_features: int,
    *,
    action_dim: int = DEFAULT_ACTION_DIM,
    max_steps: int | None = None,
) -> RLEnvironment:
    """Builds a continuous :class:`RLEnvironment` backed by the ``Box`` test env.

    The returned environment mirrors :func:`make_test_environment` but exposes
    a continuous action space, so it exercises the Gaussian-policy path (mean +
    log-std read from the decoder, ``Normal`` sampling, action clipping)
    without requiring MuJoCo.

    Args:
        n_observation_features: Observation length the genome's encoder
            expects (as returned by :func:`build_rl_genome`).
        action_dim: Continuous action dimensionality.
        max_steps: Episode length; defaults to the tiny test configuration.

    Returns:
        A configured continuous :class:`RLEnvironment`.
    """

    resolved_max_steps = int(
        max_steps if max_steps is not None else _RL_CONFIG["max_steps"]
    )
    return RLEnvironment(
        env_id=CONTINUOUS_TEST_ENV_ID,
        n_actions=action_dim,
        n_observation_features=n_observation_features,
        obs_encoder=box_observation_encoder(),
        env_kwargs={
            "n_observation_features": n_observation_features,
            "action_dim": action_dim,
            "max_steps": resolved_max_steps,
        },
        continuous=True,
        action_low=np.full(action_dim, -1.0, dtype=np.float32),
        action_high=np.full(action_dim, 1.0, dtype=np.float32),
    )


def prepare_single_update(
    trainer: ReinforcementLearningTrainer,
    genome: CircuitGenome,
) -> tuple[torch.optim.Optimizer, RLHyperparameters]:
    """Initializes a genome's model and sets up a single manual update.

    Builds the genome's hybrid model, resolves the trainer's hyperparameters
    for it (via the public :meth:`ReinforcementLearningTrainer.
    resolve_hyperparameters`), and constructs the optimizer -- everything
    :meth:`ReinforcementLearningTrainer.run_update` needs, without running it.
    Splitting setup from the update lets a caller snapshot the freshly
    initialized weights before the update is applied.

    Args:
        trainer: The trainer that will perform the update.
        genome: The genome to initialize and, later, update in place.

    Returns:
        A tuple ``(optimizer, hyperparameters)`` ready to pass to
        ``trainer.run_update(genome, environment, optimizer, 0, hp)``.
    """

    genome.initialize_model()
    hyperparameters = trainer.resolve_hyperparameters(genome)
    optimizer = torch.optim.Adam(genome.parameters(), lr=hyperparameters.learning_rate)
    return optimizer, hyperparameters


def run_single_update(
    trainer: ReinforcementLearningTrainer,
    genome: CircuitGenome,
    environment: RLEnvironment,
) -> None:
    """Initializes a genome and runs exactly one trainer update on it.

    This drives one outer-loop episode of the actual algorithm
    (``trainer.run_update``) rather than the full :meth:`train` loop, so the
    gradients from the update remain on the parameters afterwards (``train``
    would restore the best snapshot and clear the picture). After this
    returns, each hybrid-model parameter's ``.grad`` reflects the update's
    backward pass.

    Args:
        trainer: The trainer whose single update to run.
        genome: The genome to initialize and update in place.
        environment: The environment to roll out in.
    """

    optimizer, hyperparameters = prepare_single_update(trainer, genome)
    trainer.run_update(genome, environment, optimizer, 0, hyperparameters)
