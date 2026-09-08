"""Modular reinforcement-learning trainers for evolved quantum genomes.

This module mirrors :mod:`src.trainer.supervised_trainer` but for
reinforcement learning. Where ``SupervisedTrainer`` consumes dataloaders,
loss functions and metrics and calls ``genome.forward`` on labelled samples,
the trainers here drive a ``CircuitGenome`` as a *policy* (or *value*)
network inside a Gymnasium environment.

Every trainer uses the same modular ``CircuitGenome`` interface the
supervised path uses:

* ``genome.initialize_model()`` builds the ``hybrid_model`` (encoder ->
  quantum layer -> decoder);
* ``genome.forward(observation)`` produces one output value per action
  (interpreted as policy logits or Q-values);
* ``genome.parameters()`` are the only trainable parameters;
* ``genome.clone_state_dict()`` snapshots and
  ``genome.set_state_dict(state_dict)`` restores the best-performing weights.

Terminology
    The trainers use these terms consistently:

    * **step**: one interaction with the environment (observe -> act ->
      receive a reward).
    * **episode**: one full rollout through the environment, from ``reset``
      until a terminal state or ``max_steps`` steps -- i.e. a sequence of
      steps.
    * **epoch**: one weight update (a single ``optimizer.step()``).

    The outer training loop runs ``episodes`` episodes. How many epochs
    (weight updates) occur per episode depends on the algorithm:

    * REINFORCE / actor-critic: run one episode, then perform one epoch;
    * Q-learning / SARSA: run one episode, performing one epoch at every
      environment step;
    * PPO: collect several episodes into a rollout, then perform many epochs
      across ``ppo_passes`` passes over that rollout. PPO is the one algorithm
      whose outer-loop iteration spans more than one episode (see
      :class:`~src.trainer.ppo_trainer.PPOTrainer`).

Value-based advantage methods (actor-critic, PPO) need a scalar state value
in addition to the per-action policy outputs. Rather than owning a separate
value head (whose weights would live outside the genome -- never serialized,
never recombined by the encoder/decoder crossover operators, and re-created
from scratch every evaluation), those trainers ask the genome's decoder for
one *extra* output. The decoder is sized to
``n_policy_outputs + n_value_outputs``, so the value estimate is just another
row of the decoder's linear layer and
is therefore part of ``genome.hybrid_model`` -- evolved by
``crossover_encoder_decoder`` / ``torch_simplex_crossover`` and preserved
across ``to_dict``/``from_dict`` exactly like the policy weights. This does
assume a decoder whose outputs are unconstrained linear features (the default
``LinearDecoder``); a normalizing decoder such as ``ClippedDecoder`` is not
appropriate for these algorithms.

The classical observation encoder (raw env observation -> fixed-length
feature vector) is kept separate from the genome's learnable ``Encoder`` so
existing ``LinearEncoder``/``IdentityEncoder`` classes can be reused
unchanged: the observation encoder maps a Gym observation into the feature
vector the genome's ``Encoder`` then embeds into the quantum circuit.

This module provides the shared infrastructure -- the environment
abstraction, observation encoders, resolved hyperparameters, RL math helpers,
and the abstract base :class:`ReinforcementLearningTrainer`. The concrete
algorithms each live in their own module and subclass the base here:

* :class:`~src.trainer.reinforce_trainer.ReinforceTrainer` -- Monte-Carlo
  policy gradient (REINFORCE).
* :class:`~src.trainer.actor_critic_trainer.ActorCriticTrainer` -- on-policy
  advantage actor-critic.
* :class:`~src.trainer.ppo_trainer.PPOTrainer` -- proximal policy
  optimization with GAE.
* :class:`~src.trainer.q_learning_trainer.QLearningTrainer` -- semi-gradient
  Q-learning / SARSA.

They are collected by name in
:data:`src.trainer.rl_trainer_registry.TRAINER_REGISTRY`.

Action spaces
    All trainers support **discrete** action spaces (a ``Categorical`` policy).
    The three policy-gradient trainers (REINFORCE, actor-critic, PPO)
    additionally support **continuous** ``Box`` action spaces via a diagonal
    ``Normal`` policy whose per-dimension mean and log-standard-deviation are
    read from the decoder (so the policy stays entirely inside the genome). The
    value-based trainer (Q-learning / SARSA) is discrete-only, since it selects
    actions by argmax / epsilon-greedy over enumerable action values; it sets
    ``supports_continuous = False`` and :meth:`ReinforcementLearningTrainer.
    train` raises if paired with a continuous environment. The discrete vs.
    continuous branching itself lives in the shared module-level
    :func:`action_distribution` / :func:`to_env_action` / :func:`greedy_action`
    helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

import math
import numpy as np
import torch

from loguru import logger
from torch import Tensor
from torch.distributions import Categorical, Distribution, Normal

import gymnasium as gym

from src.circuits.circuit import CircuitGenome
from src.dropout.quantum_dropout import sample_quantum_dropout

#: Bounds applied to the per-dimension log-standard-deviation a continuous
#: (Gaussian) policy reads from the decoder, keeping the sampled action scale
#: numerically sane regardless of the unconstrained decoder outputs.
LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0

# ---------------------------------------------------------------------------
# Environment abstraction
# ---------------------------------------------------------------------------


@dataclass
class RLEnvironment:
    """A pluggable description of a reinforcement-learning environment.

    This wraps a Gymnasium environment id together with everything a trainer
    needs to run it against a genome policy: the number of actions (or action
    dimensions), the size of the encoded observation vector, and a callable
    that turns a raw environment observation into that fixed-length feature
    tensor.

    Keeping the observation encoder here (rather than inside the genome)
    means the genome's own learnable ``Encoder`` only ever sees a clean,
    fixed-length feature vector, so the existing ``LinearEncoder`` /
    ``IdentityEncoder`` classes work without modification.

    The environment may expose either a **discrete** action space (the
    default, e.g. CartPole) or a **continuous** ``Box`` action space (e.g.
    Pendulum and the MuJoCo tasks). For a discrete space the policy-gradient
    trainers build a ``Categorical`` over ``n_actions`` logits; for a
    continuous space they build a diagonal ``Normal`` whose per-dimension mean
    and log-standard-deviation are both read from the decoder (see
    :attr:`n_policy_outputs`), keeping the whole policy inside the genome.

    Attributes:
        env_id: Gymnasium environment id (e.g. ``"CartPole-v1"``).
        n_actions: For a discrete space, the number of discrete actions. For a
            continuous space (``continuous=True``), the action dimensionality
            (the number of ``Box`` action components).
        n_observation_features: Length of the encoded observation vector,
            i.e. the number of inputs the genome's ``Encoder`` expects.
        obs_encoder: Callable mapping a raw observation into a float tensor
            of shape ``(n_observation_features,)``.
        env_kwargs: Optional keyword arguments passed to ``gym.make``.
        deterministic: Whether the environment is fully deterministic (fixed
            initial state and transitions). When True, a greedy policy
            produces the same episode every time regardless of seed, so
            greedy evaluation runs a single episode instead of
            ``eval_episodes`` identical ones (see
            :meth:`ReinforcementLearningTrainer.evaluate`).
        continuous: Whether the action space is a continuous ``Box`` (True) or
            discrete (False, the default).
        action_low: For a continuous space, the per-dimension lower action
            bound (shape ``(n_actions,)``); actions are clipped to it before
            being passed to the environment. ``None`` for discrete spaces.
        action_high: For a continuous space, the per-dimension upper action
            bound (shape ``(n_actions,)``). ``None`` for discrete spaces.
    """

    env_id: str
    n_actions: int
    n_observation_features: int
    obs_encoder: Callable[[Any], Tensor]
    env_kwargs: Optional[dict[str, Any]] = None
    deterministic: bool = False
    continuous: bool = False
    action_low: Optional[np.ndarray] = None
    action_high: Optional[np.ndarray] = None

    @property
    def n_policy_outputs(self) -> int:
        """Number of decoder outputs the policy occupies (before any value).

        A discrete policy needs one logit per action (``n_actions``). A
        continuous diagonal-Gaussian policy needs a mean *and* a
        log-standard-deviation per action dimension (``2 * n_actions``), both
        read from the decoder so the entire policy is part of the genome.

        Returns:
            The number of leading decoder outputs devoted to the policy.
        """

        return 2 * self.n_actions if self.continuous else self.n_actions

    def make(self) -> gym.Env:
        """Instantiates the underlying Gymnasium environment.

        Returns:
            A new ``gym.Env`` instance.
        """

        return gym.make(self.env_id, **(self.env_kwargs or {}))

    def encode(self, observation: Any) -> Tensor:
        """Encodes a raw observation into the genome's input feature vector.

        Args:
            observation: A raw observation returned by the environment.

        Returns:
            A float tensor of shape ``(n_observation_features,)``.
        """

        return self.obs_encoder(observation)


def box_observation_encoder(
    scales: Optional[np.ndarray] = None,
) -> Callable[[Any], Tensor]:
    """Builds an encoder for continuous ``Box`` observations.

    Args:
        scales: Optional per-dimension scale factors. When provided, each
            observation dimension is divided by its scale and clipped to
            ``[-1, 1]``; otherwise the raw observation is returned as a float
            tensor (the genome's learnable encoder can then rescale it).

    Returns:
        A callable mapping an observation into a float tensor.
    """

    scale_array = None if scales is None else np.asarray(scales, dtype=np.float32)

    def encode(observation: Any) -> Tensor:
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if scale_array is not None:
            values = np.clip(values / scale_array, -1.0, 1.0)
        return torch.tensor(values, dtype=torch.float32)

    return encode


def onehot_observation_encoder(n_states: int) -> Callable[[Any], Tensor]:
    """Builds a one-hot encoder for discrete integer observations.

    Useful for tabular environments such as ``FrozenLake`` whose observation
    is a single integer state index.

    Args:
        n_states: Total number of discrete states.

    Returns:
        A callable mapping an integer state into a one-hot float tensor of
        shape ``(n_states,)``.
    """

    def encode(observation: Any) -> Tensor:
        vector = torch.zeros(n_states, dtype=torch.float32)
        vector[int(observation)] = 1.0
        return vector

    return encode


# ---------------------------------------------------------------------------
# Action-space helpers (shared by every policy-gradient trainer and by the
# visualization script, so the discrete/continuous branching lives in exactly
# one place).
# ---------------------------------------------------------------------------


def action_distribution(
    policy_part: Tensor, environment: RLEnvironment
) -> Distribution:
    """Builds the action distribution from the policy portion of a genome output.

    For a discrete environment this is a ``Categorical`` over ``n_actions``
    logits. For a continuous environment the ``2 * n_actions`` policy outputs
    are split into a per-dimension mean and (clamped) log-standard-deviation,
    yielding a diagonal ``Normal``.

    Args:
        policy_part: The leading ``environment.n_policy_outputs`` entries of a
            genome output (the policy, with any value output already sliced
            off).
        environment: The environment whose action space determines the
            distribution family.

    Returns:
        A ``torch.distributions`` distribution to sample actions from.
    """

    if environment.continuous:
        n = environment.n_actions
        mean = policy_part[:n]
        log_std = torch.clamp(policy_part[n : 2 * n], LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())
    return Categorical(logits=policy_part)


def distribution_log_prob(distribution: Distribution, action: Tensor) -> Tensor:
    """Returns a scalar log-probability for an action under a distribution.

    A ``Categorical`` already yields a scalar; a diagonal ``Normal`` yields a
    per-dimension vector, which is summed into the joint log-probability.

    Args:
        distribution: The action distribution.
        action: The sampled action tensor.

    Returns:
        A scalar log-probability tensor.
    """

    log_prob = distribution.log_prob(action)
    return log_prob.sum(-1) if log_prob.dim() > 0 else log_prob


def distribution_entropy(distribution: Distribution) -> Tensor:
    """Returns a scalar entropy for a distribution.

    As with :func:`distribution_log_prob`, a diagonal ``Normal``'s
    per-dimension entropy is summed into a single scalar.

    Args:
        distribution: The action distribution.

    Returns:
        A scalar entropy tensor.
    """

    entropy = distribution.entropy()
    return entropy.sum(-1) if entropy.dim() > 0 else entropy


def to_env_action(action: Tensor, environment: RLEnvironment) -> Any:
    """Converts a policy action tensor into the value ``env.step`` expects.

    For a discrete environment this is a Python ``int``. For a continuous
    environment it is a ``float32`` NumPy array clipped to the environment's
    action bounds.

    Args:
        action: The action tensor (a scalar for discrete spaces, a vector of
            shape ``(n_actions,)`` for continuous spaces).
        environment: The environment whose action space determines the format.

    Returns:
        The action in the environment's native format.
    """

    if environment.continuous:
        array = np.asarray(action.detach(), dtype=np.float32).reshape(-1)
        if environment.action_low is not None:
            array = np.clip(array, environment.action_low, environment.action_high)
        return array.astype(np.float32)
    return int(action.item())


def policy_output(
    genome: CircuitGenome, environment: RLEnvironment, observation: Any
) -> Tensor:
    """Computes the policy portion of a genome output for an observation.

    Any trailing value output (see
    :attr:`ReinforcementLearningTrainer.n_value_outputs`) is sliced off, so
    the result has length ``environment.n_policy_outputs``.

    Args:
        genome: The genome policy.
        environment: The environment (provides observation encoding).
        observation: A raw environment observation.

    Returns:
        The policy outputs of shape ``(environment.n_policy_outputs,)``.
    """

    output = genome.forward(environment.encode(observation))
    return output[: environment.n_policy_outputs]


def split_policy_value(
    output: Tensor, environment: RLEnvironment
) -> tuple[Tensor, Tensor]:
    """Splits a genome output into its policy part and a scalar state value.

    Used by advantage methods (actor-critic, PPO) whose decoder produces one
    extra output beyond the policy: the leading ``environment.n_policy_outputs``
    entries are the policy, and the next entry is the state-value estimate.

    Args:
        output: The genome's raw output vector of shape
            ``(environment.n_policy_outputs + 1,)``.
        environment: The environment whose action space sizes the policy part.

    Returns:
        A tuple ``(policy_part, value)`` where ``policy_part`` has shape
        ``(environment.n_policy_outputs,)`` and ``value`` is a scalar tensor.
    """

    n = environment.n_policy_outputs
    return output[:n], output[n]


@torch.no_grad()
def greedy_action(
    genome: CircuitGenome, environment: RLEnvironment, observation: Any
) -> Any:
    """Selects the greedy (deterministic) action for an observation.

    For a discrete environment this is the argmax over the policy logits. For
    a continuous environment it is the distribution mean, clipped to the
    action bounds. In both cases the result is already in the environment's
    native ``env.step`` format.

    Args:
        genome: The genome policy.
        environment: The environment (provides observation encoding and action
            metadata).
        observation: A raw environment observation.

    Returns:
        The greedy action in the environment's native format (an ``int`` for
        discrete spaces, a ``float32`` NumPy array for continuous spaces).
    """

    part = policy_output(genome, environment, observation)
    if environment.continuous:
        mean = part[: environment.n_actions]
        return to_env_action(mean, environment)
    return int(torch.argmax(part).item())


# ---------------------------------------------------------------------------
# Resolved hyperparameters
# ---------------------------------------------------------------------------


#: Fallback values for every reinforcement-learning training hyperparameter.
#:
#: This is the single source of truth for RL hyperparameter defaults. Each
#: genome carries its own values in ``genome.hyperparameters`` (so the
#: evolutionary search can mutate them per genome), and
#: :meth:`ReinforcementLearningTrainer.resolve_hyperparameters` falls back to
#: the value here whenever a genome does not specify a key. The example scripts
#: populate a genome's ``hyperparameters`` from the command line, so these
#: defaults normally only matter for genomes (e.g. in tests) built without a
#: full configuration.
#:
#: Keys:
#:     episodes: Number of training episodes (outer-loop iterations).
#:     learning_rate: Adam learning rate.
#:     gamma: Reward discount factor.
#:     max_steps: Maximum number of steps per episode.
#:     eval_episodes: Number of episodes used for greedy evaluation.
#:     seed: Base random seed.
#:     log_every: Logging / evaluation frequency, in episodes.
#:     ema_alpha: Smoothing factor for the exponential moving average (EMA) of
#:         episode returns reported as the training return mean. Each episode
#:         updates ``ema = alpha * return + (1 - alpha) * ema``; a smaller alpha
#:         tracks more slowly and smoothly.
#:     baseline: REINFORCE advantage baseline (``"mean"`` or ``"none"``).
#:     entropy_coef: Entropy-bonus coefficient.
#:     value_coef: Weight on the value loss (actor-critic / PPO).
#:     gae_lambda: Generalized Advantage Estimation lambda (PPO).
#:     rollout_steps: Environment steps collected per PPO episode.
#:     ppo_passes: Passes over a collected PPO rollout; each pass performs
#:         several minibatch epochs (weight updates).
#:     ppo_minibatch: PPO minibatch size (steps per weight update).
#:     ppo_clip: PPO clip range.
#:     epsilon: Initial epsilon for epsilon-greedy exploration (value-based).
#:     epsilon_min: Minimum epsilon.
#:     epsilon_decay: Per-episode multiplicative epsilon decay.
#:     improvement_cutoff: Number of episodes without an improvement in the best
#:         evaluation return after which training stops early.
#:     quantum_dropout: Master switch for quantum dropout during training. When
#:         false no quantum dropout is ever applied; when true it is sampled per
#:         training episode from the genome's ``quantum_dropout_type`` /
#:         ``quantum_dropout_rate`` and never applied during greedy evaluation.
RL_HYPERPARAMETER_DEFAULTS: dict[str, Any] = {
    "episodes": 60,
    "learning_rate": 1e-2,
    "gamma": 0.99,
    "max_steps": 500,
    "eval_episodes": 10,
    "seed": 0,
    "log_every": 10,
    "ema_alpha": 0.01,
    "baseline": "mean",
    "entropy_coef": 0.0,
    "value_coef": 0.5,
    "gae_lambda": 0.95,
    "rollout_steps": 512,
    "ppo_passes": 4,
    "ppo_minibatch": 128,
    "ppo_clip": 0.2,
    "epsilon": 0.2,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "improvement_cutoff": 30,
    "quantum_dropout": False,
}


# ---------------------------------------------------------------------------
# Small self-contained RL math helpers
# ---------------------------------------------------------------------------


def discounted_returns(rewards: list[float], gamma: float) -> Tensor:
    """Computes discounted Monte-Carlo returns for a reward sequence.

    Args:
        rewards: List of scalar step rewards, in time order.
        gamma: Discount factor.

    Returns:
        A float tensor of shape ``(len(rewards),)`` of discounted returns.
    """

    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def gae_advantages(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Computes Generalized Advantage Estimation advantages and returns.

    Args:
        rewards: Reward tensor of shape ``(T,)``.
        values: Value estimates of shape ``(T,)``.
        dones: Done flags of shape ``(T,)`` with values in ``{0.0, 1.0}``.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        A tuple ``(advantages, returns)``, each of shape ``(T,)``, where
        ``returns = advantages + values``.
    """

    n_steps = rewards.numel()
    advantages = torch.zeros(n_steps, dtype=torch.float32)
    last_advantage = 0.0
    next_value = 0.0

    for t in reversed(range(n_steps)):
        mask = 1.0 - float(dones[t].item())
        delta = (
            float(rewards[t].item())
            + gamma * next_value * mask
            - float(values[t].item())
        )
        last_advantage = delta + gamma * lam * mask * last_advantage
        advantages[t] = last_advantage
        next_value = float(values[t].item())

    returns = advantages + values
    return advantages, returns


def _normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalizes a tensor to zero mean and unit variance.

    Args:
        x: Input tensor.
        eps: Numerical-stability constant.

    Returns:
        The normalized tensor (unchanged if it has fewer than two elements).
    """

    if x.numel() < 2:
        return x
    return (x - x.mean()) / (x.std() + eps)


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------


class ReinforcementLearningTrainer(ABC):
    """Base class for reinforcement-learning trainers over circuit genomes.

    Subclasses implement a single algorithm update in :meth:`run_update`, and
    declare via :attr:`n_value_outputs` how many extra decoder outputs the
    algorithm needs (e.g. a scalar state value). This base class owns the
    generic training scaffold that mirrors ``SupervisedTrainer.train``:

    * initialize the genome's hybrid model,
    * short-circuit to evaluation-only when there are no trainable
      parameters,
    * build the optimizer over the genome's parameters,
    * run the algorithm for ``episodes`` episodes, periodically evaluating and
      snapshotting the best-performing weights,
    * restore the best weights and record metrics into ``genome.metadata``.

    See the module docstring for how the "step", "episode", and "epoch"
    (one weight update) terms are used.

    Hyperparameters are read entirely from ``genome.hyperparameters`` (so the
    evolutionary search can mutate them per genome), falling back to
    :data:`RL_HYPERPARAMETER_DEFAULTS` for any key a genome does not specify --
    see :meth:`resolve_hyperparameters`. The base trainer therefore holds no
    per-run configuration and needs no constructor; subclasses that carry an
    algorithm *variant* (rather than a hyperparameter) may still define one --
    e.g. :class:`~src.trainer.q_learning_trainer.QLearningTrainer` takes a
    ``sarsa`` flag selecting the on-policy target.

    Class Attributes:
        n_value_outputs: How many extra decoder outputs (beyond the policy
            outputs) this algorithm needs. ``0`` for policy-only / value-based
            methods; ``1`` for advantage methods that read a scalar state value
            out of the decoder. The example script sizes the genome's decoder
            as ``environment.n_policy_outputs + n_value_outputs``.
        supports_continuous: Whether the algorithm supports continuous
            (``Box``) action spaces. Policy-gradient trainers (REINFORCE,
            actor-critic, PPO) do; value-based trainers (Q-learning / SARSA),
            which enumerate discrete actions via argmax / epsilon-greedy, do
            not. :meth:`train` raises a clear error when a continuous
            environment is paired with a trainer that does not support it.
    """

    #: Extra decoder outputs required beyond the policy outputs (see above).
    n_value_outputs: int = 0

    #: Whether the algorithm supports continuous (``Box``) action spaces.
    supports_continuous: bool = True

    # -- hooks for subclasses -------------------------------------------------

    @abstractmethod
    def run_update(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        optimizer: torch.optim.Optimizer,
        episode_index: int,
        hp: SimpleNamespace,
    ) -> tuple[float, dict[str, float]]:
        """Runs one outer-loop training episode and its weight update(s).

        Called once per outer-loop episode by :meth:`train`. A subclass
        collects experience by rolling one or more episodes through the
        environment and performs one or more epochs (weight updates); see the
        module docstring for the per-algorithm breakdown.

        Args:
            genome: The genome being trained.
            environment: The environment being trained on.
            optimizer: The optimizer over the genome's parameters.
            episode_index: Zero-based index of this training episode, used to
                seed the environment reset.
            hp: Resolved hyperparameters.

        Returns:
            A tuple ``(episode_return, info)`` where ``episode_return`` is a
            representative episode return for logging/tracking (the mean over
            collected episodes for PPO) and ``info`` is a dict of extra scalar
            metrics recorded per episode.
        """

    # -- shared helpers -------------------------------------------------------

    def resolve_hyperparameters(self, genome: CircuitGenome) -> SimpleNamespace:
        """Resolves the hyperparameters to use for training a genome.

        Each hyperparameter is taken from the genome's ``hyperparameters`` dict
        when present (so the evolutionary search can mutate it per genome), and
        otherwise from :data:`RL_HYPERPARAMETER_DEFAULTS`. Only the keys in the
        defaults are read, so unrelated entries in the genome's dict (e.g. the
        quantum input/output modes) are ignored. This is the same resolution
        :meth:`train` performs internally; it is public so callers can build the
        attribute bag a single :meth:`run_update` needs (e.g. for a custom loop
        or a unit test) without reaching into private state.

        Args:
            genome: The genome whose ``hyperparameters`` dict is consulted.

        Returns:
            A fresh :class:`types.SimpleNamespace` with one attribute per key of
            :data:`RL_HYPERPARAMETER_DEFAULTS` (so fields are accessed as
            ``hp.episodes``, ``hp.gamma``, and so on).
        """

        source = getattr(genome, "hyperparameters", {}) or {}
        return SimpleNamespace(
            **{
                name: source.get(name, default)
                for name, default in RL_HYPERPARAMETER_DEFAULTS.items()
            }
        )

    def policy_logits(
        self, genome: CircuitGenome, environment: RLEnvironment, observation: Any
    ) -> Tensor:
        """Computes policy logits (or Q-values) for a raw observation.

        Only the per-action outputs are returned; if the decoder also carries
        a trailing value output (see :attr:`n_value_outputs`) it is sliced
        off here.

        Args:
            genome: The genome policy.
            environment: The environment (provides observation encoding).
            observation: A raw environment observation.

        Returns:
            The genome's per-action output vector of shape ``(n_actions,)``.
        """

        output = genome.forward(environment.encode(observation))
        return output[: environment.n_actions]

    @torch.no_grad()
    def evaluate(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        hp: SimpleNamespace,
    ) -> dict[str, float]:
        """Evaluates the genome greedily over several episodes.

        Because evaluation is greedy (deterministic policy), the only source
        of variation between episodes is the environment. For a deterministic
        environment (``environment.deterministic``) every episode is therefore
        identical, so a single episode is run instead of ``eval_episodes``
        redundant copies.

        Args:
            genome: The genome policy to evaluate.
            environment: The environment to evaluate on.
            hp: Resolved hyperparameters.

        Returns:
            A dict with ``return_mean``, ``return_std`` and
            ``best_episode_return``.
        """

        # Evaluation is always greedy on the complete evolved circuit; make
        # sure any dropout sampled during a training episode is cleared first.
        genome.clear_quantum_dropout()

        n_episodes = 1 if environment.deterministic else hp.eval_episodes

        returns: list[float] = []
        for episode in range(n_episodes):
            env = environment.make()
            observation, _ = env.reset(seed=hp.seed + 10_000 + episode)
            episode_return = 0.0
            for _ in range(hp.max_steps):
                action = greedy_action(genome, environment, observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            env.close()
            returns.append(episode_return)

        return {
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "return_std": float(np.std(returns)) if returns else 0.0,
            "best_episode_return": float(np.max(returns)) if returns else 0.0,
        }

    # -- main entry point -----------------------------------------------------

    def train(self, genome: CircuitGenome, environment: RLEnvironment) -> None:
        """Trains a genome on an environment and records metrics.

        Runs ``hp.episodes`` training episodes (each delegating to
        :meth:`run_update`), evaluates periodically, and restores the
        best-evaluated weights. On completion the genome's ``metadata``
        contains ``training_episode_metrics`` (per-episode returns),
        ``best_training_metrics`` and ``best_validation_metrics``.

        Args:
            genome: The genome to train (its model is initialized here).
            environment: The environment to train on.

        Raises:
            ValueError: If ``environment`` is continuous but this trainer does
                not support continuous action spaces
                (:attr:`supports_continuous` is False).
        """

        if environment.continuous and not self.supports_continuous:
            raise ValueError(
                f"{type(self).__name__} does not support continuous action "
                f"spaces (environment {environment.env_id!r} is continuous); "
                "use a policy-gradient trainer (reinforce, actor_critic, ppo)."
            )

        hp = self.resolve_hyperparameters(genome)

        genome.initialize_model()

        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)

        # All trainable parameters live in the genome's hybrid model (encoder,
        # quantum layer, decoder) -- including the value output for advantage
        # methods, which is an extra decoder row. There is no external head to
        # optimize, so everything trained here is also evolved by crossover
        # and preserved through genome serialization.
        trainable_parameters = list(genome.parameters())

        genome.metadata["training_episode_metrics"] = []
        genome.metadata["evaluation_episode_metrics"] = []

        n_trainable = sum(p.numel() for p in trainable_parameters if p.requires_grad)

        # The quantum weight vector carries one entry per gate parameter for
        # *every* gate, including disabled ones, but disabled gates are skipped
        # in the forward pass, so their parameters are never connected to the
        # loss. Counting them would send a genome whose only parameterized gates
        # are disabled down the training path, where backward() fails with
        # "element 0 of tensors does not require grad". Base the train-vs-eval
        # decision on the parameters that are actually used (encoder/decoder
        # plus enabled gates).
        disabled_gate_parameters = sum(
            len(gate.parameters) for gate in genome.gates if not gate.enabled
        )
        effective_trainable = n_trainable - disabled_gate_parameters

        if effective_trainable == 0:
            # nothing connected to the loss to optimize -- just evaluate
            logger.info(
                "genome has no trainable (enabled) parameters; evaluating only."
            )
            evaluation = self.evaluate(genome, environment, hp)
            genome.metadata["best_training_metrics"] = {
                "return_mean": evaluation["return_mean"],
                "best_episode_return": evaluation["best_episode_return"],
            }
            genome.metadata["best_validation_metrics"] = evaluation
            return

        optimizer = torch.optim.Adam(
            trainable_parameters, lr=hp.learning_rate, weight_decay=0.0
        )

        recent_returns: list[float] = []
        # Exponential moving average of episode returns, reported as the
        # training return mean. Seeded with the first episode's return (no
        # cold-start-at-zero bias), then updated as
        # ``ema = alpha * return + (1 - alpha) * ema`` each episode.
        ema_return: Optional[float] = None
        best_return = -math.inf
        best_state = genome.clone_state_dict()
        best_evaluation = None
        eval_every = max(1, hp.log_every)
        best_episode = 0

        for episode in range(hp.episodes):
            # Sample fresh quantum dropout for this training episode (a no-op
            # when the toggle is off). Evaluation clears it so greedy rollouts
            # always use the complete circuit.
            if hp.quantum_dropout:
                sample_quantum_dropout(genome)

            episode_return, info = self.run_update(
                genome, environment, optimizer, episode, hp
            )
            recent_returns.append(episode_return)
            ema_return = (
                episode_return
                if ema_return is None
                else hp.ema_alpha * episode_return + (1.0 - hp.ema_alpha) * ema_return
            )

            episode_metrics = {"episode": episode, "return": episode_return}
            episode_metrics.update(info)
            genome.metadata["training_episode_metrics"].append(episode_metrics)

            if (episode % eval_every == 0) or (episode == hp.episodes - 1):
                evaluation = self.evaluate(genome, environment, hp)
                evaluation["episode"] = episode

                # track evaluations for visualization purposes
                genome.metadata["evaluation_episode_metrics"].append(evaluation)

                logger.info(
                    f"[{type(self).__name__}] genome {genome.genome_number:4d} episode {episode:4d} "
                    f"train_return={episode_return:.1f} "
                    f"train_return_ema={ema_return:.1f} "
                    f"eval_return_mean={evaluation['return_mean']:.1f}"
                )

                if evaluation["return_mean"] > best_return:
                    best_return = evaluation["return_mean"]
                    best_evaluation = evaluation
                    best_state = genome.clone_state_dict()
                    best_episode = episode

                elif (
                    hp.improvement_cutoff > 0
                    and episode - best_episode > hp.improvement_cutoff
                ):
                    logger.info(
                        "Stopping at episode {} because the last improvement "
                        "occurred at episode {}.",
                        episode,
                        best_episode,
                    )
                    break

        # restore the best-evaluated weights into the genome
        genome.set_state_dict(best_state)

        genome.metadata["best_episode"] = best_episode
        genome.metadata["best_training_metrics"] = {
            "return_mean": float(ema_return) if ema_return is not None else 0.0,
            "best_episode_return": (
                float(np.max(recent_returns)) if recent_returns else 0.0
            ),
        }
        genome.metadata["best_validation_metrics"] = (
            best_evaluation
            if best_evaluation is not None
            else self.evaluate(genome, environment, hp)
        )

        # Leave the genome with no active dropout so the returned/serialized
        # policy runs the complete evolved circuit.
        genome.clear_quantum_dropout()
