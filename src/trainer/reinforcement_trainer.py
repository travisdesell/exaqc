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

import argparse

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

import secrets
import numpy as np
import torch

from loguru import logger
from torch import Tensor
from torch.distributions import Categorical, Distribution, Normal

import gymnasium as gym

from src.circuits.circuit import CircuitGenome
from src.dropout.quantum_dropout import sample_quantum_dropout
from src.trainer.running_normalizers import ObservationNormalizer

#: Bounds applied to the per-dimension log-standard-deviation a continuous
#: (Gaussian) policy reads from the decoder, keeping the sampled action scale
#: numerically sane regardless of the unconstrained decoder outputs.
#: These are the defaults of ``--log_std_min`` / ``--log_std_max``.
LOG_STD_MIN: float = -5.0
LOG_STD_MAX: float = 2.0

#: Key under which a genome's observation-normalizer statistics are kept in
#: ``genome.metadata`` (see :func:`encode_observation`), so evaluation, replay
#: and refinement normalize exactly as training did.
OBSERVATION_NORMALIZER_KEY: str = "observation_normalizer"

#: Attribute a genome's live observation normalizer is cached under while it is
#: in use; the serialized copy in ``genome.metadata`` is the source of truth.
_OBSERVATION_NORMALIZER_CACHE: str = "_observation_normalizer"

#: Learning-rate schedules ``--lr_anneal`` accepts: ``"none"`` keeps the rate
#: fixed; ``"linear"`` decays it linearly from ``learning_rate`` towards 0 over
#: the ``episodes`` outer-loop episodes.
LR_ANNEAL_CHOICES: tuple[str, ...] = ("none", "linear")

#: Action-selection regimes :meth:`ReinforcementLearningTrainer.evaluate` can
#: score a genome under.
#:
#: * ``"greedy"`` -- take the deterministic action: the argmax over the policy
#:   logits (or Q-values), or the distribution mean for a continuous policy.
#:   This is the default for every trainer, matching the common convention for
#:   evaluating a learned policy (e.g. Stable-Baselines3's ``evaluate_policy``
#:   and ``EvalCallback``, and the RL Zoo outside Atari). It also keeps
#:   evaluation from adding sampling noise to fitness or penalizing a genome
#:   for the exploration noise (``log_std``) its training relies on.
#: * ``"stochastic"`` -- sample from the policy, the regime the policy-gradient
#:   trainers collect their rollouts under. Only trainers with a policy
#:   distribution support it (see
#:   :attr:`ReinforcementLearningTrainer.supports_stochastic_eval`).
#: * ``"both"`` -- score under both regimes; greedy drives fitness and the
#:   stochastic return is recorded alongside it as a diagnostic. Costs one
#:   extra set of evaluation episodes.
EVAL_POLICY_CHOICES: tuple[str, ...] = ("greedy", "stochastic", "both")

#: Width of the seed block a trainer whose outer episode spans several
#: environment episodes (PPO) reserves per outer episode, so one outer
#: episode's seeds cannot run into the next one's.
SEED_BLOCK: int = 10_000

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
    policy_part: Tensor,
    environment: RLEnvironment,
    log_std_min: float = LOG_STD_MIN,
    log_std_max: float = LOG_STD_MAX,
) -> Distribution:
    """Builds the action distribution from the policy portion of a genome output.

    For a discrete environment this is a ``Categorical`` over ``n_actions``
    logits. For a continuous environment the ``2 * n_actions`` policy outputs
    are split into a per-dimension mean and (clamped) log-standard-deviation,
    yielding a diagonal ``Normal``.

    The outputs are read along the last dimension, so a batch of shape
    ``(batch_size, n_policy_outputs)`` yields a batch of distributions.

    Args:
        policy_part: The leading ``environment.n_policy_outputs`` entries of a
            genome output (the policy, with any value output already sliced
            off), for a single sample or a batch.
        environment: The environment whose action space determines the
            distribution family.
        log_std_min: Lower clamp on a continuous policy's log-std.
        log_std_max: Upper clamp on a continuous policy's log-std (see
            :func:`log_std_bounds` for a genome's own bounds).

    Returns:
        A ``torch.distributions`` distribution to sample actions from.
    """

    if environment.continuous:
        n = environment.n_actions
        mean = policy_part[..., :n]
        log_std = torch.clamp(policy_part[..., n : 2 * n], log_std_min, log_std_max)
        return Normal(mean, log_std.exp())
    return Categorical(logits=policy_part)


def distribution_log_prob(distribution: Distribution, action: Tensor) -> Tensor:
    """Returns a scalar log-probability for an action under a distribution.

    A ``Categorical`` already yields one log-probability per sample; a diagonal
    ``Normal`` yields one per action dimension, which are summed into the joint
    log-probability. Only the ``Normal``'s action dimension is summed, so a
    batch of ``Categorical`` log-probabilities keeps its batch dimension.

    Args:
        distribution: The action distribution (single or batched).
        action: The sampled action tensor (single or batched).

    Returns:
        A scalar log-probability tensor, or one of shape ``(batch_size,)``.
    """

    log_prob = distribution.log_prob(action)
    return log_prob.sum(-1) if isinstance(distribution, Normal) else log_prob


def distribution_entropy(distribution: Distribution) -> Tensor:
    """Returns a scalar entropy for a distribution.

    As with :func:`distribution_log_prob`, a diagonal ``Normal``'s
    per-dimension entropy is summed into a single scalar.

    Args:
        distribution: The action distribution (single or batched).

    Returns:
        A scalar entropy tensor, or one of shape ``(batch_size,)``.
    """

    entropy = distribution.entropy()
    return entropy.sum(-1) if isinstance(distribution, Normal) else entropy


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


def observation_normalizer(
    genome: CircuitGenome, n_features: int
) -> ObservationNormalizer | None:
    """Returns the genome's observation normalizer, if it normalizes at all.

    Normalization is switched on per genome by its ``normalize_observations``
    hyperparameter (``--normalize_observations``), so a saved genome is
    replayed the way it was trained. The live normalizer is cached on the
    genome; on first use it is rebuilt from the statistics saved in
    ``genome.metadata`` (inherited from a parent, or written at the end of the
    genome's own training), or started fresh at mean 0 and variance 1.

    Args:
        genome: The genome whose policy is being run.
        n_features: Length of the encoded observation, used to size a fresh
            normalizer.

    Returns:
        The normalizer, or ``None`` when the genome does not normalize
        observations.
    """

    hyperparameters = getattr(genome, "hyperparameters", {}) or {}
    if not hyperparameters.get("normalize_observations", False):
        return None

    normalizer = getattr(genome, _OBSERVATION_NORMALIZER_CACHE, None)
    if normalizer is None:
        saved = (getattr(genome, "metadata", {}) or {}).get(OBSERVATION_NORMALIZER_KEY)
        if saved is not None and len(saved["mean"]) == n_features:
            normalizer = ObservationNormalizer.from_dict(saved)
        else:
            normalizer = ObservationNormalizer(n_features)
        setattr(genome, _OBSERVATION_NORMALIZER_CACHE, normalizer)
    return normalizer


def reset_observation_normalizer(genome: CircuitGenome) -> None:
    """Drops the genome's cached normalizer so it reloads from metadata.

    Args:
        genome: The genome whose cache is cleared.

    Returns:
        None. Clears the cached normalizer attribute on ``genome``.
    """

    setattr(genome, _OBSERVATION_NORMALIZER_CACHE, None)


def snapshot_observation_normalizer(genome: CircuitGenome) -> dict[str, Any] | None:
    """Captures the genome's current observation-normalizer statistics.

    Args:
        genome: The genome whose statistics are captured.

    Returns:
        A serialized copy of the live normalizer when one is cached, otherwise
        the statistics already saved in ``genome.metadata`` (``None`` if
        there are none).
    """

    normalizer = getattr(genome, _OBSERVATION_NORMALIZER_CACHE, None)
    if normalizer is not None:
        return normalizer.to_dict()
    saved = (getattr(genome, "metadata", {}) or {}).get(OBSERVATION_NORMALIZER_KEY)
    return None if saved is None else dict(saved)


def restore_observation_normalizer(
    genome: CircuitGenome, snapshot: dict[str, Any] | None
) -> None:
    """Saves a statistics snapshot into the genome's metadata.

    Args:
        genome: The genome to update.
        snapshot: Statistics from :func:`snapshot_observation_normalizer`, or
            ``None`` to leave the genome without saved statistics.

    Returns:
        None. Sets (or removes) ``genome.metadata[OBSERVATION_NORMALIZER_KEY]``
        and clears the cached normalizer so it is rebuilt from that.
    """

    if snapshot is None:
        genome.metadata.pop(OBSERVATION_NORMALIZER_KEY, None)
    else:
        genome.metadata[OBSERVATION_NORMALIZER_KEY] = snapshot
    reset_observation_normalizer(genome)


def encode_observation(
    genome: CircuitGenome,
    environment: RLEnvironment,
    observation: Any,
    *,
    update: bool = False,
) -> Tensor:
    """Encodes a raw observation into the genome's input, normalizing if asked.

    The environment's observation encoder runs first; when the genome
    normalizes observations (see :func:`observation_normalizer`) the result is
    then standardized with the running statistics and clipped.

    Args:
        genome: The genome whose policy will consume the encoding.
        environment: The environment (provides the observation encoder).
        observation: A raw environment observation.
        update: Whether to fold this observation into the running statistics
            before normalizing it. Training rollouts update; evaluation and
            replay do not.

    Returns:
        A float tensor of shape ``(n_observation_features,)``.
    """

    encoded = environment.encode(observation)
    normalizer = observation_normalizer(genome, int(encoded.shape[-1]))
    if normalizer is None:
        return encoded

    values = encoded.detach().cpu().numpy().astype(np.float64)
    if update:
        normalizer.update(values.reshape(1, -1))
    return torch.as_tensor(normalizer.normalize(values), dtype=torch.float32)


def log_std_bounds(genome: CircuitGenome) -> tuple[float, float]:
    """Returns the log-standard-deviation clamp a genome's policy uses.

    Read from the genome's ``log_std_min`` / ``log_std_max`` hyperparameters
    (``--log_std_min`` / ``--log_std_max``), so a saved genome is replayed with
    the bounds it was trained under.

    Args:
        genome: The genome whose policy is being run.

    Returns:
        ``(log_std_min, log_std_max)``.

    Raises:
        ValueError: If ``log_std_min`` is not below ``log_std_max``.
    """

    hyperparameters = getattr(genome, "hyperparameters", {}) or {}
    low = float(hyperparameters.get("log_std_min", LOG_STD_MIN))
    high = float(hyperparameters.get("log_std_max", LOG_STD_MAX))
    if not low < high:
        raise ValueError(
            f"log_std_min ({low}) must be smaller than log_std_max ({high})."
        )
    return low, high


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

    output = genome.forward(encode_observation(genome, environment, observation))
    return output[..., : environment.n_policy_outputs]


def split_policy_value(
    output: Tensor, environment: RLEnvironment
) -> tuple[Tensor, Tensor]:
    """Splits a genome output into its policy part and a scalar state value.

    Used by advantage methods (actor-critic, PPO) whose decoder produces one
    extra output beyond the policy: the leading ``environment.n_policy_outputs``
    entries are the policy, and the next entry is the state-value estimate.

    Args:
        output: The genome's raw output vector of shape
            ``(environment.n_policy_outputs + 1,)``, or a batch of them of shape
            ``(batch_size, environment.n_policy_outputs + 1)``.
        environment: The environment whose action space sizes the policy part.

    Returns:
        A tuple ``(policy_part, value)`` where ``policy_part`` has shape
        ``(environment.n_policy_outputs,)`` and ``value`` is a scalar tensor
        (with a leading ``batch_size`` dimension on both for a batch).
    """

    n = environment.n_policy_outputs
    return output[..., :n], output[..., n]


@torch.no_grad()
def sample_from(
    distribution: Distribution, generator: torch.Generator | None = None
) -> Tensor:
    """Draws an action from an action distribution, optionally reproducibly.

    ``torch.distributions`` sampling always draws from PyTorch's global
    generator. Evaluation instead passes a generator of its own, seeded from the
    evaluation seed, so an evaluation's sampling noise is determined by that
    seed alone -- the same for every genome evaluated on it, and independent of
    (and invisible to) the training randomness. The draw has the same
    distribution either way.

    Args:
        distribution: A ``Normal`` or ``Categorical`` from
            :func:`action_distribution`.
        generator: A CPU generator to draw from, or ``None`` to use the global
            one.

    Returns:
        The sampled action tensor, shaped as ``distribution.sample()`` would
        return it.
    """

    if generator is None:
        return distribution.sample()

    if isinstance(distribution, Normal):
        noise = torch.randn(distribution.loc.shape, generator=generator)
        noise = noise.to(device=distribution.loc.device, dtype=distribution.loc.dtype)
        return distribution.loc + distribution.scale * noise

    probabilities = distribution.probs.detach().cpu()
    index = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    return index.to(distribution.probs.device)


def select_action(
    genome: CircuitGenome,
    environment: RLEnvironment,
    observation: Any,
    *,
    stochastic: bool,
    generator: torch.Generator | None = None,
) -> Any:
    """Selects an action for an observation under one action-selection regime.

    Both regimes read the same policy outputs, so this is the single place the
    policy is turned into an action; :func:`greedy_action` and
    :func:`stochastic_action` are thin wrappers over it. Keeping them together
    matters because the two regimes score *different policies*: the greedy
    action of a continuous environment ignores the learned
    log-standard-deviation half of the policy output entirely, while the
    stochastic action is drawn using it.

    Args:
        genome: The genome policy.
        environment: The environment (provides observation encoding and action
            metadata).
        observation: A raw environment observation.
        stochastic: When True, sample from the policy distribution (the
            behaviour the policy-gradient trainers collect rollouts under).
            When False, take the deterministic action: the argmax over the
            policy logits for a discrete environment, or the distribution
            mean for a continuous one.
        generator: For a stochastic action, the generator to sample from (see
            :func:`sample_from`); ``None`` uses PyTorch's global generator.
            Ignored for a deterministic action.

    Returns:
        The action in the environment's native ``env.step`` format (an ``int``
        for discrete spaces, a clipped ``float32`` NumPy array for continuous
        spaces).
    """

    part = policy_output(genome, environment, observation)

    if stochastic:
        action = sample_from(
            action_distribution(part, environment, *log_std_bounds(genome)),
            generator,
        )
        return to_env_action(action, environment)

    if environment.continuous:
        mean = part[: environment.n_actions]
        return to_env_action(mean, environment)
    return int(torch.argmax(part).item())


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

    return select_action(genome, environment, observation, stochastic=False)


def stochastic_action(
    genome: CircuitGenome, environment: RLEnvironment, observation: Any
) -> Any:
    """Samples an action from the policy for an observation.

    This is the regime the policy-gradient trainers collect their rollouts
    under, so it is what their objective actually optimizes: a ``Categorical``
    draw over the policy logits for a discrete environment, or a draw from the
    diagonal ``Normal`` built by :func:`action_distribution` (clipped to the
    action bounds) for a continuous one.

    Args:
        genome: The genome policy.
        environment: The environment (provides observation encoding and action
            metadata).
        observation: A raw environment observation.

    Returns:
        The sampled action in the environment's native format (an ``int`` for
        discrete spaces, a ``float32`` NumPy array for continuous spaces).
    """

    return select_action(genome, environment, observation, stochastic=True)


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
#:     adam_beta1: Adam first-moment decay rate (PyTorch's default).
#:     adam_beta2: Adam second-moment decay rate (PyTorch's default).
#:     adam_epsilon: Adam numerical-stability term (PyTorch's default).
#:     gamma: Reward discount factor.
#:     max_steps: Maximum number of steps per episode.
#:     eval_episodes: Number of episodes used to evaluate a genome's return.
#:     eval_policy: Action-selection regime evaluation scores a genome under,
#:         one of :data:`EVAL_POLICY_CHOICES`; ``"greedy"`` by default.
#:     training_seed: Base random seed for *training* episodes, PyTorch and
#:         NumPy, or ``None`` (the default) to draw a fresh random seed each
#:         time a genome's hyperparameters are resolved, so genomes do not all
#:         train on the same episodes. :meth:`ReinforcementLearningTrainer.train`
#:         records the seed a genome used in its ``training_seed`` metadata.
#:     eval_seed: Base random seed for *evaluation* episodes, kept in a range
#:         disjoint from the training seeds so a genome is never scored on an
#:         episode it trained on (see
#:         :meth:`ReinforcementLearningTrainer._separate_evaluation_seed`).
#:         Evaluation episode ``i`` resets from ``eval_seed + i`` and, under a
#:         stochastic policy, samples its actions from a generator seeded with
#:         the same value, so an evaluation is fully determined by the genome
#:         and this seed. ``None`` (the default) draws a fresh one per genome;
#:         setting it evaluates every genome on the same episodes with the same
#:         sampling noise, which makes their fitnesses directly comparable.
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
#:     normalize_observations: Whether to standardize observations with running
#:         statistics before the genome's encoder (see
#:         :func:`encode_observation`).
#:     lr_anneal: Learning-rate schedule, one of :data:`LR_ANNEAL_CHOICES`.
#:     log_std_min: Lower clamp on a continuous policy's log-std.
#:     log_std_max: Upper clamp on a continuous policy's log-std.
#:     ppo_target_kl: PPO stops a rollout's remaining epochs once the approximate
#:         KL divergence from the behavior policy exceeds this; 0 disables it.
#:     ppo_reward_scaling: Whether PPO divides rewards by the running standard
#:         deviation of the discounted return.
#:     quantum_dropout: Master switch for quantum dropout during training. When
#:         false no quantum dropout is ever applied; when true it is sampled per
#:         training episode from the genome's ``quantum_dropout_type`` /
#:         ``quantum_dropout_rate`` and never applied during evaluation.
RL_HYPERPARAMETER_DEFAULTS: dict[str, Any] = {
    "episodes": 60,
    "learning_rate": 1e-2,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "gamma": 0.99,
    "max_steps": 500,
    "eval_episodes": 10,
    "eval_policy": "greedy",
    "training_seed": None,
    "eval_seed": None,
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
    "normalize_observations": False,
    "lr_anneal": "none",
    "log_std_min": LOG_STD_MIN,
    "log_std_max": LOG_STD_MAX,
    "ppo_target_kl": 0.0,
    "ppo_reward_scaling": False,
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
        supports_stochastic_eval: Whether a genome trained by this algorithm
            can be evaluated under the ``"stochastic"`` regime (and so under
            ``"both"``). The policy-gradient trainers can: their outputs
            parameterize the action distribution they sample from. The
            value-based trainers cannot, because their outputs are Q-values,
            and a softmax over Q-values is not a policy they learn.
            :meth:`validate_eval_policy` rejects the combination.
    """

    #: Extra decoder outputs required beyond the policy outputs (see above).
    n_value_outputs: int = 0

    #: Whether the algorithm supports continuous (``Box``) action spaces.
    supports_continuous: bool = True

    #: Whether stochastic evaluation is meaningful for this algorithm (see above).
    supports_stochastic_eval: bool = True

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds the training-loop hyperparameters shared by every RL algorithm.

        These knobs drive the base training scaffold in :meth:`train` (and the
        per-algorithm updates), and become per-genome hyperparameters the search
        can mutate. Each algorithm's own extras are added by that subclass's
        ``initialize_parser`` (e.g.
        :meth:`~src.trainer.ppo_trainer.PPOTrainer.initialize_parser`); the
        reinforcement-learning entry point calls the base and every subclass so
        the full flag set is available regardless of ``--algo``.

        ``--entropy_coef`` and ``--value_coef`` are read only by the
        policy-gradient algorithms, but live here (rather than being added by two
        different subclasses) so they are registered exactly once.

        Args:
            parser: The parser to add the arguments to.

        Returns:
            None. Mutates ``parser`` by adding ``--episodes``,
            ``--eval_episodes``, ``--eval_policy``, ``--max_steps``,
            ``--gamma``, ``--learning_rate``/``-lr``, ``--adam_beta1``,
            ``--adam_beta2``, ``--adam_epsilon``, ``--entropy_coef``,
            ``--value_coef``, ``--eval_seed``, ``--training_seed``, ``--log_every``,
            ``--ema_alpha``, ``--improvement_cutoff``,
            ``--normalize_observations``, ``--lr_anneal``, ``--log_std_min``
            and ``--log_std_max``.
        """

        parser.add_argument(
            "--episodes",
            type=int,
            default=60,
            help=(
                "Number of training episodes (outer-loop iterations) per genome, after the "
                "episode-0 evaluation of its inherited weights; 0 scores the inherited weights only."
            ),
        )

        parser.add_argument(
            "--eval_episodes",
            type=int,
            default=10,
            help="Number of episodes used to evaluate a genome's return.",
        )

        parser.add_argument(
            "--eval_policy",
            choices=EVAL_POLICY_CHOICES,
            default="greedy",
            help=(
                "Action-selection regime a genome is evaluated under. 'greedy' "
                "(the default) takes the deterministic action: the argmax, or "
                "the distribution mean for a continuous policy. 'stochastic' "
                "samples from the policy (policy-gradient algorithms only). "
                "'both' scores under both, with greedy driving fitness and the "
                "stochastic return recorded as a diagnostic."
            ),
        )

        parser.add_argument(
            "--max_steps",
            type=int,
            default=500,
            help="Maximum number of environment steps per episode.",
        )

        parser.add_argument(
            "--gamma",
            type=float,
            default=0.99,
            help="Reward discount factor.",
        )

        parser.add_argument(
            "--learning_rate",
            "-lr",
            type=float,
            default=1e-2,
            help="Adam learning rate used when training each genome.",
        )

        parser.add_argument(
            "--adam_beta1",
            type=float,
            default=RL_HYPERPARAMETER_DEFAULTS["adam_beta1"],
            help="Adam first-moment decay rate (beta1) used when training each genome.",
        )

        parser.add_argument(
            "--adam_beta2",
            type=float,
            default=RL_HYPERPARAMETER_DEFAULTS["adam_beta2"],
            help="Adam second-moment decay rate (beta2) used when training each genome.",
        )

        parser.add_argument(
            "--adam_epsilon",
            type=float,
            default=RL_HYPERPARAMETER_DEFAULTS["adam_epsilon"],
            help="Adam epsilon (numerical-stability term) used when training each genome.",
        )

        parser.add_argument(
            "--entropy_coef",
            type=float,
            default=0.0,
            help="Coefficient on the policy entropy bonus.",
        )

        parser.add_argument(
            "--value_coef",
            type=float,
            default=0.5,
            help="Weight on the value-function loss (actor-critic and PPO).",
        )

        parser.add_argument(
            "--eval_seed",
            type=int,
            default=None,
            help=(
                "Base seed for the evaluation episodes, kept in a range "
                "disjoint from the training seeds so a genome is never scored "
                "on an episode it trained on. It also seeds a stochastic "
                "evaluation's action sampling, so an evaluation is determined by "
                "the genome and this seed. By default each genome draws its "
                "own; give one to evaluate every genome on the same episodes "
                "with the same sampling noise, which makes their fitnesses "
                "directly comparable."
            ),
        )

        parser.add_argument(
            "--training_seed",
            type=int,
            default=None,
            help=(
                "Base random seed for training episodes, PyTorch, and NumPy. By default each "
                "genome draws its own random seed (recorded as training_seed in its metadata); "
                "give a seed to train every genome on the same episodes. Evaluation episodes are "
                "seeded separately -- see --eval_seed."
            ),
        )

        parser.add_argument(
            "--log_every",
            type=int,
            default=10,
            help="Evaluate and log every this many training episodes.",
        )

        parser.add_argument(
            "--ema_alpha",
            type=float,
            default=0.05,
            help="Smoothing factor for the exponential moving average of episode "
            "returns reported as the training return mean.",
        )

        parser.add_argument(
            "--improvement_cutoff",
            type=int,
            default=30,
            help="Stop training a genome after this many episodes without an improved evaluation return.",
        )

        parser.add_argument(
            "--normalize_observations",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Standardize observations with running mean/std statistics "
                "(clipped to [-10, 10]) before the genome's encoder. The "
                "statistics are updated during training rollouts only and saved "
                "in the genome's metadata, so evaluation and replay use them too."
            ),
        )

        parser.add_argument(
            "--lr_anneal",
            choices=LR_ANNEAL_CHOICES,
            default="none",
            help=(
                "Learning-rate schedule: 'none' keeps --learning_rate fixed; "
                "'linear' decays it linearly towards 0 over --episodes."
            ),
        )

        parser.add_argument(
            "--log_std_min",
            type=float,
            default=LOG_STD_MIN,
            help="Lower clamp on a continuous policy's log standard deviation.",
        )

        parser.add_argument(
            "--log_std_max",
            type=float,
            default=LOG_STD_MAX,
            help=(
                "Upper clamp on a continuous policy's log standard deviation "
                "(0.5 is a common, tighter choice for MuJoCo tasks)."
            ),
        )

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

    def begin_training(self, hp: SimpleNamespace) -> None:
        """Resets any per-training-run state before the first update.

        Called by :meth:`train` once per genome, before its first
        :meth:`run_update`. The base trainer keeps no such state; PPO uses it
        to start a fresh reward scaler.

        Args:
            hp: Resolved hyperparameters.

        Returns:
            None.
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
            ``hp.episodes``, ``hp.gamma``, and so on). A ``seed`` that is
            ``None`` is replaced by a freshly drawn random seed, so every
            resolution can seed episodes directly.
        """

        source = getattr(genome, "hyperparameters", {}) or {}
        hp = SimpleNamespace(
            **{
                name: source.get(name, default)
                for name, default in RL_HYPERPARAMETER_DEFAULTS.items()
            }
        )
        seed_was_given = hp.training_seed is not None
        if not seed_was_given:
            # Drawn from the operating system rather than Python's, NumPy's or
            # PyTorch's generators, which training reseeds: a seed derived from
            # those could repeat from one genome to the next.
            hp.training_seed = secrets.randbits(31)

        self._separate_evaluation_seed(hp, seed_was_given=seed_was_given)
        return hp

    def training_seed_span(self, hp: SimpleNamespace) -> tuple[int, int]:
        """Returns the half-open range of seeds this trainer's training uses.

        Evaluation must not reuse an episode training has already learned from,
        so it needs to know which seeds training consumes. The base scaffold
        rolls one environment episode per outer episode, seeded
        ``hp.training_seed + episode_index``; a trainer whose outer episode
        spans several environment episodes (PPO) overrides this with its own,
        wider range.

        Args:
            hp: Resolved hyperparameters, with ``training_seed`` already drawn.

        Returns:
            ``(first, last_exclusive)`` -- an upper bound is fine and expected,
            since a genome that stops early simply uses fewer of these seeds.
        """

        return hp.training_seed, hp.training_seed + hp.episodes

    def _separate_evaluation_seed(
        self, hp: SimpleNamespace, *, seed_was_given: bool
    ) -> None:
        """Ensures evaluation episodes never reuse a training episode's seed.

        Training and evaluation seeds used to be derived from the same base, so
        they could coincide -- with PPO's ``hp.training_seed + episode_index *
        10_000 + episode``, outer episode 1 landed exactly on the evaluation seeds, and
        a genome was scored on episodes it had just trained on.

        How the overlap is resolved depends on which seeds are pinned:

        * ``eval_seed`` unset -- a fresh random evaluation seed is drawn per
          genome, rejecting any draw that would overlap training. Genomes are
          therefore scored on their own episodes, as they are trained on their
          own.
        * ``eval_seed`` set -- it is shared deliberately, so every genome faces
          the same evaluation episodes (common random numbers). It is left
          alone and the genome's *training* seed is redrawn instead, which is
          free because it was random anyway.
        * both ``eval_seed`` and ``training_seed`` set and overlapping --
          neither can move without breaking what was asked for, so this raises.

        Args:
            hp: Resolved hyperparameters, mutated in place.
            seed_was_given: Whether ``training_seed`` was pinned rather than
                drawn.

        Returns:
            None. Sets ``hp.eval_seed`` (when it was unset) or redraws
            ``hp.training_seed``, so that the two seed ranges are disjoint.

        Raises:
            ValueError: If both seeds are pinned and their ranges overlap.
        """

        def overlaps(candidate: int) -> bool:
            """Reports whether evaluating from ``candidate`` touches training.

            Args:
                candidate: The evaluation base seed to test.

            Returns:
                True when the evaluation range intersects the training range.
            """

            first, last = self.training_seed_span(hp)
            return candidate < last and first < candidate + hp.eval_episodes

        if hp.eval_seed is None:
            hp.eval_seed = secrets.randbits(31)
            while overlaps(hp.eval_seed):
                hp.eval_seed = secrets.randbits(31)
            return

        if not overlaps(hp.eval_seed):
            return

        if seed_was_given:
            first, last = self.training_seed_span(hp)
            raise ValueError(
                f"eval_seed={hp.eval_seed} evaluates on seeds "
                f"[{hp.eval_seed}, {hp.eval_seed + hp.eval_episodes}), which "
                f"overlaps the training seeds [{first}, {last}) that "
                f"training_seed={hp.training_seed} produces, so genomes would be "
                "scored on episodes they trained on. Move --eval_seed clear of "
                "that range, or leave --training_seed unset so it can be drawn "
                "around it."
            )

        # the training seed was drawn, not asked for, so move it instead and
        # keep every genome evaluating on the same episodes
        while overlaps(hp.eval_seed):
            hp.training_seed = secrets.randbits(31)

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

        output = genome.forward(encode_observation(genome, environment, observation))
        return output[: environment.n_actions]

    def validate_eval_policy(self, eval_policy: str) -> None:
        """Checks that an evaluation regime is known and usable by this trainer.

        Args:
            eval_policy: The requested regime, one of
                :data:`EVAL_POLICY_CHOICES`.

        Returns:
            None. Returns normally when the regime can be used.

        Raises:
            ValueError: If ``eval_policy`` is not a known regime, or if it
                samples actions (``"stochastic"`` or ``"both"``) and this
                trainer does not support stochastic evaluation (see
                :attr:`supports_stochastic_eval`).
        """

        if eval_policy not in EVAL_POLICY_CHOICES:
            raise ValueError(
                f"Unknown eval_policy {eval_policy!r}; choices: "
                f"{', '.join(EVAL_POLICY_CHOICES)}."
            )
        if eval_policy != "greedy" and not self.supports_stochastic_eval:
            raise ValueError(
                f"{type(self).__name__} does not support eval_policy "
                f"{eval_policy!r}: its outputs are action values, not a policy "
                "distribution to sample from, so it can only be evaluated "
                "greedily."
            )

    def selected_eval_policy(self, eval_policy: str) -> str:
        """Names the regime whose returns become the genome's fitness.

        This differs from ``eval_policy`` itself only for ``"both"``, which
        scores under two regimes but reports the greedy one as *the*
        evaluation return.

        Args:
            eval_policy: The requested regime, one of
                :data:`EVAL_POLICY_CHOICES`.

        Returns:
            ``"greedy"`` or ``"stochastic"`` -- never ``"both"``.

        Raises:
            ValueError: If ``eval_policy`` is rejected by
                :meth:`validate_eval_policy`.
        """

        self.validate_eval_policy(eval_policy)
        return "greedy" if eval_policy == "both" else eval_policy

    def _evaluate_regime(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        hp: SimpleNamespace,
        *,
        stochastic: bool,
    ) -> list[float]:
        """Rolls the evaluation episodes for one action-selection regime.

        Under a greedy policy the only source of variation between episodes is
        the environment, so a deterministic environment
        (``environment.deterministic``) is rolled once instead of
        ``eval_episodes`` identical copies. That shortcut does **not** apply to
        a stochastic policy, whose own sampling makes every episode differ even
        in a deterministic environment.

        Episode ``i`` is reset from ``hp.eval_seed + i`` and, under a stochastic
        policy, samples its actions from a generator seeded with the same value.
        An evaluation is therefore fully determined by the genome and
        ``eval_seed``: genomes sharing an ``eval_seed`` face the same starting
        states *and* the same sampling noise, and evaluating never draws from
        the global generators training uses.

        Args:
            genome: The genome policy to evaluate.
            environment: The environment to evaluate on.
            hp: Resolved hyperparameters.
            stochastic: Whether to sample actions rather than take the greedy
                action.

        Returns:
            The per-episode returns.
        """

        n_episodes = (
            1 if environment.deterministic and not stochastic else hp.eval_episodes
        )

        returns: list[float] = []
        for episode in range(n_episodes):
            env = environment.make()
            # a seed range kept disjoint from training's; see
            # :meth:`_separate_evaluation_seed`
            episode_seed = hp.eval_seed + episode
            observation, _ = env.reset(seed=episode_seed)
            generator = (
                torch.Generator().manual_seed(episode_seed) if stochastic else None
            )
            episode_return = 0.0
            for _ in range(hp.max_steps):
                action = select_action(
                    genome,
                    environment,
                    observation,
                    stochastic=stochastic,
                    generator=generator,
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            env.close()
            returns.append(episode_return)

        return returns

    @torch.no_grad()
    def evaluate(
        self,
        genome: CircuitGenome,
        environment: RLEnvironment,
        hp: SimpleNamespace,
    ) -> dict[str, float]:
        """Evaluates the genome over several episodes, under ``hp.eval_policy``.

        The regime is one of :data:`EVAL_POLICY_CHOICES`; greedy (the default)
        takes the deterministic action, so a continuous policy is scored by its
        mean and its learned log-standard-deviation does not enter fitness.
        Evaluation never updates weights, so it runs without autograd.

        Args:
            genome: The genome policy to evaluate.
            environment: The environment to evaluate on.
            hp: Resolved hyperparameters.

        Returns:
            A dict with ``return_mean``, ``return_std`` and
            ``best_episode_return`` for the selected regime. Under
            ``eval_policy="both"`` it also carries ``greedy_return_mean`` and
            ``stochastic_return_mean``, with the greedy regime providing the
            selected values.

        Raises:
            ValueError: If ``hp.eval_policy`` is rejected by
                :meth:`validate_eval_policy`.
        """

        # Evaluation always runs the complete evolved circuit; make sure any
        # dropout sampled during a training episode is cleared first.
        genome.clear_quantum_dropout()

        policy = hp.eval_policy
        selected = self.selected_eval_policy(policy)
        regimes = ("greedy", "stochastic") if policy == "both" else (policy,)

        by_regime: dict[str, list[float]] = {
            regime: self._evaluate_regime(
                genome, environment, hp, stochastic=regime == "stochastic"
            )
            for regime in regimes
        }

        returns = by_regime[selected]
        results = {
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "return_std": float(np.std(returns)) if returns else 0.0,
            "best_episode_return": float(np.max(returns)) if returns else 0.0,
        }

        if policy == "both":
            for regime, regime_returns in by_regime.items():
                results[f"{regime}_return_mean"] = (
                    float(np.mean(regime_returns)) if regime_returns else 0.0
                )

        return results

    # -- main entry point -----------------------------------------------------

    def train(self, genome: CircuitGenome, environment: RLEnvironment) -> None:
        """Evaluates a genome's inherited weights, then trains it and records metrics.

        Episode 0 is always a pre-training evaluation of the weights the genome
        inherited, so the evaluation history shows whether training improved on
        them. Training episodes are then
        numbered ``1`` to ``hp.episodes`` inclusive (each delegating to
        :meth:`run_update`), evaluated every ``hp.log_every`` episodes and on
        the last one, and the best-evaluated weights -- the inherited ones if
        training never beat them -- are restored. A genome with no trainable
        parameters, or ``hp.episodes`` of 0, is only evaluated, so its fitness
        comes from its inherited weights.

        Args:
            genome: The genome to train (its model is initialized here).
            environment: The environment to train on.

        Returns:
            None. Sets ``genome.metadata`` entries
            ``training_episode_metrics`` (per-episode returns for episodes
            ``1..hp.episodes``; there is no episode 0, since nothing was trained
            before it), ``evaluation_episode_metrics`` (starting with episode
            0), ``best_episode`` (0 when the inherited weights were never
            improved on), ``best_training_metrics``, ``best_validation_metrics``,
            ``n_trainable_parameters``, the ``training_seed`` its episodes were
            seeded from (drawn at random unless the genome's hyperparameters fix
            a ``seed``), the ``evaluation_seed`` and the ``eval_policy`` the
            evaluation returns were measured under. When nothing is trained,
            ``best_training_metrics`` holds the episode-0 evaluation's returns.

        Raises:
            ValueError: If ``environment`` is continuous but this trainer does
                not support continuous action spaces
                (:attr:`supports_continuous` is False), or if the genome's
                ``eval_policy`` is rejected by :meth:`validate_eval_policy`.
        """

        if environment.continuous and not self.supports_continuous:
            raise ValueError(
                f"{type(self).__name__} does not support continuous action "
                f"spaces (environment {environment.env_id!r} is continuous); "
                "use a policy-gradient trainer (reinforce, actor_critic, ppo)."
            )

        hp = self.resolve_hyperparameters(genome)
        log_std_bounds(genome)  # fail fast on an inverted clamp

        genome.initialize_model()

        # reload any normalizer statistics from metadata (e.g. inherited from a
        # parent) rather than reusing a stale cached copy
        reset_observation_normalizer(genome)

        # recorded so a genome's training episodes can be reproduced later
        genome.metadata["training_seed"] = hp.training_seed

        # recorded separately because it is drawn separately: evaluation runs
        # on a seed range disjoint from training's, so a genome is never scored
        # on an episode it learned from
        genome.metadata["evaluation_seed"] = hp.eval_seed

        # recorded because it decides which policy the fitness describes: a
        # genome scored greedily is not comparable with one scored
        # stochastically. This names the regime that *drove* fitness, so under
        # eval_policy="both" it is the selected one, not "both"; the genome's
        # own hyperparameters still record what was requested.
        genome.metadata["eval_policy"] = self.selected_eval_policy(hp.eval_policy)
        torch.manual_seed(hp.training_seed)
        np.random.seed(hp.training_seed)

        # All trainable parameters live in the genome's hybrid model (encoder,
        # quantum layer, decoder) -- including the value output for advantage
        # methods, which is an extra decoder row. There is no external head to
        # optimize, so everything trained here is also evolved by crossover
        # and preserved through genome serialization.
        trainable_parameters = list(genome.parameters())

        genome.metadata["training_episode_metrics"] = []
        genome.metadata["evaluation_episode_metrics"] = []

        # what the optimizer actually updates (encoder/decoder weights and enabled
        # gates' parameters), recorded the same way by every trainer
        n_trainable = genome.count_trainable_parameters()
        genome.metadata["n_trainable_parameters"] = n_trainable

        # episode 0: evaluate the inherited weights before any training
        initial_evaluation = self.evaluate(genome, environment, hp)
        initial_evaluation["episode"] = 0
        genome.metadata["evaluation_episode_metrics"].append(initial_evaluation)
        logger.info(
            f"[{type(self).__name__}] genome {genome.genome_number:4d} episode {0:4d} "
            f"pre-training eval_return_mean={initial_evaluation['return_mean']:.1f} "
            f"({genome.metadata['eval_policy']})"
        )

        best_return = initial_evaluation["return_mean"]
        best_evaluation = initial_evaluation
        best_episode = 0
        genome.metadata["best_episode"] = best_episode

        if n_trainable == 0 or hp.episodes <= 0:
            # nothing connected to the loss to optimize, or no training asked
            # for: the genome's fitness is that of its inherited weights
            if n_trainable == 0:
                logger.info(
                    "genome has no trainable (enabled) parameters; evaluating only."
                )
            else:
                logger.info("training for 0 episodes; evaluating only.")
            genome.metadata["best_training_metrics"] = {
                "return_mean": initial_evaluation["return_mean"],
                "best_episode_return": initial_evaluation["best_episode_return"],
            }
            genome.metadata["best_validation_metrics"] = initial_evaluation
            return

        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=hp.learning_rate,
            betas=(float(hp.adam_beta1), float(hp.adam_beta2)),
            eps=float(hp.adam_epsilon),
            weight_decay=0.0,
        )

        recent_returns: list[float] = []
        # Exponential moving average of episode returns, reported as the
        # training return mean. Seeded with the first episode's return (no
        # cold-start-at-zero bias), then updated as
        # ``ema = alpha * return + (1 - alpha) * ema`` each episode.
        ema_return: Optional[float] = None
        best_state = genome.clone_state_dict()
        # the normalizer statistics are part of the policy, so they are
        # snapshotted and restored together with the weights
        best_normalizer = snapshot_observation_normalizer(genome)
        eval_every = max(1, hp.log_every)
        self.begin_training(hp)

        # episode 0 was the pre-training evaluation, so training runs
        # 1..hp.episodes; run_update still takes the zero-based index, which
        # seeds the episode's reset
        for episode in range(1, hp.episodes + 1):
            if hp.lr_anneal == "linear":
                # the full rate on the first training episode, falling linearly
                for group in optimizer.param_groups:
                    group["lr"] = hp.learning_rate * (1.0 - (episode - 1) / hp.episodes)
            elif hp.lr_anneal != "none":
                raise ValueError(
                    f"Unknown lr_anneal {hp.lr_anneal!r}; choices: "
                    f"{', '.join(LR_ANNEAL_CHOICES)}."
                )

            # Sample fresh quantum dropout for this training episode (a no-op
            # when the toggle is off). Evaluation clears it so greedy rollouts
            # always use the complete circuit.
            if hp.quantum_dropout:
                sample_quantum_dropout(genome)

            episode_return, info = self.run_update(
                genome, environment, optimizer, episode - 1, hp
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

            if (episode % eval_every == 0) or (episode == hp.episodes):
                evaluation = self.evaluate(genome, environment, hp)
                evaluation["episode"] = episode

                # track evaluations for visualization purposes
                genome.metadata["evaluation_episode_metrics"].append(evaluation)

                logger.info(
                    f"[{type(self).__name__}] genome {genome.genome_number:4d} episode {episode:4d} "
                    f"train_return={episode_return:.1f} "
                    f"train_return_ema={ema_return:.1f} "
                    f"eval_return_mean={evaluation['return_mean']:.1f} "
                    f"({genome.metadata['eval_policy']})"
                )

                if evaluation["return_mean"] > best_return:
                    best_return = evaluation["return_mean"]
                    best_evaluation = evaluation
                    best_state = genome.clone_state_dict()
                    best_normalizer = snapshot_observation_normalizer(genome)
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

        # restore the best-evaluated weights (the inherited ones if training
        # never improved on them) into the genome
        genome.set_state_dict(best_state)
        restore_observation_normalizer(genome, best_normalizer)

        genome.metadata["best_episode"] = best_episode
        genome.metadata["best_training_metrics"] = {
            "return_mean": float(ema_return) if ema_return is not None else 0.0,
            "best_episode_return": (
                float(np.max(recent_returns)) if recent_returns else 0.0
            ),
        }
        genome.metadata["best_validation_metrics"] = best_evaluation

        # Leave the genome with no active dropout so the returned/serialized
        # policy runs the complete evolved circuit.
        genome.clear_quantum_dropout()
