from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributions import Categorical

from src.circuits.circuit import CircuitGenome
from src.utils.helpers import torch_params_to_genome, genome_to_torch_params
from .components import ReplayBuffer


def train_rl(
    genome: CircuitGenome, *, spec: "RLSpec", algo: str = "reinforce"
) -> CircuitGenome:
    """Train a genome with a selected RL algorithm.

    Dispatches to the matching trainer implementation based on `algo`.

    Args:
        genome: Quantum circuit genome to train. Must be compatible with
            `genome.generate_pennylane_circuit(...)` and callable via
            `genome.circuit(x, params)`.
        spec: Training configuration and environment settings.
        algo: Algorithm name. Supported values:
            - "reinforce"
            - "a2c" or "actor_critic"
            - "ppo"
            - "q_learning" or "sarsa"

    Returns:
        The updated genome with trained parameters written back and fitness
        metadata populated.

    Raises:
        ValueError: If an unknown algorithm is provided.
    """
    if algo == "reinforce":
        return train_reinforce(genome, spec=spec)
    if algo in {"a2c", "actor_critic"}:
        return train_actor_critic(genome, spec=spec)
    if algo == "ppo":
        return train_ppo(genome, spec=spec)
    if spec.algo in {"q_learning", "sarsa"}:
        return train_value_based(genome, spec=spec)
    raise ValueError(f"Unknown algo: {algo}")


# ============================
# GAE + batching utilities
# ============================


def gae_advantages(
    rewards: torch.Tensor,  # [T]
    values: torch.Tensor,  # [T]
    dones: torch.Tensor,  # [T] float {0,1}
    *,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) advantages and returns.

    This implementation follows the standard recursive definition:
        delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
        A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
        R_t = A_t + V_t

    Args:
        rewards: Reward tensor of shape [T].
        values: Value estimates tensor of shape [T].
        dones: Done flags tensor of shape [T], values in {0, 1} (float).
        gamma: Discount factor.
        lam: GAE lambda parameter.
        last_value: Bootstrap value for V_{T} (typically 0 for episodic rollouts).

    Returns:
        A tuple (advantages, returns):
            advantages: Tensor of shape [T].
            returns: Tensor of shape [T], computed as advantages + values.
    """
    T = rewards.numel()
    adv = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    next_value = float(last_value)

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t].item())
        delta = (
            float(rewards[t].item())
            + gamma * next_value * mask
            - float(values[t].item())
        )
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        next_value = float(values[t].item())

    returns = adv + values
    return adv, returns


def _stack_or_empty(
    xs: Sequence[torch.Tensor], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Stack a sequence of tensors or return an empty tensor.

    Args:
        xs: Sequence of tensors to stack.
        dtype: Dtype used for the empty tensor case.

    Returns:
        Stacked tensor if non-empty, else an empty tensor.
    """
    if len(xs) == 0:
        return torch.tensor([], dtype=dtype)
    return torch.stack(list(xs))


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a tensor to zero mean and unit variance.

    Args:
        x: Input tensor.
        eps: Numerical stability epsilon added to the standard deviation.

    Returns:
        Normalized tensor (same shape as input).
    """
    if x.numel() == 0:
        return x
    return (x - x.mean()) / (x.std() + eps)


# ============================
# Encoders
# ============================


def encode_box_to_unit_interval(
    obs: np.ndarray, scales: Optional[np.ndarray] = None
) -> torch.Tensor:
    """Encode a continuous Box observation into the unit interval.

    Produces a vector in [0, 1]^D, useful for angle encoding such as RY(pi*x).
    If `scales` is provided, performs a clipped linear scaling. Otherwise uses a
    tanh squash for a safe fallback.

    Args:
        obs: Observation array-like.
        scales: Optional scale factors per dimension. Typical use: normalize by
            plausible max magnitudes for each observation dimension.

    Returns:
        A float tensor with values in [0, 1].
    """
    obs = np.asarray(obs, dtype=np.float32)
    if scales is None:
        z = np.tanh(obs)
        z = (z + 1.0) / 2.0
        return torch.tensor(z, dtype=torch.float32)

    scales = np.asarray(scales, dtype=np.float32)
    z = np.clip(obs / scales, -1.0, 1.0)
    z = (z + 1.0) / 2.0
    return torch.tensor(z, dtype=torch.float32)


def encode_discrete_to_bits(s: int, n_bits: int) -> torch.Tensor:
    """Encode a discrete integer state into a binary bitstring tensor.

    Args:
        s: Discrete state index.
        n_bits: Number of bits to output.

    Returns:
        An int64 tensor of shape [n_bits] containing 0/1 bits.
    """
    bits = [(s >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]
    return torch.tensor(bits, dtype=torch.int64)


def encode_discrete_to_onehot_unit(s: int, n_states: int) -> torch.Tensor:
    """Encode a discrete state index into a one-hot float vector.

    Args:
        s: Discrete state index.
        n_states: Total number of states.

    Returns:
        A float tensor of shape [n_states] containing a one-hot representation.
    """
    x = torch.zeros(n_states, dtype=torch.float32)
    x[s] = 1.0
    return x


# ============================
# Policy head (logits helpers)
# ============================


def logits_from_2expvals(z: torch.Tensor) -> torch.Tensor:
    """Convert two expectation values into logits for a 2-action policy.

    Args:
        z: Expectation values on output qubits, shape [2] (each in [-1, 1]).

    Returns:
        A tensor of shape [2] suitable as logits for `torch.distributions.Categorical`.
    """
    z = torch.as_tensor(z, dtype=torch.float32).flatten()
    if z.numel() < 2:
        v = z[0] if z.numel() else torch.tensor(0.0)
        return torch.stack([v, -v])
    return z[:2]


def logits_from_kexpvals(z: torch.Tensor, n_actions: int) -> torch.Tensor:
    """Convert expectation values into logits for a K-action policy.

    Uses the first `n_actions` values as logits. If the feature vector is shorter,
    pads deterministically with zeros.

    Args:
        z: Feature tensor (expvals or any real-valued features).
        n_actions: Number of actions.

    Returns:
        A float tensor of shape [n_actions] used as logits.
    """
    z = torch.as_tensor(z, dtype=torch.float32).flatten()
    if z.numel() < n_actions:
        pad = torch.zeros(n_actions - z.numel(), dtype=z.dtype)
        z = torch.cat([z, pad], dim=0)
    return z[:n_actions]


# ============================
# Returns
# ============================


def discounted_returns(rewards: Sequence[torch.Tensor], gamma: float) -> torch.Tensor:
    """Compute discounted returns for a sequence of rewards.

    Args:
        rewards: Sequence of scalar reward tensors.
        gamma: Discount factor.

    Returns:
        Tensor of discounted returns of shape [T]. If empty, returns an empty tensor.
    """
    running = torch.tensor(0.0, dtype=torch.float32)
    out: list[torch.Tensor] = []
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return torch.stack(out) if out else torch.tensor([], dtype=torch.float32)


# ============================
# Config
# ============================


@dataclass
class RLSpec:
    """Configuration for RL training and evaluation.

    Attributes:
        env_id: Gymnasium environment id.
        n_actions: Number of discrete actions.
        algo: Algorithm name (used by some trainers).

        input_mode: How the genome encodes inputs ("angle" or "basis").
        return_expvals: If True, assume circuit returns expectation values.
            If False, assume circuit returns probabilities.

        obs_encoder: Callable that maps a raw env observation into a tensor used
            by the circuit.

        n_state_bits: Optional number of bits for discrete env basis encoding.
        box_scales: Optional scaling for continuous observations.

        episodes: Number of training episodes/updates.
        max_steps: Max steps per episode.
        gamma: Discount factor.
        lr: Learning rate.
        baseline: Baseline type for REINFORCE ("mean" or "none").
        seed: Random seed.
        log_every: Logging frequency.

        eval_episodes: Number of evaluation episodes.

        entropy_coef: Entropy regularization coefficient.

        env_kwargs: Keyword arguments passed into `gym.make(...)`.

        gae_lambda: Lambda for GAE.

        value_coef: Weight on the value loss term.

        ppo_clip: PPO clip range.
        ppo_epochs: PPO epochs per update.
        ppo_minibatch: PPO minibatch size.
        target_kl: Optional KL threshold for early stopping.

        rollout_steps: Number of steps collected per update in on-policy methods.

        epsilon: Initial epsilon for epsilon-greedy exploration in value-based RL.
        epsilon_min: Minimum epsilon.
        epsilon_decay: Multiplicative decay per episode.

        td_batch_size: TD minibatch size when sampling replay.
        replay_capacity: Replay buffer capacity.
        warmup_steps: Steps collected before learning begins.
        target_update_every: Steps between target net updates.
        train_every: Gradient step frequency (in env steps).
    """

    env_id: str
    n_actions: int
    algo: str = "reinforce"

    input_mode: str = "angle"  # "angle" or "basis"
    return_expvals: bool = True

    # observation encoder
    obs_encoder: Optional[Callable[[Any], torch.Tensor]] = None

    # optional: for discrete envs
    n_state_bits: Optional[int] = None

    # optional: for CartPole-like scaling
    box_scales: Optional[np.ndarray] = None

    # rollout/training
    episodes: int = 200
    max_steps: int = 500
    gamma: float = 0.99
    lr: float = 1e-2
    baseline: str = "mean"  # "mean" or "none"
    seed: int = 0
    log_every: int = 10

    # evaluation
    eval_episodes: int = 10

    # exploration regularizer (optional, simple)
    entropy_coef: float = 0.0

    env_kwargs: dict[str, Any] = None

    # Actor-Critic / PPO
    gae_lambda: float = 0.95

    # value function loss weight
    value_coef: float = 0.5

    # PPO
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch: int = 128
    target_kl: Optional[float] = None  # e.g. 0.02 to early stop updates

    # rollout sizing (for A2C/PPO batching)
    rollout_steps: int = (
        2048  # total steps collected before an update (PPO) or per update (A2C)
    )

    # Value-based (DQN/Q-learning/SARSA)
    epsilon: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    td_batch_size: int = 64  # if using replay
    replay_capacity: int = 10_000
    warmup_steps: int = 500  # collect before learning
    target_update_every: int = 200  # steps between target net updates
    train_every: int = 1  # gradient step frequency


# ============================
# Policy+Value head
# ============================


class PolicyValueHead(nn.Module):
    """Small policy-value head over circuit features.

    This head maps circuit output features into:
      - policy logits (for Categorical action selection)
      - a scalar value estimate

    The policy logits are derived directly from features via `logits_from_kexpvals`,
    while the value is produced by a linear layer.

    Args:
        n_actions: Number of discrete actions.
        in_dim: Feature dimension expected by the value layer.
    """

    def __init__(self, n_actions: int, in_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.value = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy logits and value estimate.

        Args:
            features: Feature tensor of shape [K] (flattened).

        Returns:
            A tuple (logits, value):
                logits: Tensor of shape [n_actions].
                value: Scalar tensor.
        """
        logits = logits_from_kexpvals(features, self.n_actions)
        v = self.value(features[: self.value.in_features]).squeeze(-1)
        return logits, v


# ============================
# Q head
# ============================


class QHead(nn.Module):
    """Linear Q-value head over circuit features.

    Args:
        n_actions: Number of discrete actions.
        in_dim: Feature dimension expected by the linear layer.
    """

    def __init__(self, n_actions: int, in_dim: int):
        super().__init__()
        self.q = nn.Linear(in_dim, n_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for each action.

        Pads features with zeros if fewer than `in_dim` are provided.

        Args:
            features: Feature tensor.

        Returns:
            Tensor of shape [n_actions] containing Q-values.
        """
        features = torch.as_tensor(features, dtype=torch.float32).flatten()
        if features.numel() < self.q.in_features:
            pad = torch.zeros(
                self.q.in_features - features.numel(), dtype=features.dtype
            )
            features = torch.cat([features, pad], dim=0)
        return self.q(features[: self.q.in_features])


# ============================
# Core rollout (generic)
# ============================


def _make_env(env_id: str, **kwargs):
    """Instantiate a Gymnasium environment.

    Args:
        env_id: Gymnasium environment id.
        **kwargs: Passed to `gym.make`.

    Returns:
        A Gymnasium environment instance.
    """
    import gymnasium as gym

    return gym.make(env_id, **kwargs)


def rollout_reinforce(
    genome: CircuitGenome,
    params: dict[str, torch.nn.Parameter],
    *,
    spec: RLSpec,
    episode_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Collect one REINFORCE episode rollout.

    For each step:
      - encode observation via `spec.obs_encoder`
      - run `genome.circuit(x, params)` to produce outputs
      - convert outputs into logits (from expvals or probs)
      - sample action, step environment
      - store log-prob and entropy for policy gradient

    Args:
        genome: CircuitGenome policy.
        params: Trainable parameters dict for the genome.
        spec: RLSpec configuration.
        episode_seed: Seed for environment reset.

    Returns:
        A tuple (logps, entropies, returns, ep_return):
            logps: Tensor [T] of log-probabilities.
            entropies: Tensor [T] of entropies.
            returns: Tensor [T] of discounted returns.
            ep_return: Episode return as float.
    """
    env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
    obs, _ = env.reset(seed=episode_seed)

    logps: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    ep_return = 0.0

    for _ in range(spec.max_steps):
        x = spec.obs_encoder(obs)

        out = genome.circuit(x, params)
        if spec.return_expvals:
            out = torch.stack(list(out))
            logits = logits_from_kexpvals(out, spec.n_actions)
        else:
            probs = torch.as_tensor(out, dtype=torch.float32).flatten()
            probs = probs[: spec.n_actions]
            probs = probs / (probs.sum() + 1e-12)
            logits = torch.log(probs.clamp_min(1e-12))

        dist = Categorical(logits=logits)
        a = dist.sample()

        obs, r, terminated, truncated, _ = env.step(int(a.item()))
        ep_return += float(r)

        logps.append(dist.log_prob(a))
        entropies.append(dist.entropy())
        rewards.append(torch.tensor(r, dtype=torch.float32))

        if terminated or truncated:
            break

    env.close()

    G = discounted_returns(rewards, spec.gamma)
    return torch.stack(logps), torch.stack(entropies), G, ep_return


# ============================
# Actor-Critic/PPO rollout (generic)
# ============================


@torch.no_grad()
def rollout_actor_critic(
    genome: CircuitGenome,
    params: dict[str, torch.nn.Parameter],
    *,
    spec: RLSpec,
    head: PolicyValueHead,
    episode_seed: int,
) -> dict[str, Any]:
    """Collect transitions for Actor-Critic or PPO.

    Stores per-step tensors for later batch processing:
      - obs tensors (encoded)
      - sampled actions
      - log probabilities under the behavior policy (old logps)
      - value estimates under the behavior policy (old values)
      - rewards
      - done flags

    Args:
        genome: CircuitGenome policy.
        params: Trainable parameters dict for the genome.
        spec: RLSpec configuration.
        head: Policy+value head over circuit features.
        episode_seed: Seed for environment reset.

    Returns:
        Dict with keys:
            obs, actions, logps_old, values_old, rewards, dones, ep_return
    """
    env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
    obs, _ = env.reset(seed=episode_seed)

    obs_tensors: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    logps_old: list[torch.Tensor] = []
    values_old: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []

    ep_return = 0.0

    for _ in range(spec.max_steps):
        x = spec.obs_encoder(obs)
        out = genome.circuit(x, params)
        feats = torch.as_tensor(out, dtype=torch.float32).flatten()

        logits, v = head(feats)
        dist = Categorical(logits=logits)
        a = dist.sample()

        next_obs, r, terminated, truncated, _ = env.step(int(a.item()))
        done = terminated or truncated

        obs_tensors.append(x.detach().cpu())
        actions.append(a.detach().cpu())
        logps_old.append(dist.log_prob(a).detach().cpu())
        values_old.append(v.detach().cpu())
        rewards.append(torch.tensor(float(r), dtype=torch.float32))
        dones.append(torch.tensor(1.0 if done else 0.0, dtype=torch.float32))

        ep_return += float(r)
        obs = next_obs
        if done:
            break

    env.close()

    return {
        "obs": obs_tensors,
        "actions": actions,
        "logps_old": logps_old,
        "values_old": values_old,
        "rewards": rewards,
        "dones": dones,
        "ep_return": ep_return,
    }


# ============================
# Train / Eval (generic)
# ============================


@torch.no_grad()
def eval_policy(
    genome: CircuitGenome,
    *,
    spec: RLSpec,
    deterministic: bool = True,
    seed: int = 1234,
) -> dict[str, float]:
    """Evaluate a policy across multiple episodes.

    Args:
        genome: CircuitGenome policy to evaluate.
        spec: RLSpec configuration.
        deterministic: If True, uses argmax action; otherwise samples from policy.
        seed: Base seed for evaluation episodes.

    Returns:
        Dict with mean and std return:
            - eval_return_mean
            - eval_return_std
    """
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(
            input_mode=spec.input_mode, measure_registers=spec.return_expvals
        )

    params = genome_to_torch_params(genome)

    returns: list[float] = []
    for ep in range(spec.eval_episodes):
        env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
        obs, _ = env.reset(seed=seed + ep)

        ep_ret = 0.0
        for _ in range(spec.max_steps):
            x = spec.obs_encoder(obs)
            out = genome.circuit(x, params)

            if spec.return_expvals:
                out = torch.stack(list(out))
                logits = logits_from_kexpvals(out, spec.n_actions)
            else:
                probs = torch.as_tensor(out, dtype=torch.float32).flatten()
                probs = probs[: spec.n_actions]
                probs = probs / (probs.sum() + 1e-12)
                logits = torch.log(probs.clamp_min(1e-12))

            if deterministic:
                a = int(torch.argmax(logits).item())
            else:
                a = int(Categorical(logits=logits).sample().item())

            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += float(r)
            if terminated or truncated:
                break

        env.close()
        returns.append(ep_ret)

    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
    }


def train_reinforce(genome: CircuitGenome, *, spec: RLSpec) -> CircuitGenome:
    """Train a genome using REINFORCE (Monte-Carlo policy gradient).

    Requires:
      - `spec.obs_encoder` to encode observations
      - `genome.generate_pennylane_circuit(...)` to create `genome.circuit`
      - circuit to return either expvals (preferred) or probabilities

    The objective is:
        L = -E[ log pi(a_t|s_t) * advantage_t ] - entropy_coef * H[pi]

    Where advantage is either `G_t - mean(G)` or `G_t` depending on baseline.

    Args:
        genome: CircuitGenome policy to train.
        spec: RLSpec configuration.

    Returns:
        Updated genome with parameters written back and fitness populated.

    Raises:
        ValueError: If `spec.obs_encoder` is not provided.
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    genome.generate_pennylane_circuit(
        input_mode=spec.input_mode, measure_registers=spec.return_expvals
    )

    params = genome_to_torch_params(genome)

    if len(params) == 0:
        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 999)
        genome.fitness = {
            "note": "no trainable params",
            "best_episode_return": ev["eval_return_mean"],
            **ev,
        }
        return genome

    opt = torch.optim.Adam(params.values(), lr=spec.lr, weight_decay=0.0)

    best_episode_return = -float("inf")
    recent_returns: list[float] = []

    for ep in range(spec.episodes):
        opt.zero_grad()

        logps, entropies, G, ep_ret = rollout_reinforce(
            genome, params, spec=spec, episode_seed=spec.seed + ep
        )
        recent_returns.append(ep_ret)
        best_episode_return = max(best_episode_return, ep_ret)

        if G.numel() == 0:
            continue

        adv = G - G.mean() if spec.baseline == "mean" else G

        loss_pg = -(logps * adv.detach()).mean()
        loss_ent = (
            -spec.entropy_coef * entropies.mean() if spec.entropy_coef > 0 else 0.0
        )
        loss = loss_pg + (loss_ent if isinstance(loss_ent, torch.Tensor) else 0.0)

        if not loss.requires_grad:
            logger.warning("Loss has no grad path. Are params used inside the QNode?")
            break

        loss.backward()
        opt.step()

        if (ep % spec.log_every) == 0:
            avg10 = (
                float(np.mean(recent_returns[-10:]))
                if len(recent_returns) >= 10
                else float(np.mean(recent_returns))
            )
            logger.info(
                f"[ep {ep:04d}] loss={float(loss.item()):.4f} return={ep_ret:.1f} avg10={avg10:.1f}"
            )

    torch_params_to_genome(genome, params)

    ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)
    if len(recent_returns) >= 20:
        train_tail = float(np.mean(recent_returns[-20:]))
    elif recent_returns:
        train_tail = float(np.mean(recent_returns))
    else:
        train_tail = 0.0

    genome.fitness = {
        "train_return_mean": train_tail,
        "best_episode_return": float(best_episode_return),
        **ev,
        "episodes": spec.episodes,
        "env_id": spec.env_id,
    }
    return genome


# ============================
# Train Actor-Critic
# ============================


def train_actor_critic(genome: CircuitGenome, *, spec: RLSpec) -> CircuitGenome:
    """Train a genome using an on-policy Actor-Critic (A2C-style) update.

    Flow:
      1) Collect `spec.rollout_steps` transitions (may span multiple episodes).
      2) Compute GAE advantages and returns.
      3) Update policy + value head jointly using:
            L = L_pi + value_coef * L_v - entropy_coef * H
         where
            L_pi = -E[ log pi(a|s) * A ]
            L_v  = 0.5 * E[ (R - V)^2 ]

    Args:
        genome: CircuitGenome policy to train.
        spec: RLSpec configuration.

    Returns:
        Updated genome with trained parameters and fitness.

    Raises:
        ValueError: If `spec.obs_encoder` is not provided.
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    genome.generate_pennylane_circuit(
        input_mode=spec.input_mode, measure_registers=spec.return_expvals
    )
    params = genome_to_torch_params(genome)

    if len(params) == 0:
        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 999)
        genome.fitness = {"note": "no trainable params", **ev, "env_id": spec.env_id}
        return genome

    with torch.no_grad():
        env0 = _make_env(spec.env_id, **(spec.env_kwargs or {}))
        o0, _ = env0.reset(seed=spec.seed)
        env0.close()
        dummy = spec.obs_encoder(o0)
        out0 = genome.circuit(dummy, params)
        in_dim = int(torch.as_tensor(out0).numel())

    head = PolicyValueHead(n_actions=spec.n_actions, in_dim=in_dim)
    opt = torch.optim.Adam(
        list(params.values()) + list(head.parameters()), lr=spec.lr, weight_decay=0.0
    )

    best_episode_return = -float("inf")
    recent_returns: list[float] = []

    for upd in range(spec.episodes):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        steps_collected = 0
        ep = 0

        while steps_collected < spec.rollout_steps:
            traj = rollout_actor_critic(
                genome,
                params,
                spec=spec,
                head=head,
                episode_seed=spec.seed + upd * 10_000 + ep,
            )
            ep += 1
            recent_returns.append(traj["ep_return"])
            best_episode_return = max(best_episode_return, traj["ep_return"])

            obs_buf += traj["obs"]
            act_buf += traj["actions"]
            logp_buf += traj["logps_old"]
            val_buf += traj["values_old"]
            rew_buf += traj["rewards"]
            done_buf += traj["dones"]

            steps_collected = len(obs_buf)

        obs_t = torch.stack([o.to(torch.float32) for o in obs_buf], dim=0)  # [T,D]
        act_t = torch.stack(act_buf).long().view(-1)  # [T]
        # logp_old_t = torch.stack(logp_buf).to(torch.float32).view(-1)  # [T]
        val_old_t = torch.stack(val_buf).to(torch.float32).view(-1)  # [T]
        rew_t = torch.stack(rew_buf).to(torch.float32).view(-1)  # [T]
        done_t = torch.stack(done_buf).to(torch.float32).view(-1)  # [T]

        adv_t, ret_t = gae_advantages(
            rew_t,
            val_old_t,
            done_t,
            gamma=spec.gamma,
            lam=spec.gae_lambda,
            last_value=0.0,
        )
        adv_t = _normalize(adv_t)

        opt.zero_grad()

        new_logps: list[torch.Tensor] = []
        new_vals: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for i in range(obs_t.shape[0]):
            x = obs_t[i]
            out = genome.circuit(x, params)
            feats = torch.as_tensor(out, dtype=torch.float32).flatten()
            logits, v = head(feats)
            dist = Categorical(logits=logits)

            new_logps.append(dist.log_prob(act_t[i]))
            entropies.append(dist.entropy())
            new_vals.append(v)

        new_logp_t = torch.stack(new_logps)  # [T]
        new_val_t = torch.stack(new_vals)  # [T]
        ent_t = torch.stack(entropies)  # [T]

        loss_pi = -(new_logp_t * adv_t.detach()).mean()
        loss_v = 0.5 * (ret_t.detach() - new_val_t).pow(2).mean()
        loss_ent = -spec.entropy_coef * ent_t.mean() if spec.entropy_coef > 0 else 0.0

        loss = (
            loss_pi
            + spec.value_coef * loss_v
            + (loss_ent if isinstance(loss_ent, torch.Tensor) else 0.0)
        )

        if not loss.requires_grad:
            logger.warning("Loss has no grad path. Are params used inside the QNode?")
            break

        loss.backward()
        opt.step()

        if (upd % spec.log_every) == 0:
            avg10 = (
                float(np.mean(recent_returns[-10:]))
                if len(recent_returns) >= 10
                else float(np.mean(recent_returns))
            )
            logger.info(
                f"[A2C upd {upd:04d}] loss={float(loss.item()):.4f} "
                f"pi={float(loss_pi.item()):.4f} v={float(loss_v.item()):.4f} avg10ret={avg10:.1f}"
            )

    torch_params_to_genome(genome, params)

    ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)
    train_tail = (
        float(np.mean(recent_returns[-20:]))
        if len(recent_returns) >= 20
        else float(np.mean(recent_returns) if recent_returns else 0.0)
    )

    genome.fitness = {
        "train_return_mean": train_tail,
        "best_episode_return": float(best_episode_return),
        **ev,
        "episodes": spec.episodes,
        "env_id": spec.env_id,
        "algo": "actor_critic",
    }
    return genome


# ============================
# Train PPO
# ============================


def train_ppo(genome: CircuitGenome, *, spec: RLSpec) -> CircuitGenome:
    """Train a genome using PPO (clipped objective) with GAE.

    Flow:
      1) Collect `spec.rollout_steps` transitions (may span multiple episodes).
      2) Compute GAE advantages and returns.
      3) Perform `spec.ppo_epochs` passes of minibatch updates using:
            ratio = exp(logp_new - logp_old)
            L_pi  = -E[min(ratio*A, clip(ratio)*A)]
            L_v   = 0.5 * E[(R - V)^2]
            L_ent = -entropy_coef * E[H]
            L = L_pi + value_coef * L_v + L_ent
      4) Optional early stop if approximate KL exceeds `spec.target_kl`.

    Args:
        genome: CircuitGenome policy to train.
        spec: RLSpec configuration.

    Returns:
        Updated genome with trained parameters and fitness.

    Raises:
        ValueError: If `spec.obs_encoder` is not provided.
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    genome.generate_pennylane_circuit(
        input_mode=spec.input_mode, measure_registers=spec.return_expvals
    )
    params = genome_to_torch_params(genome)

    if len(params) == 0:
        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 999)
        genome.fitness = {"note": "no trainable params", **ev, "env_id": spec.env_id}
        return genome

    with torch.no_grad():
        env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
        o0, _ = env.reset(seed=spec.seed)
        env.close()
        x0 = spec.obs_encoder(o0)
        out0 = genome.circuit(x0, params)
        in_dim = int(torch.as_tensor(out0).numel())

    head = PolicyValueHead(n_actions=spec.n_actions, in_dim=in_dim)
    opt = torch.optim.Adam(
        list(params.values()) + list(head.parameters()), lr=spec.lr, weight_decay=0.0
    )

    best_episode_return = -float("inf")
    recent_returns: list[float] = []

    for upd in range(spec.episodes):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        steps_collected = 0
        ep = 0

        while steps_collected < spec.rollout_steps:
            traj = rollout_actor_critic(
                genome,
                params,
                spec=spec,
                head=head,
                episode_seed=spec.seed + upd * 10_000 + ep,
            )
            ep += 1
            recent_returns.append(traj["ep_return"])
            best_episode_return = max(best_episode_return, traj["ep_return"])

            obs_buf += traj["obs"]
            act_buf += traj["actions"]
            logp_buf += traj["logps_old"]
            val_buf += traj["values_old"]
            rew_buf += traj["rewards"]
            done_buf += traj["dones"]
            steps_collected = len(obs_buf)

        T = len(obs_buf)

        obs_t = torch.stack([o.to(torch.float32) for o in obs_buf], dim=0)
        act_t = torch.stack(act_buf).long().view(-1)
        logp_old_t = torch.stack(logp_buf).to(torch.float32).view(-1)
        val_old_t = torch.stack(val_buf).to(torch.float32).view(-1)
        rew_t = torch.stack(rew_buf).to(torch.float32).view(-1)
        done_t = torch.stack(done_buf).to(torch.float32).view(-1)

        adv_t, ret_t = gae_advantages(
            rew_t,
            val_old_t,
            done_t,
            gamma=spec.gamma,
            lam=spec.gae_lambda,
            last_value=0.0,
        )
        adv_t = _normalize(adv_t)

        mb = min(spec.ppo_minibatch, T)
        idx_all = torch.randperm(T)

        for _epoch in range(spec.ppo_epochs):
            idx_all = idx_all[torch.randperm(T)]

            for start in range(0, T, mb):
                idx = idx_all[start : start + mb]

                new_logps: list[torch.Tensor] = []
                new_vals: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []

                for i in idx.tolist():
                    x = obs_t[i]
                    out = genome.circuit(x, params)
                    feats = torch.as_tensor(out, dtype=torch.float32).flatten()
                    logits, v = head(feats)
                    dist = Categorical(logits=logits)
                    a = act_t[i]
                    new_logps.append(dist.log_prob(a))
                    entropies.append(dist.entropy())
                    new_vals.append(v)

                new_logp = torch.stack(new_logps)
                new_val = torch.stack(new_vals)
                ent = torch.stack(entropies)

                ratio = torch.exp(new_logp - logp_old_t[idx])

                surr1 = ratio * adv_t[idx]
                surr2 = (
                    torch.clamp(ratio, 1.0 - spec.ppo_clip, 1.0 + spec.ppo_clip)
                    * adv_t[idx]
                )
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v = 0.5 * (ret_t[idx].detach() - new_val).pow(2).mean()

                loss_ent = (
                    -spec.entropy_coef * ent.mean() if spec.entropy_coef > 0 else 0.0
                )

                loss = (
                    loss_pi
                    + spec.value_coef * loss_v
                    + (loss_ent if isinstance(loss_ent, torch.Tensor) else 0.0)
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                if spec.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = (logp_old_t[idx] - new_logp).mean().item()
                    if approx_kl > float(spec.target_kl):
                        break

        if (upd % spec.log_every) == 0:
            avg10 = (
                float(np.mean(recent_returns[-10:]))
                if len(recent_returns) >= 10
                else float(np.mean(recent_returns))
            )
            logger.info(f"[PPO upd {upd:04d}] avg10ret={avg10:.1f} bufferT={T}")

    torch_params_to_genome(genome, params)

    ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)
    train_tail = (
        float(np.mean(recent_returns[-20:]))
        if len(recent_returns) >= 20
        else float(np.mean(recent_returns) if recent_returns else 0.0)
    )

    genome.fitness = {
        "train_return_mean": train_tail,
        "best_episode_return": float(best_episode_return),
        **ev,
        "episodes": spec.episodes,
        "env_id": spec.env_id,
        "algo": "ppo",
    }
    return genome


# ============================
# Train Q-Learning/SARSA
# ============================


def _forward_features(genome: CircuitGenome, x: torch.Tensor, params) -> torch.Tensor:
    """Compute circuit features for a given encoded observation.

    Args:
        genome: CircuitGenome with a callable `circuit`.
        x: Encoded observation tensor.
        params: Trainable parameters dict for the circuit.

    Returns:
        Flattened float feature tensor.
    """
    out = genome.circuit(x, params)
    return torch.as_tensor(out, dtype=torch.float32).flatten()


@torch.no_grad()
def _epsilon_greedy_action(q_values: torch.Tensor, epsilon: float) -> int:
    """Select an action via epsilon-greedy exploration.

    Args:
        q_values: Tensor of Q-values (shape [n_actions]).
        epsilon: Probability of sampling a random action.

    Returns:
        Selected action index.
    """
    if torch.rand(()) < epsilon:
        return int(torch.randint(low=0, high=q_values.numel(), size=(1,)).item())
    return int(torch.argmax(q_values).item())


def _infer_feature_dim(genome: CircuitGenome, spec: RLSpec) -> int:
    """Infer the circuit output dimensionality via a dummy forward pass.

    Args:
        genome: CircuitGenome policy.
        spec: RLSpec configuration.

    Returns:
        Number of scalar features produced by `genome.circuit(...)`.
    """
    env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
    obs, _ = env.reset(seed=spec.seed)
    env.close()
    x = spec.obs_encoder(obs)
    with torch.no_grad():
        params0 = genome_to_torch_params(genome)
        out0 = genome.circuit(x, params0)
    return int(torch.as_tensor(out0).numel())


def train_value_based(genome: CircuitGenome, *, spec: RLSpec) -> CircuitGenome:
    """Train a genome using value-based RL (Q-learning or SARSA).

    Uses circuit outputs as features and learns a small linear Q-function head.
    Stability is improved using a target network and optional replay buffer.

    Targets:
      - Q-learning (DQN-style): y = r + gamma * (1-done) * max_a' Q_tgt(s', a')
      - SARSA:                 y = r + gamma * (1-done) * Q_tgt(s', a'_eps)

    Args:
        genome: CircuitGenome policy to train.
        spec: RLSpec configuration.

    Returns:
        Updated genome with trained circuit parameters written back and fitness.

    Raises:
        ValueError: If `spec.obs_encoder` is not provided.
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    genome.generate_pennylane_circuit(
        input_mode=spec.input_mode, measure_registers=spec.return_expvals
    )
    params = genome_to_torch_params(genome)

    feat_dim = _infer_feature_dim(genome, spec)
    q_head = QHead(n_actions=spec.n_actions, in_dim=feat_dim)
    q_tgt = QHead(n_actions=spec.n_actions, in_dim=feat_dim)
    q_tgt.load_state_dict(q_head.state_dict())

    opt = torch.optim.Adam(
        list(params.values()) + list(q_head.parameters()), lr=spec.lr, weight_decay=0.0
    )

    rb = ReplayBuffer(spec.replay_capacity)

    epsilon = float(spec.epsilon)
    best_episode_return = -float("inf")
    recent_returns: list[float] = []

    global_step = 0

    for ep in range(spec.episodes):
        env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
        obs, _ = env.reset(seed=spec.seed + ep)

        ep_ret = 0.0

        for _ in range(spec.max_steps):
            x = spec.obs_encoder(obs)

            feats = _forward_features(genome, x, params)
            q_vals = q_head(feats)

            a = _epsilon_greedy_action(q_vals, epsilon)

            next_obs, r, terminated, truncated, _ = env.step(a)
            done = float(terminated or truncated)
            ep_ret += float(r)

            x2 = spec.obs_encoder(next_obs)
            rb.push(x, a, r, x2, done)

            obs = next_obs
            global_step += 1

            if len(rb) >= spec.warmup_steps and (global_step % spec.train_every) == 0:
                batch = rb.sample(spec.td_batch_size)

                s = torch.stack([b[0].to(torch.float32) for b in batch], dim=0)
                a_b = torch.tensor([b[1] for b in batch], dtype=torch.long)
                r_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                s2 = torch.stack([b[3].to(torch.float32) for b in batch], dim=0)
                d_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                q_sa: list[torch.Tensor] = []
                for i in range(s.shape[0]):
                    feats_i = _forward_features(genome, s[i], params)
                    q_i = q_head(feats_i)
                    q_sa.append(q_i[a_b[i]])
                q_sa_t = torch.stack(q_sa)

                with torch.no_grad():
                    q_next: list[torch.Tensor] = []
                    for i in range(s2.shape[0]):
                        feats2_i = _forward_features(genome, s2[i], params)
                        q2_i = q_tgt(feats2_i)

                        if spec.algo == "sarsa":
                            a2 = _epsilon_greedy_action(q2_i, epsilon)
                            q_next.append(q2_i[a2])
                        else:
                            q_next.append(torch.max(q2_i))

                    q_next_t = torch.stack(q_next)
                    target = r_b + spec.gamma * (1.0 - d_b) * q_next_t

                loss = torch.mean((q_sa_t - target) ** 2)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if (global_step % spec.target_update_every) == 0:
                    q_tgt.load_state_dict(q_head.state_dict())
                    q_tgt.train()

            if terminated or truncated:
                break

        env.close()
        recent_returns.append(ep_ret)
        best_episode_return = max(best_episode_return, ep_ret)

        epsilon = max(spec.epsilon_min, epsilon * spec.epsilon_decay)

        if (ep % spec.log_every) == 0:
            avg10 = (
                float(np.mean(recent_returns[-10:]))
                if len(recent_returns) >= 10
                else float(np.mean(recent_returns))
            )
            logger.info(
                f"[{spec.algo.upper()} ep {ep:04d}] return={ep_ret:.1f} avg10={avg10:.1f}"
                f"eps={epsilon:.3f} rb={len(rb)}"
            )

    torch_params_to_genome(genome, params)

    ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)
    train_tail = (
        float(np.mean(recent_returns[-20:]))
        if len(recent_returns) >= 20
        else float(np.mean(recent_returns) if recent_returns else 0.0)
    )

    genome.fitness = {
        "train_return_mean": train_tail,
        "best_episode_return": float(best_episode_return),
        **ev,
        "episodes": spec.episodes,
        "env_id": spec.env_id,
        "algo": spec.algo,
    }
    return genome


# ============================
# Ready-made specs
# ============================


def cartpole_spec(
    *, episodes: int = 100, lr: float = 1e-2, seed: int = 0, algo: str = "reinforce"
) -> RLSpec:
    """Create a ready-to-use CartPole RLSpec.

    Uses angle encoding of the 4D Box state into [0,1] with simple scaling.

    Args:
        episodes: Number of training episodes/updates.
        lr: Learning rate.
        seed: Random seed.
        algo: Algorithm name stored in the spec.

    Returns:
        RLSpec configured for "CartPole-v1".
    """
    scales = np.array([2.4, 3.0, 0.21, 3.5], dtype=np.float32)

    def encoder(obs):
        return encode_box_to_unit_interval(obs, scales=scales)

    return RLSpec(
        env_id="CartPole-v1",
        n_actions=2,
        algo=algo,
        input_mode="angle",
        return_expvals=True,
        obs_encoder=encoder,
        box_scales=scales,
        episodes=episodes,
        lr=lr,
        seed=seed,
        max_steps=500,
        eval_episodes=10,
    )


def frozenlake_spec(
    *,
    map_name: str = "4x4",
    is_slippery: bool = True,
    episodes: int = 300,
    lr: float = 2e-2,
    seed: int = 0,
    algo: str = "reinforce",
) -> RLSpec:
    """Create a ready-to-use FrozenLake RLSpec.

    FrozenLake observations are discrete states (0..n_states-1). This helper
    uses basis encoding into a fixed number of bits.

    Args:
        map_name: FrozenLake map name ("4x4" or "8x8").
        is_slippery: Whether the environment is stochastic.
        episodes: Number of training episodes/updates.
        lr: Learning rate.
        seed: Random seed.
        algo: Algorithm name stored in the spec.

    Returns:
        RLSpec configured for "FrozenLake-v1".
    """
    n_states = 16 if map_name == "4x4" else 64
    n_bits = int(np.ceil(np.log2(n_states)))

    def encoder(obs):
        return encode_discrete_to_bits(int(obs), n_bits)

    return RLSpec(
        env_id="FrozenLake-v1",
        n_actions=4,
        algo=algo,
        input_mode="basis",
        return_expvals=True,
        obs_encoder=encoder,
        n_state_bits=n_bits,
        episodes=episodes,
        lr=lr,
        seed=seed,
        max_steps=100,
        eval_episodes=20,
        env_kwargs={"is_slippery": is_slippery},
    )
