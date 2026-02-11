from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.distributions import Categorical
from loguru import logger
import torch.nn as nn

from src.circuits.circuit import CircuitGenome
from src.objectives.genome_objectives import genome_to_torch_params, torch_params_to_genome
from .components import ReplayBuffer


def train_rl(genome: CircuitGenome, *, spec: RLSpec, algo: str = "reinforce") -> CircuitGenome:
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
    rewards: torch.Tensor,      # [T]
    values: torch.Tensor,       # [T]
    dones: torch.Tensor,        # [T] float {0,1}
    *,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.
    Returns: (advantages [T], returns [T])
    """
    T = rewards.numel()
    adv = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    next_value = float(last_value)

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t].item())
        delta = float(rewards[t].item()) + gamma * next_value * mask - float(values[t].item())
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        next_value = float(values[t].item())

    returns = adv + values
    return adv, returns


def _stack_or_empty(xs, dtype=torch.float32):
    if len(xs) == 0:
        return torch.tensor([], dtype=dtype)
    return torch.stack(xs)


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return (x - x.mean()) / (x.std() + eps)



# ============================
# Encoders
# ============================

def encode_box_to_unit_interval(obs: np.ndarray, scales: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Map a Box observation to [0,1]^D (good for angle encoding RY(pi*x)).
    """
    obs = np.asarray(obs, dtype=np.float32)
    if scales is None:
        # fallback: tanh squashing
        z = np.tanh(obs)
        z = (z + 1.0) / 2.0
        return torch.tensor(z, dtype=torch.float32)

    scales = np.asarray(scales, dtype=np.float32)
    z = np.clip(obs / scales, -1.0, 1.0)
    z = (z + 1.0) / 2.0
    return torch.tensor(z, dtype=torch.float32)


def encode_discrete_to_bits(s: int, n_bits: int) -> torch.Tensor:
    """
    Encode a discrete state index into a basis bitstring tensor of length n_bits.
    Output is int64 bits.
    """
    bits = [(s >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]
    return torch.tensor(bits, dtype=torch.int64)


def encode_discrete_to_onehot_unit(s: int, n_states: int) -> torch.Tensor:
    """
    One-hot in {0,1} then cast to float; useful for angle encoding.
    """
    x = torch.zeros(n_states, dtype=torch.float32)
    x[s] = 1.0
    return x


# ============================
# Policy head
# ============================

def logits_from_2expvals(z: torch.Tensor) -> torch.Tensor:
    """
    z: expvals on output qubits, shape [2] (each in [-1,1])
    Use as logits for a 2-action policy.
    """
    z = torch.as_tensor(z, dtype=torch.float32).flatten()
    if z.numel() < 2:
        v = z[0] if z.numel() else torch.tensor(0.0)
        return torch.stack([v, -v])
    return z[:2]


def logits_from_kexpvals(z: torch.Tensor, n_actions: int) -> torch.Tensor:
    """
    General: use first n_actions expvals as logits.
    Requires len(output_qubits) >= n_actions.
    """
    z = torch.as_tensor(z, dtype=torch.float32).flatten()
    if z.numel() < n_actions:
        # pad deterministically
        pad = torch.zeros(n_actions - z.numel(), dtype=z.dtype)
        z = torch.cat([z, pad], dim=0)
    return z[:n_actions]


# ============================
# Returns
# ============================

def discounted_returns(rewards: Sequence[torch.Tensor], gamma: float) -> torch.Tensor:
    running = torch.tensor(0.0, dtype=torch.float32)
    out = []
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
    env_id: str
    n_actions: int
    algo: str = "reinforce"

    # how to build circuit output
    # assumes genome.generate_pennylane_circuit(... return_expvals=True/False etc.)
    input_mode: str = "angle"   # "angle" or "basis"
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
    baseline: str = "mean"   # "mean" or "none"
    seed: int = 0
    log_every: int = 10

    # evaluation
    eval_episodes: int = 10

    # exploration regularizer (optional, simple)
    entropy_coef: float = 0.0

    env_kwargs: dict[str, Any] = None

    # ---- Actor-Critic / PPO ----
    gae_lambda: float = 0.95

    # value function loss weight
    value_coef: float = 0.5

    # PPO
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch: int = 128
    target_kl: Optional[float] = None  # e.g. 0.02 to early stop updates

    # rollout sizing (for A2C/PPO batching)
    rollout_steps: int = 2048  # total steps collected before an update (PPO) or per update (A2C)

    # ============================
    # Value-based (DQN/Q-learning/SARSA)
    # ============================

    epsilon: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    td_batch_size: int = 64          # if using replay
    replay_capacity: int = 10_000
    warmup_steps: int = 500          # collect before learning
    target_update_every: int = 200   # steps between target net updates
    train_every: int = 1             # gradient step frequency


# ============================
# Policy+Value head
# ============================

class PolicyValueHead(nn.Module):
    """
    Turns circuit outputs (expvals or probs) into:
      - policy logits (n_actions)
      - value scalar
    We keep it tiny: linear value head over the same features.
    """
    def __init__(self, n_actions: int, in_dim: int):
        super().__init__()
        self.n_actions = n_actions
        self.value = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: [K]
        logits = logits_from_kexpvals(features, self.n_actions)
        v = self.value(features[: self.value.in_features]).squeeze(-1)
        return logits, v
    

# ============================
# Q head
# ============================

class QHead(nn.Module):
    """
    Turns circuit outputs (expvals or probs) into Q-values for each action.
    """
    def __init__(self, n_actions: int, in_dim: int):
        super().__init__()
        self.q = nn.Linear(in_dim, n_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=torch.float32).flatten()
        # if features shorter than expected, pad
        if features.numel() < self.q.in_features:
            pad = torch.zeros(self.q.in_features - features.numel(), dtype=features.dtype)
            features = torch.cat([features, pad], dim=0)
        return self.q(features[: self.q.in_features])



# ============================
# Core rollout (generic)
# ============================

def _make_env(env_id: str, **kwargs):
    import gymnasium as gym
    return gym.make(env_id, **kwargs)


def rollout_reinforce(
    genome: CircuitGenome,
    params: dict[str, torch.nn.Parameter],
    *,
    spec: RLSpec,
    episode_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    env = _make_env(spec.env_id)
    obs, _ = env.reset(seed=episode_seed)

    logps: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    ep_return = 0.0

    for _ in range(spec.max_steps):
        x = spec.obs_encoder(obs)

        # forward
        out = genome.circuit(x, params)  # expvals/probs depending on spec
        if spec.return_expvals:
            out = torch.stack(list(out))
            logits = logits_from_kexpvals(out, spec.n_actions)
        else:
            # if you decide to use probs instead, convert to logits safely
            probs = torch.as_tensor(out, dtype=torch.float32).flatten()
            probs = probs[:spec.n_actions]
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
    """
    Collect one episode worth of transitions (or until max_steps).
    Stores:
      obs_tensors: list[Tensor] each shape [D]
      actions:     list[int]
      logps_old:   list[Tensor] scalar
      values_old:  list[Tensor] scalar
      rewards:     list[Tensor] scalar
      dones:       list[Tensor] scalar {0,1}
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

    for t in range(spec.max_steps):
        x = spec.obs_encoder(obs)  # Tensor[D] float or int bits
        out = genome.circuit(x, params)  # expvals/probs
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
    # ensure qnode exists
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        genome.generate_pennylane_circuit(input_mode=spec.input_mode, measure_registers=spec.return_expvals)

    params = genome_to_torch_params(genome)

    returns = []
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
                probs = probs[:spec.n_actions]
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


def train_reinforce(
    genome: CircuitGenome,
    *,
    spec: RLSpec,
) -> CircuitGenome:
    """
    Generic REINFORCE trainer for any env.
    Needs:
      - spec.obs_encoder
      - circuit returning expvals or probs
      - spec.n_actions
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    # build qnode
    genome.generate_pennylane_circuit(input_mode=spec.input_mode, measure_registers=spec.return_expvals)

    params = genome_to_torch_params(genome)

    if len(params) == 0:
        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 999)
        genome.fitness = {
            "note": "no trainable params", 
            "best_episode_return": ev["eval_return_mean"],
            **ev
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

        if spec.baseline == "mean":
            adv = G - G.mean()
        else:
            adv = G

        # REINFORCE
        loss_pg = -(logps * adv.detach()).mean()

        # entropy bonus
        loss_ent = -spec.entropy_coef * entropies.mean() if spec.entropy_coef > 0 else 0.0

        loss = loss_pg + (loss_ent if isinstance(loss_ent, torch.Tensor) else 0.0)

        if not loss.requires_grad:
            logger.warning("Loss has no grad path. Are params used inside the QNode?")
            break

        loss.backward()
        opt.step()

        if (ep % spec.log_every) == 0:
            avg10 = float(np.mean(recent_returns[-10:])) if len(recent_returns) >= 10 else float(np.mean(recent_returns))
            logger.info(f"[ep {ep:04d}] loss={float(loss.item()):.4f} return={ep_ret:.1f} avg10={avg10:.1f}")

    # write back
    torch_params_to_genome(genome, params)

    # final eval
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

def train_actor_critic(
    genome: CircuitGenome,
    *,
    spec: RLSpec,
) -> CircuitGenome:
    """
    On-policy Actor-Critic (A2C-ish):
      - collect transitions up to spec.rollout_steps
      - compute GAE advantages + returns
      - update policy+value jointly
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    genome.generate_pennylane_circuit(input_mode=spec.input_mode, measure_registers=spec.return_expvals)
    params = genome_to_torch_params(genome)

    # handle no-params case
    if len(params) == 0:
        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 999)
        genome.fitness = {"note": "no trainable params", **ev, "env_id": spec.env_id}
        return genome

    # policy/value head input dim = number of outputs from circuit
    # for expvals: should be len(output_qubits) because circuit returns one expval per output wire.
    # we infer it by one forward pass
    with torch.no_grad():
        dummy = spec.obs_encoder(_make_env(spec.env_id, **(spec.env_kwargs or {})).reset(seed=spec.seed)[0])
        out0 = genome.circuit(dummy, params)
        in_dim = int(torch.as_tensor(out0).numel())

    head = PolicyValueHead(n_actions=spec.n_actions, in_dim=in_dim)
    opt = torch.optim.Adam(list(params.values()) + list(head.parameters()), lr=spec.lr, weight_decay=0.0)

    best_episode_return = -float("inf")
    recent_returns: list[float] = []

    # We run multiple updates, reusing spec.episodes as "num updates"
    for upd in range(spec.episodes):
        # collect rollout_steps transitions total
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        steps_collected = 0
        ep = 0

        while steps_collected < spec.rollout_steps:
            traj = rollout_actor_critic(
                genome, params, spec=spec, head=head, episode_seed=spec.seed + upd * 10_000 + ep
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

        # tensors
        obs_t = torch.stack([o.to(torch.float32) for o in obs_buf], dim=0)  # [T,D] (basis bits will cast)
        act_t = torch.stack(act_buf).long().view(-1)                        # [T]
        logp_old_t = torch.stack(logp_buf).to(torch.float32).view(-1)       # [T]
        val_old_t = torch.stack(val_buf).to(torch.float32).view(-1)         # [T]
        rew_t = torch.stack(rew_buf).to(torch.float32).view(-1)             # [T]
        done_t = torch.stack(done_buf).to(torch.float32).view(-1)           # [T]

        # GAE
        adv_t, ret_t = gae_advantages(
            rew_t, val_old_t, done_t, gamma=spec.gamma, lam=spec.gae_lambda, last_value=0.0
        )
        adv_t = _normalize(adv_t)

        # ---- update ----
        opt.zero_grad()

        new_logps = []
        new_vals = []
        entropies = []

        for i in range(obs_t.shape[0]):
            x = obs_t[i]
            out = genome.circuit(x, params)
            feats = torch.as_tensor(out, dtype=torch.float32).flatten()
            logits, v = head(feats)
            dist = Categorical(logits=logits)

            new_logps.append(dist.log_prob(act_t[i]))
            entropies.append(dist.entropy())
            new_vals.append(v)

        new_logp_t = torch.stack(new_logps)      # [T]
        new_val_t = torch.stack(new_vals)        # [T]
        ent_t = torch.stack(entropies)           # [T]

        # policy loss
        loss_pi = -(new_logp_t * adv_t.detach()).mean()

        # value loss
        loss_v = 0.5 * (ret_t.detach() - new_val_t).pow(2).mean()

        # entropy bonus
        loss_ent = -spec.entropy_coef * ent_t.mean() if spec.entropy_coef > 0 else 0.0

        loss = loss_pi + spec.value_coef * loss_v + (loss_ent if isinstance(loss_ent, torch.Tensor) else 0.0)

        if not loss.requires_grad:
            logger.warning("Loss has no grad path. Are params used inside the QNode?")
            break

        loss.backward()
        opt.step()

        if (upd % spec.log_every) == 0:
            avg10 = float(np.mean(recent_returns[-10:])) if len(recent_returns) >= 10 else float(np.mean(recent_returns))
            logger.info(f"[A2C upd {upd:04d}] loss={float(loss.item()):.4f} pi={float(loss_pi.item()):.4f} v={float(loss_v.item()):.4f} avg10ret={avg10:.1f}")

    # write back circuit params
    torch_params_to_genome(genome, params)

    ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)
    train_tail = float(np.mean(recent_returns[-20:])) if len(recent_returns) >= 20 else float(np.mean(recent_returns) if recent_returns else 0.0)

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

def train_ppo(
    genome: CircuitGenome,
    *,
    spec: RLSpec,
) -> CircuitGenome:
    """
    PPO (clip) with GAE and multiple epochs over collected rollout buffer.
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    genome.generate_pennylane_circuit(input_mode=spec.input_mode, measure_registers=spec.return_expvals)
    params = genome_to_torch_params(genome)

    if len(params) == 0:
        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 999)
        genome.fitness = {"note": "no trainable params", **ev, "env_id": spec.env_id}
        return genome

    # infer feature dim from one forward pass
    with torch.no_grad():
        env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
        o0, _ = env.reset(seed=spec.seed)
        env.close()
        x0 = spec.obs_encoder(o0)
        out0 = genome.circuit(x0, params)
        in_dim = int(torch.as_tensor(out0).numel())

    head = PolicyValueHead(n_actions=spec.n_actions, in_dim=in_dim)
    opt = torch.optim.Adam(list(params.values()) + list(head.parameters()), lr=spec.lr, weight_decay=0.0)

    best_episode_return = -float("inf")
    recent_returns: list[float] = []

    for upd in range(spec.episodes):
        # -------- collect rollout buffer --------
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        steps_collected = 0
        ep = 0

        while steps_collected < spec.rollout_steps:
            traj = rollout_actor_critic(
                genome, params, spec=spec, head=head, episode_seed=spec.seed + upd * 10_000 + ep
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

        obs_t = torch.stack([o.to(torch.float32) for o in obs_buf], dim=0)      # [T,D]
        act_t = torch.stack(act_buf).long().view(-1)                            # [T]
        logp_old_t = torch.stack(logp_buf).to(torch.float32).view(-1)           # [T]
        val_old_t = torch.stack(val_buf).to(torch.float32).view(-1)             # [T]
        rew_t = torch.stack(rew_buf).to(torch.float32).view(-1)                 # [T]
        done_t = torch.stack(done_buf).to(torch.float32).view(-1)               # [T]

        adv_t, ret_t = gae_advantages(
            rew_t, val_old_t, done_t, gamma=spec.gamma, lam=spec.gae_lambda, last_value=0.0
        )
        adv_t = _normalize(adv_t)

        # indices for minibatches
        mb = min(spec.ppo_minibatch, T)
        idx_all = torch.randperm(T)

        # -------- PPO updates --------
        for epoch in range(spec.ppo_epochs):
            idx_all = idx_all[torch.randperm(T)]  # reshuffle each epoch

            for start in range(0, T, mb):
                idx = idx_all[start : start + mb]

                # recompute under current params (grad-enabled)
                new_logps = []
                new_vals = []
                entropies = []

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

                new_logp = torch.stack(new_logps)   # [B]
                new_val = torch.stack(new_vals)     # [B]
                ent = torch.stack(entropies)        # [B]

                ratio = torch.exp(new_logp - logp_old_t[idx])

                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - spec.ppo_clip, 1.0 + spec.ppo_clip) * adv_t[idx]
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v = 0.5 * (ret_t[idx].detach() - new_val).pow(2).mean()

                loss_ent = -spec.entropy_coef * ent.mean() if spec.entropy_coef > 0 else 0.0

                loss = loss_pi + spec.value_coef * loss_v + (loss_ent if isinstance(loss_ent, torch.Tensor) else 0.0)

                opt.zero_grad()
                loss.backward()
                opt.step()

                # optional early stop on KL
                if spec.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = (logp_old_t[idx] - new_logp).mean().item()
                    if approx_kl > float(spec.target_kl):
                        break

        if (upd % spec.log_every) == 0:
            avg10 = float(np.mean(recent_returns[-10:])) if len(recent_returns) >= 10 else float(np.mean(recent_returns))
            logger.info(f"[PPO upd {upd:04d}] avg10ret={avg10:.1f} bufferT={T}")

    # write back circuit params
    torch_params_to_genome(genome, params)

    ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)
    train_tail = float(np.mean(recent_returns[-20:])) if len(recent_returns) >= 20 else float(np.mean(recent_returns) if recent_returns else 0.0)

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

def _forward_features(genome: CircuitGenome, x: torch.Tensor, params):
    out = genome.circuit(x, params)  # expvals/probs
    return torch.as_tensor(out, dtype=torch.float32).flatten()


@torch.no_grad()
def _epsilon_greedy_action(q_values: torch.Tensor, epsilon: float) -> int:
    if torch.rand(()) < epsilon:
        return int(torch.randint(low=0, high=q_values.numel(), size=(1,)).item())
    return int(torch.argmax(q_values).item())

def _infer_feature_dim(genome: CircuitGenome, spec: RLSpec) -> int:
    env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
    obs, _ = env.reset(seed=spec.seed)
    env.close()
    x = spec.obs_encoder(obs)
    with torch.no_grad():
        params0 = genome_to_torch_params(genome)
        out0 = genome.circuit(x, params0)
    return int(torch.as_tensor(out0).numel())

def train_value_based(
    genome: CircuitGenome,
    *,
    spec: RLSpec,
) -> CircuitGenome:
    """
    Value-based learning with function approximation:
      - algo="q_learning": DQN-style TD target uses max_a' Q_target(s', a')
      - algo="sarsa":      TD target uses Q_target(s', a'_epsilon_greedy)  (on-policy)

    Uses:
      - circuit outputs -> features
      - small linear QHead -> Q-values
      - optional replay buffer + target QHead
    """
    if spec.obs_encoder is None:
        raise ValueError("spec.obs_encoder must be provided")

    # Build qnode: expvals are ideal as features; probs also work.
    genome.generate_pennylane_circuit(input_mode=spec.input_mode, measure_registers=spec.return_expvals)

    params = genome_to_torch_params(genome)

    # If no trainable circuit params exist, we can still train QHead, but you probably want params.
    # We'll allow it: QHead learns on top of fixed circuit.
    feat_dim = _infer_feature_dim(genome, spec)
    q_head = QHead(n_actions=spec.n_actions, in_dim=feat_dim)
    q_tgt = QHead(n_actions=spec.n_actions, in_dim=feat_dim)
    q_tgt.load_state_dict(q_head.state_dict())

    opt = torch.optim.Adam(list(params.values()) + list(q_head.parameters()), lr=spec.lr, weight_decay=0.0)

    rb = ReplayBuffer(spec.replay_capacity)

    epsilon = float(spec.epsilon)
    best_episode_return = -float("inf")
    recent_returns: list[float] = []

    global_step = 0

    for ep in range(spec.episodes):
        env = _make_env(spec.env_id, **(spec.env_kwargs or {}))
        obs, _ = env.reset(seed=spec.seed + ep)

        ep_ret = 0.0

        for t in range(spec.max_steps):
            x = spec.obs_encoder(obs)

            # current Q(s, ·)
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

            # learn
            if len(rb) >= spec.warmup_steps and (global_step % spec.train_every) == 0:
                batch = rb.sample(spec.td_batch_size)

                s = torch.stack([b[0].to(torch.float32) for b in batch], dim=0)
                a_b = torch.tensor([b[1] for b in batch], dtype=torch.long)
                r_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                s2 = torch.stack([b[3].to(torch.float32) for b in batch], dim=0)
                d_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                # compute Q(s,a)
                q_sa = []
                for i in range(s.shape[0]):
                    feats_i = _forward_features(genome, s[i], params)
                    q_i = q_head(feats_i)
                    q_sa.append(q_i[a_b[i]])
                q_sa = torch.stack(q_sa)  # [B]

                # TD target
                with torch.no_grad():
                    q_next = []
                    for i in range(s2.shape[0]):
                        feats2_i = _forward_features(genome, s2[i], params)
                        q2_i = q_tgt(feats2_i)

                        if spec.algo == "sarsa":
                            # on-policy next action via epsilon-greedy under current QHead (or target)
                            # using target net for stability is fine.
                            a2 = _epsilon_greedy_action(q2_i, epsilon)
                            q_next.append(q2_i[a2])
                        else:
                            # q_learning / DQN target: max
                            q_next.append(torch.max(q2_i))

                    q_next = torch.stack(q_next)  # [B]

                    target = r_b + spec.gamma * (1.0 - d_b) * q_next

                loss = torch.mean((q_sa - target) ** 2)

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

        # epsilon schedule
        epsilon = max(spec.epsilon_min, epsilon * spec.epsilon_decay)

        if (ep % spec.log_every) == 0:
            avg10 = float(np.mean(recent_returns[-10:])) if len(recent_returns) >= 10 else float(np.mean(recent_returns))
            logger.info(
                f"[{spec.algo.upper()} ep {ep:04d}] return={ep_ret:.1f} avg10={avg10:.1f} eps={epsilon:.3f} rb={len(rb)}"
            )

    # write back circuit params
    torch_params_to_genome(genome, params)

    ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)
    train_tail = float(np.mean(recent_returns[-20:])) if len(recent_returns) >= 20 else float(np.mean(recent_returns) if recent_returns else 0.0)

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
    *,
    episodes: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
    algo: str = "reinforce",
) -> RLSpec:
    # CartPole obs dim=4, actions=2
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
    """
    FrozenLake observations are discrete states: 0..n_states-1
    We'll basis-encode into bits => input_mode="basis".
    """
    # 4x4 => 16 states => 4 bits
    n_states = 16 if map_name == "4x4" else 64
    n_bits = int(np.ceil(np.log2(n_states)))

    def encoder(obs):
        # obs is an int state index
        return encode_discrete_to_bits(int(obs), n_bits)

    # IMPORTANT: gymnasium FrozenLake config is set in env creation, not here.
    # If you need map_name / is_slippery, you can implement a custom _make_env.
    # For now, use standard FrozenLake-v1 defaults, or patch _make_env accordingly.
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
        env_kwargs={
            "is_slippery": is_slippery,
        }
    )
