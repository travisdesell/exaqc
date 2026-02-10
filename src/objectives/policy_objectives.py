from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.distributions import Categorical
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.objectives.genome_objectives import genome_to_torch_params, torch_params_to_genome


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


def train_policy_gradient(
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
# Ready-made specs
# ============================

def cartpole_spec(
    *,
    episodes: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
) -> RLSpec:
    # CartPole obs dim=4, actions=2
    scales = np.array([2.4, 3.0, 0.21, 3.5], dtype=np.float32)

    def encoder(obs):
        return encode_box_to_unit_interval(obs, scales=scales)

    return RLSpec(
        env_id="CartPole-v1",
        n_actions=2,
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
