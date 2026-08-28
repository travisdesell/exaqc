"""Visualize a trained RL circuit genome acting in its target environment.

Loads a circuit-genome JSON produced by ``src.examples.reinforcement_learning``
(i.e. by ``CircuitGenome.to_dict``), rebuilds the quantum circuit, reconnects
it to the Gymnasium environment it was trained on, and rolls the greedy policy
so you can *watch* the evolved circuit control the environment.

The environment the genome was designed for is recorded in the genome's
``fitness["env_id"]`` (e.g. ``"CartPole-v1"``), so it is auto-detected from the
JSON; ``--env`` can override it. The genome's own ``encoder``/``decoder`` and
gate parameters are restored from the JSON, so the policy plays exactly as it
was evolved.

Two output modes:

* **Live** (default): renders an interactive window as the policy plays --
  ``python -m src.examples.visualize_rl path/to/genome.json``.
* **Saved** (``--output_file PATH``): headless-friendly; writes an animated
  GIF of the rollout to ``PATH``.

Example::

    python -m src.examples.visualize_rl genome_368.json --episodes 3
    python -m src.examples.visualize_rl genome_368.json --output_file rollout.gif
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

import gymnasium as gym
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.examples.reinforcement_learning import ENV_IDS, make_environment
from src.trainer.reinforcement_trainer import RLEnvironment, greedy_action

#: Maps a Gymnasium environment id back to the friendly ``--env`` name, derived
#: by reversing ``src.examples.reinforcement_learning.ENV_IDS`` (the single
#: source of truth). Covers every supported environment -- discrete and
#: continuous alike -- so a genome trained on any of them can be auto-detected
#: from its recorded ``fitness["env_id"]``.
ENV_ID_TO_NAME: dict[str, str] = {env_id: name for name, env_id in ENV_IDS.items()}


def load_genome(json_path: str) -> CircuitGenome:
    """Loads and initializes a circuit genome from a saved JSON file.

    Args:
        json_path: Path to a genome JSON produced by ``CircuitGenome.to_dict``.

    Returns:
        The reconstructed :class:`CircuitGenome` with its ``hybrid_model``
        already initialized (ready for ``genome.forward``).
    """

    with open(json_path) as json_file:
        serialized = json.load(json_file)

    genome = CircuitGenome.from_dict(serialized)
    genome.initialize_model()
    return genome


def resolve_environment(
    genome: CircuitGenome,
    env_name: str | None,
    *,
    map_name: str = "4x4",
    is_slippery: bool = False,
) -> RLEnvironment:
    """Rebuilds the :class:`RLEnvironment` the genome was designed for.

    The environment name is taken from ``env_name`` when given, otherwise
    auto-detected from the genome's ``fitness["env_id"]``. The rebuilt
    environment is validated against the genome's encoder so a mismatched
    ``--env`` fails with a clear message rather than deep inside a forward
    pass.

    Args:
        genome: The loaded genome (used for auto-detection and validation).
        env_name: Friendly environment name (e.g. ``"cartpole"``), or ``None``
            to auto-detect from the genome.
        map_name: FrozenLake map (``"4x4"`` or ``"8x8"``); ignored otherwise.
        is_slippery: FrozenLake slipperiness; ignored otherwise.

    Returns:
        A configured :class:`RLEnvironment`.

    Raises:
        ValueError: If the environment cannot be determined, or the rebuilt
            environment's observation size does not match the genome encoder.
    """

    if env_name is None:
        env_id = (genome.fitness or {}).get("env_id")
        if env_id is None:
            raise ValueError(
                "Could not determine the environment: the genome has no "
                "fitness['env_id']. Pass --env explicitly."
            )
        if env_id not in ENV_ID_TO_NAME:
            raise ValueError(
                f"Genome was trained on env_id={env_id!r}, which has no known "
                f"--env mapping (known: {sorted(ENV_ID_TO_NAME.values())}). "
                "Pass --env explicitly."
            )
        env_name = ENV_ID_TO_NAME[env_id]
        logger.info(f"auto-detected environment '{env_name}' (env_id={env_id})")

    environment = make_environment(env_name, map_name=map_name, is_slippery=is_slippery)

    # The genome's encoder expects exactly this many observation features; a
    # mismatch means the wrong environment was selected.
    expected = genome.encoder.n_inputs
    if environment.n_observation_features != expected:
        raise ValueError(
            f"Environment '{env_name}' produces "
            f"{environment.n_observation_features} observation features, but "
            f"the genome's encoder expects {expected}. This genome was likely "
            "trained on a different environment/configuration."
        )

    recorded_env_id = (genome.fitness or {}).get("env_id")
    if recorded_env_id is not None and recorded_env_id != environment.env_id:
        logger.warning(
            f"selected environment {environment.env_id} differs from the "
            f"genome's recorded env_id {recorded_env_id}."
        )

    return environment


def play_episode(
    genome: CircuitGenome,
    environment: RLEnvironment,
    gym_env: gym.Env,
    *,
    seed: int,
    max_steps: int,
    collect_frames: bool,
) -> tuple[float, list[np.ndarray]]:
    """Plays one greedy episode, optionally collecting rendered frames.

    Args:
        genome: The initialized genome policy.
        environment: The :class:`RLEnvironment` (encoding/action metadata).
        gym_env: The Gymnasium environment to reset and step through.
        seed: Seed for ``gym_env.reset``.
        max_steps: Maximum number of steps to take.
        collect_frames: If True, append ``gym_env.render()`` frames (requires
            the env to have been created with ``render_mode="rgb_array"``).

    Returns:
        A tuple ``(episode_return, frames)``; ``frames`` is empty unless
        ``collect_frames`` is True.
    """

    observation, _ = gym_env.reset(seed=seed)
    frames: list[np.ndarray] = []
    episode_return = 0.0

    for _ in range(max_steps):
        if collect_frames:
            frames.append(np.asarray(gym_env.render()))

        # greedy_action returns the action already in the env's native format:
        # an int for discrete spaces, a clipped float array for continuous ones.
        action = greedy_action(genome, environment, observation)
        observation, reward, terminated, truncated, _ = gym_env.step(action)
        episode_return += float(reward)

        if terminated or truncated:
            break

    if collect_frames:
        frames.append(np.asarray(gym_env.render()))

    return episode_return, frames


def visualize(
    genome: CircuitGenome,
    environment: RLEnvironment,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
    render_mode: str,
) -> tuple[list[float], list[np.ndarray]]:
    """Rolls the greedy policy for several episodes, rendering each one.

    Args:
        genome: The initialized genome policy.
        environment: The :class:`RLEnvironment` to play in.
        episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        seed: Base seed; episode ``i`` uses ``seed + i``.
        render_mode: ``"human"`` for a live window, or ``"rgb_array"`` to
            collect frames for saving.

    Returns:
        A tuple ``(returns, frames)`` where ``returns`` holds each episode's
        total reward and ``frames`` holds the concatenated rendered frames
        (empty for ``render_mode="human"``).
    """

    collect_frames = render_mode == "rgb_array"
    returns: list[float] = []
    frames: list[np.ndarray] = []

    for episode in range(episodes):
        gym_env = gym.make(
            environment.env_id,
            render_mode=render_mode,
            **(environment.env_kwargs or {}),
        )
        episode_return, episode_frames = play_episode(
            genome,
            environment,
            gym_env,
            seed=seed + episode,
            max_steps=max_steps,
            collect_frames=collect_frames,
        )
        gym_env.close()

        returns.append(episode_return)
        frames.extend(episode_frames)
        logger.info(f"episode {episode:02d}: return={episode_return:.1f}")

    return returns, frames


def save_gif(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Writes rendered frames to an animated GIF.

    Args:
        frames: RGB frames of shape ``(H, W, 3)`` (uint8).
        path: Destination ``.gif`` path.
        fps: Frames per second for playback.
    """

    from PIL import Image

    images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
    )
    logger.info(f"saved rollout animation ({len(frames)} frames) to {path}")


def main() -> None:
    """Parses arguments, loads the genome, and visualizes it in its env."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--genome_json",
        type=str,
        help="Path to a genome JSON produced by the RL example script.",
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENV_ID_TO_NAME.values()),
        default=None,
        help="Environment to evaluate in (default: auto-detected from the genome).",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to play.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per episode (default: from the genome, else 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base environment seed (default: a random seed).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output GIF path. When given, saves the rollout to this file "
        "(headless); when omitted, renders a live window instead.",
    )
    parser.add_argument("--fps", type=int, default=30, help="GIF frames per second.")
    # FrozenLake options (used only when the environment is FrozenLake)
    parser.add_argument("--map_name", choices=["4x4", "8x8"], default="4x4")
    parser.add_argument("--is_slippery", action="store_true")
    parser.add_argument("--logging_level", type=str, default="INFO")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    genome = load_genome(args.genome_json)
    logger.info(
        f"loaded genome {genome.genome_number} (target={genome.target}); "
        f"recorded fitness: {genome.fitness}"
    )

    environment = resolve_environment(
        genome,
        args.env,
        map_name=args.map_name,
        is_slippery=args.is_slippery,
    )

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else int(genome.hyperparameters.get("max_steps", 500))
    )

    seed = args.seed if args.seed is not None else random.randrange(2**31)
    logger.info(f"using seed {seed}")

    render_mode = "rgb_array" if args.output_file is not None else "human"

    try:
        returns, frames = visualize(
            genome,
            environment,
            episodes=args.episodes,
            max_steps=max_steps,
            seed=seed,
            render_mode=render_mode,
        )
    except Exception:
        if render_mode == "human":
            logger.error(
                "live rendering failed; on a headless machine pass "
                "--output_file PATH to save a GIF instead."
            )
        raise

    logger.info(
        f"evaluated {len(returns)} episodes on {environment.env_id}: "
        f"return_mean={np.mean(returns):.1f} return_std={np.std(returns):.1f} "
        f"(recorded eval_return_mean="
        f"{(genome.fitness or {}).get('eval_return_mean')})"
    )

    if args.output_file is not None and frames:
        directory = os.path.dirname(args.output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        save_gif(frames, args.output_file, fps=args.fps)


if __name__ == "__main__":
    main()
