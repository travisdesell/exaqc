"""Evolve quantum genomes for reinforcement learning with EXAQC.

This is the reinforcement-learning counterpart to
:mod:`src.examples.classification`, refactored to reuse the same modular
building blocks:

* the genome's ``initialize_model`` / ``forward`` hybrid-model interface,
* the existing ``LinearEncoder`` / ``LinearDecoder`` (and identity/clipped
  variants) for embedding observations into the circuit and mapping circuit
  outputs to per-action values,
* an :class:`~src.evolution.objective.Objective` that wraps a *trainer* and
  sets genome fitness,
* the same ``master_worker`` evolutionary driver.

The RL algorithms live in :mod:`src.trainer.reinforcement_trainer` as
pluggable trainer classes (REINFORCE, actor-critic, PPO, Q-learning), exactly
mirroring how ``SupervisedTrainer`` is a pluggable component of the
classification objective. The environment is described by a pluggable
:class:`~src.trainer.reinforcement_trainer.RLEnvironment`, which the
:class:`ReinforcementLearningObjective` receives together with a trainer.

Example (single-process smoke run is driven programmatically via the
objective; the ``__main__`` block wires everything into ``master_worker`` for
an MPI evolutionary search)::

    mpirun -n 4 python -m src.examples.reinforcement_learning \\
        --env cartpole --algo ppo -ms "uniform 1 3" -ps "uniform 2 3" \\
        --batch_placeholder steady_state
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import torch

from collections.abc import Callable
from typing import Any

from loguru import logger
from torch import Tensor

from src.trainer.reinforcement_trainer import RLEnvironment
from src.trainer.rl_trainer_registry import TRAINER_REGISTRY

from src.examples.reinforcement_learning import (
    build_trainer,
    ENV_CHOICES,
    make_environment,
)

# visualize() and save_gif() are model-agnostic -- they drive the policy only
# through greedy_action(model, environment, observation), which needs nothing
# more than model.forward -- so they are reused unchanged for the classical
# ClassicalModel here, exactly as for a CircuitGenome.
from src.examples.visualize_rl import save_gif, visualize

# ---------------------------------------------------------------------
# Generic Classical RL MLP Model
# ---------------------------------------------------------------------


class ClassicalModel(torch.nn.Module):
    def __init__(
        self,
        hyperparameters: dict[str, Any],
        n_inputs: int,
        n_outputs: int,
        layer_sizes: list[int] = [64, 64],
        hidden_activation: Callable[[Tensor], Tensor] = torch.tanh,
        output_activation: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """
        Builds a classical multi-layer perceptron usable by the RL trainers.

        Args:
            hyperparameters: include all training and environment hyperparameters for the classical model.
            n_inputs: how many classical input features will be used.
            n_outputs: how many outputs the model produces (the policy outputs
                plus any value output the trainer expects).
            layer_sizes: the size of each hidden layer in the model.
            hidden_activation: activation applied after every hidden layer
                (every layer except the last). Defaults to ``torch.tanh``, a
                common and stable choice for policy-gradient RL networks
                (``torch.relu`` is the usual alternative).
            output_activation: activation applied after the final output layer,
                or None (the default) for a raw linear output. The default is
                None because the trainers interpret the raw output differently
                per environment/algorithm -- as unbounded policy logits, as
                Q-values, or as continuous-policy means + log-stds together with
                a state value -- none of which should be squashed. Only pass an
                output activation (e.g. ``torch.tanh``) if you have deliberately
                sized the output to be values that belong in that activation's
                range (e.g. action means in a ``[-1, 1]`` action space).
        """
        # initialize both superclasses
        torch.nn.Module.__init__(self)

        self.genome_number = 0
        self.hyperparameters = hyperparameters
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # the trainer will set metadata for the model and its training statistics
        self.metadata = {}

        logger.info(
            f"creating RL model with n_inputs: {n_inputs}, hidden layer sizes {layer_sizes} and n_outputs: {n_outputs}"
        )

        # add the input and output layer sizes so we can create all the layers
        # in a single loop
        layer_sizes = [n_inputs] + layer_sizes + [n_outputs]

        # create the layers -- held in an nn.ModuleList (NOT a plain list) so
        # that the Linear layers are registered as submodules and their
        # parameters are discoverable via self.parameters(). A plain Python
        # list is invisible to nn.Module registration, so parameters() would
        # return nothing and the trainer would see zero trainable parameters.
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            logger.info(
                f"\tcreating a layer with {layer_sizes[i]} inputs and {layer_sizes[i + 1]} outputs."
            )

            # initalize the layer weights randomly
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

            self.layers.append(layer)

        n_trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        logger.info(f"n parameters: {n_trainable_parameters}")

    def initialize_model(self) -> None:
        """
        This is called on QuantumCircuits to set up the hybrid model. We don't need to do anything here but
        need this method so that we can re-use the trainer.
        """
        return

    def set_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Sets the model's state dict to a state dict previously
        obtained with :meth:`clone_state_dict`.

        Args:
            state_dict: a state_dict previously obtained from this
                model.
        """
        self.load_state_dict(state_dict)

    def clone_state_dict(self) -> dict[str, Tensor]:
        """Returns a detached, cloned snapshot of the model's state.

        The returned mapping is a deep copy of ``self.state_dict()``
        (every tensor detached and cloned), so it is safe to hold onto while
        the live model keeps training and later restore via
        :meth:`set_state_dict`. Trainers use this to snapshot the
        best-performing weights during a run.

        Returns:
            A mapping from parameter name to a detached clone of its tensor,
            suitable to pass to :meth:`set_state_dict`.
        """

        with torch.no_grad():
            return {
                name: tensor.detach().clone()
                for name, tensor in self.state_dict().items()
            }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies all the layers of the model to the inputs and returns the
        model output.

        The hidden activation is applied after every hidden layer (so a stack
        of linear layers does not collapse into a single linear map), and the
        output activation (if any) is applied after the final layer.

        Args:
            inputs: the x (input) tensor for a sample

        Returns:
            The output tensor of the model
        """

        x = inputs
        last_index = len(self.layers) - 1

        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < last_index:
                x = self.hidden_activation(x)
            elif self.output_activation is not None:
                x = self.output_activation(x)

        return x


# ---------------------------------------------------------------------
# Visualization (reuses visualize_rl.visualize / save_gif)
# ---------------------------------------------------------------------


def run_visualization(
    model: torch.nn.Module,
    environment: RLEnvironment,
    *,
    render_mode: str,
    episodes: int,
    max_steps: int,
    seed: int,
    output_file: str | None = None,
    fps: int = 30,
) -> list[float]:
    """Rolls the greedy policy for several episodes and logs best-episode stats.

    Reuses ``visualize`` / ``save_gif`` from :mod:`src.examples.visualize_rl`
    to drive the model greedily in the requested render mode, then logs the
    rollout's best-episode return along with the mean and standard deviation.
    When ``render_mode`` is ``"rgb_array"`` and ``output_file`` is given, the
    collected frames are written to that GIF path.

    Args:
        model: The trained model to visualize (its best weights); only its
            ``forward`` is used, via ``greedy_action``.
        environment: The environment to roll episodes in.
        render_mode: ``"human"`` for a live window, or ``"rgb_array"`` to
            collect frames for a GIF.
        episodes: Number of episodes to roll.
        max_steps: Maximum number of steps per episode.
        seed: Base seed; episode ``i`` uses ``seed + i``.
        output_file: Destination GIF path, used only when ``render_mode`` is
            ``"rgb_array"``; ignored otherwise.
        fps: Frames per second for the saved GIF.

    Returns:
        The per-episode returns of the rollout (empty only if ``episodes`` is
        zero).
    """

    returns, frames = visualize(
        model,
        environment,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        render_mode=render_mode,
    )

    if returns:
        return_std = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        logger.info(
            f"[{render_mode}] visualization over {len(returns)} episode(s): "
            f"best_episode_return={max(returns):.2f} "
            f"return_mean={statistics.fmean(returns):.2f} "
            f"return_std={return_std:.2f} "
            f"env={environment.env_id}"
        )

    if render_mode == "rgb_array" and output_file is not None and frames:
        directory = os.path.dirname(output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        save_gif(frames, output_file, fps=fps)

    return returns


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--env",
        choices=list(ENV_CHOICES),
        required=True,
    )

    p.add_argument(
        "--algo",
        choices=sorted(TRAINER_REGISTRY.keys()),
        required=True,
        default="reinforce",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs",
    )

    # RL hyperparameters (become genome.hyperparameters, mutable by the search)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning_rate", "-lr", type=float, default=1e-2)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--baseline", choices=["mean", "none"], default="mean")
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument(
        "--ema_alpha",
        type=float,
        default=0.05,
        help="Smoothing factor for the exponential moving average of episode "
        "returns reported as the training return mean.",
    )

    # PPO extras
    p.add_argument("--rollout_steps", type=int, default=512)
    p.add_argument(
        "--ppo_passes",
        type=int,
        default=4,
        help="Passes over each PPO rollout (PPO literature calls these 'epochs').",
    )
    p.add_argument("--ppo_minibatch", type=int, default=128)
    p.add_argument("--ppo_clip", type=float, default=0.2)
    p.add_argument("--gae_lambda", type=float, default=0.95)

    # Value-based extras
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_min", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.995)

    # FrozenLake options
    p.add_argument("--map_name", choices=["4x4", "8x8"], default="4x4")
    p.add_argument("--is_slippery", action="store_true")

    p.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="DEBUG/INFO/WARNING/ERROR/CRITICAL",
    )

    # Visualization of the trained (best-weights) model
    p.add_argument(
        "--visualize_episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize the trained model for.",
    )
    p.add_argument(
        "--visualize_seed",
        type=int,
        default=None,
        help="Base seed for the visualization episodes (default: a random seed).",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Show a live human-rendered window of the trained model. "
        "Independent of --output_file (you can do both); if neither is given, "
        "a live window is shown by default.",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="If set, save the rollout to this GIF path (headless).",
    )
    p.add_argument("--fps", type=int, default=30, help="GIF frames per second.")

    args = p.parse_args()

    logger.remove()
    os.makedirs(args.out_dir, exist_ok=True)
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"))

    # -----------------------------------------------------------------
    # Environment + trainer + objective
    # -----------------------------------------------------------------
    environment = make_environment(
        args.env,
        map_name=args.map_name,
        is_slippery=args.is_slippery,
    )

    if environment.deterministic and args.eval_episodes > 1:
        logger.warning(
            f"environment {environment.env_id} is deterministic, so greedy "
            f"evaluation yields identical episodes; --eval_episodes="
            f"{args.eval_episodes} will be reduced to 1 during evaluation."
        )

    trainer = build_trainer(
        args.algo,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        log_every=args.log_every,
        ema_alpha=args.ema_alpha,
        entropy_coef=args.entropy_coef,
        baseline=args.baseline,
        value_coef=args.value_coef,
        gae_lambda=args.gae_lambda,
        rollout_steps=args.rollout_steps,
        ppo_passes=args.ppo_passes,
        ppo_minibatch=args.ppo_minibatch,
        ppo_clip=args.ppo_clip,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    # Value-based trainers (q_learning / sarsa) enumerate discrete actions and
    # cannot drive a continuous Box-action environment; fail fast with a clear
    # message rather than deep inside the first weight update.
    if environment.continuous and not trainer.supports_continuous:
        p.error(
            f"algorithm {args.algo!r} does not support the continuous "
            f"environment {args.env!r}; use reinforce, actor_critic, or ppo."
        )

    # These become each genome's hyperparameters, so the evolutionary search
    # can carry/mutate them per genome (mirroring the classification example).
    hyperparameters = {
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "baseline": args.baseline,
        "value_coef": args.value_coef,
        "gae_lambda": args.gae_lambda,
        "rollout_steps": args.rollout_steps,
        "ppo_passes": args.ppo_passes,
        "ppo_minibatch": args.ppo_minibatch,
        "ppo_clip": args.ppo_clip,
        "epsilon": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "seed": args.seed,
        "log_every": args.log_every,
        "ema_alpha": args.ema_alpha,
    }

    classical_model = ClassicalModel(
        hyperparameters=hyperparameters,
        n_inputs=environment.n_observation_features,
        n_outputs=environment.n_policy_outputs + trainer.n_value_outputs,
    )

    trainer.train(classical_model, environment)

    # trainer.train restored the best-evaluated weights into the model, so
    # classical_model now holds the best policy found during training.
    training_metrics = classical_model.metadata["best_training_metrics"]
    validation_metrics = classical_model.metadata["best_validation_metrics"]
    best_episode = classical_model.metadata.get("best_episode")

    logger.info(
        f"best_episode={best_episode} "
        f"train_return_ema={training_metrics['return_mean']:.2f} "
        f"best_episode_return={training_metrics['best_episode_return']:.2f} "
        f"eval_return_mean={validation_metrics['return_mean']:.2f} "
        f"eval_return_std={validation_metrics['return_std']:.2f} "
        f"env={environment.env_id}"
    )

    # -----------------------------------------------------------------
    # Visualize the trained (best-weights) model
    # -----------------------------------------------------------------
    # --live shows a live window; --output_file saves a GIF. They are
    # independent (you can do both); when neither is given we default to a live
    # window so there is always a visualization.
    if environment.deterministic and args.visualize_episodes > 1:
        logger.warning(
            f"environment {environment.env_id} is deterministic, so the greedy "
            f"policy yields identical visualization episodes; consider "
            f"--visualize_episodes 1."
        )

    visualize_seed = (
        args.visualize_seed
        if args.visualize_seed is not None
        else random.randrange(2**31)
    )
    save_gif_requested = args.output_file is not None
    show_live = args.live
    logger.info(
        f"visualizing best model (seed={visualize_seed}, live={show_live}, "
        f"gif={save_gif_requested})"
    )

    # Save the GIF first (headless, always works) so a live-render failure on a
    # headless machine does not lose the saved animation.
    if save_gif_requested:
        run_visualization(
            classical_model,
            environment,
            render_mode="rgb_array",
            episodes=args.visualize_episodes,
            max_steps=args.max_steps,
            seed=visualize_seed,
            output_file=args.output_file,
            fps=args.fps,
        )

    if show_live:
        try:
            run_visualization(
                classical_model,
                environment,
                render_mode="human",
                episodes=args.visualize_episodes,
                max_steps=args.max_steps,
                seed=visualize_seed,
            )
        except Exception:
            logger.error(
                "live rendering failed; on a headless machine pass "
                "--output_file PATH to save a GIF instead (and omit --live)."
            )
            raise
