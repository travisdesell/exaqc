from __future__ import annotations

import argparse
import os
import sys
from loguru import logger

from src.evolution.master_worker import master_worker
from src.evolution.steady_state_islands import SteadyStateIslands
from src.evolution.steady_state_population import SteadyStatePopulation
from src.evolution.objective import Objective
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.circuit import CircuitGenome

from src.objectives.policy_objectives import (
    RLSpec,
    cartpole_spec,
    frozenlake_spec,
    mountaincar_continuous_spec,
    halfcheetah_spec,
    walker2d_spec,
    eval_policy,
    train_rl,
    minigrid_spec,
)


def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """
    Sort genomes by evaluation return descending (higher is better).

    Args:
        genome1: First genome.
        genome2: Second genome.

    Returns:
        Negative if genome1 should come before genome2, positive if genome2
        should come before genome1, and 0 if equivalent.
    """
    r1 = float(genome1.fitness.get("eval_return_mean", -1e9))
    r2 = float(genome2.fitness.get("eval_return_mean", -1e9))

    if r1 > r2:
        return -1
    if r1 < r2:
        return 1
    return 0


class RLObjective(Objective):
    """
    Objective wrapper for RL environments.

    This class maps genome hyperparameters into an ``RLSpec``, trains the
    genome using the selected algorithm, evaluates it, and stores fitness.

    Attributes:
        spec: Base RL specification.
        target: Backend target string used by the broader framework.
        input_size: Number of encoded input features expected by the circuit.
        n_actions: Number of discrete actions (0 for continuous-action tasks).
    """

    def __init__(self, *, spec: RLSpec):
        """
        Initialize RL objective.

        Args:
            spec: Base RL specification.
        """
        self.spec = spec
        self.target = "pennylane"

        # Determine encoded input size
        if spec.input_mode in ["angle", "amplitude"]:
            if hasattr(spec, "n_state_bits") and spec.n_state_bits is not None:
                self.input_size = int(spec.n_state_bits)
            elif hasattr(spec, "box_scales") and spec.box_scales is not None:
                self.input_size = int(len(spec.box_scales))
            else:
                if spec.obs_dim is not None:
                    self.input_size = int(spec.obs_dim)
                elif spec.box_scales is not None:
                    self.input_size = int(len(spec.box_scales))
                elif spec.env_id == "CartPole-v1":
                    self.input_size = 4
                elif spec.env_id == "MountainCarContinuous-v0":
                    self.input_size = 2
                elif spec.env_id == "HalfCheetah-v5":
                    self.input_size = 17
                elif spec.env_id == "Walker2d-v5":
                    self.input_size = 17

                else:
                    raise ValueError(
                        f"Could not infer input size for env_id={spec.env_id}. "
                        "Set box_scales or expose an explicit input size in the spec."
                    )
        elif spec.input_mode == "basis":
            if spec.n_state_bits is None:
                raise ValueError("spec.n_state_bits required for basis mode")
            self.input_size = int(spec.n_state_bits)
        else:
            raise ValueError(f"Unknown input_mode: {spec.input_mode}")

        self.n_actions = int(spec.n_actions)

    def __call__(self, genome: CircuitGenome):
        """
        Train and evaluate a genome on the configured RL environment.

        Args:
            genome: Genome to train and evaluate.
        """
        hp = genome.hyperparameters
        spec = RLSpec(**{**self.spec.__dict__})

        spec.episodes = int(hp["episodes"])
        spec.lr = float(hp["learning_rate"])
        spec.gamma = float(hp["gamma"])
        spec.max_steps = int(hp["max_steps"])
        spec.eval_episodes = int(hp["eval_episodes"])
        spec.entropy_coef = float(hp["entropy_coef"])
        spec.baseline = str(hp["baseline"])
        spec.log_every = int(hp["log_every"])
        spec.seed = int(hp["seed"])

        if "rollout_steps" in hp:
            spec.rollout_steps = int(hp["rollout_steps"])
        if "ppo_epochs" in hp:
            spec.ppo_epochs = int(hp["ppo_epochs"])
        if "ppo_minibatch" in hp:
            spec.ppo_minibatch = int(hp["ppo_minibatch"])
        if "ppo_clip" in hp:
            spec.ppo_clip = float(hp["ppo_clip"])
        if "gae_lambda" in hp:
            spec.gae_lambda = float(hp["gae_lambda"])
        if "value_coef" in hp:
            spec.value_coef = float(hp["value_coef"])

        if "epsilon" in hp:
            spec.epsilon = float(hp["epsilon"])
        if "epsilon_min" in hp:
            spec.epsilon_min = float(hp["epsilon_min"])
        if "epsilon_decay" in hp:
            spec.epsilon_decay = float(hp["epsilon_decay"])

        spec.env_kwargs = hp.get("env_kwargs", spec.env_kwargs)

        train_rl(genome, spec=spec, algo=spec.algo)

        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)

        genome.fitness = {
            **(genome.fitness or {}),
            "eval_return_mean": float(ev["eval_return_mean"]),
            "eval_return_std": float(ev["eval_return_std"]),
            "env_id": spec.env_id,
            "n_actions": spec.n_actions,
            "input_mode": spec.input_mode,
            "action_space": getattr(spec, "action_space", "discrete"),
        }

        logger.info(
            f"[{genome.genome_number:04d}] "
            f"train_return_mean={float(genome.fitness.get('train_return_mean', 0.0)):.2f} "
            f"best_episode_return={float(genome.fitness.get('best_episode_return', 0.0)):.2f} "
            f"eval_return_mean={genome.fitness['eval_return_mean']:.2f} "
            f"eval_return_std={genome.fitness['eval_return_std']:.2f} "
            f"env={spec.env_id}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env",
        choices=[
            "cartpole",
            "frozenlake",
            "mountaincar_continuous",
            "halfcheetah",
            "walker2d",
            "minigrid",
        ],
        required=True,
    )
    p.add_argument("--minigrid_env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    p.add_argument(
        "--minigrid_obs_wrapper",
        choices=["flat", "image"],
        default="flat",
    )
    p.add_argument(
        "--algo",
        choices=["reinforce", "a2c", "ppo", "q_learning", "sarsa"],
        required=True,
        default="reinforce",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs; "
        "make sure to mention the run number as 'run#' at the end",
    )

    # Evolution
    p.add_argument("--max_population_size", type=int, default=30)
    p.add_argument("--number_genomes", type=int, default=500)

    p.add_argument(
        "--mutation_strategy",
        "-ms",
        type=str,
        nargs="+",
        required=True,
    )

    # Registers
    p.add_argument("--input_qubits", type=int, default=6)
    p.add_argument("--output_qubits", type=int, default=None)

    # Islands
    subparsers = p.add_subparsers(
        dest="population_strategy",
        help="Specify how genomes will be handled.",
        required=True,
    )

    steady_state_parser = subparsers.add_parser(
        "steady_state", help="Use a single steady state population."
    )
    steady_state_parser.add_argument("--max_population_size", type=int, default=30)

    islands_parser = subparsers.add_parser(
        "islands", help="Use multiple islands of steady state opulations."
    )
    islands_parser.add_argument("--n_islands", type=int, default=10)
    islands_parser.add_argument("--max_island_size", type=int, default=10)
    islands_parser.add_argument("--genomes_before_extinction", type=int, default=100)
    islands_parser.add_argument("--genomes_for_next_extinction", type=int, default=200)
    islands_parser.add_argument("--islands_to_extinct", type=int, default=1)
    islands_parser.add_argument(
        "--intra_island_crossover_rate", type=float, default=0.5
    )

    # RL hyperparams
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning_rate", "--lr", type=float, default=1e-2)
    p.add_argument("--entropy_coef", type=float, default=0.00)
    p.add_argument("--baseline", choices=["mean", "none"], default="mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)

    # PPO / A2C extras
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--ppo_epochs", type=int, default=10)
    p.add_argument("--ppo_minibatch", type=int, default=256)
    p.add_argument("--ppo_clip", type=float, default=0.2)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--value_coef", type=float, default=0.5)

    # Q-learning / SARSA extras
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_min", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.995)

    # FrozenLake kwargs
    p.add_argument("--map_name", choices=["4x4", "8x8"], default="4x4")
    p.add_argument("--is_slippery", action="store_true")

    # Logging
    p.add_argument(
        "--logging_level",
        type=str,
        required=False,
        default="INFO",
        help="DEBUG/INFO/WARNING/ERROR/CRITICAL",
    )

    args = p.parse_args()

    logger.remove()
    os.makedirs(args.out_dir, exist_ok=True)
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"))

    # -------------------------------------------------------------
    # Build environment-specific spec
    # -------------------------------------------------------------
    if args.env == "cartpole":
        spec = cartpole_spec(
            episodes=args.episodes,
            lr=args.learning_rate,
            seed=args.seed,
            algo=args.algo,
        )
        default_out_qubits = 2

    elif args.env == "frozenlake":
        spec = frozenlake_spec(
            map_name=args.map_name,
            is_slippery=args.is_slippery,
            episodes=args.episodes,
            lr=args.learning_rate,
            seed=args.seed,
            algo=args.algo,
        )
        spec.env_kwargs = {
            "map_name": args.map_name,
            "is_slippery": args.is_slippery,
        }
        default_out_qubits = 4

    elif args.env == "mountaincar_continuous":
        spec = mountaincar_continuous_spec(
            episodes=args.episodes,
            lr=args.learning_rate,
            seed=args.seed,
            algo=args.algo,
        )
        # Continuous actions; circuit outputs features, not direct action bins.
        default_out_qubits = 2

    elif args.env == "halfcheetah":
        spec = halfcheetah_spec(
            episodes=args.episodes,
            lr=args.learning_rate,
            seed=args.seed,
            algo=args.algo,
        )
        # Continuous action env; 2-6 output feature qubits is a reasonable start.
        default_out_qubits = 4

    elif args.env == "walker2d":
        spec = walker2d_spec(
            episodes=args.episodes,
            lr=args.learning_rate,
            seed=args.seed,
            algo=args.algo,
        )
        default_out_qubits = 4

    elif args.env == "minigrid":
        spec = minigrid_spec(
            env_id=args.minigrid_env_id,
            obs_wrapper=args.minigrid_obs_wrapper,
            episodes=args.episodes,
            lr=args.learning_rate,
            seed=args.seed,
            algo=args.algo,
        )
        # discrete env; output features/logits
        default_out_qubits = 4

    else:
        raise ValueError(args.env)

    objective = RLObjective(spec=spec)

    out_qubits = (
        int(args.output_qubits)
        if args.output_qubits is not None
        else default_out_qubits
    )

    hyperparameters = {
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "baseline": args.baseline,
        "seed": args.seed,
        "log_every": args.log_every,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "ppo_minibatch": args.ppo_minibatch,
        "ppo_clip": args.ppo_clip,
        "gae_lambda": args.gae_lambda,
        "value_coef": args.value_coef,
        "epsilon": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "env_kwargs": getattr(spec, "env_kwargs", None),
    }

    input_registers = {"input": min(args.input_qubits, objective.input_size)}
    output_registers = {"output": out_qubits}

    logger.info(
        f"env={spec.env_id} "
        f"algo={spec.algo} "
        f"input_registers={input_registers} "
        f"output_registers={output_registers}"
    )

    population = None
    print(f"args.population_strategy: {args.population_strategy}")

    if args.population_strategy == "steady_state":
        population = SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
        )
    elif args.population_strategy == "islands":
        population = SteadyStateIslands(
            n_islands=args.n_islands,
            max_island_size=args.max_island_size,
            genomes_before_extinction=args.genomes_before_extinction,
            genomes_for_next_extinction=args.genomes_for_next_extinction,
            islands_to_extinct=args.islands_to_extinct,
            compare=compare,
            out_dir=args.out_dir,
        )

    # run
    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=population,
        objective=objective,
        hyperparameters=hyperparameters,
        mutation_strategy=args.mutation_strategy,
        run_for=args.number_genomes,
        input_registers=input_registers,
        output_registers=output_registers,
        target="pennylane",
    )
