from __future__ import annotations

import argparse
import os
import sys
import math
from typing import Optional

import numpy as np
from loguru import logger

from src.evolution.master_worker import master_worker
from src.evolution.steady_state_population import SteadyStatePopulation
from src.evolution.objective import Objective
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.circuit import CircuitGenome

from src.objectives.policy_objectives import (
    RLSpec,
    cartpole_spec,
    frozenlake_spec,
    train_policy_gradient,
    eval_policy,
)

# ---------------------------------------------------------------------
# Comparison (sorting) for steady-state population
# ---------------------------------------------------------------------

def compare(genome1: CircuitGenome, genome2: CircuitGenome) -> int:
    """
    Sort genomes by test/eval return descending (higher is better).
    Your SteadyStatePopulation expects:
      negative => genome1 before genome2
      positive => genome2 before genome1
    """
    r1 = float(genome1.fitness.get("eval_return_mean", -1e9))
    r2 = float(genome2.fitness.get("eval_return_mean", -1e9))
    # higher return should come first => reverse
    if r1 > r2:
        return -1
    elif r1 < r2:
        return 1
    return 0

# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------

class RLObjective(Objective):
    def __init__(self, *, spec: RLSpec):
        self.spec = spec
        self.target = "pennylane"  # consistent with your other scripts

        # For register sizing in main
        # - CartPole encoder outputs 4 floats (angle input)
        # - FrozenLake encoder outputs n_bits int64 (basis input)
        # We'll set input_size based on spec.
        if spec.input_mode == "angle":
            # CartPole obs encoder -> length 4 floats
            self.input_size = 4
        elif spec.input_mode == "basis":
            if spec.n_state_bits is None:
                raise ValueError("spec.n_state_bits required for basis mode")
            self.input_size = int(spec.n_state_bits)
        else:
            raise ValueError(f"Unknown input_mode: {spec.input_mode}")

        self.n_actions = int(spec.n_actions)

    def __call__(self, genome: CircuitGenome):
        """
        Trains the circuit as a policy (REINFORCE) and sets fitness keys:
          - train_return_mean
          - eval_return_mean / eval_return_std
        """
        hp = genome.hyperparameters
        # Map your hyperparameters into the RLSpec (do not mutate original spec in place)
        spec = RLSpec(**{**self.spec.__dict__})

        # Override from genome hyperparameters
        spec.episodes = int(hp["episodes"])
        spec.lr = float(hp["learning_rate"])
        spec.gamma = float(hp["gamma"])
        spec.max_steps = int(hp["max_steps"])
        spec.eval_episodes = int(hp["eval_episodes"])
        spec.entropy_coef = float(hp["entropy_coef"])
        spec.baseline = str(hp["baseline"])
        spec.log_every = int(hp["log_every"])
        spec.seed = int(hp["seed"])

        # For FrozenLake, pass env kwargs (map_name/is_slippery) through
        spec.env_kwargs = hp.get("env_kwargs", spec.env_kwargs)

        # Train with your RL module
        train_policy_gradient(genome, spec=spec)

        logger.debug(
            f"Genome Fitness in Objective: {genome.fitness}"
        )

        # Ensure we have an evaluation metric (train_policy_gradient already writes eval_*,
        # but keep it explicit and consistent with classification code style)
        ev = eval_policy(genome, spec=spec, deterministic=True, seed=spec.seed + 9999)

        # train_policy_gradient sets genome.fitness; we can enrich/standardize keys:
        genome.fitness = {
            **(genome.fitness or {}),
            "eval_return_mean": float(ev["eval_return_mean"]),
            "eval_return_std": float(ev["eval_return_std"]),
            "env_id": spec.env_id,
            "n_actions": spec.n_actions,
            "input_mode": spec.input_mode,
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
    p.add_argument("--env", choices=["cartpole", "frozenlake"], required=True)

    p.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Output directory to store results from runs",
    )

    # Evolution
    p.add_argument("--max_population_size", type=int, default=30)
    p.add_argument("--number_genomes", type=int, default=500)

    # Registers
    p.add_argument("--input_qubits", type=int, default=6)
    p.add_argument("--out_qubits", type=int, default=None)  # if None, we pick sensible default

    # RL hyperparams
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning_rate", "-lr", type=float, default=1e-2)
    p.add_argument("--entropy_coef", type=float, default=0.00)
    p.add_argument("--baseline", choices=["mean", "none"], default="mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)

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

    # logging
    logger.remove()
    os.makedirs(args.out_dir, exist_ok=True)
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "run.log"))

    # Build RL spec from rl_objectives.py
    if args.env == "cartpole":
        spec = cartpole_spec(episodes=args.episodes, lr=args.learning_rate, seed=args.seed)
        # 2 actions => can use 2 output qubits as logits directly
        default_out_qubits = 2
    else:
        # FrozenLake actions=4 => best is >=4 output qubits (logits), OR rely on padding in logits_from_kexpvals.
        spec = frozenlake_spec(
            map_name=args.map_name,
            is_slippery=args.is_slippery,
            episodes=args.episodes,
            lr=args.learning_rate,
            seed=args.seed,
        )
        default_out_qubits = 4  # recommended

        spec.env_kwargs = {"map_name": args.map_name, "is_slippery": args.is_slippery}

    # wrap objective
    objective = RLObjective(spec=spec)

    # choose output qubits
    out_qubits = int(args.out_qubits) if args.out_qubits is not None else default_out_qubits

    # hyperparameters injected into each genome (like your classification script)
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
        # optional env kwargs for FrozenLake
        "env_kwargs": getattr(spec, "env_kwargs", None),
    }

    # registers
    input_registers = {"input": min(args.input_qubits, objective.input_size)}
    output_registers = {"output": out_qubits}

    logger.info(f"env={spec.env_id} input_registers={input_registers} output_registers={output_registers}")

    # run
    master_worker(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=args.max_population_size,
            compare=compare,
            out_dir=args.out_dir,
        ),
        objective=objective,
        hyperparameters=hyperparameters,
        run_for=args.number_genomes,
        input_registers=input_registers,
        output_registers=output_registers,
        target="pennylane",
    )
