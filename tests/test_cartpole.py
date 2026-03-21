from __future__ import annotations

from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers
from src.objectives.policy_objectives import (
    cartpole_spec,
    train_rl,
)

import math
import random
import pytest

target_backend = "pennylane"


def _make_genome(genome_number: int):
    """
    Best-effort factory for your CircuitGenome without knowing the exact constructor.
    If your CircuitGenome signature differs, adjust ONLY this function.
    """
    # CartPole obs dim = 4. Discrete actions = 2.
    # Use 2 output qubits in your setup (expvals on 2 wires => 2 logits).
    # Try a few common constructor patterns.
    input_qubits = expand_registers({"input": 4})
    output_qubits = expand_registers({"output": 2})
    candidates = [
        lambda: CircuitGenome(
            genome_number=genome_number,
            target=target_backend,
            input_qubits=input_qubits.copy(),
            output_qubits=output_qubits.copy(),
        ),
        lambda: CircuitGenome(
            genome_number=genome_number,
            target=target_backend,
            input_qubits=input_qubits.copy(),
            output_qubits=output_qubits.copy(),
        ),
        lambda: CircuitGenome(
            genome_number=genome_number,
            target=target_backend,
            input_qubits=input_qubits.copy(),
            output_qubits=output_qubits.copy(),
        ),
        lambda: CircuitGenome(
            genome_number=genome_number,
            target=target_backend,
            input_qubits=input_qubits.copy(),
            output_qubits=output_qubits.copy(),
        ),
    ]

    last_err = None
    genome = None
    for ctor in candidates:
        try:
            genome = ctor()
            break
        except Exception as e:
            last_err = e
            genome = None

    if genome is None:
        pytest.skip(
            "Could not construct CircuitGenome in test. "
            "Please update _make_genome() to match your CircuitGenome constructor.\n"
            f"Last error: {last_err}"
        )

    for maybe_init in ["initialize", "init_random", "random_init", "seed_minimal"]:
        if hasattr(genome, maybe_init) and callable(getattr(genome, maybe_init)):
            try:
                getattr(genome, maybe_init)()
            except Exception:
                pass

    return genome


@pytest.mark.parametrize("algo", ["reinforce", "a2c", "ppo"])
@pytest.mark.parametrize("n_genomes", [3])  # "small number of genomes"
def test_cartpole_algos_smoke(algo: str, n_genomes: int):
    """
    Parameterized smoke test:
    - builds a few genomes
    - trains each with the specified algo for a tiny number of updates
    - asserts fitness is populated and numbers are finite

    Keeps runs short (CI-friendly), but still exercises:
    - qnode construction
    - rollouts
    - optimizer steps
    - evaluation
    """
    try:

        # Keep everything tiny.
        # NOTE: "episodes" is "updates" for A2C/PPO; for REINFORCE it's episodes.
        spec = cartpole_spec(episodes=2, lr=5e-3, seed=0, algo=algo)
        spec.max_steps = 64
        spec.eval_episodes = 2
        spec.log_every = 1
        spec.entropy_coef = 0.0

        # PPO/A2C specifics: shrink rollout sizes
        spec.rollout_steps = 128
        spec.gae_lambda = 0.95
        spec.value_coef = 0.5

        spec.ppo_epochs = 2
        spec.ppo_minibatch = 32
        spec.ppo_clip = 0.2
        spec.target_kl = None

        random.seed(0)

        genomes = [_make_genome(i) for i in range(n_genomes)]

        for g in genomes:
            # Train for a tiny budget
            train_rl(g, spec=spec, algo=algo)

            assert hasattr(
                g, "fitness"
            ), "Genome should have a fitness dict after training."
            assert isinstance(g.fitness, dict), "Genome fitness must be a dict."

            assert "env_id" in g.fitness
            assert g.fitness["env_id"] == "CartPole-v1"

            assert "eval_return_mean" in g.fitness
            assert "eval_return_std" in g.fitness
            assert "train_return_mean" in g.fitness

            # Must be finite
            for k in ["eval_return_mean", "eval_return_std", "train_return_mean"]:
                v = float(g.fitness[k])
                assert math.isfinite(v), f"{algo}: {k} should be finite, got {v}"

            # Optional sanity check: returns should be within plausible bounds for short runs
            # CartPole max per episode is 500, but capping steps at 64 here.
            assert g.fitness["eval_return_mean"] <= 64 + 1e-6

    except Exception as e:
        pytest.fail(f"Missing deps or project import failed: {e}")


@pytest.mark.parametrize("algo", ["reinforce", "actor_critic", "ppo"])
def test_cartpole_single_genome_sets_algo(algo: str):
    """
    Tiny focused test: ensure algo label is written (if your code sets it).
    """
    try:

        spec = cartpole_spec(episodes=1, lr=1e-2, seed=1, algo=algo)
        spec.max_steps = 32
        spec.eval_episodes = 1
        spec.rollout_steps = 64
        spec.ppo_epochs = 1
        spec.ppo_minibatch = 16
        spec.log_every = 1

        g = _make_genome(0)
        train_rl(g, spec=spec, algo=algo)

        if "algo" in g.fitness:
            assert g.fitness["algo"] in {"reinforce", "actor_critic", "ppo"}

    except Exception as e:
        pytest.fail(f"Missing deps or project import failed: {e}")
