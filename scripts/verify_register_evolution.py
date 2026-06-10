"""Smoke test: run a Stage-C-like mutation+crossover loop and ensure no crash.

Hammers EXAQC.mutate and EXAQC.generate_genome with register-evolution on, to
confirm shrink_register + enable_gate and grow_register + crossover no longer
raise ValueError on stale-qubit gate lookups.
"""
from __future__ import annotations

import random
import sys
import tempfile

from src.circuits.circuit import CircuitGenome
from src.evolution.exaqc import EXAQC
from src.evolution.steady_state_population import SteadyStatePopulation
from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.utils.profiler import EXAQCProfiler


def _compare(a, b):
    return a.fitness["test_loss"] - b.fitness["test_loss"]


def main(n_iters: int = 500, seed: int = 0):
    random.seed(seed)

    # cheap objective: assign random fitness, don't actually train
    def cheap_objective(genome: CircuitGenome) -> None:
        genome.fitness = {"test_loss": random.random(), "test_acc": random.random()}

    scratch = tempfile.mkdtemp(prefix="verify_register_evolution_")
    pop = SteadyStatePopulation(
        max_population_size=8,
        compare=_compare,
        out_dir=None,
        profiler=EXAQCProfiler(out_dir=scratch),
    )

    exaqc = EXAQC(
        gate_specifications=pennylane_gate_specifications,
        population=pop,
        objective=cheap_objective,
        hyperparameters={"epochs": 1, "learning_rate": 0.05},
        input_qubits=[("q", i) for i in range(4)],
        output_qubits=[("q", 0), ("q", 1)],
        target="pennylane",
        allow_register_evolution=True,
    )

    n_grow = 0
    n_shrink = 0
    n_rejected = 0
    n_ok = 0

    for i in range(n_iters):
        child = exaqc.generate_genome()
        cheap_objective(child)
        pop.insert_genome(child, current_genome_number=exaqc.genome_number)
        n_ok += 1
        gens = child.metadata.get("generated_by", [])
        n_grow += sum(1 for g in gens if g == "grow_register")
        n_shrink += sum(1 for g in gens if g == "shrink_register")
        if i % 50 == 0:
            print(
                f"  [{i:4d}] genome #{child.genome_number} "
                f"qubits={len(child.qubits)} gates={len(child.gates)} "
                f"gen_by={gens}"
            )

    print()
    print(f"survived {n_iters} iterations with register evolution.")
    print(f"  grow_register fired:   {n_grow}")
    print(f"  shrink_register fired: {n_shrink}")
    print(f"  rejected (logged):     see warnings above")
    print(f"  ok inserted:           {n_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
