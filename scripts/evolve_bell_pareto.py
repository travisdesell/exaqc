"""Milestone 1 of the architecture-aware circuit-evolution project: evolve
Bell-state preparation circuits under a device card with NSGA-II.

Objectives (all minimised): p_fail, n_entanglers, total_time_ns, with
task_loss (1 - noiseless Bell fidelity) as a feasibility CONSTRAINT
(<= --task-tol), so the front is the hardware trade-off among circuits
that actually prepare |Phi+>.  Gate set restricted to what the card has.

    .venv_win/Scripts/python scripts/evolve_bell_pareto.py \
        --card ../../Arch_Aware_Circuit_Evolution/device_cards/device_card_transmon_lindblad_markov.json \
        --genomes 400 --pop 40 --out results/bell_pareto/lindblad_markov

Run once per card; compare the fronts across cards/architectures.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications  # noqa: E402
from src.circuits.registers import expand_registers  # noqa: E402
from src.evolution.exaqc import EXAQC  # noqa: E402
from src.evolution.pareto_population import ParetoPopulation  # noqa: E402
from src.objectives.bell_prep_objective import BellPrepObjective  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--genomes", type=int, default=400)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--wires", type=int, default=2)
    ap.add_argument("--task-tol", type=float, default=1e-3,
                    help="feasibility: task_loss <= tol")
    ap.add_argument("--gates", nargs="+", default=None,
                    help="restrict gate set (default: every gate in the card)")
    ap.add_argument("--no-noisy", action="store_true", help="skip tier-1 default.mixed eval")
    ap.add_argument("--virtual-z", action="store_true", help="z/s/t/sdg/tdg error-free & instant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        logger.add(os.path.join(args.out, "evolve.log"), level="INFO")
    os.makedirs(args.out, exist_ok=True)

    card = DeviceCard.load(args.card)
    gates = args.gates or [g for g in card.gates if g in pennylane_gate_specifications.specifications]
    specs = pennylane_gate_specifications.use_only(gates)
    logger.warning(f"card {card.architecture}/{card.model}; gate set {sorted(gates)}")

    objective = BellPrepObjective(card, noisy=not args.no_noisy, virtual_z_family=args.virtual_z)
    population = ParetoPopulation(
        max_population_size=args.pop,
        objective_keys=["p_fail", "n_entanglers", "total_time_ns"],
        constraints={"task_loss": args.task_tol},
        out_dir=args.out,
    )
    qubits = expand_registers({"q": args.wires})
    exaqc = EXAQC(
        gate_specifications=specs,
        population=population,
        objective=objective,
        hyperparameters={"epochs": 1, "learning_rate": 0.0, "log_every": 10 ** 9},
        input_qubits=qubits,
        output_qubits=qubits.copy(),
        target="pennylane",
    )
    t0 = time.time()
    exaqc.run_for(args.genomes)
    elapsed = time.time() - t0

    front = sorted(population.pareto_front(),
                   key=lambda g: (g.fitness["p_fail"], g.fitness["n_entanglers"], g.fitness["total_time_ns"]))
    population.save(os.path.join(args.out, "pareto_front.json"))
    summary = {
        "card": args.card, "model": card.model, "architecture": card.architecture,
        "genomes": args.genomes, "pop": args.pop, "task_tol": args.task_tol,
        "gates": sorted(gates), "virtual_z": args.virtual_z, "seed": args.seed,
        "elapsed_s": elapsed, "archive": len(population.population),
        "front": [{"fitness": g.fitness,
                   "gates": [(x.method_name, [q[1] for q in x.qubits]) for x in g.gates if x.enabled]}
                  for g in front],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n== {card.architecture}/{card.model}: {args.genomes} genomes in {elapsed:.0f}s, "
          f"archive {len(population.population)}, feasible Pareto front {len(front)}")
    for g in front:
        f_ = g.fitness
        print(f"  p_fail={f_['p_fail']:.3e}  cx={int(f_['n_entanglers'])}  T={f_['total_time_ns']:.0f} ns  "
              f"task_loss={f_['task_loss']:.1e}  noisyF={f_.get('noisy_fidelity', float('nan')):.5f}  "
              f"gates={[(x.method_name, [q[1] for q in x.qubits]) for x in g.gates if x.enabled]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
