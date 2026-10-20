"""Evolve state-preparation circuits (Bell / GHZ_n) under a device card.

Objectives minimised: p_fail, n_entanglers, total_time_ns.  task_loss
(1 - noiseless GHZ_n fidelity) is a feasibility constraint, so the reported
front is the hardware trade-off among circuits that actually prepare GHZ_n.
n = 2 reproduces the Bell run.

    .venv_win/Scripts/python scripts/evolve_ghz_pareto.py \
        --card ../../Arch_Aware_Circuit_Evolution/device_cards/device_card_transmon_mixed.json \
        --n 3 --genomes 60000 --pop 80 --seed 1 --out results/ghz3/mixed_s1

Restarts matter (the landscape has feasible local optima that no single
gate deletion escapes): run several --seed values and take the union of
the fronts with scripts/merge_pareto.py.
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
from src.objectives.state_prep_objective import (  # noqa: E402
    StatePrepObjective, TARGETS, TARGET_WIRES)
from src.objectives.hardware_objectives import DeviceCard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--target", default=None,
                    help="named target state: phi_plus, phi_minus, psi_plus, "
                         "psi_minus, ghz3, ghz4 (overrides --n)")
    ap.add_argument("--n", type=int, default=3, help="GHZ size (target wires)")
    ap.add_argument("--wires", type=int, default=None,
                    help="circuit wires (default: --n, i.e. no helper wires)")
    ap.add_argument("--genomes", type=int, default=60000)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--task-tol", type=float, default=1e-3)
    ap.add_argument("--gates", nargs="+", default=None)
    ap.add_argument("--no-noisy", action="store_true")
    ap.add_argument("--virtual-z", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        logger.add(os.path.join(args.out, "evolve.log"), level="INFO")

    target = args.target or f"ghz{args.n}"
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; have {sorted(TARGETS)}")
    n_target = TARGET_WIRES[target]
    n_wires = args.wires or n_target
    card = DeviceCard.load(args.card)
    gates = args.gates or [g for g in card.gates
                           if g in pennylane_gate_specifications.specifications]
    specs = pennylane_gate_specifications.use_only(gates)
    objective = StatePrepObjective(card, target, noisy=not args.no_noisy,
                                   virtual_z_family=args.virtual_z)
    population = ParetoPopulation(
        max_population_size=args.pop,
        objective_keys=["p_fail", "n_entanglers", "total_time_ns"],
        constraints={"task_loss": args.task_tol},
        out_dir=args.out,
    )
    qubits = expand_registers({"q": n_wires})
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
                   key=lambda g: (g.fitness["p_fail"], g.fitness["n_entanglers"],
                                  g.fitness["total_time_ns"]))
    population.save(os.path.join(args.out, "pareto_front.json"))
    summary = {
        "task": target, "card": args.card, "model": card.model,
        "architecture": card.architecture, "n_target": n_target, "wires": n_wires,
        "genomes": args.genomes, "pop": args.pop, "task_tol": args.task_tol,
        "gates": sorted(gates), "virtual_z": args.virtual_z, "seed": args.seed,
        "elapsed_s": elapsed, "archive": len(population.population),
        "front": [{"fitness": g.fitness,
                   "gates": [(x.method_name, [q[1] for q in x.qubits]) for x in g.gates if x.enabled]}
                  for g in front],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        with open(args.trace, "w") as fh:
            json.dump({"task": summary["task"], "model": card.model,
                       "architecture": card.architecture, "n_wires": n_wires,
                       "seed": args.seed, "genomes": args.genomes,
                       "trace": trace}, fh, indent=2, default=str)
        print(f"wrote {args.trace}: {len(trace)} improving circuits")

    print(f"\n== {target} {card.architecture}/{card.model} seed {args.seed}: "
          f"{args.genomes} genomes in {elapsed:.0f}s, front {len(front)}")
    for g in front:
        f_ = g.fitness
        print(f"  p_fail={f_['p_fail']:.4e}  cx={int(f_['n_entanglers'])}  "
              f"T={f_['total_time_ns']:.0f} ns  layers={int(f_['n_layers'])}  "
              f"noisyF={f_.get('noisy_fidelity', float('nan')):.5f}  "
              f"gates={[(x.method_name, [q[1] for q in x.qubits]) for x in g.gates if x.enabled]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
