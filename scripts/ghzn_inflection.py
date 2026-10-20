"""The GHZ_n benchmark: where does hardware-aware refinement start to matter?

For each n we compare, under every open-system model:

  textbook cascade   H_0 then a linear CNOT cascade -- what a textbook or a
                     naive compiler emits. Depth n-1 CNOT layers.
  textbook fan-out    the balanced copy tree. Same n-1 entanglers,
                     ceil(log2 n) CNOT layers. Also a closed-system
                     derivation, so it is a fair "standard" baseline.
  learned (card)      the search seeded FROM the cascade and refined against
                     that model's device card.
  learned (naive)     the same search seeded from the cascade but refined
                     against a uniform-depolarizing or noiseless card -- the
                     control that isolates what the device simulation adds.

Every row is a scheduled tier-1 density-matrix infidelity under the model named
by the column, so all four are judged on the same yardstick.

The quantity of interest is the ratio cascade / learned(card) as a function of
n: if hardware conditioning matters at all, that ratio should grow with n,
because the cascade's duration grows linearly in n while the optimal structure
grows logarithmically, and decoherence compounds over that difference.

    .venv_win/Scripts/python scripts/ghzn_inflection.py --out <path>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics  # noqa: E402
from src.objectives.state_prep_objective import (  # noqa: E402
    StatePrepObjective, ghz_state, state_fidelity)
from scripts.ablate_drives import greedy_prune  # noqa: E402
from scripts.reference_circuits import chain, tree, genome  # noqa: E402

CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")
MODELS = ["lindblad_markov", "lindblad_nm", "mixed"]
SHORT = {"schrod": "Schrod", "lindblad_markov": "Lind-M",
         "lindblad_nm": "Lind-NM", "mixed": "Mixed"}
Gate = Tuple[str, List[int]]


def learned(n: int, card_name: str) -> Optional[List[Gate]]:
    rows = []
    for p in glob.glob(f"results/ghzn/n{n}__{card_name}_s*/summary.json"):
        rows.extend(json.load(open(p))["front"])
    if not rows:
        return None
    b = min(rows, key=lambda r: r["fitness"]["p_fail"])
    return [(g, list(w)) for g, w in b["gates"]]


def tier1(gates: List[Gate], n: int, card: DeviceCard,
          nm: DeviceCardNoiseModel) -> Dict[str, float]:
    target = ghz_state(n)
    obj = StatePrepObjective(card, f"ghz{n}", noisy=False)
    G = genome(list(gates), n)
    obj(G)
    assert G.fitness["task_loss"] < 1e-9, \
        f"circuit does not prepare GHZ_{n}: {G.fitness['task_loss']:.2e}"
    hm = hardware_metrics(genome(list(gates), n), card)
    dev, qnode = genome(list(gates), n).generate_pennylane_circuit(
        noise_model=nm, input_mode="basis", return_probs=False,
        measure_registers=False)
    rho = qnode(torch.zeros(n, dtype=torch.int64), None)
    rho = rho.detach().cpu().numpy() if hasattr(rho, "detach") else np.asarray(rho)
    F = state_fidelity(rho, target, n)
    return {"infidelity": 1.0 - F, "n_gates": len(gates),
            "time_ns": hm["total_time_ns"], "n_entanglers": hm["n_entanglers"],
            "p_fail": hm["p_fail"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out: Dict[str, dict] = {}
    print("GHZ_n: tier-1 state infidelity, and cascade/learned ratio vs n\n")
    hdr = (f"{'n':>2s} {'model':8s} {'cascade':>11s} {'fan-out':>11s} "
           f"{'learned(card)':>13s} {'learned(depol)':>14s} {'learned(none)':>13s} "
           f"{'casc/learn':>10s} {'T casc':>7s} {'T learn':>7s}")
    print(hdr)
    for n in range(args.n_min, args.n_max + 1):
        refs = {"cascade": chain(n), "fanout": tree(n)}
        for model in MODELS:
            cp = CARDS / f"device_card_transmon_detuned_{model}.json"
            if not cp.exists():
                continue
            card = DeviceCard.load(cp)
            nm = DeviceCardNoiseModel(card)
            row: Dict[str, dict] = {}
            for k, g in refs.items():
                row[k] = tier1(g, n, card, nm)
            for tag, cname in (("learned_card", f"transmon_detuned_{model}"),
                               ("learned_depol", "naive_depol"),
                               ("learned_none", "naive_closed")):
                g = learned(n, cname)
                if g is None:
                    continue
                gp = greedy_prune(g, n, card, only_1q=True, target=f"ghz{n}")
                a, b = tier1(g, n, card, nm), tier1(gp, n, card, nm)
                row[tag] = a if a["infidelity"] <= b["infidelity"] else b
            if "learned_card" not in row:
                continue
            ratio = row["cascade"]["infidelity"] / row["learned_card"]["infidelity"]
            fmt = lambda k: (f"{row[k]['infidelity']:11.4e}" if k in row else f"{'--':>11s}")
            print(f"{n:2d} {SHORT[model]:8s} {fmt('cascade')} {fmt('fanout')} "
                  f"{row['learned_card']['infidelity']:13.4e} "
                  f"{row.get('learned_depol',{}).get('infidelity',float('nan')):14.4e} "
                  f"{row.get('learned_none',{}).get('infidelity',float('nan')):13.4e} "
                  f"{ratio:10.3f} {row['cascade']['time_ns']:7.0f} "
                  f"{row['learned_card']['time_ns']:7.0f}")
            out[f"n{n}_{model}"] = {"n": n, "model": model, "rows": row,
                                    "cascade_over_learned": ratio}
        print()

    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
