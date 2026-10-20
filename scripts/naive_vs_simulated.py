"""Does the device simulation earn its place in the loop?

The control experiment.  Three search configurations are compared, all using the
identical search algorithm, gate set, objectives and genome budget, differing
only in what noise model the search was given:

  closed        no noise at all -- the search minimises gate count and duration
                only (what circuit synthesis does without any device model)
  depolarizing  uniform depolarizing at the true average 1q and 2q rates, with
                true gate durations and no idle cost (standard noise-aware
                practice: a spec-sheet error rate)
  simulated     the device card, i.e. per-gate Pauli-twirled channels and an
                idle channel, both derived from open-system optimal control

Every resulting circuit is then evaluated under the TRUE card for each master
equation, which is the common yardstick.  The question the table answers is:
how much accuracy does a practitioner lose by not simulating the device?

    .venv_win/Scripts/python scripts/naive_vs_simulated.py --task ghz3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard, hardware_metrics  # noqa: E402
from scripts.ablate_drives import feasible, greedy_prune  # noqa: E402
from scripts.compare_distributions import distributions  # noqa: E402
from scripts.reference_circuits import chain, tree, genome  # noqa: E402

CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")
# Card family: "transmon_detuned" is the valid architecture; plain
# "transmon" is the resonant one kept only as a counterexample.
FAMILY = "transmon_detuned"
TRUE_MODELS = ["lindblad_markov", "lindblad_nm", "mixed"]
SHORT = {"lindblad_markov": "Lind-M", "lindblad_nm": "Lind-NM", "mixed": "Mixed"}
Gate = Tuple[str, List[int]]


def best(pattern: str) -> Optional[List[Gate]]:
    rows = []
    for p in glob.glob(pattern):
        rows.extend(json.load(open(p))["front"])
    if not rows:
        return None
    b = min(rows, key=lambda r: r["fitness"]["p_fail"])
    return [(g, list(w)) for g, w in b["gates"]]


def main() -> int:
    global FAMILY
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ghz3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--family", default=FAMILY,
                    help="device-card family "
                         "(transmon_detuned = valid; transmon = resonant)")
    args = ap.parse_args()
    FAMILY = args.family
    print(f"card family: {FAMILY}")
    n = {"ghz3": 3, "ghz4": 4}[args.task]
    d = args.task

    true_cards = {m: DeviceCard.load(CARDS / f"device_card_{FAMILY}_{m}.json")
                  for m in TRUE_MODELS}
    noise = {m: DeviceCardNoiseModel(c) for m, c in true_cards.items()}

    configs: Dict[str, List[Gate]] = {}
    c = best(f"results/{d}/naive_closed_s*/summary.json")
    if c:
        configs["search: closed (no noise)"] = c
    c = best(f"results/{d}/naive_depol_s*/summary.json")
    if c:
        configs["search: uniform depolarizing"] = c
    # "simulated" is per-model: the circuit learned against that model's own card.
    per_model_sim = {m: best(f"results/{d}/{m}_s*/summary.json") for m in TRUE_MODELS}
    configs["textbook cascade"] = chain(n)
    configs["textbook fan-out"] = tree(n)

    rows: Dict[str, Dict[str, float]] = {}
    for name, gates in configs.items():
        rows[name] = {}
        for m in TRUE_MODELS:
            if not feasible(gates, n, true_cards[m]):
                rows[name][m] = float("nan")
                continue
            rows[name][m] = 1.0 - distributions(gates, n, noise[m])["fidelity"]
    rows["search: simulated (own card)"] = {}
    rows["search: simulated + prune"] = {}
    for m in TRUE_MODELS:
        g = per_model_sim[m]
        if g is None:
            rows["search: simulated (own card)"][m] = float("nan")
            rows["search: simulated + prune"][m] = float("nan")
            continue
        rows["search: simulated (own card)"][m] = 1.0 - distributions(g, n, noise[m])["fidelity"]
        gp = greedy_prune(g, n, true_cards[m], only_1q=True, target=d)
        rows["search: simulated + prune"][m] = 1.0 - distributions(gp, n, noise[m])["fidelity"]

    order = ["search: closed (no noise)", "search: uniform depolarizing",
             "search: simulated (own card)", "search: simulated + prune",
             "textbook cascade", "textbook fan-out"]
    order = [k for k in order if k in rows]
    w = max(len(k) for k in order) + 1
    print(f"\n=== {args.task}: state infidelity 1-F under the TRUE card "
          f"(rows = what the search was told)")
    print(" " * w + "  ".join(f"{SHORT[m]:>12s}" for m in TRUE_MODELS))
    for k in order:
        print(f"{k:{w}s}" + "  ".join(f"{rows[k][m]:12.4e}" for m in TRUE_MODELS))

    print("\n--- cost of not simulating (naive / simulated, >1 means simulation helped)")
    for m in TRUE_MODELS:
        sim = rows["search: simulated + prune"][m]
        for naive in ("search: closed (no noise)", "search: uniform depolarizing"):
            if naive in rows:
                print(f"  {SHORT[m]:8s} {naive:30s} {rows[naive][m] / sim:6.3f}x")

    if args.out:
        json.dump({"task": args.task, "models": TRUE_MODELS, "infidelity": rows,
                   "circuits": {k: [[a, list(b)] for a, b in v]
                                for k, v in configs.items()}},
                  open(args.out, "w"), indent=2)
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
