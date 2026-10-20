"""Build the device card a practitioner WITHOUT a device simulation would use.

This is the control experiment for the claim that the quantum simulation is
load-bearing.  Standard practice in noise-aware circuit search is a uniform
depolarizing model: one error rate for single-qubit gates, one for two-qubit
gates, taken from a spec sheet or a calibration summary, with no per-gate
structure, no error bias, and no cost for idling.  We construct exactly that,
calibrated to be *as favourable as possible* to the naive practitioner:

  - the 1q rate is the average of the true per-gate rates on the reference card,
    so the naive model is not simply wrong about the magnitude;
  - the 2q rate is the true CNOT rate;
  - gate durations are the true durations, so the naive search still sees the
    correct circuit timing and can still minimise duration;
  - the error channel is depolarizing, i.e. unbiased, and idle is free.

Everything the naive card lacks is therefore exactly what the device simulation
supplies: which Pauli errors a gate actually makes, and what a wire costs while
it waits.  A search run against this card and then evaluated on the true card
measures the price of not simulating.

Two variants:
  --mode depolarizing   uniform depolarizing, idle free   (the usual practice)
  --mode closed         no noise at all                   (gate-count-only search)

    py -3.14 scripts/make_naive_card.py --reference mixed --mode depolarizing \
        --out ../../Arch_Aware_Circuit_Evolution/device_cards/device_card_naive_depol.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict

CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")


def depolarizing_pauli(n_qubits: int, p_err: float) -> Dict[str, float]:
    """Uniform depolarizing: identity gets 1 - p, every other Pauli p/(4^n - 1)."""
    labels = ["".join(c) for c in itertools.product("IXYZ", repeat=n_qubits)]
    ident = "I" * n_qubits
    share = p_err / (len(labels) - 1)
    return {lab: (1.0 - p_err if lab == ident else share) for lab in labels}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="mixed",
                    help="true card whose durations and average rates are copied")
    ap.add_argument("--mode", choices=["depolarizing", "closed"], default="depolarizing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = json.load(open(CARDS / f"device_card_transmon_{args.reference}.json"))
    one_q = [e["infidelity"] for g, e in ref["gates"].items()
             if int(e.get("wires", 1)) == 1 and "alias_of" not in e]
    p1 = sum(one_q) / len(one_q)
    cx = ref["gates"].get("cx")
    p2 = cx["infidelity"] if cx else 10 * p1
    if args.mode == "closed":
        p1 = p2 = 0.0

    card = {
        "architecture": f"naive_{args.mode}",
        "model": f"naive_{args.mode}",
        "model_label": ("uniform depolarizing, idle free"
                        if args.mode == "depolarizing" else "no noise (gate count only)"),
        "method": "none",
        "derived_from": f"durations and average rates of device_card_transmon_{args.reference}",
        "conventions": {
            "purpose": "control experiment: the noise model a search would use WITHOUT "
                       "an open-system device simulation",
            "p_1q": p1, "p_2q": p2,
            "idle": "absent by construction (a naive model charges nothing for idling)",
        },
        "gates": {},
        "idle": {},
        "sources": {},
    }
    for g, e in ref["gates"].items():
        n_w = int(e.get("wires", 1))
        p = p2 if n_w == 2 else p1
        entry = {
            "T_ns": e["T_ns"], "wires": n_w,
            "infidelity": p, "leakage": 0.0, "process_infidelity": p,
            "trace_loss": 0.0,
            "pauli_error": depolarizing_pauli(n_w, p),
            "ptm_error_diag": {},
        }
        if "alias_of" in e:
            entry["alias_of"] = e["alias_of"]
        card["gates"][g] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(card, open(args.out, "w"), indent=2)
    print(f"naive card ({args.mode}): p_1q={p1:.4e} p_2q={p2:.4e}, "
          f"{len(card['gates'])} gates, no idle model")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
