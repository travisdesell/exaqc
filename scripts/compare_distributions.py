"""Measurement-outcome distributions: on-paper circuit vs learned circuit.

The point of the comparison is the one a fidelity number hides.  A textbook
preparation circuit is derived in the closed system, where it is exactly
correct; run it under an open-system model and it is not.  A circuit learned
against that model's device card is a different circuit --- usually with more
gates --- that prepares the same state in the closed system and a better
approximation of it in the open one.  We therefore forward-solve three
circuits under each card:

  on-paper       the textbook circuit (linear CNOT cascade), and the
                 log-depth fan-out variant, both closed-system derivations
  learned        the best evolved circuit for that card
  card-informed  the fan-out with the spectator-drive trick the search
                 discovered applied by hand

and report, for each, the full distribution over the $2^n$ outcomes in the
$Z$ basis, the same in the $X$ basis, and the state fidelity.

Both bases are necessary and the reason is instructive.  A $Z$ error acts
trivially on computational-basis probabilities, so the $Z$-basis distribution
is blind to dephasing --- which is the dominant error channel under every
dissipative card here.  The $X$-basis distribution carries the coherence
between $|0\\cdots0\\rangle$ and $|1\\cdots1\\rangle$ that dephasing destroys.
We report the Z-basis peak weight and the X-basis parity as the two
experimentally accessible summaries, and the state fidelity as the
quantity they are proxies for.  The single-setting combination of the
two is NOT a fidelity lower bound -- the textbook GHZ witness needs a
parity oscillation over many measurement angles -- and the numbers
below show it failing as one.

    .venv_win/Scripts/python scripts/compare_distributions.py \\
        --out ../../Arch_Aware_Circuit_Evolution/paper/generated/distributions.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits.circuit import CircuitGenome  # noqa: E402
from src.noise.DeviceCardNoiseModel import DeviceCardNoiseModel  # noqa: E402
from src.objectives.ghz_prep_objective import ghz_fidelity_from_state  # noqa: E402
from src.objectives.hardware_objectives import DeviceCard  # noqa: E402
from scripts.reference_circuits import chain, tree, with_drive, genome  # noqa: E402

MODELS = ["schrod", "lindblad_markov", "lindblad_nm", "mixed"]
CARDS = Path("../../Arch_Aware_Circuit_Evolution/device_cards")
TASKS = {"bell": (2, "bell_pareto/final"), "ghz3": (3, "ghz3"), "ghz4": (4, "ghz4")}
Gate = Tuple[str, List[int]]


def _run(g: CircuitGenome, noise_model, want: str) -> np.ndarray:
    dev, qnode = g.generate_pennylane_circuit(
        noise_model=noise_model, input_mode="basis",
        return_probs=(want != "state"))
    zeros = torch.zeros(len(g.input_indexes), dtype=torch.int64)
    out = qnode(zeros, None)
    a = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
    return a


def distributions(gates: Sequence[Gate], n: int, noise_model) -> Dict[str, object]:
    """Z-basis probabilities, X-basis probabilities, and the state/density
    matrix fidelity against GHZ_n."""
    pz = np.real(_run(genome(gates, n), noise_model, "probs")).astype(float)
    # X basis: a Hadamard on every wire immediately before measurement. It is
    # appended to the genome so the noise model charges it like any other gate.
    x_gates = list(gates) + [("h", [w]) for w in range(n)]
    px = np.real(_run(genome(x_gates, n), noise_model, "probs")).astype(float)
    state = _run(genome(gates, n), noise_model, "state")
    F = ghz_fidelity_from_state(state, n)
    # Two experimentally accessible summaries. P_pop is the weight the Z-basis
    # measurement puts on the two allowed outcomes; parity is the expectation
    # of (-1)^{|x|} in the X basis, which is +1 for the ideal state and 0 for a
    # fully dephased mixture of |0..0> and |1..1>, so it reports the coherence
    # that the Z basis cannot see. NOTE: these are diagnostics, not a fidelity
    # bound -- the textbook GHZ witness needs a parity OSCILLATION over many
    # measurement angles, not this single setting, and the numbers below show
    # the single-setting combination is not a lower bound on F.
    weights = np.array([bin(i).count("1") for i in range(2 ** n)])
    parity = float(((-1.0) ** weights * px).sum())
    P_pop = float(pz[0] + pz[-1])
    return {
        "z": pz.tolist(),
        "x": px.tolist(),
        "P_pop": P_pop,
        "parity": parity,
        "fidelity": float(F),
        "z_forbidden_weight": float(pz.sum() - pz[0] - pz[-1]),
    }


def best_evolved_gates(task_dir: str, model: str) -> List[Gate] | None:
    rows = []
    for path in glob.glob(f"results/{task_dir}/{model}_s*/summary.json"):
        rows.extend(json.load(open(path))["front"])
    if not rows:
        return None
    best = min(rows, key=lambda r: r["fitness"]["p_fail"])
    return [(g, list(w)) for g, w in best["gates"]]


def tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--closed-tol", type=float, default=1e-9)
    args = ap.parse_args()

    out: Dict[str, dict] = {}
    for tag, (n, task_dir) in TASKS.items():
        ideal_z = np.zeros(2 ** n)
        ideal_z[0] = ideal_z[-1] = 0.5
        per_model = {}
        for m in MODELS:
            cp = CARDS / f"device_card_transmon_{m}.json"
            if not cp.exists():
                continue
            card = DeviceCard.load(cp)
            learned = best_evolved_gates(task_dir, m)
            if learned is None:
                continue
            circuits = {
                "on-paper (cascade)": chain(n),
                "on-paper (fan-out)": tree(n),
                "card-informed (fan-out + drive)": with_drive(tree(n), n, card),
                "learned": learned,
            }
            nm = DeviceCardNoiseModel(card)
            rows = {}
            for name, gates in circuits.items():
                closed = distributions(gates, n, None)
                noisy = distributions(gates, n, nm)
                rows[name] = {
                    "n_gates": len(gates),
                    "gates": [[a, list(b)] for a, b in gates],
                    "closed": closed,
                    "card": noisy,
                    "closed_z_tvd_from_ideal": tvd(closed["z"], ideal_z),
                    "card_z_tvd_from_ideal": tvd(noisy["z"], ideal_z),
                }
            # Logical equivalence: every circuit must be the same in the closed
            # system, since feasibility was enforced there.
            base = rows["on-paper (cascade)"]["closed"]["z"]
            worst = max(tvd(base, r["closed"]["z"]) for r in rows.values())
            rows["_closed_z_tvd_max"] = worst
            status = "OK" if worst <= args.closed_tol else "MISMATCH"
            print(f"{tag:5s} {m:16s} closed-system agreement {worst:.1e} [{status}]")
            for name in circuits:
                r = rows[name]
                print(f"        {name:32s} gates={r['n_gates']:2d}  "
                      f"F={r['card']['fidelity']:.5f}  "
                      f"P_pop={r['card']['P_pop']:.5f}  "
                      f"parity={r['card']['parity']:+.5f}  "
                      f"Zforbidden={r['card']['z_forbidden_weight']:.2e}")
            per_model[m] = rows
        if per_model:
            out[tag] = {"n": n, "ideal_z": ideal_z.tolist(), "models": per_model}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
