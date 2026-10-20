"""Emit one search command per line, for a SLURM array on tigris.

Each line is the argument string passed to python by search_tigris.sbatch, so
array index i runs line i+1.  Cells whose summary.json already exists are
skipped unless --all is given, which makes a requeue cheap.

    py -3.14 scripts/make_search_cells.py --campaign bell4 --out cells_bell4.txt
    py -3.14 scripts/make_search_cells.py --campaign ghz   --out cells_ghz.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
CARDS = "../../Arch_Aware_Circuit_Evolution/device_cards"

REAL = ["transmon_detuned_schrod", "transmon_detuned_lindblad_markov",
        "transmon_detuned_lindblad_nm", "transmon_detuned_mixed"]
NAIVE = ["naive_depol", "naive_closed"]
BELL = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]


def bell_cells(seeds: int, genomes: int, pop: int) -> List[tuple]:
    out = []
    for target in BELL:
        for card in REAL + NAIVE:
            for s in range(1, seeds + 1):
                d = f"results/bell4/{target}__{card}_s{s}"
                out.append((d,
                            f"scripts/evolve_state_pareto.py "
                            f"--card {CARDS}/device_card_{card}.json "
                            f"--target {target} --genomes {genomes} --pop {pop} "
                            f"--seed {s} --out {d} --quiet"))
    return out


def ghz_cells(seeds: int, genomes: int, pop: int) -> List[tuple]:
    out = []
    for n in (3, 4):
        for card in REAL + NAIVE:
            for s in range(1, seeds + 1):
                d = f"results/ghz{n}/{card}_s{s}"
                out.append((d,
                            f"scripts/evolve_ghz_pareto.py "
                            f"--card {CARDS}/device_card_{card}.json "
                            f"--n {n} --genomes {genomes} --pop {pop} "
                            f"--seed {s} --out {d} --quiet"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=["bell4", "ghz"], required=True)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--genomes", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--all", action="store_true",
                    help="include cells that already have a summary.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.campaign == "bell4":
        seeds, genomes, pop = args.seeds or 3, args.genomes or 12000, args.pop or 50
        cells = bell_cells(seeds, genomes, pop)
    else:
        seeds, genomes, pop = args.seeds or 6, args.genomes or 40000, args.pop or 80
        cells = ghz_cells(seeds, genomes, pop)

    kept = [(d, c) for d, c in cells
            if args.all or not (ROOT / d / "summary.json").exists()]
    Path(args.out).write_text("\n".join(c for _, c in kept) + "\n", encoding="utf-8")
    print(f"{args.campaign}: {len(cells)} cells, {len(kept)} written to {args.out}")
    if kept:
        print(f"submit with:  sbatch -M tigris --array=0-{len(kept) - 1} "
              f"~/Non-Markov/code/remote/slurm/search_tigris.sbatch "
              f"~/exaqc/{Path(args.out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
