"""Bounded-worker campaign driver.

Launching the whole search matrix at once with shell backgrounding overwhelmed
the machine and lost runs silently (2026-09-26: 36 of 72 Bell searches died).
This driver keeps a fixed number of workers busy, retries a failed cell once,
and reports exactly which cells produced a summary.json.

    .venv_win/Scripts/python scripts/run_campaign.py --campaign bell4 --workers 6
    .venv_win/Scripts/python scripts/run_campaign.py --campaign ghz_naive --workers 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv_win" / "Scripts" / "python.exe")
CARDS = "../../Arch_Aware_Circuit_Evolution/device_cards"

REAL = ["transmon_detuned_schrod", "transmon_detuned_lindblad_markov",
        "transmon_detuned_lindblad_nm", "transmon_detuned_mixed"]
NAIVE = ["naive_depol", "naive_closed"]
BELL = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]


def bell_cells(seeds: int, genomes: int, pop: int) -> List[Tuple[str, List[str], Path]]:
    cells = []
    for target in BELL:
        for card in REAL + NAIVE:
            for s in range(1, seeds + 1):
                out = ROOT / "results" / "bell4" / f"{target}__{card}_s{s}"
                cells.append((f"{target}/{card}/s{s}", [
                    PY, str(ROOT / "scripts" / "evolve_state_pareto.py"),
                    "--card", f"{CARDS}/device_card_{card}.json",
                    "--target", target, "--genomes", str(genomes),
                    "--pop", str(pop), "--seed", str(s),
                    "--out", str(out), "--quiet"], out))
    return cells


def ghz_cells(seeds: int, genomes: int, pop: int) -> List[Tuple[str, List[str], Path]]:
    """GHZ_3 / GHZ_4 on the detuned cards plus the naive controls."""
    cells = []
    for n in (3, 4):
        for card in REAL + NAIVE:
            for s in range(1, seeds + 1):
                out = ROOT / "results" / f"ghz{n}" / f"{card}_s{s}"
                cells.append((f"ghz{n}/{card}/s{s}", [
                    PY, str(ROOT / "scripts" / "evolve_ghz_pareto.py"),
                    "--card", f"{CARDS}/device_card_{card}.json",
                    "--n", str(n), "--genomes", str(genomes),
                    "--pop", str(pop), "--seed", str(s),
                    "--out", str(out), "--quiet"], out))
    return cells


def ghz_naive_cells(seeds: int, genomes: int, pop: int) -> List[Tuple[str, List[str], Path]]:
    cells = []
    for n in (3, 4):
        for card in NAIVE:
            for s in range(1, seeds + 1):
                tag = card.replace("naive_", "naive_")
                out = ROOT / "results" / f"ghz{n}" / f"{tag}_s{s}"
                cells.append((f"ghz{n}/{card}/s{s}", [
                    PY, str(ROOT / "scripts" / "evolve_ghz_pareto.py"),
                    "--card", f"{CARDS}/device_card_{card}.json",
                    "--n", str(n), "--genomes", str(genomes),
                    "--pop", str(pop), "--seed", str(s),
                    "--out", str(out), "--quiet"], out))
    return cells


def run_cell(cell) -> Tuple[str, bool, str]:
    label, cmd, out = cell
    out.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    for attempt in (1, 2):
        with open(log, "w") as fh:
            p = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        if (out / "summary.json").exists():
            d = json.load(open(out / "summary.json"))
            return label, True, f"front={len(d['front'])} {d['elapsed_s']:.0f}s"
        if attempt == 1:
            continue
    return label, False, f"rc={p.returncode}; see {log}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=["bell4", "ghz", "ghz_naive"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--genomes", type=int, default=12000)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--fresh", action="store_true",
                    help="delete existing cell outputs before running")
    args = ap.parse_args()

    builder = {"bell4": bell_cells, "ghz": ghz_cells,
               "ghz_naive": ghz_naive_cells}[args.campaign]
    cells = builder(args.seeds, args.genomes, args.pop)
    if args.fresh:
        import shutil
        for _, _, out in cells:
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
    todo = [c for c in cells if not (c[2] / "summary.json").exists()]
    print(f"{args.campaign}: {len(cells)} cells, {len(todo)} to run, "
          f"{args.workers} workers", flush=True)
    ok = bad = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for label, good, msg in ex.map(run_cell, todo):
            ok, bad = ok + int(good), bad + int(not good)
            print(f"  [{'ok ' if good else 'FAIL'}] {label:44s} {msg}", flush=True)
    done = sum(1 for _, _, out in cells if (out / "summary.json").exists())
    print(f"{args.campaign}: {done}/{len(cells)} cells have summary.json "
          f"(this run: {ok} ok, {bad} failed)")
    return 0 if done == len(cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
