from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import os
import csv
import math
import glob

import numpy as np
import matplotlib.pyplot as plt

from src.circuits.circuit import CircuitGenome


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _gate_counts(genome: CircuitGenome) -> dict[str, float]:
    """
    Try to extract total gates + common breakdowns if present.
    Falls back to total enabled gates.
    """
    total = 0
    cnot = 0
    rot = 0

    for g in getattr(genome, "gates", []):
        if not getattr(g, "enabled", True):
            continue
        total += 1
        name = str(getattr(g, "method_name", "")).lower()

        if name in {"cx", "cnot"}:
            cnot += 1
        if name in {"rx", "ry", "rz", "u", "u3"}:
            rot += 1

    return {"gates_total": float(total), "gates_cnot": float(cnot), "gates_rot": float(rot)}


def default_fitness_extractor(genome: CircuitGenome) -> float:
    """
    Generic scalar to plot.
    Prefer RL metrics if present; otherwise classification loss.
    """
    fit = getattr(genome, "fitness", None) or {}

    # RL-style
    if isinstance(fit, dict):
        if "eval_return_mean" in fit:
            return _safe_float(fit["eval_return_mean"])
        if "train_return_mean" in fit:
            return _safe_float(fit["train_return_mean"])
        if "best_episode_return" in fit:
            return _safe_float(fit["best_episode_return"])

        # classification-style
        if "test_loss" in fit:
            return _safe_float(fit["test_loss"])
        if "loss" in fit:
            return _safe_float(fit["loss"])

    # fallback: try genome.fitness scalar
    return _safe_float(getattr(genome, "fitness", np.nan))


def _is_minimization(fitness_mode: str) -> bool:
    return fitness_mode.lower() in {"min", "minimize", "loss"}


def _sort_population(pop: list[CircuitGenome], fitness_fn: Callable[[CircuitGenome], float], minimize: bool) -> list[CircuitGenome]:
    vals = [(fitness_fn(g), g) for g in pop]
    vals.sort(key=lambda t: t[0], reverse=not minimize)
    return [g for _, g in vals]



@dataclass
class EXAQCPoint:
    step: int

    # population curves
    best: float
    top5_mean: float
    pop_mean: float

    # gate complexity
    gates_total: float
    gates_cnot: float
    gates_rot: float


class EXAQCProfiler:
    """
    Minimal-intrusion tracker for QNEAT-like plots:
      - top-5 average
      - best
      - population mean
      - gate counts

    Writes:
      <out_dir>/exaqc_history.csv

    Then you can aggregate multiple runs via EXAQCTracker.aggregate_and_plot(...)
    """

    def __init__(
        self,
        *,
        out_dir: str,
        run_name: str = "run0",
        fitness_fn: Callable[[CircuitGenome], float] = default_fitness_extractor,
        fitness_mode: str = "max",  # "max" for returns, "min" for losses
        topk: int = 5,
    ):
        self.out_dir = out_dir
        self.run_name = run_name
        self.fitness_fn = fitness_fn
        self.minimize = _is_minimization(fitness_mode)
        self.topk = int(topk)

        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, f"exaqc_history_{self.run_name}.csv")

        self.history: list[EXAQCPoint] = []

        # write header once
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(asdict(EXAQCPoint(0,0,0,0,0,0,0)).keys()))
                w.writeheader()

    def record(self, *, step: int, population: list[CircuitGenome]):
        if not population:
            return

        ordered = _sort_population(population, self.fitness_fn, self.minimize)
        vals = [self.fitness_fn(g) for g in ordered]

        best = vals[0]
        topk = vals[: min(self.topk, len(vals))]
        top5_mean = float(np.mean(topk)) if topk else float("nan")
        pop_mean = float(np.mean(vals)) if vals else float("nan")

        # gate counts: track for best
        gc = _gate_counts(ordered[0])

        pt = EXAQCPoint(
            step=int(step),
            best=_safe_float(best),
            top5_mean=_safe_float(top5_mean),
            pop_mean=_safe_float(pop_mean),
            gates_total=_safe_float(gc["gates_total"]),
            gates_cnot=_safe_float(gc["gates_cnot"]),
            gates_rot=_safe_float(gc["gates_rot"]),
        )
        self.history.append(pt)

        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(pt).keys()))
            w.writerow(asdict(pt))

    def plot_single_run(self, *, out_path: Optional[str] = None, title: Optional[str] = None):
        if not self.history:
            return

        steps = np.array([p.step for p in self.history], dtype=np.int32)
        best = np.array([p.best for p in self.history], dtype=np.float32)
        top5 = np.array([p.top5_mean for p in self.history], dtype=np.float32)
        popm = np.array([p.pop_mean for p in self.history], dtype=np.float32)

        fig = plt.figure()
        plt.plot(steps, top5, label="Top-5 mean")
        plt.plot(steps, best, label="Best")
        plt.plot(steps, popm, label="Population mean")
        plt.xlabel("Insertion / step")
        plt.ylabel("Fitness" if not self.minimize else "Loss")
        plt.title(title or f"EXAQC run: {self.run_name}")
        plt.legend()
        plt.grid(True, alpha=0.25)

        if out_path is None:
            out_path = os.path.join(self.out_dir, f"exaqc_curves_{self.run_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


    @staticmethod
    def _load_csv(path: str) -> list[dict[str, float]]:
        rows = []
        with open(path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append({k: _safe_float(v) for k, v in row.items()})
        return rows

    @staticmethod
    def aggregate_and_plot(
        *,
        csv_glob: str,
        out_path: str,
        metric: str = "top5_mean",
        conf: str = "std",  # "std" or "95ci"
        title: str = "EXAQC (mean ± confidence)",
    ):
        """
        Combine multiple runs (multiple CSVs) and plot mean + upper/lower confidence lines.

        conf:
          - "std": mean ± std
          - "95ci": mean ± 1.96*std/sqrt(n_runs)
        """
        paths = sorted(glob.glob(csv_glob))
        if not paths:
            raise FileNotFoundError(f"No CSVs matched: {csv_glob}")

        # load runs -> align by step
        runs = [EXAQCProfiler._load_csv(p) for p in paths]

        # build a common step grid (intersection is safest)
        step_sets = [{int(rw["step"]) for rw in run} for run in runs]
        common_steps = sorted(set.intersection(*step_sets))
        if not common_steps:
            raise RuntimeError("No common steps across runs. (Try using same run length.)")

        # matrix: [n_runs, n_steps]
        Y = []
        for run in runs:
            m = {int(rw["step"]): rw.get(metric, np.nan) for rw in run}
            Y.append([m[s] for s in common_steps])

        Y = np.array(Y, dtype=np.float32)
        mu = np.nanmean(Y, axis=0)
        sd = np.nanstd(Y, axis=0)
        n = Y.shape[0]

        if conf.lower() == "std":
            lo = mu - sd
            hi = mu + sd
            label = "±1 std"
        else:
            sem = sd / max(math.sqrt(n), 1.0)
            lo = mu - 1.96 * sem
            hi = mu + 1.96 * sem
            label = "±95% CI"

        fig = plt.figure()
        plt.plot(common_steps, mu, label=f"mean({metric})")
        plt.plot(common_steps, lo, linestyle="--", label=f"lower {label}")
        plt.plot(common_steps, hi, linestyle="--", label=f"upper {label}")
        plt.fill_between(common_steps, lo, hi, alpha=0.15)

        plt.xlabel("Insertion / step")
        plt.ylabel(metric)
        plt.title(title + f"  (n_runs={n})")
        plt.legend()
        plt.grid(True, alpha=0.25)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
