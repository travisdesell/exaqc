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
    """Safely cast a value to ``float``.

    Args:
        x: Value to convert to a float.
        default: Value to return if conversion fails. Defaults to ``np.nan``.

    Returns:
        The converted float value if possible; otherwise ``default``.
    """
    try:
        return float(x)
    except Exception:
        return default


def _gate_counts(genome: CircuitGenome) -> dict[str, float]:
    """Compute simple enabled-gate complexity statistics for a genome.

    This helper attempts to extract:
      - total enabled gate count
      - count of CNOT-like gates (``cx``, ``cnot``)
      - count of common rotation/parameterized single-qubit gates (``rx``, ``ry``, ``rz``, ``u``, ``u3``)

    If the genome lacks a ``gates`` attribute, counts default to 0.

    Args:
        genome: Circuit genome whose enabled gates are counted.

    Returns:
        A dictionary with float-valued counts:
        ``{"gates_total": ..., "gates_cnot": ..., "gates_rot": ...}``.
    """
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

    return {
        "gates_total": float(total),
        "gates_cnot": float(cnot),
        "gates_rot": float(rot),
    }


def default_fitness_extractor(genome: CircuitGenome) -> float:
    """Extract a scalar fitness value from a ``CircuitGenome``.

    This provides a generic "metric to plot" across different experiment styles.

    Preference order when ``genome.fitness`` is a dict:
      1) RL-style metrics: ``eval_return_mean``, ``train_return_mean``, ``best_episode_return``
      2) Classification-style metrics: ``test_loss``, ``loss``

    If no known keys exist (or fitness is not a dict), it falls back to attempting to
    treat ``genome.fitness`` as a scalar.

    Args:
        genome: Circuit genome that holds fitness information.

    Returns:
        Extracted scalar value as a float. Returns ``np.nan`` if extraction fails.
    """
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


def _sort_population(
    pop: list[CircuitGenome],
    fitness_fn: Callable[[CircuitGenome], float],
    minimize: bool,
) -> list[CircuitGenome]:
    """Sort a population by fitness according to an extractor and objective direction.

    Args:
        pop: List of genomes to sort.
        fitness_fn: Function mapping a genome to a scalar fitness value.
        minimize: If True, lower fitness values are considered better; if False, higher is better.

    Returns:
        A new list of genomes sorted from best to worst according to the chosen direction.
    """
    vals = [(fitness_fn(g), g) for g in pop]
    vals.sort(key=lambda t: t[0], reverse=not minimize)
    return [g for _, g in vals]


@dataclass
class EXAQCPoint:
    """Single recorded datapoint for EXAQC profiling history.

    Attributes:
        step: Insertion index / step counter at which the datapoint was recorded.

        best: Best (top-1) fitness value in the population at this step.
        top5_mean: Mean fitness of the top-k genomes (k defaults to 5 in the profiler).
        pop_mean: Mean fitness across the entire population.

        gates_total: Total enabled gate count of the best genome.
        gates_cnot: Number of enabled CNOT-like gates (``cx``/``cnot``) in the best genome.
        gates_rot: Number of enabled rotation-like gates (``rx``/``ry``/``rz``/``u``/``u3``) in the best genome.
    """

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
    """Track EXAQC run history and generate QNEAT-style performance plots.

    This class is designed to be "minimal intrusion": you call ``record(step=..., population=...)``
    at each iteration/step, and it logs scalar summaries for plotting:

      - Best fitness in the population
      - Top-k mean fitness (k defaults to 5)
      - Population mean fitness
      - Simple gate-count complexity metrics of the best genome

    It writes a per-run CSV history at:
      ``<out_dir>/exaqc_history_<run_name>.csv``

    You can later combine multiple runs (multiple CSVs) with
    ``EXAQCProfiler.aggregate_and_plot(...)`` to produce mean curves with confidence bands.

    Args:
        out_dir: Output directory where CSV and plots are written.
        run_name: Identifier appended to output filenames. Defaults to ``"run0"``.
        fitness_fn: Function mapping a genome to a scalar fitness value. Defaults to
            ``default_fitness_extractor``.
        fitness_mode: ``"max"`` for higher-is-better metrics (e.g., returns) or ``"min"``
            for lower-is-better metrics (e.g., losses). Defaults to ``"max"``.
        topk: The k used for the "top-k mean" curve. Defaults to 5.

    Attributes:
        out_dir: Output directory where artifacts are saved.
        run_name: Run identifier used in filenames.
        fitness_fn: Fitness extraction callable.
        minimize: Whether the objective is minimization (derived from ``fitness_mode``).
        topk: Integer k used for the top-k mean.
        csv_path: Full path to the per-run CSV file.
        history: In-memory list of recorded ``EXAQCPoint`` values.
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
        self.minimize = fitness_mode.lower() == "min"
        self.topk = int(topk)

        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, f"exaqc_history_{self.run_name}.csv")

        self.history: list[EXAQCPoint] = []

        # write header once
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=list(asdict(EXAQCPoint(0, 0, 0, 0, 0, 0, 0)).keys())
                )
                w.writeheader()

    def record(self, *, step: int, population: list[CircuitGenome]):
        """Record a new profiling datapoint for the given population.

        This computes:
          - best fitness (top-1)
          - top-k mean fitness (k = ``self.topk``)
          - population mean fitness
          - gate-count metrics for the best genome

        The datapoint is appended to ``self.history`` and written as a row to ``self.csv_path``.

        Args:
            step: Insertion index / iteration step to associate with this snapshot.
            population: Current population of genomes at this step.

        Returns:
            None. If ``population`` is empty, the method returns early without writing.
        """
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

    def plot_single_run(
        self, *, out_path: Optional[str] = None, title: Optional[str] = None
    ):
        """Plot performance curves for the currently recorded run history.

        The plot includes:
          - Top-k mean curve
          - Best curve
          - Population mean curve

        The figure is saved to disk and then closed.

        Args:
            out_path: Output file path for the figure. If None, defaults to
                ``<out_dir>/exaqc_curves_<run_name>.png``.
            title: Optional title override. If None, uses ``"EXAQC run: <run_name>"``.

        Returns:
            None. If no history has been recorded, the method returns early.
        """
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
        """Load an EXAQC history CSV into a list of float-valued dictionaries.

        Args:
            path: Path to a CSV produced by ``EXAQCProfiler`` (or matching schema).

        Returns:
            List of rows, where each row is a dict mapping column names to float values.
            Non-parsable values are converted to ``np.nan`` via ``_safe_float``.
        """
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
        """Aggregate multiple run CSVs and plot mean with confidence bounds.

        This utility loads multiple CSV histories (typically one per run), aligns them by
        the intersection of available ``step`` values, and plots:
          - mean(metric)
          - lower confidence bound
          - upper confidence bound
          - shaded band between bounds

        The output is saved to ``out_path`` and the figure is closed.

        Args:
            csv_glob: Glob pattern matching run CSV files (e.g., ``"runs/*/exaqc_history_*.csv"``).
            out_path: File path to write the aggregated plot (e.g., PNG).
            metric: Column/field name to aggregate (e.g., ``"top5_mean"``, ``"best"``, ``"pop_mean"``).
                Defaults to ``"top5_mean"``.
            conf: Confidence style to plot:
                - ``"std"``: mean ± 1 standard deviation
                - ``"95ci"``: mean ± 1.96 * std / sqrt(n_runs)
                Defaults to ``"std"``.
            title: Plot title prefix. The final title appends ``(n_runs=<n>)``.

        Raises:
            FileNotFoundError: If no CSV files match ``csv_glob``.
            RuntimeError: If the matched CSVs have no overlapping ``step`` values.

        Returns:
            None.
        """
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
        runs = [EXAQCProfiler._load_csv(p) for p in paths if p.endswith(".csv")]

        # build a common step grid (intersection is safest)
        step_sets = [{int(rw["step"]) for rw in run} for run in runs]
        common_steps = sorted(set.intersection(*step_sets))
        if not common_steps:
            raise RuntimeError(
                "No common steps across runs. (Try using same run length.)"
            )

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
