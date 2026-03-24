from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import csv
import glob
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.circuits.circuit import CircuitGenome
from src.utils.helpers import GATE_COMPLEXITY


def _safe_float(x, default=np.nan) -> float:
    """Safely cast a value to ``float``.

    Args:
        x: Value to convert.
        default: Fallback value returned if conversion fails.

    Returns:
        The converted float value if possible, otherwise ``default``.
    """
    try:
        return float(x)
    except Exception:
        return default


def _gate_counts(genome: CircuitGenome) -> dict[str, float]:
    """Compute simple enabled-gate statistics for a genome.

    This helper attempts to extract:
      - total enabled gate count
      - count of CNOT-like gates (``cx``, ``cnot``)
      - count of common parameterized single-qubit rotation gates
        (``rx``, ``ry``, ``rz``, ``u``, ``u3``)

    Args:
        genome: Circuit genome whose gates should be counted.

    Returns:
        A dictionary containing float-valued gate counts with keys:
        ``"gates_total"``, ``"gates_cnot"``, and ``"gates_rot"``.
    """
    total = 0
    cnot = 0
    rot = 0

    for gate in getattr(genome, "gates", []):
        if not getattr(gate, "enabled", True):
            continue

        name = str(getattr(gate, "method_name", "")).lower()

        gate_specs = GATE_COMPLEXITY[name]

        total += gate_specs["gate_count"]
        cnot += gate_specs["cnot_count"]
        rot += gate_specs["rot_count"]

    return {
        "gates_total": float(total),
        "gates_cnot": float(cnot),
        "gates_rot": float(rot),
    }


def _num_enabled_gates(genome: CircuitGenome) -> float:
    """Return the total number of enabled gates in a genome.

    Args:
        genome: Circuit genome to inspect.

    Returns:
        Number of enabled gates as a float.
    """
    return _gate_counts(genome)["gates_total"]


def _param_count(genome: CircuitGenome) -> float:
    """Count trainable genome parameters.

    This uses ``genome_to_torch_params(...)`` so the reported count matches
    the exact parameter set exposed to the PyTorch training code.

    Args:
        genome: Circuit genome to inspect.

    Returns:
        Number of trainable scalar parameters as a float.
    """
    total = 0
    for gate in genome.gates:
        if gate.enabled:
            total += len(gate.parameters)
    return float(total)


def _extract_train_fitness(genome: CircuitGenome) -> float:
    """Extract a train-fitness-like scalar from a genome.

    Preference order:
      - RL-style: ``train_return_mean``, ``eval_return_mean``,
        ``best_episode_return``
      - classification-style: ``train_loss``, ``test_loss``, ``loss``
      - scalar ``genome.fitness`` fallback

    Args:
        genome: Circuit genome holding fitness information.

    Returns:
        Extracted scalar train fitness value, or ``np.nan`` if unavailable.
    """
    fit = getattr(genome, "fitness", None)

    if isinstance(fit, dict):
        for key in (
            "train_return_mean",
            "eval_return_mean",
            "best_episode_return",
            "train_loss",
            "test_loss",
            "loss",
        ):
            if key in fit:
                return _safe_float(fit[key])

    return _safe_float(fit)


def default_fitness_extractor(genome: CircuitGenome) -> tuple[float, str]:
    """Extract a scalar fitness value and task mode from a genome.

    Preference order when ``genome.fitness`` is a dict:
      1. RL-style metrics: ``eval_return_mean``, ``train_return_mean``,
         ``best_episode_return``
      2. Classification-style metrics: ``test_loss``, ``train_loss``, ``loss``

    The returned mode is used to interpret whether higher is better
    (``"rl"``) or lower is better (``"class"`` / ``"default"``).

    Args:
        genome: Circuit genome that holds fitness information.

    Returns:
        A tuple ``(fitness_value, mode)`` where:
          - ``fitness_value`` is the extracted scalar
          - ``mode`` is one of ``"rl"``, ``"class"``, or ``"default"``
    """
    fit = getattr(genome, "fitness", None) or {}

    if isinstance(fit, dict):
        if "eval_return_mean" in fit:
            return _safe_float(fit["eval_return_mean"]), "rl"
        if "train_return_mean" in fit:
            return _safe_float(fit["train_return_mean"]), "rl"
        if "best_episode_return" in fit:
            return _safe_float(fit["best_episode_return"]), "rl"

        if "test_loss" in fit:
            return _safe_float(fit["test_loss"]), "class"
        if "train_loss" in fit:
            return _safe_float(fit["train_loss"]), "class"
        if "loss" in fit:
            return _safe_float(fit["loss"]), "class"

    return _safe_float(getattr(genome, "fitness", np.nan)), "default"


def _extract_population_fitness(
    pop: list[CircuitGenome],
    fitness_fn: Callable[[CircuitGenome], tuple[float, str]],
) -> tuple[list[tuple[float, CircuitGenome]], str]:
    """Extract fitness values for an already-sorted population.

    This profiler assumes the incoming population is already sorted in
    ascending order according to the same scalar metric represented by
    ``fitness_fn``.

    Interpretation:
      - For classification/loss settings, the best genome is the first genome.
      - For RL/reward settings, the best genome is the last genome.

    Args:
        pop: Population of genomes already sorted in ascending order.
        fitness_fn: Callable that maps a genome to ``(fitness_value, mode)``.

    Returns:
        A tuple ``(vals, mode)`` where:
          - ``vals`` is a list of ``(fitness_value, genome)`` pairs in the
            same order as the input population.
          - ``mode`` indicates whether the task is ``"rl"``, ``"class"``,
            or ``"default"``.
    """
    _, mode = fitness_fn(pop[0])
    vals = [(fitness_fn(genome)[0], genome) for genome in pop]
    return vals, mode


@dataclass
class EXAQCPoint:
    """Single recorded profiling datapoint for one population snapshot.

    Attributes:
        step: Insertion index or step counter.
        current_time: Elapsed wall-clock time in seconds since profiler creation.
        inserted_genomes: Number of times ``record()`` has been called.

        best_genome_train_fitness: Best train fitness across the population.
        avg_genome_train_fitness: Mean train fitness across the population.
        worst_genome_train_fitness: Worst train fitness across the population.

        best: Best ranking fitness in the population according to ``fitness_fn``.
        top5_mean: Mean fitness of the top-k genomes, with k controlled by
            ``EXAQCProfiler.topk``.
        pop_mean: Mean ranking fitness across the full population.

        gates_best: Number of enabled gates in the best genome.
        gates_avg: Average enabled gate count across all genomes.
        gates_worst: Number of enabled gates in the worst genome.
        gates_smallest: Smallest enabled gate count in the population.
        gates_largest: Largest enabled gate count in the population.

        params_best: Number of parameters in the best genome.
        params_avg: Average parameter count across all genomes.
        params_worst: Number of parameters in the worst genome.
        params_smallest: Smallest parameter count in the population.
        params_largest: Largest parameter count in the population.

        gates_cnot_best: Number of enabled CNOT-like gates in the best genome.
        gates_rot_best: Number of enabled rotation-like gates in the best genome.
    """

    step: int
    current_time: float
    inserted_genomes: int

    best_genome_train_fitness: float
    avg_genome_train_fitness: float
    worst_genome_train_fitness: float

    best: float
    top5_mean: float
    pop_mean: float

    gates_best: float
    gates_avg: float
    gates_worst: float
    gates_smallest: float
    gates_largest: float

    params_best: float
    params_avg: float
    params_worst: float
    params_smallest: float
    params_largest: float

    gates_cnot_best: float
    gates_rot_best: float


class EXAQCProfiler:
    """Track EXAQC run history and generate performance plots.

    This profiler is designed to be minimally intrusive: call ``record()``
    each time you want to log a population snapshot. It stores both fitness
    summaries and structural complexity summaries, then writes them to a CSV.

    Important:
        This profiler assumes the provided population is already sorted in
        ascending order according to the same scalar criterion used by
        ``fitness_fn``.

        Therefore:
          - for loss/classification tasks, best genome = ``population[0]``
          - for reward/RL tasks, best genome = ``population[-1]``

    Args:
        out_dir: Output directory where CSV and plots should be written.
        fitness_fn: Callable that extracts ``(fitness_value, mode)`` from a
            genome. The mode controls whether lower or higher is better.
        topk: Number of top genomes used for the top-k mean metric.

    Attributes:
        out_dir: Output directory for profiler artifacts.
        fitness_fn: Fitness extractor callable.
        topk: Number of genomes used for the top-k mean.
        csv_path: Path to the CSV file storing per-step history.
        history: In-memory list of recorded ``EXAQCPoint`` objects.
        start_time: Wall-clock timestamp when the profiler was created.
        inserted_genomes: Count of how many times ``record()`` has been called.
        mode: Current profiling mode, such as ``"rl"`` or ``"class"``.
    """

    def __init__(
        self,
        *,
        out_dir: str,
        fitness_fn: Callable[
            [CircuitGenome], tuple[float, str]
        ] = default_fitness_extractor,
        topk: int = 5,
    ):
        """Initialize the profiler.

        Args:
            out_dir: Output directory for CSV and plots.
            fitness_fn: Callable that returns ``(fitness_value, mode)`` for a
                genome.
            topk: Number of top genomes to include in the top-k mean metric.
        """
        self.out_dir = out_dir
        self.fitness_fn = fitness_fn
        self.topk = int(topk)

        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, "exaqc_history.csv")

        self.history: list[EXAQCPoint] = []
        self.start_time = time.time()
        self.inserted_genomes = 0
        self.mode = "default"

        if not os.path.exists(self.csv_path):
            dummy = EXAQCPoint(
                step=0,
                current_time=0.0,
                inserted_genomes=0,
                best_genome_train_fitness=0.0,
                avg_genome_train_fitness=0.0,
                worst_genome_train_fitness=0.0,
                best=0.0,
                top5_mean=0.0,
                pop_mean=0.0,
                gates_best=0.0,
                gates_avg=0.0,
                gates_worst=0.0,
                gates_smallest=0.0,
                gates_largest=0.0,
                params_best=0.0,
                params_avg=0.0,
                params_worst=0.0,
                params_smallest=0.0,
                params_largest=0.0,
                gates_cnot_best=0.0,
                gates_rot_best=0.0,
            )
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(dummy).keys()))
                writer.writeheader()

    def record(self, *, step: int, population: list[CircuitGenome]) -> None:
        """Record a new profiling datapoint for the given population.

        This method assumes the population is already sorted in ascending order.

        It computes:
          - elapsed time and insertion count
          - best / average / worst train fitness
          - best fitness, top-k mean fitness, and population mean fitness
          - gate-count and parameter-count statistics

        The resulting datapoint is appended to ``self.history`` and written
        as a row in ``self.csv_path``.

        Args:
            step: Insertion index or iteration step associated with this
                population snapshot.
            population: Sorted population of genomes at the current step.

        Returns:
            None. If the population is empty, the method returns immediately.
        """
        if not population:
            return

        self.inserted_genomes += 1
        current_time = time.time() - self.start_time

        ordered, mode = _extract_population_fitness(population, self.fitness_fn)
        self.mode = mode

        vals = [value for value, _ in ordered]
        genomes = [genome for _, genome in ordered]

        if mode == "rl":
            best_genome = genomes[-1]
            worst_genome = genomes[0]
            best = vals[-1]
            topk_vals = vals[-min(self.topk, len(vals)) :]
        else:
            best_genome = genomes[0]
            worst_genome = genomes[-1]
            best = vals[0]
            topk_vals = vals[: min(self.topk, len(vals))]

        top5_mean = float(np.mean(topk_vals)) if topk_vals else float("nan")
        pop_mean = float(np.mean(vals)) if vals else float("nan")

        train_fitness_vals = np.array(
            [_extract_train_fitness(genome) for genome in population],
            dtype=np.float64,
        )

        if mode == "rl":
            best_train = float(np.nanmax(train_fitness_vals))
            worst_train = float(np.nanmin(train_fitness_vals))
        else:
            best_train = float(np.nanmin(train_fitness_vals))
            worst_train = float(np.nanmax(train_fitness_vals))

        avg_train = float(np.nanmean(train_fitness_vals))

        gate_vals = np.array(
            [_num_enabled_gates(genome) for genome in population],
            dtype=np.float64,
        )
        gates_best = float(_num_enabled_gates(best_genome))
        gates_worst = float(_num_enabled_gates(worst_genome))
        gates_avg = float(np.nanmean(gate_vals))
        gates_smallest = float(np.nanmin(gate_vals))
        gates_largest = float(np.nanmax(gate_vals))

        param_vals = np.array(
            [_param_count(genome) for genome in population],
            dtype=np.float64,
        )
        params_best = float(_param_count(best_genome))
        params_worst = float(_param_count(worst_genome))
        params_avg = float(np.nanmean(param_vals))
        params_smallest = float(np.nanmin(param_vals))
        params_largest = float(np.nanmax(param_vals))

        gc_best = _gate_counts(best_genome)

        point = EXAQCPoint(
            step=int(step),
            current_time=_safe_float(current_time),
            inserted_genomes=int(self.inserted_genomes),
            best_genome_train_fitness=_safe_float(best_train),
            avg_genome_train_fitness=_safe_float(avg_train),
            worst_genome_train_fitness=_safe_float(worst_train),
            best=_safe_float(best),
            top5_mean=_safe_float(top5_mean),
            pop_mean=_safe_float(pop_mean),
            gates_best=_safe_float(gates_best),
            gates_avg=_safe_float(gates_avg),
            gates_worst=_safe_float(gates_worst),
            gates_smallest=_safe_float(gates_smallest),
            gates_largest=_safe_float(gates_largest),
            params_best=_safe_float(params_best),
            params_avg=_safe_float(params_avg),
            params_worst=_safe_float(params_worst),
            params_smallest=_safe_float(params_smallest),
            params_largest=_safe_float(params_largest),
            gates_cnot_best=_safe_float(gc_best["gates_cnot"]),
            gates_rot_best=_safe_float(gc_best["gates_rot"]),
        )

        self.history.append(point)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(point).keys()))
            writer.writerow(asdict(point))

    def plot_single_run(
        self,
        *,
        out_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Plot fitness curves for the currently recorded run.

        The plot includes:
          - top-k mean fitness
          - best fitness
          - population mean fitness

        Args:
            out_path: Output figure path. If ``None``, the figure is saved to
                ``<out_dir>/exaqc_curves.png``.
            title: Optional plot title. If omitted, a default title is used.

        Returns:
            None. If no history exists, the method returns immediately.
        """
        if not self.history:
            return

        steps = np.array([point.step for point in self.history], dtype=np.int32)
        best = np.array([point.best for point in self.history], dtype=np.float32)
        top5 = np.array([point.top5_mean for point in self.history], dtype=np.float32)
        popm = np.array([point.pop_mean for point in self.history], dtype=np.float32)

        fig = plt.figure()
        plt.plot(steps, top5, label=f"Top-{self.topk} mean")
        plt.plot(steps, best, label="Best")
        plt.plot(steps, popm, label="Population mean")
        plt.xlabel("Insertion / step")
        plt.ylabel("Reward" if self.mode == "rl" else "Loss")
        plt.title(title or "EXAQC Run")
        plt.legend()
        plt.grid(True, alpha=0.25)

        if out_path is None:
            out_path = os.path.join(self.out_dir, "exaqc_curves.png")

        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _load_csv(path: str) -> list[dict[str, float]]:
        """Load an EXAQC history CSV into float-valued rows.

        Args:
            path: Path to a CSV file produced by ``EXAQCProfiler``.

        Returns:
            A list of rows, where each row is a dictionary mapping column
            names to float values. Non-parsable values become ``np.nan``.
        """
        rows = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({key: _safe_float(value) for key, value in row.items()})
        return rows

    @staticmethod
    def aggregate_and_plot(
        *,
        csv_glob: str,
        out_path: str,
        metric: Optional[str] = "top5_mean",
        conf: str = "std",
        title: str = "EXAQC (mean ± confidence)",
    ) -> None:
        """Aggregate multiple run CSVs and plot mean with confidence bounds.

        This method aligns runs by the intersection of their available step
        values, then plots the mean of the selected metric with either
        standard deviation bands or 95% confidence intervals.

        Args:
            csv_glob: Glob pattern matching profiler CSV files.
            out_path: Output path for the aggregated plot.
            metric: Name of the metric column to aggregate. If ``None``,
                plots ``"top5_mean"``, ``"best"``, and ``"pop_mean"``.
            conf: Confidence style. Supported values are:
                - ``"std"`` for mean ± 1 standard deviation
                - ``"95ci"`` for mean ± 1.96 * std / sqrt(n_runs)
            title: Title prefix for the plot.

        Raises:
            FileNotFoundError: If no matching CSV files are found.
            RuntimeError: If the matched runs have no common step values.

        Returns:
            None.
        """
        paths = sorted(glob.glob(csv_glob))
        if not paths:
            raise FileNotFoundError(f"No CSVs matched: {csv_glob}")

        runs = [
            EXAQCProfiler._load_csv(path) for path in paths if path.endswith(".csv")
        ]

        step_sets = [{int(row["step"]) for row in run} for run in runs]
        common_steps = sorted(set.intersection(*step_sets))
        if not common_steps:
            raise RuntimeError(
                "No common steps across runs. Try using the same run length."
            )

        fig = plt.figure()
        metrics = ["top5_mean", "best", "pop_mean"] if metric is None else [metric]

        n_runs = len(runs)

        for metric_name in metrics:
            Y = []
            for run in runs:
                step_to_value = {
                    int(row["step"]): row.get(metric_name, np.nan) for row in run
                }
                Y.append([step_to_value[step] for step in common_steps])

            Y = np.array(Y, dtype=np.float32)
            mu = np.nanmean(Y, axis=0)
            sd = np.nanstd(Y, axis=0)

            if conf.lower() == "std":
                lo = mu - sd
                hi = mu + sd
            else:
                sem = sd / max(math.sqrt(n_runs), 1.0)
                lo = mu - 1.96 * sem
                hi = mu + 1.96 * sem

            plt.plot(common_steps, mu, label=f"mean({metric_name})")
            plt.fill_between(common_steps, lo, hi, alpha=0.15)

        plt.xlabel("Insertion / step")
        plt.ylabel("Reward / Loss")
        plt.title(f"{title}  (n_runs={n_runs})")
        plt.legend()
        plt.grid(True, alpha=0.25)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
