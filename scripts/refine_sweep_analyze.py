"""Summarizes the fine-tuning sweep prepared by ``refine_sweep_prepare.py``.

For every (genome, replicate) pair, the ``control`` arm (learning rate 0) scores
the inherited weights on the same evaluation episodes the other arms use, so
each arm's evaluations are compared with the control's mean as a paired
difference. Reported per tier and arm:

* ``d@<episode>`` -- mean eval return at that episode minus the control mean.
  The columns are the episodes of :data:`CANDIDATE_EPISODES` the sweep
  evaluated, plus the last one analyzed, so they fit a sweep of any length.
* ``d_best`` -- the arm's best evaluation minus the *best* control evaluation,
  which compares like with like: both are a maximum over the same number of
  noisy evaluations, so the selection bias cancels.
* ``win`` -- fraction of pairs whose best evaluation beats the control's best.
* ``p_ep0`` -- fraction of pairs whose best evaluation came at episode 0.
* ``ep_best`` -- median episode of the arm's best evaluation, showing how long
  training kept paying off.

It also reports, per tier, how far the genomes' recorded search fitness sits
above the control's mean: the selection bias of taking a best-of-N evaluation
as fitness.

**Unfinished tasks.** A task's refined genome is only written when it finishes,
but ``refine_genome`` logs every evaluation to ``refine.log`` in the task's
output directory as it happens, so a task that is still running -- or was
killed at its time limit -- is read from that log instead. Because tasks stop
at different episodes, each pair's arms are cut to the shortest of their
curves (and to ``--max_episode``, when given) before anything is compared, so
``d_best``, ``win`` and ``p_ep0`` always compare the same number of
evaluations. Give ``--max_episode`` to put every pair on one horizon, so that
every column is computed over the same pairs.

**Analyzing a copy.** Paths in ``tasks.tsv`` are the cluster's, so every task's
files are looked up under ``--sweep_dir`` by their place in the sweep
(``results/<genome>_r<replicate>/<arm>/``). Only ``tasks.tsv`` and the
``refine.log`` files are needed; the genome's search fitness is read from the
log when the genome files are not there. For example::

    rsync -av --include='*/' --include='tasks.tsv' --include='refine.log' \\
        --include='refined_genome_*.json' --exclude='*' \\
        tjdvse@tigris.rc.rit.edu:/home/tjdvse/refine_sweep_1000/ ./refine_sweep_1000/
    python3 -m scripts.refine_sweep_analyze --sweep_dir ./refine_sweep_1000
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import statistics
from typing import Any
from collections import defaultdict

#: Episodes shown as ``d@`` columns when the sweep evaluated them; the last
#: episode analyzed is always shown as well (see :func:`shown_episodes`).
CANDIDATE_EPISODES: list[int] = [0, 10, 25, 50, 100, 250, 500, 750]

#: One evaluation logged by the RL trainer, e.g. ``... episode   35
#: train_return=... eval_return_mean=412.7 (stochastic)``.
EVALUATION_LINE = re.compile(
    r" episode\s+(\d+) .*eval_return_mean=(-?[0-9.]+(?:e[-+]?\d+)?)"
)

#: The line ``refine_genome`` logs before training, carrying the genome's
#: recorded search fitness as a dict literal.
STARTING_FITNESS_LINE = re.compile(r"starting fitness: (\{.*\})\s*$")

#: What a (run, genome, replicate) pair is keyed by.
Pair = tuple[str, str, str]


def shown_episodes(curves: list[dict[int, float]]) -> list[int]:
    """Picks the episodes to show as columns for the curves being analyzed.

    Args:
        curves: Every curve being compared, mapping evaluated episode to return.

    Returns:
        The candidate episodes that were evaluated and come before the last
        episode analyzed, followed by that last episode; empty when there are
        no curves.
    """

    evaluated = {episode for curve in curves for episode in curve}
    if not evaluated:
        return []
    final = max(evaluated)
    return [
        episode
        for episode in CANDIDATE_EPISODES
        if episode in evaluated and episode < final
    ] + [final]


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the sweep summary.

    Returns:
        The configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sweep_dir",
        required=True,
        help="The sweep directory (or a copy of it) holding tasks.tsv and results/.",
    )
    parser.add_argument(
        "--max_episode",
        type=int,
        default=None,
        help=(
            "Only analyze evaluations up to this episode, putting every pair on "
            "the same horizon (default: each pair's own shortest arm)."
        ),
    )
    return parser


def local_paths(task: dict[str, str], sweep_dir: str) -> tuple[str, str]:
    """Locates a task's output directory and genome file under ``sweep_dir``.

    Args:
        task: A row of ``tasks.tsv``, whose paths are where the sweep was
            prepared.
        sweep_dir: The sweep directory being analyzed, possibly a copy.

    Returns:
        ``(out_dir, genome_json)`` resolved under ``sweep_dir`` by their place
        in the sweep's layout.
    """

    genome_json = os.path.basename(task["genome_json"])
    stem = os.path.splitext(genome_json)[0]
    out_dir = os.path.join(sweep_dir, "results", stem, task["arm"])
    return out_dir, os.path.join(sweep_dir, "genomes", genome_json)


def read_log(path: str) -> tuple[dict[int, float], float | None]:
    """Reads the evaluations and starting fitness a ``refine.log`` recorded.

    A log that holds more than one run (a task that was run again) is read from
    its last run only.

    Args:
        path: The task's ``refine.log``.

    Returns:
        ``(curve, search_fitness)``: the last run's evaluations, mapping
        episode to mean return, and the genome's recorded target metric (None
        when the log does not record it).
    """

    curve: dict[int, float] = {}
    search_fitness: float | None = None
    # read as bytes and skip every line that cannot match before decoding it: a
    # log written at DEBUG is almost entirely forward-pass messages and can run
    # to hundreds of megabytes
    with open(path, "rb") as log_file:
        for raw in log_file:
            if b"eval_return_mean=" not in raw and b"starting fitness" not in raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            starting = STARTING_FITNESS_LINE.search(line)
            if starting:
                # a new run begins; anything logged before it was a previous run
                curve = {}
                try:
                    search_fitness = float(
                        ast.literal_eval(starting.group(1))["target_metric"]
                    )
                except (ValueError, SyntaxError, KeyError, TypeError):
                    search_fitness = None
                continue
            evaluation = EVALUATION_LINE.search(line)
            if evaluation:
                curve[int(evaluation.group(1))] = float(evaluation.group(2))
    return curve, search_fitness


def read_json(path: str) -> dict[str, Any] | None:
    """Reads a JSON file, tolerating one that is empty or cut short.

    A task that finishes after its disk has filled up leaves an empty (or
    truncated) refined genome behind, which must not stop the analysis.

    Args:
        path: The JSON file to read.

    Returns:
        The parsed object, or None when the file is missing, is not valid JSON,
        or does not hold an object.
    """

    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as json_file:
            loaded = json.load(json_file)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def refined_curve(refined: dict[str, Any]) -> dict[int, float] | None:
    """Reads the evaluation curve out of a refined genome.

    Args:
        refined: A refined genome, as ``refine_genome`` writes it.

    Returns:
        Its evaluations mapping episode to mean return, or None when it holds
        none (or not in the expected form).
    """

    try:
        curve = {
            int(evaluation["episode"]): float(evaluation["return_mean"])
            for evaluation in refined["metadata"]["evaluation_episode_metrics"]
        }
    except (KeyError, TypeError, ValueError):
        return None
    return curve or None


def load_curve(
    task: dict[str, str], sweep_dir: str
) -> tuple[dict[int, float], float, str] | None:
    """Loads one task's evaluation curve, whether or not it has finished.

    Args:
        task: A row of ``tasks.tsv``.
        sweep_dir: The sweep directory being analyzed, possibly a copy.

    Returns:
        ``(curve, search_fitness, source)``: the task's evaluations mapping
        episode to mean return, the target metric the genome was recorded with
        in the search, and where the curve came from -- ``"refined"`` for a
        finished task's refined genome, ``"log"`` for an unfinished task's log,
        or ``"unreadable"`` for a log read because the task's refined genome
        exists but cannot be read. None when the task has no evaluations yet,
        or its search fitness cannot be found.
    """

    out_dir, genome_json = local_paths(task, sweep_dir)
    refined_path = os.path.join(out_dir, f"refined_genome_{task['genome_number']}.json")
    log_path = os.path.join(out_dir, "refine.log")

    search_fitness: float | None = None
    original = read_json(genome_json)
    if original is not None:
        try:
            search_fitness = float(original["fitness"]["target_metric"])
        except (KeyError, TypeError, ValueError):
            search_fitness = None

    refined = read_json(refined_path)
    curve = refined_curve(refined) if refined is not None else None
    if curve is not None:
        source = "refined"
        if search_fitness is None and os.path.exists(log_path):
            search_fitness = read_log(log_path)[1]
    elif os.path.exists(log_path):
        curve, logged_fitness = read_log(log_path)
        if search_fitness is None:
            search_fitness = logged_fitness
        source = "unreadable" if os.path.exists(refined_path) else "log"
    else:
        return None

    if not curve or search_fitness is None:
        return None
    return curve, search_fitness, source


def truncate(curve: dict[int, float], horizon: int) -> dict[int, float]:
    """Keeps a curve's evaluations up to and including an episode.

    Args:
        curve: Evaluations mapping episode to mean return.
        horizon: The last episode to keep.

    Returns:
        The evaluations at episodes no later than ``horizon``.
    """

    return {episode: value for episode, value in curve.items() if episode <= horizon}


def mean(values: list[float]) -> float:
    """Returns the mean of a list, or NaN when it is empty.

    Args:
        values: The values to average.

    Returns:
        Their arithmetic mean, or NaN for an empty list.
    """

    return statistics.fmean(values) if values else float("nan")


def main() -> None:
    """Prints the paired arm-versus-control summary for each tier.

    Returns:
        None. Prints the tables to standard output.
    """

    args = build_parser().parse_args()
    with open(
        os.path.join(args.sweep_dir, "tasks.tsv"), encoding="utf-8", newline=""
    ) as task_file:
        tasks = list(csv.DictReader(task_file, delimiter="\t"))

    curves: dict[Pair, dict[str, dict[int, float]]] = defaultdict(dict)
    tiers: dict[Pair, str] = {}
    search_fitness: dict[Pair, float] = {}
    arms: list[str] = []
    sources: dict[str, int] = defaultdict(int)
    for task in tasks:
        if task["arm"] not in arms:
            arms.append(task["arm"])
        loaded = load_curve(task, args.sweep_dir)
        if loaded is None:
            continue
        curve, fitness, source = loaded
        sources[source] += 1
        pair = (task["run_name"], task["genome_number"], task["replicate"])
        curves[pair][task["arm"]] = curve
        search_fitness[pair] = fitness
        tiers[pair] = task["tier"]

    print(
        f"{sources['refined']} of {len(tasks)} tasks finished, "
        f"{sources['log']} unfinished read from their logs, "
        f"{sources['unreadable']} read from their logs because their refined "
        f"genome is empty or unreadable, "
        f"{len(tasks) - sum(sources.values())} with no evaluations yet"
    )

    # cut every arm of a pair to the pair's shortest arm (and --max_episode), so
    # each comparison is over the same evaluations
    horizons: dict[Pair, int] = {}
    for pair, arm_curves in curves.items():
        horizon = min(max(curve) for curve in arm_curves.values())
        if args.max_episode is not None:
            horizon = min(horizon, args.max_episode)
        horizons[pair] = horizon
        curves[pair] = {
            arm: truncate(curve, horizon) for arm, curve in arm_curves.items()
        }

    for tier in sorted(set(tiers.values())):
        pairs = [
            pair
            for pair, tier_of in tiers.items()
            if tier_of == tier and "control" in curves[pair]
        ]
        if not pairs:
            continue

        tier_horizons = sorted(horizons[pair] for pair in pairs)
        control_means = {
            pair: mean(list(curves[pair]["control"].values())) for pair in pairs
        }
        control_sd = mean(
            [
                statistics.stdev(curves[pair]["control"].values())
                for pair in pairs
                if len(curves[pair]["control"]) > 1
            ]
        )
        bias = mean([search_fitness[pair] - control_means[pair] for pair in pairs])
        print(
            f"\n== {tier}: {len(pairs)} pairs, inherited-weights return "
            f"{mean(list(control_means.values())):.0f} (sd across its evals {control_sd:.0f}), "
            f"search fitness minus inherited {bias:+.0f}"
        )
        print(
            f"   analyzed through episode {tier_horizons[0]} (shortest pair) to "
            f"{tier_horizons[-1]} (longest), median {statistics.median(tier_horizons):.0f}"
        )

        episodes = shown_episodes(
            [curve for pair in pairs for curve in curves[pair].values()]
        )
        print(
            "   pairs reaching each column: "
            + " ".join(
                f"d@{episode}={sum(horizons[pair] >= episode for pair in pairs)}"
                for episode in episodes
            )
        )

        header = (
            f"  {'arm':18s} {'n':>3s} "
            + " ".join(f"{'d@' + str(episode):>7s}" for episode in episodes)
            + f" {'d_best':>7s} {'win':>5s} {'p_ep0':>5s} {'ep_best':>7s}"
        )
        print(header)
        for arm in arms:
            arm_pairs = [pair for pair in pairs if arm in curves[pair]]
            if not arm_pairs:
                continue
            columns = []
            for episode in episodes:
                differences = [
                    curves[pair][arm][episode] - control_means[pair]
                    for pair in arm_pairs
                    if episode in curves[pair][arm]
                ]
                columns.append(f"{mean(differences):+7.0f}")
            best_differences = [
                max(curves[pair][arm].values()) - max(curves[pair]["control"].values())
                for pair in arm_pairs
            ]
            best_episodes = [
                max(curves[pair][arm], key=curves[pair][arm].__getitem__)
                for pair in arm_pairs
            ]
            print(
                f"  {arm:18s} {len(arm_pairs):3d} "
                + " ".join(columns)
                + f" {mean(best_differences):+7.0f}"
                + f" {mean([float(d > 0) for d in best_differences]):5.2f}"
                + f" {mean([float(b == 0) for b in best_episodes]):5.2f}"
                + f" {statistics.median(best_episodes):7.0f}"
            )


if __name__ == "__main__":
    main()
