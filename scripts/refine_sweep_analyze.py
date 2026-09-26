"""Summarizes the fine-tuning sweep prepared by ``refine_sweep_prepare.py``.

For every (genome, replicate) pair, the ``control`` arm (learning rate 0) scores
the inherited weights on the same evaluation episodes the other arms use, so
each arm's evaluations are compared with the control's mean as a paired
difference. Reported per tier and arm:

* ``d@<episode>`` -- mean eval return at that episode minus the control mean.
* ``d_best`` -- the arm's best evaluation minus the *best* control evaluation,
  which compares like with like: both are a maximum over the same number of
  noisy evaluations, so the selection bias cancels.
* ``win`` -- fraction of pairs whose best evaluation beats the control's best.
* ``p_ep0`` -- fraction of pairs whose best evaluation came at episode 0.

It also reports, per tier, how far the genomes' recorded search fitness sits
above the control's mean: the selection bias of taking a best-of-N evaluation
as fitness.

    python3 -m scripts.refine_sweep_analyze --sweep_dir /home/tjdvse/refine_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

#: Evaluation episodes shown as columns (the last is the final evaluation).
SHOWN_EPISODES: list[int] = [0, 5, 10, 20, 30, 49]


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the sweep summary.

    Returns:
        The configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sweep_dir",
        required=True,
        help="The directory refine_sweep_prepare.py wrote tasks.tsv into.",
    )
    return parser


def load_curve(task: dict[str, str]) -> tuple[dict[int, float], float] | None:
    """Loads one finished task's evaluation curve and its genome's search fitness.

    Args:
        task: A row of ``tasks.tsv``.

    Returns:
        ``(curve, search_fitness)``, where ``curve`` maps each evaluated episode
        to its mean return, and ``search_fitness`` is the target metric the
        genome was recorded with in the search; or None if the task has not
        finished.
    """

    path = os.path.join(task["out_dir"], f"refined_genome_{task['genome_number']}.json")
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as refined_file:
        refined = json.load(refined_file)
    with open(task["genome_json"], encoding="utf-8") as original_file:
        original = json.load(original_file)

    curve = {
        int(evaluation["episode"]): float(evaluation["return_mean"])
        for evaluation in refined["metadata"]["evaluation_episode_metrics"]
    }
    return curve, float(original["fitness"]["target_metric"])


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

    # (run, genome, replicate) -> arm -> curve
    curves: dict[tuple[str, str, str], dict[str, dict[int, float]]] = defaultdict(dict)
    tiers: dict[tuple[str, str, str], str] = {}
    search_fitness: dict[tuple[str, str, str], float] = {}
    arms: list[str] = []
    finished = 0
    for task in tasks:
        if task["arm"] not in arms:
            arms.append(task["arm"])
        loaded = load_curve(task)
        if loaded is None:
            continue
        finished += 1
        pair = (task["run_name"], task["genome_number"], task["replicate"])
        curves[pair][task["arm"]], search_fitness[pair] = loaded
        tiers[pair] = task["tier"]

    print(f"{finished} of {len(tasks)} tasks finished")

    for tier in sorted(set(tiers.values())):
        pairs = [
            pair
            for pair, tier_of in tiers.items()
            if tier_of == tier and "control" in curves[pair]
        ]
        if not pairs:
            continue

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

        header = (
            f"  {'arm':18s} {'n':>3s} "
            + " ".join(f"{'d@' + str(episode):>7s}" for episode in SHOWN_EPISODES)
            + f" {'d_best':>7s} {'win':>5s} {'p_ep0':>5s}"
        )
        print(header)
        for arm in arms:
            arm_pairs = [pair for pair in pairs if arm in curves[pair]]
            if not arm_pairs:
                continue
            columns = []
            for episode in SHOWN_EPISODES:
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
            best_at_zero = [
                max(curves[pair][arm], key=curves[pair][arm].__getitem__) == 0
                for pair in arm_pairs
            ]
            print(
                f"  {arm:18s} {len(arm_pairs):3d} "
                + " ".join(columns)
                + f" {mean(best_differences):+7.0f}"
                + f" {mean([float(d > 0) for d in best_differences]):5.2f}"
                + f" {mean([float(b) for b in best_at_zero]):5.2f}"
            )


if __name__ == "__main__":
    main()
