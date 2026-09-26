"""Prepares a sweep of ``refine_genome`` runs testing whether PPO fine-tuning helps evolved genomes.

The search's own training data suggests that PPO fine-tuning improves weak
genomes but degrades strong ones. This sweep tests that directly by refining a
fixed set of genomes (``refine_sweep_genomes.tsv``: each run's best, plus one
mid and one weak genome from late in the search) under several learning rates
and entropy coefficients.

Two things make the arms comparable:

* A ``learning_rate=0`` control arm. Adam with a zero learning rate leaves every
  weight where it is, so each of that arm's evaluations scores the *inherited*
  weights. This is the "no training" baseline the search itself never
  measures, since its first evaluation follows one PPO update.
* Pinned seeds. Every arm of a (genome, replicate) pair uses the same training
  and evaluation seeds, so the arms face the same evaluation episodes (common
  random numbers). The seeds are written into each genome JSON's
  hyperparameters rather than passed with ``--set``, because ``--set`` coerces
  a value to the type already stored, and an unset seed is stored as ``None``.

Run from the repository root on the machine that holds the archives:

    python3 -m scripts.refine_sweep_prepare --sweep_dir /home/tjdvse/refine_sweep

It writes ``<sweep_dir>/genomes/*.json`` (one per genome and replicate) and
``<sweep_dir>/tasks.tsv`` (one row per job), and prints the ``sbatch`` command
that submits ``scripts/refine_sweep_job.sh`` as an array over the tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Any

from src.utils.genome_archive import load_genome_dict

#: Each arm's name, learning rate and entropy coefficient. The search ran with
#: learning_rate=0.0005 and entropy_coef=0.015; the ``control`` arm freezes the
#: weights, so its entropy coefficient does not matter.
ARMS: list[tuple[str, float, float]] = [
    ("control", 0.0, 0.015),
    ("lr5e-4_ent0.015", 5e-4, 0.015),
    ("lr5e-4_ent0", 5e-4, 0.0),
    ("lr1e-4_ent0.015", 1e-4, 0.015),
    ("lr1e-4_ent0", 1e-4, 0.0),
    ("lr5e-5_ent0.015", 5e-5, 0.015),
    ("lr5e-5_ent0", 5e-5, 0.0),
]

#: Columns of the task list the job script reads, in order.
TASK_COLUMNS: list[str] = [
    "task_id",
    "genome_json",
    "out_dir",
    "run_name",
    "genome_number",
    "tier",
    "replicate",
    "arm",
    "learning_rate",
    "entropy_coef",
]

#: Offset between the training and evaluation seeds. PPO's training seeds span
#: ``episodes * block`` values from the training seed, far less than this, so
#: the two ranges never overlap (the trainer raises if they would).
EVAL_SEED_OFFSET = 2**30


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for preparing the sweep.

    Returns:
        The configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sweep_dir",
        required=True,
        help="Directory to write the genome files, task list and results into.",
    )
    parser.add_argument(
        "--archive_dir",
        default="/home/tjdvse/genome_archives",
        help="Directory holding each run's archive directory, named by run_name.",
    )
    parser.add_argument(
        "--genomes",
        default=os.path.join(os.path.dirname(__file__), "refine_sweep_genomes.tsv"),
        help="Tab-separated list of run_name, genome_number and tier to refine.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="Seed replicates per genome; every arm of a replicate shares its seeds.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Training episodes per refinement (evaluated every log_every episodes).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260925,
        help="Seed for drawing each replicate's training and evaluation seeds.",
    )
    return parser


def read_genome_list(path: str) -> list[dict[str, str]]:
    """Reads the genomes to refine.

    Args:
        path: Tab-separated file with at least run_name, genome_number and tier
            columns.

    Returns:
        One dict per row, keyed by column name.
    """

    with open(path, encoding="utf-8", newline="") as genome_file:
        return list(csv.DictReader(genome_file, delimiter="\t"))


def pinned_genome(
    serialized: dict[str, Any], seed: int, episodes: int
) -> dict[str, Any]:
    """Returns a copy of a serialized genome with its seeds and schedule pinned.

    Args:
        serialized: The genome as stored in its run's archive.
        seed: Training seed; the evaluation seed is offset from it by
            :data:`EVAL_SEED_OFFSET`.
        episodes: Training episodes to run.

    Returns:
        The modified copy. Its ``seed``, ``eval_seed``, ``episodes`` and
        ``improvement_cutoff`` hyperparameters are set (the cutoff to 0, so
        every arm records a full-length curve); everything else is unchanged.
    """

    genome = json.loads(json.dumps(serialized))
    hyperparameters = genome["hyperparameters"]
    hyperparameters["seed"] = seed
    hyperparameters["eval_seed"] = seed + EVAL_SEED_OFFSET
    hyperparameters["episodes"] = episodes
    hyperparameters["improvement_cutoff"] = 0
    return genome


def main() -> None:
    """Writes the pinned genome files and the task list, then prints the submit command.

    Returns:
        None. Writes ``genomes/*.json`` and ``tasks.tsv`` under ``--sweep_dir``.
    """

    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    genome_dir = os.path.join(args.sweep_dir, "genomes")
    os.makedirs(genome_dir, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for row in read_genome_list(args.genomes):
        run_name = row["run_name"]
        genome_number = int(row["genome_number"])
        serialized = load_genome_dict(
            archive=os.path.join(args.archive_dir, run_name),
            genome_number=genome_number,
        )

        for replicate in range(args.replicates):
            seed = rng.randrange(0, EVAL_SEED_OFFSET - 1)
            stem = f"{run_name}_g{genome_number}_r{replicate}"
            genome_json = os.path.join(genome_dir, f"{stem}.json")
            with open(genome_json, "w", encoding="utf-8") as genome_file:
                json.dump(pinned_genome(serialized, seed, args.episodes), genome_file)

            for arm, learning_rate, entropy_coef in ARMS:
                tasks.append(
                    {
                        "task_id": len(tasks) + 1,
                        "genome_json": genome_json,
                        "out_dir": os.path.join(args.sweep_dir, "results", stem, arm),
                        "run_name": run_name,
                        "genome_number": genome_number,
                        "tier": row["tier"],
                        "replicate": replicate,
                        "arm": arm,
                        "learning_rate": learning_rate,
                        "entropy_coef": entropy_coef,
                    }
                )

    task_path = os.path.join(args.sweep_dir, "tasks.tsv")
    with open(task_path, "w", encoding="utf-8", newline="") as task_file:
        writer = csv.DictWriter(task_file, fieldnames=TASK_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(tasks)

    print(f"wrote {len(tasks)} tasks to {task_path}")
    print("submit with (Slurm does not create the log directory itself):")
    print("  mkdir -p /home/tjdvse/logs/exaqc_refine_sweep")
    print(f"  sbatch --array=1-{len(tasks)} scripts/refine_sweep_job.sh {task_path}")


if __name__ == "__main__":
    main()
