"""Plot evolution of quantum circuit gate and parameter counts.

Generates PPSN-style plots showing the best and average number of gates
and parameters throughout the evolutionary search. Statistics are
aggregated across multiple independent runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def natural_sort_key(path: Path) -> list:
    """Generate a natural sorting key for a path.

    Args:
        path: Path to sort.

    Returns:
        Natural sorting key.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def count_parameters(genome: dict) -> int:
    """Count trainable quantum gate parameters in a genome.

    Args:
        genome: Genome dictionary loaded from JSON.

    Returns:
        Number of gate parameters.
    """
    return sum(
        len(gate.get("parameters", []))
        for gate in genome.get("gates", [])
        if gate.get("enabled", True)
    )


def count_gates(genome: dict) -> int:
    """Count enabled gates in a genome.

    Args:
        genome: Genome dictionary loaded from JSON.

    Returns:
        Number of enabled gates.
    """
    return sum(
        1
        for gate in genome.get("gates", [])
        if gate.get("enabled", True)
    )


def get_metric(genome: dict, metric: str) -> float:
    """Get the requested fitness metric.

    Args:
        genome: Genome dictionary.
        metric: Fitness metric name.

    Returns:
        Fitness value.
    """
    return float(genome["fitness"][metric])


def load_run(
    run_directory: Path,
    metric: str,
    maximize: bool,
) -> dict[str, np.ndarray]:
    """Load the evolutionary trajectory from one run.

    Args:
        run_directory: Directory containing ``all_genomes``.
        metric: Fitness metric used to identify the best genome.
        maximize: Whether larger metric values are better.

    Returns:
        Evolutionary trajectories for gates and parameters.
    """
    genome_directory = run_directory / "all_genomes"

    genome_files = sorted(
        genome_directory.glob("*.json"),
        key=natural_sort_key,
    )

    if not genome_files:
        raise RuntimeError(
            f"No genome JSON files found in {genome_directory}"
        )

    gate_counts = []
    parameter_counts = []
    metric_values = []

    for genome_file in genome_files:
        try:
            with open(genome_file, "r", encoding="utf-8") as file:
                genome = json.load(file)
        except (json.JSONDecodeError, OSError) as error:
            print(
                "Skipping invalid genome file '%s': %s",
                genome_file,
                error,
            )
            continue

        if not isinstance(genome, dict):
            print(
                "Skipping genome file '%s': expected JSON object.",
                genome_file,
            )
            continue

        if "gates" not in genome:
            print(
                "Skipping genome file '%s': missing 'gates'.",
                genome_file,
            )
            continue

        gate_counts.append(count_gates(genome))
        parameter_counts.append(count_parameters(genome))
        metric_values.append(get_metric(genome, metric))

    gate_counts = np.asarray(gate_counts, dtype=float)
    parameter_counts = np.asarray(parameter_counts, dtype=float)
    metric_values = np.asarray(metric_values, dtype=float)

    # --------------------------------------------------------------
    # Best-so-far genome.
    # --------------------------------------------------------------

    best_gate_counts = np.zeros(len(genome_files))
    best_parameter_counts = np.zeros(len(genome_files))

    if maximize:
        best_metric = -np.inf
    else:
        best_metric = np.inf

    best_gates = 0
    best_parameters = 0

    for i in range(len(genome_files)):
        metric_value = metric_values[i]

        if maximize:
            improved = metric_value > best_metric
        else:
            improved = metric_value < best_metric

        if improved:
            best_metric = metric_value
            best_gates = gate_counts[i]
            best_parameters = parameter_counts[i]

        best_gate_counts[i] = best_gates
        best_parameter_counts[i] = best_parameters

    # --------------------------------------------------------------
    # Running average.
    #
    # This is the average complexity of all generated genomes up
    # through the current insertion.
    # --------------------------------------------------------------

    average_gate_counts = np.cumsum(gate_counts) / np.arange(
        1,
        len(gate_counts) + 1,
    )

    average_parameter_counts = np.cumsum(
        parameter_counts
    ) / np.arange(
        1,
        len(parameter_counts) + 1,
    )

    return {
        "best_gates": best_gate_counts,
        "average_gates": average_gate_counts,
        "best_parameters": best_parameter_counts,
        "average_parameters": average_parameter_counts,
    }


def truncate_runs(
    runs: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Truncate all runs to the shortest run length.

    Args:
        runs: Run trajectory dictionaries.

    Returns:
        Truncated run trajectories.
    """
    min_length = min(
        len(run["best_gates"])
        for run in runs
    )

    print(f"Using {min_length} insertion steps.")

    truncated = []

    for run in runs:
        truncated.append(
            {
                key: values[:min_length]
                for key, values in run.items()
            }
        )

    return truncated


def mean_std(
    runs: list[dict[str, np.ndarray]],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mean and standard deviation across runs.

    Args:
        runs: Evolutionary run trajectories.
        key: Metric key.

    Returns:
        Mean and standard deviation arrays.
    """
    values = np.stack(
        [run[key] for run in runs],
        axis=0,
    )

    return (
        np.mean(values, axis=0),
        np.std(values, axis=0),
    )


def plot_trajectory(
    runs: list[dict[str, np.ndarray]],
    output_file: Path,
    title: str,
) -> None:
    """Generate PPSN-style gate and parameter evolution plot.

    Args:
        runs: Evolutionary trajectories.
        output_file: Output figure path.
        title: Plot title.
    """
    runs = truncate_runs(runs)

    best_gates_mean, best_gates_std = mean_std(
        runs,
        "best_gates",
    )

    avg_gates_mean, avg_gates_std = mean_std(
        runs,
        "average_gates",
    )

    best_params_mean, best_params_std = mean_std(
        runs,
        "best_parameters",
    )

    avg_params_mean, avg_params_std = mean_std(
        runs,
        "average_parameters",
    )

    steps = np.arange(len(best_gates_mean))

    fig, ax = plt.subplots(
        figsize=(7.0, 4.5),
    )

    # --------------------------------------------------------------
    # Best gates.
    # --------------------------------------------------------------

    ax.plot(
        steps,
        best_gates_mean,
        label="Best gates",
        linewidth=1.5,
    )

    ax.fill_between(
        steps,
        best_gates_mean - best_gates_std,
        best_gates_mean + best_gates_std,
        alpha=0.18,
    )

    # --------------------------------------------------------------
    # Average gates.
    # --------------------------------------------------------------

    ax.plot(
        steps,
        avg_gates_mean,
        linestyle="--",
        label="Average gates",
        linewidth=1.5,
    )

    ax.fill_between(
        steps,
        avg_gates_mean - avg_gates_std,
        avg_gates_mean + avg_gates_std,
        alpha=0.18,
    )

    # --------------------------------------------------------------
    # Best parameters.
    # --------------------------------------------------------------

    ax.plot(
        steps,
        best_params_mean,
        label="Best parameters",
        linewidth=1.5,
    )

    ax.fill_between(
        steps,
        best_params_mean - best_params_std,
        best_params_mean + best_params_std,
        alpha=0.18,
    )

    # --------------------------------------------------------------
    # Average parameters.
    # --------------------------------------------------------------

    ax.plot(
        steps,
        avg_params_mean,
        linestyle="--",
        label="Average parameters",
        linewidth=1.5,
    )

    ax.fill_between(
        steps,
        avg_params_mean - avg_params_std,
        avg_params_mean + avg_params_std,
        alpha=0.18,
    )

    ax.set_xlabel("Insertion / Step")
    ax.set_ylabel("Count")
    ax.set_title(title)

    ax.grid(
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        loc="upper left",
    )

    fig.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    # Vector version for LaTeX.
    fig.savefig(
        output_file.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")
    print(f"Saved: {output_file.with_suffix('.pdf')}")


def main() -> None:
    """Run gate/parameter trajectory analysis."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_directories",
        "-i",
        nargs="+",
        required=True,
        help="Independent EXAQC run directories.",
    )

    parser.add_argument(
        "--metric",
        default="valid_acc",
        help="Fitness metric used to determine the best genome.",
    )

    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Treat the metric as a minimization objective.",
    )

    parser.add_argument(
        "--title",
        default="EXAQC Gates and Parameters",
        help="Figure title.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_plots/gates_parameters.png"
        ),
        help="Output PNG path.",
    )

    args = parser.parse_args()

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.2,
    )

    runs = []

    for directory in args.input_directories:
        directory = Path(directory)

        print(f"Reading: {directory}")

        run = load_run(
            directory,
            metric=args.metric,
            maximize=not args.minimize,
        )

        runs.append(run)

    if not runs:
        raise RuntimeError(
            "No valid runs were provided."
        )

    print(
        f"Loaded {len(runs)} independent runs."
    )

    plot_trajectory(
        runs,
        args.output,
        args.title,
    )


if __name__ == "__main__":
    main()