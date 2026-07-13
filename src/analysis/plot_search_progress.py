from __future__ import annotations

import argparse
import csv
import numpy as np
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

from loguru import logger


def get_group_metrics(
    input_directories: list[str], column_max_length: int, group: str = None
) -> (list[float], list[float], list[float]):
    """
    Parses through the given input directories for runs with the 'group' flag and for all
    runs in the group, calculate the avg, min and max for each insertion time step so these
    can make progress plots.

    Args:
        input_directories: the directories to look through
        column_max_length: is the required length for each column, any values after this will
            be clipped.
        group: the string to glob for run directories to use in the metrics calculation if
            provided.

    Returns:
        Three lists, the minimum list, average list, and maximum list of fitnesses for each
        time step.
    """

    print(f"PARSING FILES FOR GROUP '{group}'")

    count = 0

    group_progress = []
    for directory in args.input_directories:
        if group is not None:
            # skip the directory if it doesnt contain the search string
            if group not in directory:
                continue

        best_column = []
        filename = directory + "/exaqc_history.csv"
        print(f"\topening: {filename}")
        with open(filename, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                best_column.append(float(row["best_genome_train_fitness"]))

        if len(best_column) < column_max_length:
            print(
                f"ERROR: this file only had {len(best_column)} rows, less than the "
                f"minimum required column max length: {column_max_length}"
            )
            exit(1)

        print(f"\tcapping file with length {len(best_column)} to {column_max_length}")

        group_progress.append(best_column[0:column_max_length])

        count += 1

    group_array = np.array(group_progress)
    print(f"\tgroup array shape: {group_array.shape}")

    min_list = []
    avg_list = []
    max_list = []
    for i in range(column_max_length):
        row = group_array[:, i]
        row_min = np.min(row)
        row_avg = np.mean(row)
        row_max = np.max(row)
        print(
            f"row[{i}] shape is: {row.shape}, min: {row_min}, avg: {row_avg}, max: {row_max}"
        )

        min_list.append(row_min)
        avg_list.append(row_avg)
        max_list.append(row_max)

    return (min_list, avg_list, max_list)


if __name__ == "__main__":
    """
    This will parse all the provided input directories, reading all the genomes in the
    `all_genomes` subdirectory to calculate statistics about which crossovers and
    mutations had the best results (i.e., global best, local best, inserted or discarded)
    during the evolution process.

    It will also calculate the min/avg/max/stddev of the provided fitness metric.
    """

    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_directories",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="Input run output directories to analyze results from runs",
    )

    p.add_argument(
        "--groups",
        "-g",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Keyword to divide up runs by for table generation",
    )

    p.add_argument(
        "--column_max_length",
        type=int,
        required=True,
        help="""How many rows to use for each column in generating the progress plots.""",
    )

    p.add_argument(
        "--logging_level",
        type=str,
        required=False,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    args = p.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    if args.groups is None:
        (
            min_list,
            avg_list,
            max_list,
        ) = get_group_metrics(args.input_directories, args.column_max_length)
    else:

        group_mins = {}
        group_avgs = {}
        group_maxs = {}
        for group in args.groups:
            (
                min_list,
                avg_list,
                max_list,
            ) = get_group_metrics(args.input_directories, args.column_max_length, group)

            group_mins[group] = min_list
            group_avgs[group] = avg_list
            group_maxs[group] = max_list

        fig, ax = plt.subplots(1)

        y_min = None
        y_max = None
        xs = range(0, args.column_max_length)
        colors = mpl.color_sequences["Accent"]

        title = "Search Progress"
        position = 0
        for group in args.groups:
            ax.plot(xs, group_avgs[group], lw=2, label=group, color=colors[position])
            ax.fill_between(
                xs,
                group_mins[group],
                group_maxs[group],
                facecolor=colors[position],
                alpha=0.25,
            )
            position += 1

        ax.relim()
        if y_max is not None and y_min is not None:
            plt.ylim(ymax=y_max, ymin=y_min)

        if y_min is not None:
            plt.ylim(ymin=y_min)

        ax.set_title(title)
        legend_loc = "upper right"
        ax.legend(loc=legend_loc)
        ax.set_xlabel("Genomes Evaluated")
        ax.set_ylabel("Validation Loss")
        ax.grid()

        plt.show()

        """
        plt.savefig(filename)
        plt.clf()
        plt.cla()
        plt.close()
        """
