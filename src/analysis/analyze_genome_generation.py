from __future__ import annotations

import argparse
import numpy as np
import json
import sys

from loguru import logger
from pathlib import Path

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
        "--metric",
        type=str,
        required=False,
        default="test_acc",
        help="""The fitness metric to track statistics for, 'test_acc' by default.""",
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

    metric = args.metric

    insert_counts = {}

    best_metrics = []

    for directory in args.input_directories:
        logger.info(f"parsing directory: {directory}")

        genome_directory = Path(directory + "/all_genomes/")

        best_metric = 0
        for genome_json in genome_directory.glob("*.json"):
            logger.info(f"\tgenome: {genome_json}")

            with open(genome_json, "r") as file:
                genome = json.load(file)

                metric_value = genome["fitness"][metric]
                if metric_value > best_metric:
                    best_metric = metric_value

                metadata = genome["metadata"]
                insert_type = metadata["insert_type"]

                for gen_type in metadata["generated_by"]:
                    if gen_type not in insert_counts:
                        insert_counts[gen_type] = {}

                    if "total" not in insert_counts[gen_type]:
                        insert_counts[gen_type]["total"] = 1
                    else:
                        insert_counts[gen_type]["total"] += 1

                    if insert_type not in insert_counts[gen_type]:
                        insert_counts[gen_type][insert_type] = 1
                    else:
                        insert_counts[gen_type][insert_type] += 1

        best_metrics.append(best_metric)

    for gen_type in sorted(insert_counts.keys()):
        total = insert_counts[gen_type]["total"]
        for insert_type in sorted(insert_counts[gen_type].keys()):
            if insert_type == "total":
                continue

            count = insert_counts[gen_type][insert_type]

            print(f"{gen_type} : {insert_type} : {count} : {100.0 * count / total:.3f}")

    print("\n\n")
    print(f"best metrics: {best_metrics}")
    print(f"min {metric}: {np.min(best_metrics)}")
    print(f"avg {metric}: {np.max(best_metrics)}")
    print(f"max {metric}: {np.mean(best_metrics)}")
    print(f"stddev {metric}: {np.std(best_metrics)}")
