from __future__ import annotations

import argparse
import numpy as np
import json
import sys

from loguru import logger
from pathlib import Path

from src.circuits.circuit import CircuitGenome


def get_group_metrics(
    input_directories: list[str], metric: str, group: str = None
) -> (dict[str, dict[str, int]], dict[str, list[float]], int, int):
    """
    Parses through the given input directories for runs with the 'group' flag and then
    generates the count statistics for mutation insertion rates. If the group flag is not
    provided then use all input directories.

    Args:
        input_directories: the directories to look through
        metric: the metric to track (e.g., test_acc, valid_acc)
        group: the string to glob for run directories to use in the metrics calculation if
            provided.

    Returns:
        The mutation counts by type dict, a dict of the different metric lists, and
        the best genome n gates and n parameters
    """

    print(f"PARSING FILES FOR GROUP '{group}' WITH METRIC '{metric}'")

    insert_counts = {}

    best_metrics = []
    best_n_gates_list = []
    best_n_parameters_list = []

    overall_best_metric = 0
    overall_best_n_gates = 100000
    overall_best_n_parameters = 100000

    overall_best_json = ""

    for directory in args.input_directories:
        if group is not None:
            # skip the directory if it doesnt contain the search string
            if group not in directory:
                continue

        genome_directory = Path(directory + "/all_genomes/")
        logger.info(f"parsing directory: {genome_directory}")

        best_metric = 0
        best_n_gates = 10000
        best_n_parameters = 10000

        for genome_json in genome_directory.glob("*.json"):

            with open(genome_json, "r") as file:
                genome = json.load(file)

                metric_value = genome["fitness"][metric]
                n_gates = len(genome["gates"])
                n_parameters = sum(
                    [len(gate["parameters"]) for gate in genome["gates"]]
                )

                # logger.info(f"\tgenome: {genome_json} had metric '{metric}': {metric_value}")

                if metric_value > best_metric:
                    logger.info(
                        f"genome {genome_json} had NEW best metric {metric_value} with n "
                        f"gates {n_gates} and n parameters {n_parameters}"
                    )

                    best_metric = metric_value
                    best_n_gates = n_gates
                    best_n_parameters = n_parameters

                if metric_value == best_metric:
                    if n_gates + n_parameters < best_n_gates + best_n_parameters:
                        logger.info(
                            f"genome {genome_json} had SMALLER best metric {metric_value} with n "
                            f"gates {n_gates} and n parameters {n_parameters}"
                        )
                        best_n_gates = n_gates
                        best_n_parameters = n_parameters

                if metric_value > overall_best_metric:
                    logger.info(
                        f"genome {genome_json} had NEW overall best metric {metric_value} with n "
                        f"gates {n_gates} and n parameters {n_parameters}"
                    )
                    overall_best_json = genome_json
                    overall_best_metric = metric_value

                    overall_best_n_gates = n_gates
                    overall_best_n_parameters = n_parameters

                if metric_value == overall_best_metric:
                    if (
                        n_gates + n_parameters
                        < overall_best_n_gates + overall_best_n_parameters
                    ):
                        logger.info(
                            f"genome {genome_json} had SMALLER overall best metric {metric_value} "
                            f"with n gates {n_gates} and n parameters {n_parameters}"
                        )
                        overall_best_json = genome_json
                        overall_best_n_gates = n_gates
                        overall_best_n_parameters = n_parameters

                        circuit_genome = CircuitGenome.from_dict(genome)
                        circuit_genome.save_circuit("analysis_smallest", "./")

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
        best_n_gates_list.append(best_n_gates)
        best_n_parameters_list.append(best_n_parameters)

    for gen_type in sorted(insert_counts.keys()):
        total = insert_counts[gen_type]["total"]
        for insert_type in sorted(insert_counts[gen_type].keys()):
            if insert_type == "total":
                continue

            count = insert_counts[gen_type][insert_type]

            print(f"{gen_type} : {insert_type} : {count} : {100.0 * count / total:.3f}")

    print("\n\n")
    print(f"best {metric} list: {best_metrics}")
    print(f"min {metric}: {np.min(best_metrics)}")
    print(f"max {metric}: {np.max(best_metrics)}")
    print(f"avg {metric}: {np.mean(best_metrics)}")
    print(f"stddev {metric}: {np.std(best_metrics)}")

    print("\n")
    print(f"best n gates list: {best_n_gates_list}")
    print(f"min n gates: {np.min(best_n_gates_list)}")
    print(f"max n gates : {np.max(best_n_gates_list)}")
    print(f"avg n gates: {np.mean(best_n_gates_list)}")
    print(f"stddev n_gates: {np.std(best_n_gates_list)}")

    print("\n")
    print(f"best n parameters list: {best_n_parameters_list}")
    print(f"min n parameters: {np.min(best_n_parameters_list)}")
    print(f"max n parameters : {np.max(best_n_parameters_list)}")
    print(f"avg n parameters: {np.mean(best_n_parameters_list)}")
    print(f"stddev n_parameters: {np.std(best_n_parameters_list)}")

    print("\n")
    print(f"best genome n parameters: {overall_best_n_parameters}")
    print(f"best genome n gates: {overall_best_n_gates}")

    print("\n")
    print(f"best genome json: {overall_best_json}")

    best_lists = {}
    best_lists[metric] = best_metrics
    best_lists["n_gates"] = best_n_gates_list
    best_lists["n_parameters"] = best_n_parameters_list

    return insert_counts, best_lists, overall_best_n_gates, overall_best_n_parameters


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

    if args.groups is None:
        input_counts = get_group_metrics(args.input_directories, metric)
    else:

        group_counts = {}
        for group in args.groups:
            (
                insert_counts,
                best_lists,
                overall_best_n_gates,
                overall_best_n_parameters,
            ) = get_group_metrics(args.input_directories, metric, group)
            print(f"insert counts for '{group}' were: {insert_counts}")
            group_counts[group] = insert_counts

        # create the summary table

        operators = set()
        result_types = set()

        for group in args.groups:
            for operator, insert_counts in group_counts[group].items():
                print(f"insert counts for '{operator}' were: {insert_counts}")

                operators.add(operator)
                print(f"added {operator} to operator set: {operators}")

                print(f"adding result types: {insert_counts.keys()}")
                result_types.update(insert_counts.keys())
                print(f"result types set now: {result_types}")

        print(f"operators: {operators}")
        print(f"result_types: {result_types}")

        print("\\begin{tabular}{lp{2cm}", end="")
        print("p{1.5cm}" * len(args.groups), end="")
        print("}")

        print("\\toprule")
        print(" &")
        for group in args.groups:
            print(f" & {{\\bf {group} }}", end="")
        print("\\\\")
        print("\\midrule")

        for operator in operators:
            cleaned_operator = operator.replace("n_ary", "n-ary").replace("_", "\\\\")
            global_best_line = (
                f"\\multirowcell{{3}}{{{cleaned_operator}}} & global best"
            )
            inserted_line = "& inserted"
            discarded_line = "& discarded"

            for group in args.groups:
                total_count = float(group_counts[group][operator]["total"])
                global_best_p = (
                    group_counts[group][operator]["global_best"] / total_count
                )
                inserted_p = group_counts[group][operator]["inserted"] / total_count
                discarded_p = group_counts[group][operator]["discarded"] / total_count

                global_best_line += f" & {global_best_p:.3f}"
                inserted_line += f" & {inserted_p:.3f}"
                discarded_line += f" & {discarded_p:.3f}"

            print(f"{global_best_line} \\\\")
            print(f"{inserted_line}\\\\")
            print(f"{discarded_line} \\\\")
            print("\\hline")

        print("\\end{tabular}")