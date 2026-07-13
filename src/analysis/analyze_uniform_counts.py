"""
This was used for the 2026 GPTP book chapter to generate the line plots for insertion counts for the uniform strategies
with:

```
python3 -m src.analysis.analyze_uniform_counts \
    --input_directories ~/Data/2026_gptp_exaqc/wine_uniform/wine_i1_p64_per_class_* \
    ~/Data/2026_gptp_exaqc/breast_uniform/breast_cancer_i1_p64_per_class_* --groups uniform -e wine breast
```

And the histograms comparing uniform to exponential with:

```
python3 -m src.analysis.analyze_uniform_counts \
    --input_directories ~/Data/2026_gptp_exaqc/wine_uniform/wine_i1_p64_per_class_* \
    ~/Data/2026_gptp_exaqc/breast_norepop/breast_cancer_i1_p64* \
    ~/Data/2026_gptp_exaqc/wine_norepop/wine_i1_p64_per_class_* \
    ~/Data/2026_gptp_exaqc/breast_uniform/breast_cancer_i1_p64_per_class_* \
    --groups norepop uniform -e wine breast
```
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
from scipy import stats

from pathlib import Path


def get_group_metrics(
    mutation_df: dict[str, list[any]],
    crossover_df: dict[str, list[any]],
    input_directories: list[str],
    experiment: str,
    group: str,
):
    """
    Parses through the given input directories for runs with the 'group' flag and then
    generates the count statistics for mutation insertion rates. If the group flag is not
    provided then use all input directories.

    Args:
        mutation_df: a dict for the mutation dataframe which has 'Experiment',
        'N', and 'Rate (%)' columns, which will have rate rows appended to it.
        crossover_df: a dict for the crossover dataframe which has 'Experiment',
        'N', and 'Rate (%)' columns, which will have rate rows appended to it.
        input_directories: the directories to look through
        experiment: the first string to glob for to select the correct directories to parse
        group: the second string to glob for to select correct directories to parse
    """

    best_n_gates_list = []
    best_n_parameters_list = []
    best_fitness_list = []

    for directory in args.input_directories:
        # skip the directory if it doesnt contain the experiment search string
        if experiment not in directory:
            continue

        # skip the directory if it doesnt contain the group search string
        if group is not None and group not in directory:
            continue

        best_metric = 0
        best_n_gates = 10000
        best_n_parameters = 10000

        genome_directory = Path(directory + "/all_genomes/")
        print(f"\tparsing directory: {genome_directory}")

        crossover_global_count = {}
        crossover_insert_count = {}
        crossover_total = {}

        mutation_global_count = {}
        mutation_insert_count = {}
        mutation_total = {}

        # initialize the lists for summing up counts
        for i in range(1, 10):
            mutation_global_count[i] = 0
            mutation_insert_count[i] = 0
            mutation_total[i] = 0

        for i in range(2, 10):
            crossover_global_count[i] = 0
            crossover_insert_count[i] = 0
            crossover_total[i] = 0

        for genome_json in genome_directory.glob("*.json"):
            # print(f"\t\tloading json: {genome_json}")
            with open(genome_json, "r") as file:
                genome = json.load(file)

                # track the best genome
                metric_value = genome["fitness"]["test_acc"]
                n_gates = len(genome["gates"])
                n_parameters = sum(
                    [len(gate["parameters"]) for gate in genome["gates"]]
                )

                if metric_value > best_metric:
                    print(
                        f"genome {genome_json} had NEW best metric {metric_value} with n "
                        f"gates {n_gates} and n parameters {n_parameters}"
                    )

                    best_metric = metric_value
                    best_n_gates = n_gates
                    best_n_parameters = n_parameters

                if metric_value == best_metric:
                    if n_gates + n_parameters < best_n_gates + best_n_parameters:
                        print(
                            f"genome {genome_json} had SMALLER best metric {metric_value} with n "
                            f"gates {n_gates} and n parameters {n_parameters}"
                        )
                        best_n_gates = n_gates
                        best_n_parameters = n_parameters

                metadata = genome["metadata"]
                insert_type = metadata["insert_type"]

                generated_by = metadata["generated_by"]

                if "n_ary_crossover" in generated_by:
                    n_parents = len(metadata["parent_genomes"])

                    crossover_total[n_parents] += 1
                    if insert_type == "global_best":
                        crossover_global_count[n_parents] += 1
                    elif insert_type == "inserted":
                        crossover_insert_count[n_parents] += 1

                elif "exponential_crossover" not in generated_by:
                    # this was then a mutation

                    n_mutations = len(generated_by)

                    mutation_total[n_mutations] += 1
                    if insert_type == "global_best":
                        mutation_global_count[n_mutations] += 1
                    elif insert_type == "inserted":
                        mutation_insert_count[n_mutations] += 1

        best_n_gates_list.append(best_n_gates)
        best_n_parameters_list.append(best_n_parameters)
        best_fitness_list.append(best_metric)

        print(f"\t\tmutation global counts: {mutation_global_count}")
        print(f"\t\tmutation insert counts: {mutation_insert_count}")
        print(f"\t\tmutation total: {mutation_total}")
        print()
        print(f"\t\tcrossover global counts: {crossover_global_count}")
        print(f"\t\tcrossover insert counts: {crossover_insert_count}")
        print(f"\t\tcrossover total: {crossover_total}")
        print()

        for i in range(1, 10):
            name = experiment
            if experiment == "wine":
                name = "Wine"
            elif experiment == "breast":
                name = "Breast Cancer"

            if group is not None:
                name += f" {group}"

            # hack to prevent divide by 0
            if mutation_total[i] == 0:
                mutation_total[i] = 1

            mutation_df["Experiment"].append(name + " Global Best")
            mutation_df["N"].append(i)
            global_rate = float(mutation_global_count[i]) / mutation_total[i]
            mutation_df["Rate (%)"].append(global_rate)

            mutation_df["Experiment"].append(name + " Insertion")
            mutation_df["N"].append(i)
            insert_rate = float(mutation_insert_count[i]) / mutation_total[i]
            mutation_df["Rate (%)"].append(insert_rate)
            print(f"\t\tmutation  {experiment} {group} {i} {global_rate} {insert_rate}")

        for i in range(2, 10):
            name = experiment
            if experiment == "wine":
                name = "Wine"
            elif experiment == "breast":
                name = "Breast Cancer"

            if group is not None:
                name += f" {group}"

            # hack to prevent divide by 0
            if crossover_total[i] == 0:
                crossover_total[i] = 1

            crossover_df["Experiment"].append(name + " Global Best")
            crossover_df["N"].append(i)

            global_rate = float(crossover_global_count[i]) / crossover_total[i]
            crossover_df["Rate (%)"].append(global_rate)

            crossover_df["Experiment"].append(name + " Insertion")
            crossover_df["N"].append(i)
            insert_rate = float(crossover_insert_count[i]) / crossover_total[i]
            crossover_df["Rate (%)"].append(insert_rate)
            print(f"\t\tcrossover {experiment} {group} {i} {global_rate} {insert_rate}")

    return best_n_gates_list, best_n_parameters_list, best_fitness_list


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
        "--experiments",
        "-e",
        type=str,
        nargs="+",
        required=True,
        help="Keyword to divide up runs by for table generation",
    )

    p.add_argument(
        "--groups",
        "-g",
        type=str,
        nargs="+",
        required=True,
        help="Keyword to divide up runs by for table generation",
    )

    args = p.parse_args()

    mutation_df = {
        "Experiment": [],
        "N": [],
        "Rate (%)": [],
    }

    crossover_df = {
        "Experiment": [],
        "N": [],
        "Rate (%)": [],
    }

    best_n_gates = []
    best_n_parameters = []
    best_fitness = []

    for experiment in args.experiments:
        for group in args.groups:
            print(f"getting results for experiemnt {experiment} and group {group}:")
            if len(args.groups) == 1:
                best_n_gates_list, best_n_parameters_list, best_fitness_list = (
                    get_group_metrics(
                        mutation_df,
                        crossover_df,
                        args.input_directories,
                        experiment,
                        None,
                    )
                )
            else:
                best_n_gates_list, best_n_parameters_list, best_fitness_list = (
                    get_group_metrics(
                        mutation_df,
                        crossover_df,
                        args.input_directories,
                        experiment,
                        group,
                    )
                )

            print(f"{experiment} {group} best_n_gates: {best_n_gates_list}")
            print(f"{experiment} {group} best_n_parameters: {best_n_parameters_list}")

            best_n_gates.append(best_n_gates_list)
            best_n_parameters.append(best_n_parameters_list)
            best_fitness.append(best_fitness_list)

    print("\n\b")

    print(f"mutation_df: {mutation_df}")
    print(f"crossover_df: {crossover_df}")
    print()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)

    sns.set(rc={"figure.figsize": (12, 8)})

    # sns.set_theme(style="ticks", palette="pastel")
    sns.set_theme(style="ticks", palette="Paired")

    print("df dict:")
    df = pd.DataFrame(mutation_df)
    print(df)

    # Draw a nested boxplot to show bills by day and time
    ax = sns.lineplot(
        x="N", y="Rate (%)", hue="Experiment", data=df  # palette=["m", "g"],
    )

    for cat in df["Experiment"].unique():
        subset = df[df["Experiment"] == cat]

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subset["N"], subset["Rate (%)"]
        )

        sns.regplot(
            data=subset,
            x="N",
            y="Rate (%)",
            scatter=False,
            label=cat + f" Trend (y={slope:.4f}+{intercept:.2f}, R^2={r_value**2:.2f})",
        )

    sns.despine(offset=0, trim=True)
    ax.set_title("Insertion Rates per Number Mutations", fontsize=18)
    ax.legend()
    plt.tight_layout()
    plt.show()

    plt.close()
    print("df dict:")
    df = pd.DataFrame(crossover_df)
    print(df)

    # Draw a nested boxplot to show bills by day and time
    ax = sns.lineplot(
        x="N", y="Rate (%)", hue="Experiment", data=df  # palette=["m", "g"],
    )

    for cat in df["Experiment"].unique():
        subset = df[df["Experiment"] == cat]

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subset["N"], subset["Rate (%)"]
        )

        sns.regplot(
            data=subset,
            x="N",
            y="Rate (%)",
            scatter=False,
            label=cat + f" Trend (y={slope:.4f}+{intercept:.2f}, R^2={r_value**2:.2f})",
        )

    sns.despine(offset=0, trim=True)
    ax.set_title("Crossover Insertion Rates per Number Parents", fontsize=18)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("best n gates:")
    print(best_n_gates)

    print("best n parameters:")
    print(best_n_parameters)

    print("best fitness:")
    print(best_fitness)

    params_df = {
        "Dataset": [],
        "Strategy": [],
        "# Parameters": [],
    }

    gates_df = {
        "Dataset": [],
        "Strategy": [],
        "# Gates": [],
    }

    fitness_df = {
        "Dataset": [],
        "Strategy": [],
        "Test Accuracy (%)": [],
    }

    count = 0
    for experiment in args.experiments:
        for group in args.groups:
            strategy = ""
            if group == "norepop":
                strategy = "Exponential"
            else:
                strategy = "Uniform"

            dataset = ""
            if experiment == "wine":
                dataset = "Wine"
            else:
                dataset = "Breast Cancer"

            gates_list = best_n_gates[count]
            params_list = best_n_parameters[count]
            fitness_list = best_fitness[count]

            for n_gates in gates_list:
                gates_df["Dataset"].append(dataset)
                gates_df["Strategy"].append(strategy)
                gates_df["# Gates"].append(n_gates)

            for n_params in params_list:
                params_df["Dataset"].append(dataset)
                params_df["Strategy"].append(strategy)
                params_df["# Parameters"].append(n_params)

            for fitness in fitness_list:
                fitness_df["Dataset"].append(dataset)
                fitness_df["Strategy"].append(strategy)
                fitness_df["Test Accuracy (%)"].append(fitness)

            count += 1

    df = pd.DataFrame(gates_df)
    print(df)

    # Draw a nested boxplot to show bills by day and time
    ax = sns.boxplot(x="Strategy", y="# Gates", hue="Dataset", data=df)
    sns.despine(offset=0, trim=True)
    ax.set_title("Gate Counts (Uniform vs Exponential)", fontsize=18)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(params_df)
    print(df)

    # Draw a nested boxplot to show bills by day and time
    ax = sns.boxplot(x="Strategy", y="# Parameters", hue="Dataset", data=df)
    sns.despine(offset=0, trim=True)
    ax.set_title("Parameter Counts (Uniform vs Exponential)", fontsize=18)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(fitness_df)
    print(df)

    # Draw a nested boxplot to show bills by day and time
    ax = sns.boxplot(x="Strategy", y="Test Accuracy (%)", hue="Dataset", data=df)
    sns.despine(offset=0, trim=True)
    ax.set_title("Test Accuracy (Uniform vs Exponential)", fontsize=18)
    plt.tight_layout()
    plt.show()
