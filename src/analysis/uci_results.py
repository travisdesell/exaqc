"""Figure 2 of the SHO paper (UCI convergence and evolved learning rate),
straight from the EXAQC run archives.

Islands + SHO uses only the runs of one topology (--islands_sho_topology,
default fully_connected: 10 seeds per dataset). Islands runs from the SHO
ablations (directory names containing sho_training, sho_mutations or
sho_operators) are left out, so Islands + SHO is the full-SHO run set.

Usage (from the repository root):
    python make_figure2.py artifacts/uci_runs artifacts/sho --out_dir plots/sho

Based on uci_results.py; the original description follows.


Reads every run archive (genomes.sqlar) under the given directories, works out
which of the four groups each run belongs to from the settings saved in the
archive, and writes one figure (convergence and evolved learning rate) and the
results table:

    Steady-state         population_strategy steady_state, hyperparameter_strategy fixed
    Steady-state + SHO   population_strategy steady_state, hyperparameter_strategy simplex
    Islands              population_strategy islands,      hyperparameter_strategy fixed
    Islands + SHO        population_strategy islands,      hyperparameter_strategy simplex

Folder and run names do not matter. A run is used if it is a UCI
classification run (iris, seeds, wine, breast_cancer) that evaluated at least
--budget genomes; everything after the first --budget genomes is ignored.
Runs that stopped early or have no genomes are listed and left out.

Island topologies are pooled. By default only topologies that both island
groups ran are kept (so Islands and Islands + SHO cover the same topologies);
--all_topologies keeps every topology.

What is read from each archive:
    run_info 'arguments'   dataset, seed, population and hyperparameter
                           strategy, topology, number of genomes
    genomes table          insertion order, fitness (loss, target_metric),
                           insert_type, n_enabled_gates
    sqlar all_genomes/genome_<N>.json
                           the genome's learning rate (SHO runs only)

Fitness is the genome's loss: the mean of its best training and best
validation loss (lower is better). Accuracy is its target_metric: the mean of
its training and validation mean-class accuracy.

Outputs in --out_dir:
    uci-convergence-hyperparameters.pdf/.png
                                  one 3 x 2 figure: (a)-(d) best fitness so
                                  far per dataset, median and IQR over runs;
                                  (e) spread of log10 learning rate within a
                                  run; (f) each run's final learning rate
    uci-runs.csv                  one row per run: group, dataset, topology,
                                  seed, best fitness and its accuracy, gates
    uci-summary.csv               best and mean +/- std per group and dataset
    uci-table.tex                 the same as LaTeX table rows
    uci-tests.csv                 Vargha-Delaney A12 and Mann-Whitney U with
                                  Holm correction over datasets, per pair of
                                  groups with at least 3 runs each

Extracting is the slow part, so the per-genome records are cached in
<out_dir>/uci-genomes-<budget>.csv.gz; a rerun only reads archives that are not in the
cache yet. Delete the cache (or pass --no_cache) to re-read everything.

Usage (from the repository root):
    python uci_results.py artifacts/uci_runs artifacts/genome_archives/sho --out_dir figures
    python uci_results.py artifacts/uci_runs --budget 2000 --out_dir results/uci

Needs pandas, numpy, scipy, matplotlib and seaborn.
"""

import argparse
import glob
import json
import os
import sqlite3
import zlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

import matplotlib

matplotlib.use("Agg")

DATASETS = [("iris", "Iris"), ("seeds", "Seeds"), ("wine", "Wine"), ("breast_cancer", "Breast Cancer")]
GROUPS = [  # (group, label, dashes)
    ("ss", "Steady-state", (1.2, 1.2)),
    ("ss_sho", "Steady-state + SHO-TO", (4, 1.5)),
    ("islands", "Islands", (6, 1.5, 1.2, 1.5)),
    ("islands_sho", "Islands + SHO-TO", None),
]
LABELS = {g: label for g, label, _ in GROUPS}
LOG_Y = {"iris", "seeds", "wine"}  # Breast Cancer's loss has a floor near 0.34
# at most three y ticks per convergence panel; every panel shares the x ticks 0, 1k, 2k
Y_TICKS = {"iris": [0.03, 0.1, 0.3], "seeds": [0.03, 0.1, 0.3], "wine": [0.01, 0.1],
           "breast_cancer": [0.34, 0.37, 0.40]}
ABLATIONS = ("sho_training", "sho_mutations", "sho_operators")
GENOME_COLUMNS = ["run", "group", "dataset", "topology", "seed", "insertion", "loss",
                  "target_metric", "insert_type", "n_enabled_gates", "learning_rate"]


# ---------------------------------------------------------------------------
# extraction

def find_archives(roots: list) -> list:
    paths = []
    for root in roots:
        paths += glob.glob(os.path.join(root, "**", "genomes.sqlar"), recursive=True)
    return sorted(set(os.path.abspath(p) for p in paths))


def read_archive(path: str, budget: int) -> tuple:
    """Returns (records, None) for a usable run, or (None, reason) otherwise."""
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if any(tag in path for tag in ABLATIONS):
            return None, "SHO ablation run"
        row = c.execute("select value from run_info where key='arguments'").fetchone()
        if row is None:
            return None, "no run_info"
        args = json.loads(row[0])
        dataset = args.get("dataset")
        if dataset not in dict(DATASETS):
            return None, f"not a UCI run ({dataset})"
        population = args.get("population_strategy")
        sho = args.get("hyperparameter_strategy") == "simplex"
        if population == "steady_state":
            group, topology = ("ss_sho" if sho else "ss"), "steady_state"
        elif population == "islands":
            group = "islands_sho" if sho else "islands"
            topology = "_".join(str(t) for t in args.get("topology") or ["unknown"])
        else:
            return None, f"unknown population strategy {population}"

        rows = c.execute(
            "select genome_number, insertion, fitness, insert_type, n_enabled_gates from genomes "
            "where fitness is not null and insertion <= ? order by insertion", (budget,)).fetchall()
        if len(rows) < budget:
            return None, f"only {len(rows)} genomes"

        learning_rates = {}
        if sho:  # fixed runs all use the same learning rate; skip the slow part
            numbers = {g for g, *_ in rows}
            for name, data, size in c.execute(
                    "select name, data, sz from sqlar where name like 'all_genomes/genome_%.json'"):
                g = int(name.rsplit("_", 1)[1].split(".")[0])
                if g in numbers:
                    genome = json.loads(zlib.decompress(data) if len(data) < size else data)
                    learning_rates[g] = genome.get("hyperparameters", {}).get("learning_rate")
        fixed_lr = args.get("learning_rate")

        records = []
        for g, insertion, fitness, insert_type, gates in rows:
            fitness = json.loads(fitness)
            records.append((path, group, dataset, topology, args.get("seed"), insertion,
                            fitness["loss"], fitness.get("target_metric"), insert_type, gates,
                            learning_rates.get(g, fixed_lr)))
        return records, None
    finally:
        c.close()


def extract(roots: list, budget: int, cache: str) -> pd.DataFrame:
    cached = pd.read_csv(cache) if cache and os.path.exists(cache) else pd.DataFrame(columns=GENOME_COLUMNS)
    done = set(cached.run)
    frames = [cached]
    for path in find_archives(roots):
        if path in done:
            continue
        records, reason = read_archive(path, budget)
        if records is None:
            print(f"skipped {path}: {reason}")
            continue
        frame = pd.DataFrame(records, columns=GENOME_COLUMNS)
        frames.append(frame)
        print(f"read {path}")
        if cache:  # save as we go so an interrupted run keeps its progress
            pd.concat(frames, ignore_index=True).to_csv(cache, index=False)
    genomes = pd.concat(frames, ignore_index=True)
    numeric = ["seed", "insertion", "loss", "target_metric", "n_enabled_gates", "learning_rate"]
    genomes[numeric] = genomes[numeric].apply(pd.to_numeric, errors="coerce")
    return genomes[genomes.insertion <= budget]


def match_topologies(genomes: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the island topologies that both island groups ran."""
    islands = genomes[genomes.group.isin(["islands", "islands_sho"])]
    sets = [set(islands[islands.group == g].topology) for g in ("islands", "islands_sho")]
    if not all(sets):
        return genomes
    shared = sets[0] & sets[1]
    dropped = (sets[0] | sets[1]) - shared
    if dropped:
        print("topologies left out (not run by both island groups):", ", ".join(sorted(dropped)))
    return genomes[(genomes.topology == "steady_state") | genomes.topology.isin(shared)]


# ---------------------------------------------------------------------------
# results

def per_run(genomes: pd.DataFrame) -> pd.DataFrame:
    """One row per run: the genome with the lowest fitness and its accuracy."""
    best = genomes.loc[genomes.groupby("run").loss.idxmin()]
    discarded = genomes.groupby("run").insert_type.apply(lambda s: (s == "discarded").mean())
    runs = best[["run", "group", "dataset", "topology", "seed", "loss", "target_metric", "n_enabled_gates"]]
    runs = runs.rename(columns={"loss": "fitness", "target_metric": "accuracy", "n_enabled_gates": "gates"})
    return runs.set_index("run").join(discarded.rename("discarded")).reset_index()


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    return (runs.groupby(["dataset", "group"])
            .agg(runs=("fitness", "size"),
                 fitness_best=("fitness", "min"), fitness_mean=("fitness", "mean"), fitness_std=("fitness", "std"),
                 accuracy_best=("accuracy", "max"), accuracy_mean=("accuracy", "mean"),
                 accuracy_std=("accuracy", "std"),
                 gates_median=("gates", "median"), discarded_mean=("discarded", "mean"))
            .reset_index())


def latex_rows(summary: pd.DataFrame) -> str:
    def sd(value, fmt):
        return "--" if pd.isna(value) else format(value, fmt)
    lines = []
    for dataset, title in DATASETS:
        if not (summary.dataset == dataset).any():
            continue
        lines.append(rf"    \multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        for group, label, _ in GROUPS:
            r = summary[(summary.dataset == dataset) & (summary.group == group)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                rf"    \quad {label} & {r.fitness_best:.3f} & {r.fitness_mean:.3f} $\pm$ {sd(r.fitness_std, '.3f')}"
                rf" & {100 * r.accuracy_best:.1f} & {100 * r.accuracy_mean:.1f} $\pm$ {sd(100 * r.accuracy_std, '.1f')} \\")
        lines.append(r"    \addlinespace")
    return "\n".join(lines[:-1]) + "\n"


def a12(x, y) -> float:
    """Probability that a value from x is lower (better) than one from y."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def tests(runs: pd.DataFrame, min_runs: int = 3) -> pd.DataFrame:
    pairs = [("islands_sho", "islands"), ("ss_sho", "ss"), ("ss_sho", "islands_sho"),
             ("ss_sho", "islands"), ("ss", "islands")]
    rows = []
    for first, second in pairs:
        part = []
        for dataset, _ in DATASETS:
            x = runs[(runs.dataset == dataset) & (runs.group == first)].fitness
            y = runs[(runs.dataset == dataset) & (runs.group == second)].fitness
            if len(x) < min_runs or len(y) < min_runs:
                continue
            part.append([f"{LABELS[first]} vs {LABELS[second]}", dataset, len(x), len(y), a12(x, y),
                         mannwhitneyu(x, y).pvalue])
        if part:
            p = holm(np.array([r[-1] for r in part]))
            rows += [r + [q] for r, q in zip(part, p)]
    return pd.DataFrame(rows, columns=["comparison", "dataset", "n_first", "n_second", "a12", "p", "p_holm"])


# ---------------------------------------------------------------------------
# figures

def colours() -> dict:
    # rocket runs from near-black to near-white; sample its middle so every
    # line is visible on white
    return {g: matplotlib.colormaps["rocket"](x)
            for (g, _, _), x in zip(GROUPS, np.linspace(0.08, 0.78, len(GROUPS)))}


def set_style() -> None:
    sns.set_theme(style="ticks", context="paper", font="serif",
                  rc={"axes.spines.top": False, "axes.spines.right": False})


def save(fig: plt.Figure, out_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def genome_ticks(ax: plt.Axes, budget: int) -> None:
    """Three x ticks on every panel: 0, half the budget and the budget (0, 1k, 2k)."""
    ticks = np.linspace(0, budget, 3)
    ax.set_xticks(ticks, [f"{v / 1000:g}k" if v else "0" for v in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def finish(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="#e6e5e1", lw=0.4)


def draw_convergence(axes, genomes: pd.DataFrame, budget: int) -> None:
    """Best fitness so far, one panel per dataset: median over runs (line) and
    interquartile range (band). A +/- std band would cross zero on the log axes."""
    g = genomes.sort_values(["run", "insertion"]).copy()
    g["genomes"] = g.groupby("run").cumcount() + 1
    g["best"] = g.groupby("run").loss.cummin()
    colour = colours()
    for ax, (dataset, title) in zip(axes, DATASETS):
        data = g[g.dataset == dataset]
        for group, label, dashes in GROUPS:
            stats = data[data.group == group].groupby("genomes").best.quantile([0.25, 0.5, 0.75]).unstack()
            if stats.empty:
                continue
            ax.fill_between(stats.index, stats[0.25], stats[0.75], color=colour[group], alpha=0.15, lw=0)
            line, = ax.plot(stats.index, stats[0.5], color=colour[group], lw=1.2, label=label)
            if dashes:
                line.set_dashes(dashes)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(0, budget)
        genome_ticks(ax, budget)
        if data.empty:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        elif dataset in LOG_Y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.335, 0.40)
        if not data.empty:
            ax.set_yticks(Y_TICKS[dataset], [f"{v:g}" for v in Y_TICKS[dataset]])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        finish(ax)


def draw_hyperparameters(ax_a: plt.Axes, ax_b: plt.Axes, genomes: pd.DataFrame, budget: int) -> None:
    """(e) standard deviation of log10 learning rate over each window of 100
    genomes, median over runs and datasets; (f) each run's median learning rate
    over its last 10% of genomes, by dataset, with the median over runs as a bar.
    Only the SHO groups evolve the learning rate."""
    g = genomes[genomes.group.isin(["ss_sho", "islands_sho"])].copy()
    ax_a.set_title(r"(e) Spread of $\eta$ in a run", fontsize=8.5)
    ax_b.set_title(r"(f) Final $\eta$ per run", fontsize=8.5)
    if g.empty:
        print("no SHO runs: hyperparameter panels left empty")
        return
    g["loglr"] = np.log10(g.learning_rate.astype(float))
    g["window"] = (g.insertion - 1) // 100
    spread = (g.groupby(["group", "run", "window"]).loglr.std()
              .groupby(["group", "window"]).median().unstack(0))
    final = g[g.insertion > 0.9 * budget].groupby(["group", "dataset", "run"]).loglr.median().reset_index()
    colour = colours()
    groups = [x for x in GROUPS if x[0] in spread.columns]

    for group, label, dashes in groups:
        line, = ax_a.plot((spread.index + 1) * 100, spread[group], color=colour[group], lw=1.2, label=label)
        if dashes:
            line.set_dashes(dashes)
    ax_a.set_xlim(0, budget)
    genome_ticks(ax_a, budget)
    ax_a.set_ylim(bottom=0)
    ax_a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax_a.set_xlabel("Genomes evaluated", fontsize=7.5)
    ax_a.set_ylabel(r"SD of $\log_{10}\eta$", fontsize=7.5)

    offsets = dict(zip([x[0] for x in groups], np.linspace(-0.18, 0.18, len(groups)) if len(groups) > 1 else [0]))
    rng = np.random.default_rng(0)
    for i, (dataset, _) in enumerate(DATASETS):
        for group, _, _ in groups:
            v = final[(final.dataset == dataset) & (final.group == group)].loglr.to_numpy()
            if not len(v):
                continue
            x = i + offsets[group] + rng.uniform(-0.07, 0.07, len(v))
            ax_b.scatter(x, 10 ** v, s=4, color=colour[group], alpha=0.6, lw=0)
            ax_b.plot([i + offsets[group] - 0.12, i + offsets[group] + 0.12], [10 ** np.median(v)] * 2,
                      color="black", lw=1.2)
    ax_b.set_yscale("log")
    ax_b.set_yticks([3e-3, 1e-2, 3e-2], ["0.003", "0.01", "0.03"])
    ax_b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_b.set_xticks(range(len(DATASETS)), ["Iris", "Seeds", "Wine", "BC"])
    ax_b.set_xlabel("Dataset", fontsize=7.5)
    ax_b.set_ylabel(r"Learning rate $\eta$", fontsize=7.5)
    for ax in (ax_a, ax_b):
        finish(ax)


def plot_figure(genomes: pd.DataFrame, budget: int, out_dir: str) -> None:
    """One 3 x 2 figure for a single-column manuscript (LNCS text width 12.2 cm):
    rows 1-2 are the convergence panels (a)-(d), sharing one x and one y label;
    row 3 is the evolved learning rate (e, f). All six panels are the same size."""
    fig = plt.figure(figsize=(4.8, 5.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[2, 1.08], hspace=0.04)
    conv = top.subplots(2, 2, sharex=True).ravel()
    draw_convergence(conv, genomes, budget)
    for ax, (_, title), letter in zip(conv, DATASETS, "abcd"):
        ax.set_title(f"({letter}) {title}", fontsize=8.5)
    top.supxlabel("Genomes evaluated", fontsize=7.5)
    top.supylabel("Best fitness so far", fontsize=7.5)
    ax_a, ax_b = bottom.subplots(1, 2)
    draw_hyperparameters(ax_a, ax_b, genomes, budget)

    handles, names = [], []
    for ax in conv:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    order = [label for _, label, _ in GROUPS]
    pairs = sorted(zip(handles, names), key=lambda p: order.index(p[1]))
    fig.legend([h for h, _ in pairs], [n for _, n in pairs], loc="outside upper center", ncol=len(pairs),
               frameon=False, fontsize=7, handlelength=2.5, columnspacing=1.2)
    save(fig, out_dir, "uci-convergence-hyperparameters")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UCI results (plots and table) from EXAQC run archives.")
    parser.add_argument("archives", nargs="+", help="directories searched (recursively) for genomes.sqlar")
    parser.add_argument("--out_dir", default="uci_results")
    parser.add_argument("--budget", type=int, default=2_000, help="genomes per run to use [2000]")
    parser.add_argument("--all_topologies", action="store_true",
                        help="keep island topologies that only one island group ran")
    parser.add_argument("--no_cache", action="store_true", help="re-read every archive")
    parser.add_argument("--islands_sho_topology", default="fully_connected",
                        help="topology whose runs are used for Islands + SHO ['fully_connected'; 'all' keeps every topology]")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache = None if args.no_cache else os.path.join(args.out_dir, f"uci-genomes-{args.budget}.csv.gz")
    genomes = extract(args.archives, args.budget, cache)
    if genomes.empty:
        raise SystemExit("no usable UCI runs found")
    if not args.all_topologies:
        genomes = match_topologies(genomes)
    if args.islands_sho_topology != "all":
        keep = (genomes.group != "islands_sho") | (genomes.topology == args.islands_sho_topology)
        genomes = genomes[keep]

    runs = per_run(genomes)
    summary = summarize(runs)
    print("\nruns per dataset and group:")
    print(runs.groupby(["dataset", "group"]).size().unstack().fillna(0).astype(int).to_string())
    runs.to_csv(os.path.join(args.out_dir, "uci-runs.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "uci-summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "uci-table.tex"), "w") as f:
        f.write(latex_rows(summary))
    t = tests(runs)
    t.to_csv(os.path.join(args.out_dir, "uci-tests.csv"), index=False)
    print("\n" + t.round(4).to_string(index=False))

    set_style()
    plot_figure(genomes, args.budget, args.out_dir)
    print(f"\nwrote results to {args.out_dir}/")


if __name__ == "__main__":
    main()
