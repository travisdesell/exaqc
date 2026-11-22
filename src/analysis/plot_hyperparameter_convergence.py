"""How every SHO-tuned hyperparameter evolves during the search, one figure
for the UCI tasks and one for the RL environments.

One panel per hyperparameter. Each line is the mean over runs of each run's
mean value among the genomes evaluated in each window of 4% of the budget (80
genomes on UCI, 400 on RL), and the band is +/- one standard deviation over
runs. The learning rate is averaged in log10 space (geometric mean), so its
band stays positive on the log axis. The dotted grey line is EXAQC's
default where it lies in the range.

hyperparameter-convergence-uci   training settings (learning rate, Adam), the
                                 ones both groups tune: one column per
                                 dataset, one line per group, Steady-state +
                                 SHO (solid, 5 runs per task) and Islands +
                                 SHO with the fully connected topology
                                 (dotted, 10 runs per task).
hyperparameter-convergence-uci-steady-state
                                 Steady-state + SHO alone: training settings,
                                 2 x 2 panels, colour = dataset (5 runs each).
hyperparameter-convergence-uci-islands
                                 Islands + SHO alone (all topologies pooled,
                                 49-50 runs per task): all 16 tuned settings,
                                 4 x 4 panels, colour = dataset.
hyperparameter-convergence-uci-operators
                                 operator settings, tuned by Islands + SHO
                                 only (all topologies pooled): colour = dataset.
                                 UCI: 2,000 genomes; runs that stopped earlier
                                 are left out.
hyperparameter-convergence-rl    colour = environment; the sho02 batch
                                 (Islands + SHO), up to 10,000 genomes.

Panels:
    training     learning rate, Adam beta1, beta2, epsilon
    RL only      discount gamma, entropy coefficient, GAE lambda, PPO clip
    operators    binary / n-ary / exponential crossover rate, mutation count
    mutations    each mutation's share of the total mutation weight

Inputs (plain or gzipped) in the data directory:
    uci_runs.csv       Steady-state + SHO (written by extract_uci_runs.py)
    sho_<dataset>.csv  Islands + SHO on UCI (written by extract_uci.py)
    rl.csv             RL runs (written by extract_rl.py)

Usage:
    python plot_hyperparameter_convergence.py artifacts/sho_paper_analysis figures
    python plot_hyperparameter_convergence.py artifacts/sho_paper_analysis figures --median
The second form draws the median over runs with an interquartile band instead
of the mean +/- std, and adds "-median" to every file name.
writes figures/hyperparameter-convergence-uci.pdf/.png,
figures/hyperparameter-convergence-uci-operators.pdf/.png and
figures/hyperparameter-convergence-rl.pdf/.png
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import matplotlib

matplotlib.use("Agg")

WINDOWS = 25  # windows of 4% of the budget
STAT = "mean"  # "mean" (band +/- std) or "median" (band IQR); set by --median
SUFFIX = {"mean": "", "median": "-median"}
UCI_BUDGET, RL_BUDGET = 2_000, 10_000
UCI = [("iris", "Iris"), ("seeds", "Seeds"), ("wine", "Wine"), ("breast_cancer", "Breast Cancer")]
RL = [("acrobot", "Acrobot"), ("cartpole", "CartPole"), ("frozenlake", "FrozenLake"),
      ("mountaincar_continuous", "MountainCarCont."), ("walker2d", "Walker2d")]
UCI_GROUPS = [("ss_sho", "Steady-state + SHO", "-"), ("islands_sho", "Islands + SHO (fully connected)", (0, (1, 1.2)))]
COMPARISON_TOPOLOGY = "fully_connected"  # the islands compared with the steady state
MUTATIONS = ["add_gate", "reorder_gate", "qubit_swap", "enable_gate", "disable_gate", "clone",
             "mutate_some_weights", "mutate_all_weights"]
MUTATION_TITLES = ["Add gate", "Reorder gate", "Qubit swap", "Enable gate", "Disable gate", "Clone",
                   "Mutate some weights", "Mutate all weights"]
# defaults from MUTATION_WEIGHTS in src/evolution/exaqc.py
DEFAULT_WEIGHTS = dict(zip(MUTATIONS, [11, 2, 2, 1, 2, 2, 2, 2]))

TRAINING_PANELS = [  # (column, title, log y, default)
    ("learning_rate", r"Learning rate $\eta$", True, None),
    ("adam_beta1", r"Adam $\beta_1$", False, None),
    ("adam_beta2", r"Adam $\beta_2$", False, None),
    ("adam_epsilon", r"Adam $\epsilon$ ($\times10^{-9}$)", False, None),
]
RL_PANELS = [
    ("gamma", r"Discount $\gamma$", False, None),
    ("entropy_coef", "Entropy coef.", False, None),  # linear: SHO can set it to 0
    ("gae_lambda", r"GAE $\lambda$", False, None),
    ("ppo_clip", "PPO clip", False, None),
]
OPERATOR_PANELS = [
    ("binary_crossover_rate", "Binary crossover", False, 0.1),
    ("n_ary_crossover_rate", r"$n$-ary crossover", False, 0.1),
    ("exponential_crossover_rate", "Exp. crossover", False, 0.1),
    ("mutation_count", "Mutation count", False, None),
] + [(f"{m}_share", title, False, DEFAULT_WEIGHTS[m] / sum(DEFAULT_WEIGHTS.values()))
     for m, title in zip(MUTATIONS, MUTATION_TITLES)]
HYPERPARAMETERS = ([c for c, *_ in TRAINING_PANELS + RL_PANELS + OPERATOR_PANELS[:4]]
                   + [f"{m}_weight" for m in MUTATIONS])


def read_csv(directory: str, name: str, columns: list) -> pd.DataFrame:
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        path += ".gz"
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, usecols=[c for c in columns if c in header])


def prepare(g: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Keeps the first `budget` genomes, adds mutation shares and the window."""
    g = g.drop_duplicates(["run", "insertion"])
    g = g[g.insertion <= budget].copy()
    weights = [f"{m}_weight" for m in MUTATIONS]
    if all(w in g for w in weights):
        total = g[weights].sum(axis=1)
        for m in MUTATIONS:
            g[f"{m}_share"] = g[f"{m}_weight"] / total
    g["window"] = np.minimum((g.insertion - 1) * WINDOWS // budget, WINDOWS - 1)
    return g


def load_uci(directory: str) -> pd.DataFrame:
    ss = read_csv(directory, "uci_runs.csv", ["run", "dataset", "mode", "topology", "insertion"] + HYPERPARAMETERS)
    ss = ss[(ss["mode"] == "sho") & (ss.topology == "steady_state")].assign(group="ss_sho")
    frames = [ss]
    for dataset, _ in UCI:
        d = read_csv(directory, f"sho_{dataset}.csv", ["dataset", "topology", "seed", "insertion"] + HYPERPARAMETERS)
        d["run"] = dataset + "_" + d.topology + "_" + d.seed.astype(str)
        frames.append(d.assign(group="islands_sho"))
    g = pd.concat(frames, ignore_index=True).rename(columns={"dataset": "benchmark"})
    g = g[g.groupby("run").insertion.transform("max") >= UCI_BUDGET]  # complete runs only
    return prepare(g, UCI_BUDGET)


def load_rl(directory: str) -> pd.DataFrame:
    r = read_csv(directory, "rl.csv", ["run", "env", "tag", "insertion"] + HYPERPARAMETERS)
    r = r[r.tag == "sho02"]  # the matched batch (two older CartPole runs left out)
    r = r.rename(columns={"env": "benchmark"}).assign(group="islands_sho")
    return prepare(r, RL_BUDGET)


def summary(g: pd.DataFrame, column: str, log: bool) -> pd.DataFrame:
    """Per (benchmark, group, window): columns mean, low, high for the line and
    band. With STAT "mean": mean over runs of each run's per-window mean, band
    mean +/- std; log-scale settings are averaged in log10 space (geometric
    mean), so the band stays positive. With STAT "median": median over runs of
    each run's per-window median (mean for the integer mutation count), band
    the interquartile range."""
    if column not in g:
        return pd.DataFrame(columns=["mean", "low", "high"])
    v = g.dropna(subset=[column])
    scale = 1e9 if column == "adam_epsilon" else 1  # epsilon is plotted in units of 1e-9
    if STAT == "median":
        windows = (v[column] * scale).groupby([v.benchmark, v.group, v.run, v.window])
        per_run = windows.mean() if column == "mutation_count" else windows.median()
        q = per_run.groupby(level=["benchmark", "group", "window"]).quantile([0.25, 0.5, 0.75]).unstack()
        return pd.DataFrame({"mean": q[0.5], "low": q[0.25], "high": q[0.75]})
    values = np.log10(v[column]) if log else v[column] * scale
    per_run = values.groupby([v.benchmark, v.group, v.run, v.window]).mean()
    stats = per_run.groupby(level=["benchmark", "group", "window"]).agg(["mean", "std"]).fillna({"std": 0})
    out = pd.DataFrame({"mean": stats["mean"], "low": stats["mean"] - stats["std"],
                        "high": stats["mean"] + stats["std"]})
    return 10 ** out if log else out


def plot_line(ax: plt.Axes, s: pd.DataFrame, x: np.ndarray, colour, ls, label=None, band_alpha=0.15,
              lw: float = 1.0, band_edges: bool = False) -> None:
    ax.fill_between(x[s.index], s["low"], s["high"], color=colour, alpha=band_alpha, lw=0)
    if band_edges:  # thin edges in the line's style, so overlapping bands stay apart
        for edge in ("low", "high"):
            ax.plot(x[s.index], s[edge], color=colour, lw=0.5, ls=ls, alpha=0.6)
    ax.plot(x[s.index], s["mean"], color=colour, lw=lw, ls=ls, label=label)


def style_axis(ax: plt.Axes, log: bool, default) -> None:
    if default is not None:
        ax.axhline(default, color="#8a8984", lw=0.6, ls=":")
    if log:
        ax.set_yscale("log")
        low, high = ax.get_ylim()
        subs = (1,) if high / low > 30 else (1, 2, 5)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=subs))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100], ["0", "50", "100"])
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.grid(axis="y", color="#e6e5e1", lw=0.4)


def draw(g: pd.DataFrame, panels: list, benchmarks: list, groups: list, budget: int,
         out_dir: str, name: str, ncols: int = 4, panel_height: float = 1.3) -> None:
    """One panel per hyperparameter in a grid `ncols` wide (single-column
    manuscript, LNCS text width 12.2 cm); colour = benchmark, style = group."""
    rows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(4.8, panel_height * rows + 0.4), sharex=True, squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.0)  # wider panels, so the titles fit
    colour = dict(zip([b for b, _ in benchmarks],
                      [matplotlib.colormaps["rocket"](x) for x in np.linspace(0.12, 0.78, len(benchmarks))]))
    x = (np.arange(WINDOWS) + 0.5) / WINDOWS * 100
    for ax, (column, title, log, default) in zip(axes.ravel(), panels):
        stats = summary(g, column, log)
        for group, _, ls in groups:
            for b, _ in benchmarks:
                if (b, group) in stats.index.droplevel("window"):
                    plot_line(ax, stats.loc[(b, group)], x, colour[b], ls, band_alpha=0.1)
        ax.set_title(title, fontsize=6.5 if ncols > 2 else 8)
        style_axis(ax, log, default)
        if ncols <= 2:
            ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.supxlabel(f"Genomes evaluated (% of {budget:,})", fontsize=7)
    fig.supylabel(f"{STAT.capitalize()} value over runs", fontsize=7)

    handles = [Line2D([], [], color=colour[b], lw=1.2) for b, _ in benchmarks]
    labels = [label for _, label in benchmarks]
    if len(groups) > 1:
        handles += [Line2D([], [], color="#4a4a4a", lw=1.0, ls=ls) for _, _, ls in groups]
        labels += [label for _, label, _ in groups]
    fig.legend(handles, labels, loc="outside upper center", ncol=min(len(labels), 3 if len(groups) > 1 else 5),
               frameon=False, fontsize=6, handlelength=2.2, columnspacing=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}{SUFFIX[STAT]}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out_dir, name + SUFFIX[STAT] + ".pdf"))


def draw_uci_by_dataset(g: pd.DataFrame, out_dir: str, name: str) -> None:
    """Training settings on UCI: one column per dataset, one row per setting,
    one line per group (line and band as set by STAT: mean +/- std or median + IQR).
    These are the settings both SHO groups tune."""
    # the two ends of the rocket range used elsewhere, for the most contrast
    colour = {"ss_sho": matplotlib.colormaps["rocket"](0.25), "islands_sho": matplotlib.colormaps["rocket"](0.7)}
    x = (np.arange(WINDOWS) + 0.5) / WINDOWS * 100
    fig, axes = plt.subplots(len(TRAINING_PANELS), len(UCI), figsize=(4.8, 4.4), sharex=True, sharey="row",
                             layout="constrained")
    for row, (column, title, log, default) in enumerate(TRAINING_PANELS):
        stats = summary(g, column, log)
        for col, (dataset, dataset_title) in enumerate(UCI):
            ax = axes[row, col]
            for group, label, ls in UCI_GROUPS:
                if (dataset, group) not in stats.index.droplevel("window"):
                    continue
                plot_line(ax, stats.loc[(dataset, group)], x, colour[group], ls, label,
                          band_alpha=0.08, lw=1.4, band_edges=True)
            style_axis(ax, log, default)
            if row == 0:
                ax.set_title(dataset_title, fontsize=7.5)
            if col == 0:
                ax.set_ylabel(title, fontsize=6.5)
    fig.supxlabel(f"Genomes evaluated (% of {UCI_BUDGET:,})", fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, fontsize=6.5,
               handlelength=2.5, columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}{SUFFIX[STAT]}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out_dir, name + SUFFIX[STAT] + ".pdf"))


def main(directory: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font="serif",
                  rc={"axes.spines.top": False, "axes.spines.right": False})

    uci = load_uci(directory)
    print("UCI runs:\n", uci.groupby(["benchmark", "group"]).run.nunique().unstack().to_string())
    comparison = uci[(uci.group == "ss_sho") | (uci.topology == COMPARISON_TOPOLOGY)]
    print("compared:\n", comparison.groupby(["benchmark", "group"]).run.nunique().unstack().to_string())
    draw_uci_by_dataset(comparison, out_dir, "hyperparameter-convergence-uci")
    # Steady-state + SHO on its own: training settings, one line per dataset
    draw(uci[uci.group == "ss_sho"], TRAINING_PANELS, UCI, [("ss_sho", "Steady-state + SHO", "-")], UCI_BUDGET,
         out_dir, "hyperparameter-convergence-uci-steady-state", ncols=2, panel_height=1.6)
    # operator settings are tuned by Islands + SHO only: one line per dataset
    draw(uci[uci.group == "islands_sho"], OPERATOR_PANELS, UCI, [("islands_sho", "Islands + SHO", "-")], UCI_BUDGET, out_dir,
         "hyperparameter-convergence-uci-operators")
    # Islands + SHO on its own: every tuned setting, one line per dataset
    draw(uci[uci.group == "islands_sho"], TRAINING_PANELS + OPERATOR_PANELS, UCI,
         [("islands_sho", "Islands + SHO", "-")], UCI_BUDGET, out_dir, "hyperparameter-convergence-uci-islands")

    rl = load_rl(directory)
    print("RL runs:\n", rl.groupby("benchmark").run.nunique().to_string())
    draw(rl, TRAINING_PANELS + RL_PANELS + OPERATOR_PANELS, RL, [("islands_sho", "Islands + SHO", "-")],
         RL_BUDGET, out_dir, "hyperparameter-convergence-rl")


if __name__ == "__main__":
    if "--median" in sys.argv:
        sys.argv.remove("--median")
        STAT = "median"
    main(sys.argv[1], sys.argv[2])
