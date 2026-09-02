from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome

#: Line color for the loss series (shared by classification and RL plots).
_LOSS_COLOR = "#d62728"

#: Line color for the reinforcement-learning return series.
_RETURN_COLOR = "#1f77b4"

#: Line color for the classification mean-class-accuracy series.
_ACCURACY_COLOR = "#2ca02c"

#: Line color for the reinforcement-learning best-episode-return series.
_BEST_RETURN_COLOR = "#ff7f0e"

#: Line color for the reinforcement-learning evaluation return-std series.
_RETURN_STD_COLOR = "#9467bd"

#: Line colors cycled across the metrics of a multi-metric plot. The same
#: metric keeps the same color in the training and validation panels, so the
#: two can be read against each other.
_METRIC_COLORS = (
    "#d62728",  # loss is listed first, and keeps the shared loss color
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#17becf",
)

#: Metric entries that describe the record rather than a measured quantity, and
#: so are never plotted.
_NON_METRIC_KEYS = frozenset({"epoch"})


def save_training_plot(
    out_dir: str,
    genome: CircuitGenome,
    filename: str,
) -> None:
    """Saves a per-epoch/episode training line plot for a trained genome.

    Which plot is drawn depends on the genome's task:

    * **reinforcement learning** -- return and loss per episode on twin y-axes,
      beside an evaluation panel when evaluation metrics were recorded;
    * **classification** -- loss and mean class accuracy per epoch, as
      side-by-side training and validation twin-axis panels;
    * **anything else that trains per epoch** (quantum teacher imitation) --
      every recorded metric gets its own panel, with the training and validation
      curves drawn together. Those tasks report several measures at once
      (fidelity, Bures angle, KL divergence, mean squared error), so nothing is
      hard-coded and none of them is dropped.

    The genome's own ``task`` selects the plot when it is set; otherwise the
    shape of the recorded metrics is used, so a genome saved before tasks were
    recorded still plots.

    Args:
        out_dir: Directory to write the plot into.
        genome: The trained genome whose metadata holds the training history.
        filename: File name (within ``out_dir``) for the saved PNG.

    Returns:
        None. Writes the plot on success; logs a warning and returns without
        writing when no recognized metrics are present or plotting fails (the
        plot is best-effort and must never abort saving).
    """
    metadata = getattr(genome, "metadata", {}) or {}
    genome_number = getattr(genome, "genome_number", "?")
    task = getattr(genome, "task", None)

    plot = _PLOTS_BY_TASK.get(task) or _plot_for_metrics(metadata)

    if plot is None:
        logger.warning(
            "No training metrics found for genome {}; skipping training plot.",
            genome_number,
        )
        return

    try:
        plot(out_dir, genome, filename, metadata)
    except Exception as error:
        logger.warning(
            "Could not save training plot for genome {}: {}",
            genome_number,
            error,
        )


def _plot_rl_training_panel(
    return_axis: Axes,
    episode_metrics: list[dict[str, Any]],
) -> None:
    """Draws return-per-episode and loss-per-episode on one twin-axis panel.

    Args:
        return_axis: The panel's primary (left) axes; return is drawn here and a
            twin right axis is created for the loss.
        episode_metrics: The per-episode metric dicts (``return`` and ``loss``);
            may be empty.
    """
    loss_axis = return_axis.twinx()

    return_axis.set_xlabel("Episode")
    return_axis.set_ylabel("Return", color=_RETURN_COLOR)
    loss_axis.set_ylabel("Loss", color=_LOSS_COLOR)
    return_axis.tick_params(axis="y", colors=_RETURN_COLOR)
    loss_axis.tick_params(axis="y", colors=_LOSS_COLOR)
    return_axis.set_title("Training")
    return_axis.grid(True, alpha=0.3)

    if not episode_metrics:
        return

    episodes = [
        metric.get("episode", index) for index, metric in enumerate(episode_metrics)
    ]
    returns = [metric.get("return", float("nan")) for metric in episode_metrics]
    losses = [metric.get("loss", float("nan")) for metric in episode_metrics]

    (return_line,) = return_axis.plot(
        episodes, returns, color=_RETURN_COLOR, label="Return"
    )
    (loss_line,) = loss_axis.plot(episodes, losses, color=_LOSS_COLOR, label="Loss")
    return_axis.legend(handles=[return_line, loss_line], loc="best", fontsize=9)


def _plot_rl_evaluation_panel(
    return_axis: Axes,
    evaluation_metrics: list[dict[str, Any]],
) -> None:
    """Draws evaluation return mean/best and return std on one twin-axis panel.

    The mean and best episode return (both return quantities) share the left
    y-axis; the return standard deviation uses the right y-axis. Each evaluation
    is placed at the training episode it was generated from.

    Args:
        return_axis: The panel's primary (left) axes for the return series; a
            twin right axis is created for the return std.
        evaluation_metrics: The per-evaluation metric dicts (``episode``,
            ``return_mean``, ``best_episode_return``, ``return_std``); may be
            empty.
    """
    std_axis = return_axis.twinx()

    return_axis.set_xlabel("Episode")
    return_axis.set_ylabel("Return", color=_RETURN_COLOR)
    std_axis.set_ylabel("Return Std", color=_RETURN_STD_COLOR)
    return_axis.tick_params(axis="y", colors=_RETURN_COLOR)
    std_axis.tick_params(axis="y", colors=_RETURN_STD_COLOR)
    return_axis.set_title("Evaluation")
    return_axis.grid(True, alpha=0.3)

    if not evaluation_metrics:
        return

    evaluations = [
        metric.get("episode", index) for index, metric in enumerate(evaluation_metrics)
    ]
    return_mean = [
        metric.get("return_mean", float("nan")) for metric in evaluation_metrics
    ]
    best_return = [
        metric.get("best_episode_return", float("nan")) for metric in evaluation_metrics
    ]
    return_std = [
        metric.get("return_std", float("nan")) for metric in evaluation_metrics
    ]

    (mean_line,) = return_axis.plot(
        evaluations, return_mean, color=_RETURN_COLOR, label="Return Mean"
    )
    (best_line,) = return_axis.plot(
        evaluations,
        best_return,
        color=_BEST_RETURN_COLOR,
        label="Best Episode Return",
    )
    (std_line,) = std_axis.plot(
        evaluations, return_std, color=_RETURN_STD_COLOR, label="Return Std"
    )
    return_axis.legend(handles=[mean_line, best_line, std_line], loc="best", fontsize=9)


def _plot_reinforcement_learning(
    out_dir: str,
    genome: CircuitGenome,
    filename: str,
    metadata: dict[str, Any],
) -> None:
    """Draws the RL training (and, when present, evaluation) panels.

    The training panel shows return and loss per episode. When
    ``evaluation_episode_metrics`` is present, a second panel is drawn beside it
    showing the periodic evaluation return mean, best episode return, and return
    std.

    Args:
        out_dir: Directory to write the plot into.
        genome: The genome being plotted (used for the title).
        filename: Output file name within ``out_dir``.
        metadata: The genome metadata containing ``training_episode_metrics``
            and (optionally) ``evaluation_episode_metrics``.
    """
    episode_metrics = metadata.get("training_episode_metrics", [])
    evaluation_metrics = metadata.get("evaluation_episode_metrics", [])
    if not episode_metrics and not evaluation_metrics:
        return

    if evaluation_metrics:
        figure, (training_axis, evaluation_axis) = plt.subplots(1, 2, figsize=(14, 5))
        _plot_rl_training_panel(training_axis, episode_metrics)
        _plot_rl_evaluation_panel(evaluation_axis, evaluation_metrics)
    else:
        figure, training_axis = plt.subplots(figsize=(8, 5))
        _plot_rl_training_panel(training_axis, episode_metrics)

    figure.suptitle(f"Genome {genome.genome_number} Training", fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close(figure)


def _plot_for_metrics(metadata: dict[str, Any]) -> Callable[..., None] | None:
    """Chooses a plot from the shape of the recorded metrics.

    Only needed for a genome that records no task -- one saved before the task
    was stamped onto genomes, or built by hand. A run with per-episode records
    is reinforcement learning; per-epoch records with a mean class accuracy are
    classification; anything else that trained per epoch reports a set of
    measures and gets them all plotted.

    Args:
        metadata: The genome's metadata.

    Returns:
        The plotting function to use, or ``None`` when nothing was recorded.
    """

    if metadata.get("training_episode_metrics") is not None:
        return _plot_reinforcement_learning

    if metadata.get("training_epoch_metrics") is None:
        return None

    for key in ("training_epoch_metrics", "validation_epoch_metrics"):
        for record in metadata.get(key) or []:
            if "mean_class_accuracy" in record:
                return _plot_classification

    return _plot_all_metrics


def _metric_value(record: dict[str, Any], name: str) -> float:
    """Reads one metric out of an epoch record.

    Metrics are recorded either as a bare number (e.g. ``loss``) or as a dict
    carrying a ``"mean"`` entry (the shape every
    :class:`~src.metrics.metric.Metric` reports), so both are unwrapped here.

    Args:
        record: One epoch's metric record.
        name: The metric to read.

    Returns:
        The metric's value, or NaN when it is missing or not a number.
    """

    value = record.get(name)

    if isinstance(value, dict):
        value = value.get("mean")

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_names(*records: list[dict[str, Any]]) -> list[str]:
    """Finds every plottable metric across one or more sets of epoch records.

    Metrics are discovered from the records rather than hard-coded, so a task
    that reports a different set (or gains a new measure) is plotted in full
    without this module needing to know about it. ``loss`` is placed first when
    present, since it is the quantity being optimized.

    Args:
        *records: Lists of per-epoch metric records to scan.

    Returns:
        The metric names to plot, in a stable order.
    """

    names: list[str] = []
    for record_list in records:
        for record in record_list:
            for key, value in record.items():
                if key in _NON_METRIC_KEYS or key in names:
                    continue
                # keep only entries that resolve to a number
                if isinstance(value, (int, float)) or (
                    isinstance(value, dict) and "mean" in value
                ):
                    names.append(key)

    if "loss" in names:
        names.remove("loss")
        names.insert(0, "loss")

    return names


def _plot_metrics_panel(
    axes: Axes,
    title: str,
    records: list[dict[str, Any]],
    names: list[str],
) -> None:
    """Draws every metric of one split (training or validation) on one panel.

    All measures share the panel's y-axis, so they are directly comparable
    against each other over the run. That reads well while the measures occupy
    similar ranges; a metric whose scale dwarfs the others (an early KL
    divergence, say) will flatten the rest.

    Args:
        axes: The panel to draw into.
        title: Panel title, naming the split.
        records: That split's per-epoch metric records; may be empty.
        names: The metrics to draw, in order.

    Returns:
        None. Draws onto ``axes``.
    """

    axes.set_title(title)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Metric value")
    axes.grid(True, alpha=0.3)

    if not records:
        return

    epochs = [record.get("epoch", index) for index, record in enumerate(records)]

    for index, name in enumerate(names):
        axes.plot(
            epochs,
            [_metric_value(record, name) for record in records],
            color=_METRIC_COLORS[index % len(_METRIC_COLORS)],
            label=name,
        )


def _plot_all_metrics(
    out_dir: str,
    genome: CircuitGenome,
    filename: str,
    metadata: dict[str, Any],
) -> None:
    """Draws every recorded metric, as side-by-side training and validation panels.

    Used for tasks that report several measures per epoch -- quantum teacher
    imitation reports fidelity, Bures angle, KL divergence and mean squared
    error alongside the loss -- so the plot shows all of them rather than a
    single hard-coded pair. The two splits are drawn as separate panels sharing
    one y-scale, so training and validation can be read against each other the
    same way the classification plot presents them.

    Args:
        out_dir: Directory to write the plot into.
        genome: The genome being plotted (used for the title).
        filename: Output file name within ``out_dir``.
        metadata: The genome metadata holding the per-epoch records.

    Returns:
        None. Writes the plot, or returns without writing when there is nothing
        to draw.
    """

    training = metadata.get("training_epoch_metrics", [])
    validation = metadata.get("validation_epoch_metrics", [])
    if not training and not validation:
        return

    names = _metric_names(training, validation)
    if not names:
        return

    # sharey so the two panels are directly comparable rather than each being
    # autoscaled to its own split
    figure, (training_axes, validation_axes) = plt.subplots(
        1, 2, figsize=(14, 5), sharey=True
    )
    _plot_metrics_panel(training_axes, "Training", training, names)
    _plot_metrics_panel(validation_axes, "Validation", validation, names)

    # one shared legend rather than repeating it on both panels
    handles, labels = training_axes.get_legend_handles_labels()
    if not handles:
        handles, labels = validation_axes.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper right", fontsize=9)

    figure.suptitle(f"Genome {genome.genome_number} Training", fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close(figure)


def _plot_epoch_panel(
    loss_axis: Axes,
    metrics: list[dict[str, Any]],
    title: str,
) -> None:
    """Draws loss and mean class accuracy per epoch on one twin-axis panel.

    Args:
        loss_axis: The panel's primary (left) axes; loss is drawn here and a
            twin right axis is created for the accuracy.
        metrics: The per-epoch metric dicts (``loss`` and
            ``mean_class_accuracy.mean``) to plot; may be empty.
        title: The panel title.
    """
    accuracy_axis = loss_axis.twinx()

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss", color=_LOSS_COLOR)
    accuracy_axis.set_ylabel("Mean Class Accuracy", color=_ACCURACY_COLOR)
    loss_axis.tick_params(axis="y", colors=_LOSS_COLOR)
    accuracy_axis.tick_params(axis="y", colors=_ACCURACY_COLOR)
    loss_axis.set_title(title)
    loss_axis.grid(True, alpha=0.3)

    if not metrics:
        return

    epochs = [metric.get("epoch", index) for index, metric in enumerate(metrics)]
    losses = [metric.get("loss", float("nan")) for metric in metrics]
    accuracies = [
        metric.get("mean_class_accuracy", {}).get("mean", float("nan"))
        for metric in metrics
    ]

    (loss_line,) = loss_axis.plot(epochs, losses, color=_LOSS_COLOR, label="Loss")
    (accuracy_line,) = accuracy_axis.plot(
        epochs, accuracies, color=_ACCURACY_COLOR, label="Mean Class Accuracy"
    )
    loss_axis.legend(handles=[loss_line, accuracy_line], loc="best", fontsize=9)


def _plot_classification(
    out_dir: str,
    genome: CircuitGenome,
    filename: str,
    metadata: dict[str, Any],
) -> None:
    """Draws side-by-side training and validation loss/accuracy-per-epoch panels.

    Two panels are drawn: the training metrics on the left and the validation
    metrics on the right, each showing loss and mean class accuracy per epoch on
    twin y-axes.

    Args:
        out_dir: Directory to write the plot into.
        genome: The genome being plotted (used for the title).
        filename: Output file name within ``out_dir``.
        metadata: The genome metadata containing ``training_epoch_metrics`` and
            ``validation_epoch_metrics``.
    """
    training = metadata.get("training_epoch_metrics", [])
    validation = metadata.get("validation_epoch_metrics", [])
    if not training and not validation:
        return

    figure, (training_axis, validation_axis) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_epoch_panel(training_axis, training, "Training")
    _plot_epoch_panel(validation_axis, validation, "Validation")

    figure.suptitle(f"Genome {genome.genome_number} Training", fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close(figure)


#: Which plot each task gets. A genome carries its own ``task`` (stamped by
#: EXAQC), so this is the whole of the routing; a genome without one falls back
#: to :func:`_plot_for_metrics`.
_PLOTS_BY_TASK: dict[str, Callable[..., None]] = {
    "classification": _plot_classification,
    "reinforcement_learning": _plot_reinforcement_learning,
    "teacher": _plot_all_metrics,
}
