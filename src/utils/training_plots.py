from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

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


def save_training_plot(
    out_dir: str,
    genome: CircuitGenome,
    filename: str,
) -> None:
    """Saves a per-epoch/episode training line plot for a trained genome.

    The task is auto-detected from the genome's ``metadata``: a
    reinforcement-learning genome carries ``training_episode_metrics`` (return
    and loss per episode, drawn on twin y-axes of a single plot) while a
    classification genome carries ``training_epoch_metrics`` /
    ``validation_epoch_metrics`` (loss and mean class accuracy per epoch, drawn
    as side-by-side training and validation twin-axis panels).

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

    try:
        if "training_episode_metrics" in metadata:
            _plot_reinforcement_learning(out_dir, genome, filename, metadata)
        elif "training_epoch_metrics" in metadata:
            _plot_classification(out_dir, genome, filename, metadata)
        else:
            logger.warning(
                "No training metrics found for genome {}; skipping training " "plot.",
                genome_number,
            )
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
