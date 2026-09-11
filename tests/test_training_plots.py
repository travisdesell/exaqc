"""Tests for the optional per-epoch/episode training-history line plots.

``src.utils.training_plots.save_training_plot`` chooses a plot from the
genome's task and writes it: loss and mean class accuracy per epoch for
classification, return and loss per episode for reinforcement learning, and --
for a task that reports several measures at once, such as quantum teacher
imitation -- every recorded metric drawn across side-by-side training and
validation panels. ``CircuitGenome.save_circuit`` writes it too when its
``save_training_plot`` flag is set.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from src.utils.training_plots import save_training_plot  # noqa: E402
from tests.supervised_trainer_test_utils import (  # noqa: E402
    build_classification_genome,
)

#: The first eight bytes of any PNG file (the PNG signature).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _FakeGenome:
    """Minimal genome stand-in carrying only ``metadata`` and a number."""

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.genome_number = 7


def _assert_valid_png(path) -> None:
    """Asserts that ``path`` exists, is non-empty, and is a valid PNG.

    Args:
        path: Path to the file to check.
    """
    assert path.is_file()
    assert path.stat().st_size > 0
    with open(path, "rb") as handle:
        assert handle.read(len(_PNG_MAGIC)) == _PNG_MAGIC


def test_classification_training_plot_written(tmp_path) -> None:
    """A genome with per-epoch metrics gets a valid classification plot.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome = _FakeGenome(
        {
            "training_epoch_metrics": [
                {
                    "epoch": e,
                    "loss": 1.0 / (e + 1),
                    "mean_class_accuracy": {"mean": 0.5},
                }
                for e in range(6)
            ],
            "validation_epoch_metrics": [
                {
                    "epoch": e,
                    "loss": 1.1 / (e + 1),
                    "mean_class_accuracy": {"mean": 0.4},
                }
                for e in range(6)
            ],
        }
    )
    save_training_plot(str(tmp_path), genome, "training.png")
    _assert_valid_png(tmp_path / "training.png")


def test_reinforcement_learning_training_plot_written(tmp_path) -> None:
    """A genome with per-episode metrics gets a valid RL plot.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome = _FakeGenome(
        {
            "training_episode_metrics": [
                {"episode": e, "return": float(e), "loss": 2.0 / (e + 1)}
                for e in range(15)
            ]
        }
    )
    save_training_plot(str(tmp_path), genome, "training.png")
    _assert_valid_png(tmp_path / "training.png")


def test_reinforcement_learning_evaluation_plot_written(tmp_path) -> None:
    """A genome with per-episode and evaluation metrics gets a valid two-panel plot.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome = _FakeGenome(
        {
            "training_episode_metrics": [
                {"episode": e, "return": float(e), "loss": 2.0 / (e + 1)}
                for e in range(15)
            ],
            "evaluation_episode_metrics": [
                {
                    "episode": e,
                    "return_mean": float(e),
                    "return_std": 0.5,
                    "best_episode_return": float(e) + 1.0,
                }
                for e in range(0, 15, 5)
            ],
        }
    )
    save_training_plot(str(tmp_path), genome, "training.png")
    _assert_valid_png(tmp_path / "training.png")


def test_no_metrics_skips_without_error(tmp_path) -> None:
    """With no recognized metrics, no file is written and nothing is raised.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    save_training_plot(str(tmp_path), _FakeGenome({}), "training.png")
    assert not (tmp_path / "training.png").is_file()


def test_save_circuit_emits_training_plot_when_enabled(tmp_path, monkeypatch) -> None:
    """``save_circuit(save_training_plot=True)`` writes the training plot too.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: used to ``chdir`` into ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome, _ = build_classification_genome(
        genome_number=7,
        target="pennylane",
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.metadata = {
        "best_training_metrics": {"loss": 0.1, "mean_class_accuracy": {"mean": 0.9}},
        "best_validation_metrics": {"loss": 0.2, "mean_class_accuracy": {"mean": 0.8}},
        "training_epoch_metrics": [
            {"epoch": e, "loss": 1.0 / (e + 1), "mean_class_accuracy": {"mean": 0.5}}
            for e in range(4)
        ],
        "validation_epoch_metrics": [
            {"epoch": e, "loss": 1.1 / (e + 1), "mean_class_accuracy": {"mean": 0.45}}
            for e in range(4)
        ],
    }
    genome.initialize_model()
    genome.save_circuit(
        insert_type="best", out_dir=str(out_dir), save_training_plot=True
    )

    training_pngs = [
        path for path in out_dir.iterdir() if path.name.endswith("_training.png")
    ]
    assert len(training_pngs) == 1
    _assert_valid_png(training_pngs[0])


def test_save_circuit_skips_training_plot_by_default(tmp_path, monkeypatch) -> None:
    """Without the flag, ``save_circuit`` writes no training plot.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: used to ``chdir`` into ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome, _ = build_classification_genome(
        genome_number=7,
        target="pennylane",
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.metadata = {
        "best_training_metrics": {"loss": 0.1, "mean_class_accuracy": {"mean": 0.9}},
        "best_validation_metrics": {"loss": 0.2, "mean_class_accuracy": {"mean": 0.8}},
        "training_epoch_metrics": [
            {"epoch": 0, "loss": 1.0, "mean_class_accuracy": {"mean": 0.5}}
        ],
    }
    genome.initialize_model()
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    assert not any(path.name.endswith("_training.png") for path in out_dir.iterdir())


# ---------------------------------------------------------------------
# Multi-metric (quantum teacher) plots
# ---------------------------------------------------------------------


def teacher_epoch_metrics(n_epochs: int = 5, offset: float = 0.0) -> list[dict]:
    """Builds per-epoch records shaped like a teacher run's metrics.

    Args:
        n_epochs: How many epoch records to build.
        offset: Added to every value so training and validation differ.

    Returns:
        A list of per-epoch metric records carrying loss plus all four
        teacher measures.
    """
    return [
        {
            "epoch": e,
            "loss": 1.0 / (e + 1) + offset,
            "fidelity": {"mean": 1.0 - 1.0 / (e + 2) - offset},
            "angle": {"mean": 0.8 / (e + 1) + offset},
            "kl": {"mean": 2.0 / (e + 1) + offset},
            "mse": {"mean": 0.05 / (e + 1) + offset},
        }
        for e in range(n_epochs)
    ]


class _TeacherGenome:
    """Genome stand-in that records a task, as EXAQC now stamps."""

    def __init__(self, metadata: dict, task: str | None = "teacher") -> None:
        self.metadata = metadata
        self.task = task
        self.genome_number = 3


def test_metric_names_finds_every_measure_with_loss_first() -> None:
    """All recorded measures are discovered, with the loss ordered first."""
    from src.utils.training_plots import _metric_names

    names = _metric_names(teacher_epoch_metrics(), teacher_epoch_metrics())

    assert names[0] == "loss"
    assert set(names) == {"loss", "fidelity", "angle", "kl", "mse"}
    # 'epoch' labels the record, it is not a measured quantity
    assert "epoch" not in names


def test_metric_value_unwraps_both_shapes() -> None:
    """Bare numbers and ``{"mean": ...}`` metrics both read back as floats."""
    from src.utils.training_plots import _metric_value

    record = {"loss": 0.25, "fidelity": {"mean": 0.75}}

    assert _metric_value(record, "loss") == 0.25
    assert _metric_value(record, "fidelity") == 0.75
    # a missing metric is NaN rather than an error
    assert _metric_value(record, "absent") != _metric_value(record, "absent")


def test_teacher_plot_written_with_all_metrics(tmp_path) -> None:
    """A teacher genome's plot is written and covers every recorded measure.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome = _TeacherGenome(
        {
            "training_epoch_metrics": teacher_epoch_metrics(),
            "validation_epoch_metrics": teacher_epoch_metrics(offset=0.05),
        }
    )

    save_training_plot(str(tmp_path), genome, "teacher.png")

    _assert_valid_png(tmp_path / "teacher.png")


def test_teacher_plot_draws_training_and_validation_panels(
    tmp_path, monkeypatch
) -> None:
    """Both splits get a panel, each carrying every discovered metric.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to observe the panels that were drawn.
    """
    import src.utils.training_plots as training_plots

    drawn = []
    original = training_plots._plot_metrics_panel

    def record_panel(axes, title, records, names):
        """Records the split title and metrics each panel was drawn with."""
        drawn.append((title, list(names), len(records)))
        return original(axes, title, records, names)

    monkeypatch.setattr(training_plots, "_plot_metrics_panel", record_panel)

    genome = _TeacherGenome(
        {
            "training_epoch_metrics": teacher_epoch_metrics(),
            "validation_epoch_metrics": teacher_epoch_metrics(offset=0.05),
        }
    )
    save_training_plot(str(tmp_path), genome, "teacher.png")

    expected = ["loss", "fidelity", "angle", "kl", "mse"]
    assert [title for title, _, _ in drawn] == ["Training", "Validation"]
    # every metric appears on both panels
    assert all(names == expected for _, names, _ in drawn)
    # and each panel got its own split's records
    assert all(n_records == 5 for _, _, n_records in drawn)


def test_each_task_routes_to_its_own_plot() -> None:
    """A genome's task selects its plot, with no metric sniffing involved."""
    import src.utils.training_plots as training_plots

    assert training_plots._PLOTS_BY_TASK == {
        "classification": training_plots._plot_classification,
        "reinforcement_learning": training_plots._plot_reinforcement_learning,
        "teacher": training_plots._plot_all_metrics,
    }


def test_plot_is_inferred_when_the_genome_records_no_task() -> None:
    """Genomes saved before tasks were recorded are routed by their metrics."""
    import src.utils.training_plots as training_plots

    classification_metadata = {
        "training_epoch_metrics": [
            {"epoch": 0, "loss": 1.0, "mean_class_accuracy": {"mean": 0.5}}
        ]
    }
    reinforcement_metadata = {
        "training_episode_metrics": [{"episode": 0, "return": 1.0, "loss": 0.5}]
    }
    teacher_metadata = {"training_epoch_metrics": teacher_epoch_metrics()}

    chooser = training_plots._plot_for_metrics

    assert (
        chooser(reinforcement_metadata) is training_plots._plot_reinforcement_learning
    )
    assert chooser(classification_metadata) is training_plots._plot_classification
    assert chooser(teacher_metadata) is training_plots._plot_all_metrics
    # nothing recorded at all -> no plot to draw
    assert chooser({}) is None


def test_classification_genome_keeps_its_own_plot(tmp_path) -> None:
    """A classification genome still gets the loss/accuracy panels drawn.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome = _TeacherGenome(
        {
            "training_epoch_metrics": [
                {
                    "epoch": e,
                    "loss": 1.0 / (e + 1),
                    "mean_class_accuracy": {"mean": 0.5},
                }
                for e in range(4)
            ],
            "validation_epoch_metrics": [
                {
                    "epoch": e,
                    "loss": 1.1 / (e + 1),
                    "mean_class_accuracy": {"mean": 0.4},
                }
                for e in range(4)
            ],
        },
        task="classification",
    )

    save_training_plot(str(tmp_path), genome, "cls.png")

    _assert_valid_png(tmp_path / "cls.png")


def test_empty_epoch_metrics_do_not_raise(tmp_path) -> None:
    """A genome that recorded no epochs is skipped rather than crashing.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome = _TeacherGenome(
        {"training_epoch_metrics": [], "validation_epoch_metrics": []}
    )

    save_training_plot(str(tmp_path), genome, "empty.png")

    assert not (tmp_path / "empty.png").is_file()
