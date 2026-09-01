"""Tests for the optional per-epoch/episode training-history line plots.

``src.utils.training_plots.save_training_plot`` auto-detects the task from a
genome's metadata and writes a twin-axis line plot: loss and mean class
accuracy per epoch for classification, or return and loss per episode for
reinforcement learning. ``CircuitGenome.save_circuit`` writes it too when its
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
