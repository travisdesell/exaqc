"""Tests for the epoch-0 (pre-training) evaluation of a genome's inherited weights.

Both trainers evaluate a genome before training it, so the recorded history
shows whether training improved on the weights the genome inherited:

* ``SupervisedTrainer`` records the evaluation as epoch 0 of
  ``validation_epoch_metrics`` (``training_epoch_metrics`` starts at epoch 1,
  since nothing was trained before it) and numbers its training epochs
  ``1..epochs`` inclusive.
* ``ReinforcementLearningTrainer`` records it as episode 0 of
  ``evaluation_episode_metrics`` and numbers its training episodes
  ``1..episodes`` inclusive.

The inherited weights compete with the trained ones, so a genome keeps them
when training never improves on them, and a genome trained for 0 epochs or
episodes gets its fitness purely from them -- which is what lets EXAQC run
without any training.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.circuits.circuit import CircuitGenome
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.objectives.classification_objective import ClassificationObjective
from src.objectives.reinforcement_learning_objective import (
    ReinforcementLearningObjective,
)
from src.trainer.reinforcement_trainer import RLEnvironment

from tests.reinforcement_trainer_test_utils import (
    build_rl_genome,
    build_trainer,
    make_test_environment,
)
from tests.supervised_trainer_test_utils import (
    build_classification_genome,
    cross_entropy_on_logits,
    make_balanced_binary_dataloaders,
    snapshot_gate_parameters,
)


def _classification_setup(
    epochs: int,
) -> tuple[CircuitGenome, ClassificationObjective]:
    """Builds a trainable classification genome and an objective to score it.

    Args:
        epochs: The ``epochs`` hyperparameter to give the genome.

    Returns:
        The genome (with early stopping disabled, so every epoch runs) and a
        classification objective over balanced binary dataloaders.
    """

    genome, n_features = build_classification_genome(
        genome_number=1,
        target="pennylane",
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        include_parametric=True,
        epochs=epochs,
    )
    genome.hyperparameters["improvement_cutoff"] = 0

    train_loader, val_loader = make_balanced_binary_dataloaders(n_features=n_features)
    objective = ClassificationObjective(
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        training_loss_function=cross_entropy_on_logits,
        validation_loss_function=cross_entropy_on_logits,
        metrics={"mean_class_accuracy": MeanClassAccuracy(n_labels=2)},
    )
    return genome, objective


def _rl_setup(
    episodes: int,
) -> tuple[CircuitGenome, ReinforcementLearningObjective, RLEnvironment]:
    """Builds a trainable RL genome, an objective to score it, and its environment.

    Args:
        episodes: The ``episodes`` hyperparameter to give the genome.

    Returns:
        The genome, a REINFORCE-backed objective, and the deterministic test
        environment it is trained on.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    genome.hyperparameters["episodes"] = episodes
    environment = make_test_environment(observation_features)
    objective = ReinforcementLearningObjective(environment=environment, trainer=trainer)
    return genome, objective, environment


@pytest.mark.parametrize("epochs", [1, 3])
def test_supervised_epochs_run_from_one_through_epochs_inclusive(epochs: int) -> None:
    """Epoch 0 is the pre-training evaluation, then exactly ``epochs`` epochs train.

    Args:
        epochs: How many training epochs to run.
    """

    genome, objective = _classification_setup(epochs)

    objective(genome)

    training = genome.metadata["training_epoch_metrics"]
    validation = genome.metadata["validation_epoch_metrics"]
    assert [entry["epoch"] for entry in training] == list(range(1, epochs + 1))
    assert [entry["epoch"] for entry in validation] == list(range(0, epochs + 1))


def test_supervised_zero_epochs_scores_the_inherited_weights() -> None:
    """With 0 epochs nothing trains and fitness comes from the inherited weights."""

    genome, objective = _classification_setup(epochs=0)
    initial_gate_parameters = snapshot_gate_parameters(genome)

    objective(genome)

    assert genome.metadata["training_epoch_metrics"] == []
    validation = genome.metadata["validation_epoch_metrics"]
    assert [entry["epoch"] for entry in validation] == [0]
    assert genome.metadata["best_epoch"] == 0
    assert genome.metadata["best_validation_metrics"] is validation[0]
    # untouched, not even round-tripped through the torch model
    assert snapshot_gate_parameters(genome) == initial_gate_parameters

    training_loss = genome.metadata["best_training_metrics"]["loss"]
    assert genome.fitness["loss"] == pytest.approx(
        (training_loss + validation[0]["loss"]) / 2.0
    )


def test_supervised_keeps_inherited_weights_when_training_never_improves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genome whose training only gets worse keeps its epoch-0 weights and metrics.

    Args:
        monkeypatch: Used to make every training epoch score worse than epoch 0.
    """

    genome, objective = _classification_setup(epochs=2)
    trainer = objective.trainer
    real_get_metrics = trainer.get_metrics

    def worsening_get_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Runs the real metrics, then inflates the loss of every training epoch.

        Args:
            *args: Forwarded to :meth:`SupervisedTrainer.get_metrics`.
            **kwargs: Forwarded to :meth:`SupervisedTrainer.get_metrics`.

        Returns:
            The real metrics, with ``loss`` raised by 100 after epoch 0.
        """

        results = real_get_metrics(*args, **kwargs)
        if results.get("epoch", 0) > 0:
            results["loss"] += 100.0
        return results

    monkeypatch.setattr(trainer, "get_metrics", worsening_get_metrics)
    objective(genome)

    validation = genome.metadata["validation_epoch_metrics"]
    assert len(validation) == 3
    assert genome.metadata["best_epoch"] == 0
    assert genome.metadata["best_validation_metrics"] is validation[0]


@pytest.mark.parametrize("episodes", [1, 3])
def test_rl_episodes_run_from_one_through_episodes_inclusive(episodes: int) -> None:
    """Episode 0 is the pre-training evaluation, then exactly ``episodes`` episodes train.

    The test configuration evaluates after every episode (``log_every=1``),
    so the evaluation history covers every episode from 0.

    Args:
        episodes: How many training episodes to run.
    """

    genome, objective, _ = _rl_setup(episodes)

    objective(genome)

    training = genome.metadata["training_episode_metrics"]
    evaluation = genome.metadata["evaluation_episode_metrics"]
    assert [entry["episode"] for entry in training] == list(range(1, episodes + 1))
    assert [entry["episode"] for entry in evaluation] == list(range(0, episodes + 1))


def test_rl_evaluates_every_log_every_episodes_and_the_last() -> None:
    """Evaluations land on episode 0, each multiple of ``log_every``, and the last."""

    genome, objective, _ = _rl_setup(episodes=5)
    genome.hyperparameters["log_every"] = 2

    objective(genome)

    evaluation = genome.metadata["evaluation_episode_metrics"]
    assert [entry["episode"] for entry in evaluation] == [0, 2, 4, 5]


def test_rl_zero_episodes_scores_the_inherited_weights() -> None:
    """With 0 episodes nothing trains and fitness comes from the inherited weights."""

    genome, objective, _ = _rl_setup(episodes=0)
    initial_gate_parameters = snapshot_gate_parameters(genome)

    objective(genome)

    assert genome.metadata["training_episode_metrics"] == []
    evaluation = genome.metadata["evaluation_episode_metrics"]
    assert [entry["episode"] for entry in evaluation] == [0]
    assert genome.metadata["best_episode"] == 0
    assert snapshot_gate_parameters(genome) == initial_gate_parameters

    initial_return = evaluation[0]["return_mean"]
    assert genome.metadata["best_training_metrics"]["return_mean"] == initial_return
    assert genome.fitness["eval_return_mean"] == initial_return
    assert genome.fitness["train_return_mean"] == initial_return
    assert genome.fitness["loss"] == pytest.approx(-initial_return)


def test_rl_keeps_inherited_weights_when_training_never_improves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genome whose evaluations only get worse keeps its episode-0 weights.

    Args:
        monkeypatch: Used to make every post-training evaluation score worse
            than the pre-training one.
    """

    genome, objective, _ = _rl_setup(episodes=3)
    trainer = objective.trainer
    real_evaluate = trainer.evaluate
    calls = {"count": 0}

    def worsening_evaluate(*args: Any, **kwargs: Any) -> dict[str, float]:
        """Runs the real evaluation, then lowers every one after the first.

        Args:
            *args: Forwarded to :meth:`ReinforcementLearningTrainer.evaluate`.
            **kwargs: Forwarded to :meth:`ReinforcementLearningTrainer.evaluate`.

        Returns:
            The real results, with ``return_mean`` lowered by 100 after the
            pre-training evaluation.
        """

        results = real_evaluate(*args, **kwargs)
        if calls["count"] > 0:
            results["return_mean"] -= 100.0
        calls["count"] += 1
        return results

    monkeypatch.setattr(trainer, "evaluate", worsening_evaluate)
    objective(genome)

    evaluation = genome.metadata["evaluation_episode_metrics"]
    assert genome.metadata["best_episode"] == 0
    assert genome.metadata["best_validation_metrics"] is evaluation[0]
    assert genome.fitness["eval_return_mean"] == evaluation[0]["return_mean"]
