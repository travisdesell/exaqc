"""Tests for the public RL-trainer hyperparameter API.

``RLHyperparameters`` and
``ReinforcementLearningTrainer.resolve_hyperparameters`` are public so callers
(custom training loops, tests) can build the exact hyperparameters a single
:meth:`~src.trainer.reinforcement_trainer.ReinforcementLearningTrainer.run_update`
needs without reaching into private state. These tests pin that contract:

* ``RLHyperparameters`` is a public dataclass with the expected fields; and
* ``resolve_hyperparameters`` prefers per-genome overrides and otherwise
  falls back to the trainer's construction defaults, returning a fresh,
  independent instance each call.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers
from src.trainer.reinforcement_trainer import RLHyperparameters
from src.trainer.reinforce_trainer import ReinforceTrainer


def _genome_with_hyperparameters(hyperparameters: dict[str, object]) -> CircuitGenome:
    """Builds a minimal genome carrying the given hyperparameters dict.

    ``resolve_hyperparameters`` only reads ``genome.hyperparameters``, so the
    genome needs no gates, encoder, decoder, or initialized model.

    Args:
        hyperparameters: The mapping to attach as ``genome.hyperparameters``.

    Returns:
        A :class:`CircuitGenome` with its ``hyperparameters`` set.
    """

    genome = CircuitGenome(
        genome_number=1,
        target="pennylane",
        input_qubits=expand_registers({"q": 1}),
    )
    genome.hyperparameters = dict(hyperparameters)
    return genome


def test_rl_hyperparameters_is_a_public_dataclass() -> None:
    """``RLHyperparameters`` is a public dataclass with the expected fields."""

    assert is_dataclass(RLHyperparameters)

    field_names = {field.name for field in fields(RLHyperparameters)}
    assert {
        "episodes",
        "learning_rate",
        "gamma",
        "max_steps",
        "eval_episodes",
        "ema_alpha",
        "ppo_passes",
        "ppo_minibatch",
        "epsilon",
    } <= field_names
    # the field was renamed from the PPO-literature "epochs" term
    assert "ppo_epochs" not in field_names

    defaults = RLHyperparameters()
    assert defaults.episodes == 60
    assert defaults.ppo_passes == 4
    assert defaults.ema_alpha == 0.01


def test_resolve_hyperparameters_falls_back_to_trainer_defaults() -> None:
    """A genome with no hyperparameters resolves to the trainer's defaults."""

    trainer = ReinforceTrainer(episodes=7, learning_rate=0.123, gamma=0.9)
    genome = _genome_with_hyperparameters({})

    resolved = trainer.resolve_hyperparameters(genome)

    assert isinstance(resolved, RLHyperparameters)
    assert resolved.episodes == 7
    assert resolved.learning_rate == 0.123
    assert resolved.gamma == 0.9


def test_resolve_hyperparameters_prefers_genome_overrides() -> None:
    """Per-genome values win; unspecified fields fall back to trainer defaults."""

    trainer = ReinforceTrainer(episodes=7, gamma=0.9)
    genome = _genome_with_hyperparameters(
        {"episodes": 3, "gamma": 0.5, "unrelated_key": 123}
    )

    resolved = trainer.resolve_hyperparameters(genome)

    # overrides taken from the genome
    assert resolved.episodes == 3
    assert resolved.gamma == 0.5
    # unspecified field falls back to the trainer's default
    assert resolved.learning_rate == trainer.defaults.learning_rate
    # keys that are not hyperparameter fields are ignored (no error, no attr)
    assert not hasattr(resolved, "unrelated_key")


def test_resolve_hyperparameters_returns_independent_instances() -> None:
    """Each call returns a fresh instance that can be mutated safely."""

    trainer = ReinforceTrainer()
    genome = _genome_with_hyperparameters({})

    first = trainer.resolve_hyperparameters(genome)
    second = trainer.resolve_hyperparameters(genome)

    assert first is not second

    first.episodes = 999
    assert second.episodes != 999
    # the trainer's stored defaults are also untouched by the mutation
    assert trainer.defaults.episodes != 999
