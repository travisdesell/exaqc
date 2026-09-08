"""Tests for the public RL-trainer hyperparameter API.

``RL_HYPERPARAMETER_DEFAULTS`` and
``ReinforcementLearningTrainer.resolve_hyperparameters`` are public so callers
(custom training loops, tests) can build the exact hyperparameters a single
:meth:`~src.trainer.reinforcement_trainer.ReinforcementLearningTrainer.run_update`
needs without reaching into private state. These tests pin that contract:

* ``RL_HYPERPARAMETER_DEFAULTS`` is a mapping carrying the expected keys; and
* ``resolve_hyperparameters`` prefers per-genome overrides and otherwise falls
  back to those defaults, returning a fresh, independent attribute bag each
  call whose attributes cover exactly the default keys.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers
from src.trainer.reinforcement_trainer import RL_HYPERPARAMETER_DEFAULTS
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


def test_rl_hyperparameter_defaults_has_expected_keys() -> None:
    """``RL_HYPERPARAMETER_DEFAULTS`` carries the expected keys and values."""

    assert isinstance(RL_HYPERPARAMETER_DEFAULTS, dict)

    keys = set(RL_HYPERPARAMETER_DEFAULTS)
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
        "improvement_cutoff",
        "quantum_dropout",
    } <= keys
    # the field was renamed from the PPO-literature "epochs" term
    assert "ppo_epochs" not in keys

    assert RL_HYPERPARAMETER_DEFAULTS["episodes"] == 60
    assert RL_HYPERPARAMETER_DEFAULTS["ppo_passes"] == 4
    assert RL_HYPERPARAMETER_DEFAULTS["ema_alpha"] == 0.01
    assert RL_HYPERPARAMETER_DEFAULTS["improvement_cutoff"] == 30
    assert RL_HYPERPARAMETER_DEFAULTS["quantum_dropout"] is False


def test_resolve_hyperparameters_falls_back_to_defaults() -> None:
    """A genome with no hyperparameters resolves to the module defaults."""

    trainer = ReinforceTrainer()
    genome = _genome_with_hyperparameters({})

    resolved = trainer.resolve_hyperparameters(genome)

    assert isinstance(resolved, SimpleNamespace)
    # every default key is present as an attribute, and only those keys
    assert vars(resolved) == dict(RL_HYPERPARAMETER_DEFAULTS)
    assert resolved.episodes == RL_HYPERPARAMETER_DEFAULTS["episodes"]
    assert (
        resolved.improvement_cutoff == RL_HYPERPARAMETER_DEFAULTS["improvement_cutoff"]
    )
    assert resolved.quantum_dropout is False


def test_resolve_hyperparameters_prefers_genome_overrides() -> None:
    """Per-genome values win; unspecified fields fall back to the defaults."""

    trainer = ReinforceTrainer()
    genome = _genome_with_hyperparameters(
        {
            "episodes": 3,
            "gamma": 0.5,
            "improvement_cutoff": 2,
            "quantum_dropout": True,
            "unrelated_key": 123,
        }
    )

    resolved = trainer.resolve_hyperparameters(genome)

    # overrides taken from the genome
    assert resolved.episodes == 3
    assert resolved.gamma == 0.5
    assert resolved.improvement_cutoff == 2
    assert resolved.quantum_dropout is True
    # unspecified field falls back to the module default
    assert resolved.learning_rate == RL_HYPERPARAMETER_DEFAULTS["learning_rate"]
    # keys that are not hyperparameter defaults are ignored (no error, no attr)
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
    # the shared module defaults are also untouched by the mutation
    assert RL_HYPERPARAMETER_DEFAULTS["episodes"] != 999
