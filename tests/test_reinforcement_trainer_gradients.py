"""Gradient-flow tests for the reinforcement-learning trainers.

These tests verify requirement (3): that running one epoch of an actual RL
algorithm on a simple environment backpropagates gradients through the
genome's **encoder**, **quantum circuit**, and **decoder**.

Rather than reconstructing an algorithm's loss by hand, each test drives the
trainer's own :meth:`run_update` for a single update on the deterministic
test environment, then inspects the parameters: after the update, every
trainable stage (a ``LinearEncoder``, the quantum weight tensor, and a
``LinearDecoder``) must have received a non-zero gradient and been moved by
the optimizer. Using ``run_update`` (instead of the full ``train`` loop)
keeps the post-update gradients on the parameters for inspection, since
``train`` would restore the best snapshot afterwards.

The genome is built with a trainable ``LinearEncoder`` and ``LinearDecoder``
so all three stages actually carry parameters to check; encoder/decoder
variety (including stateless coders) is covered in
``test_reinforcement_trainer_epochs.py``.
"""

from __future__ import annotations

import numpy as np
import torch

import pytest

from src.circuits.circuit import CircuitGenome
from src.trainer.reinforcement_trainer import greedy_action

from tests.reinforcement_trainer_test_utils import (
    COMPLEXITY_LEVELS,
    CONTINUOUS_TRAINER_NAMES,
    TRAINER_NAMES,
    build_rl_genome,
    build_trainer,
    circuit_trainable_parameters,
    decoder_trainable_parameters,
    encoder_trainable_parameters,
    make_continuous_test_environment,
    make_test_environment,
    prepare_single_update,
    run_single_update,
)

TARGETS: tuple[str, ...] = ("pennylane", "qiskit")


def _assert_stage_received_gradient(
    parameters: list[torch.nn.Parameter], stage: str
) -> None:
    """Asserts a stage has trainable parameters that all received a gradient.

    Args:
        parameters: The stage's trainable parameters.
        stage: Human-readable stage name for assertion messages.

    Raises:
        AssertionError: If the stage has no parameters, or any parameter has a
            missing or all-zero gradient.
    """

    assert parameters, f"expected the {stage} to have trainable parameters"
    for parameter in parameters:
        assert parameter.grad is not None, f"{stage} parameter received no gradient"
        assert torch.any(parameter.grad != 0), f"{stage} gradient was entirely zero"


def _snapshot(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Detached clones of the given parameters, for before/after comparison.

    Args:
        parameters: Parameters to snapshot.

    Returns:
        A list of detached, cloned tensors.
    """

    return [parameter.detach().clone() for parameter in parameters]


def _any_changed(before: list[torch.Tensor], after: list[torch.nn.Parameter]) -> bool:
    """Returns whether any parameter moved from its snapshot.

    Args:
        before: Snapshotted tensors captured before the update.
        after: The live parameters after the update, in the same order.

    Returns:
        True if at least one parameter changed value.
    """

    return any(not torch.allclose(old, new) for old, new in zip(before, after))


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS)
@pytest.mark.parametrize("trainer_name", TRAINER_NAMES)
@pytest.mark.parametrize("target", TARGETS)
def test_gradients_flow_through_encoder_circuit_decoder(
    target: str, trainer_name: str, complexity: str
) -> None:
    """One RL update backpropagates through encoder, quantum circuit, decoder.

    Builds a fully-trainable genome (``LinearEncoder`` + parametric circuit +
    ``LinearDecoder``), runs a single update of the given algorithm on the
    deterministic environment, and asserts each stage received a non-zero
    gradient and was updated by the optimizer.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        trainer_name: The RL algorithm to exercise.
        complexity: The circuit complexity level to build.
    """

    trainer = build_trainer(trainer_name)
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target=target,
        complexity=complexity,
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)

    # prepare_single_update initializes the model and builds the optimizer via
    # the trainer's public hyperparameter resolution, without running the
    # update -- so the snapshots below capture the freshly initialized weights.
    optimizer, hp = prepare_single_update(trainer, genome)

    encoder_params = encoder_trainable_parameters(genome)
    (quantum_weight,) = circuit_trainable_parameters(genome)
    decoder_params = decoder_trainable_parameters(genome)

    encoder_before = _snapshot(encoder_params)
    quantum_before = _snapshot([quantum_weight])
    decoder_before = _snapshot(decoder_params)

    trainer.run_update(genome, environment, optimizer, 0, hp)

    # gradients reached every stage
    _assert_stage_received_gradient(encoder_params, "encoder")
    _assert_stage_received_gradient([quantum_weight], "quantum circuit")
    _assert_stage_received_gradient(decoder_params, "decoder")

    # and the optimizer moved every stage
    assert _any_changed(
        encoder_before, encoder_params
    ), "encoder weights did not change"
    assert _any_changed(
        quantum_before, [quantum_weight]
    ), "quantum weights did not change"
    assert _any_changed(
        decoder_before, decoder_params
    ), "decoder weights did not change"


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS)
@pytest.mark.parametrize("trainer_name", CONTINUOUS_TRAINER_NAMES)
@pytest.mark.parametrize("target", TARGETS)
def test_gradients_flow_through_encoder_circuit_decoder_continuous(
    target: str, trainer_name: str, complexity: str
) -> None:
    """One update on a continuous env backprops through all three stages.

    The continuous counterpart of
    :func:`test_gradients_flow_through_encoder_circuit_decoder`: the genome's
    decoder produces a mean *and* a log-std per action dimension (plus any
    value output), the trainer builds a diagonal ``Normal`` policy, and a
    single update must still push a non-zero gradient into the encoder, the
    quantum weight tensor, and the decoder -- and move every one of them. Only
    the policy-gradient trainers are exercised, since value-based methods do
    not support continuous action spaces.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        trainer_name: The (policy-gradient) RL algorithm to exercise.
        complexity: The circuit complexity level to build.
    """

    trainer = build_trainer(trainer_name)
    genome, observation_features = build_rl_genome(
        genome_number=1,
        target=target,
        complexity=complexity,
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
        continuous=True,
    )
    environment = make_continuous_test_environment(observation_features)

    optimizer, hp = prepare_single_update(trainer, genome)

    encoder_params = encoder_trainable_parameters(genome)
    (quantum_weight,) = circuit_trainable_parameters(genome)
    decoder_params = decoder_trainable_parameters(genome)

    encoder_before = _snapshot(encoder_params)
    quantum_before = _snapshot([quantum_weight])
    decoder_before = _snapshot(decoder_params)

    trainer.run_update(genome, environment, optimizer, 0, hp)

    _assert_stage_received_gradient(encoder_params, "encoder")
    _assert_stage_received_gradient([quantum_weight], "quantum circuit")
    _assert_stage_received_gradient(decoder_params, "decoder")

    assert _any_changed(
        encoder_before, encoder_params
    ), "encoder weights did not change"
    assert _any_changed(
        quantum_before, [quantum_weight]
    ), "quantum weights did not change"
    assert _any_changed(
        decoder_before, decoder_params
    ), "decoder weights did not change"


@pytest.mark.parametrize("trainer_name", TRAINER_NAMES)
@pytest.mark.parametrize("target", TARGETS)
def test_quantum_circuit_gradient_flows_with_stateless_coders(
    target: str, trainer_name: str
) -> None:
    """The circuit still trains when the encoder/decoder are stateless.

    With an ``IdentityEncoder`` and ``ClippedDecoder`` (no trainable
    parameters of their own), gradients must still flow *through* them to the
    quantum circuit's weight tensor. Value methods are skipped here because a
    normalizing ``ClippedDecoder`` is not an appropriate value readout (see
    the trainer module docstring).

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        trainer_name: The RL algorithm to exercise.
    """

    trainer = build_trainer(trainer_name)
    if trainer.n_value_outputs != 0:
        pytest.skip("value methods require an unconstrained (linear) decoder")

    genome, observation_features = build_rl_genome(
        genome_number=2,
        target=target,
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)

    run_single_update(trainer, genome, environment)

    assert encoder_trainable_parameters(genome) == []
    assert decoder_trainable_parameters(genome) == []
    _assert_stage_received_gradient(
        circuit_trainable_parameters(genome), "quantum circuit"
    )


@pytest.mark.parametrize("target", TARGETS)
def test_forward_output_is_differentiable_end_to_end(target: str) -> None:
    """A direct forward/backward through the genome is differentiable.

    A framework-agnostic sanity check that does not depend on any RL
    algorithm: a scalar built from ``genome.forward`` backpropagates a
    non-zero gradient to encoder, quantum, and decoder parameters.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=3,
        target=target,
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)
    genome.initialize_model()

    observation, _ = environment.make().reset(seed=0)
    output = genome.forward(environment.encode(observation))
    loss = output.pow(2).sum()
    loss.backward()

    _assert_stage_received_gradient(encoder_trainable_parameters(genome), "encoder")
    _assert_stage_received_gradient(
        circuit_trainable_parameters(genome), "quantum circuit"
    )
    _assert_stage_received_gradient(decoder_trainable_parameters(genome), "decoder")


def test_greedy_action_discrete_returns_valid_action_index() -> None:
    """The shared greedy action is a valid discrete action index.

    ``greedy_action`` is what both ``evaluate`` and the visualization script
    use to drive the environment; for a discrete space it must return a Python
    ``int`` within ``[0, n_actions)``.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=5,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)
    genome.initialize_model()

    observation, _ = environment.make().reset(seed=0)
    action = greedy_action(genome, environment, observation)

    assert isinstance(action, int)
    assert 0 <= action < environment.n_actions


def test_greedy_action_continuous_is_clipped_to_bounds() -> None:
    """The shared greedy action is a clipped, correctly-shaped float array.

    For a continuous space ``greedy_action`` must return the policy mean as a
    ``float32`` array of shape ``(n_actions,)`` clipped to the environment's
    action bounds -- exactly what ``env.step`` (and the visualization script)
    expects.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=6,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
        continuous=True,
    )
    environment = make_continuous_test_environment(observation_features)
    genome.initialize_model()

    observation, _ = environment.make().reset(seed=0)
    action = greedy_action(genome, environment, observation)

    assert isinstance(action, np.ndarray)
    assert action.shape == (environment.n_actions,)
    assert action.dtype == np.float32
    assert np.all(action >= environment.action_low)
    assert np.all(action <= environment.action_high)


def test_run_single_update_helper_returns_none() -> None:
    """The shared ``run_single_update`` helper drives an update without error.

    A light smoke test of the test utility itself (used by the parametrized
    cases above), ensuring it initializes and updates a genome in place.
    """

    trainer = build_trainer("reinforce")
    genome, observation_features = build_rl_genome(
        genome_number=4,
        target="pennylane",
        complexity="minimal",
        encoder_name="linear",
        decoder_name="linear",
        trainer=trainer,
    )
    environment = make_test_environment(observation_features)

    result = run_single_update(trainer, genome, environment)
    assert result is None
    assert isinstance(genome, CircuitGenome)
    assert genome.hybrid_model is not None
