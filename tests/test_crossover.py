"""Tests for ``src.evolution.crossover.torch_simplex_crossover``.

Simplex crossover blends the trainable weights of several parent
encoders/decoders into a child using

    child = primary + r * (mean(others) - primary)

applied element-wise to every parameter tensor. These tests pin two things
that a previous no-op bug got wrong:

* the child's parameters are actually *updated* to the blended values (the
  bug rebound a local ``state_dict`` dict entry without writing back into the
  module, so the child silently kept the primary's weights); and
* every parameter row is blended -- including a trailing "value" output row
  on a multi-output decoder, which is exactly what the RL trainers rely on.

Parents must be left untouched, since the child is meant to be an
independent new module.
"""

from __future__ import annotations

import torch

import pytest

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import ClippedDecoder, initialize_decoder
from src.circuits.encoder import IdentityEncoder, initialize_encoder
from src.circuits.registers import expand_registers
from src.evolution.crossover import crossover_encoder_decoder, torch_simplex_crossover


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Returns a detached, cloned copy of a module's ``state_dict``.

    Args:
        module: The module whose parameters should be snapshotted.

    Returns:
        A mapping from parameter name to a detached clone of its tensor.
    """

    return {
        name: tensor.detach().clone() for name, tensor in module.state_dict().items()
    }


def _expected_simplex(
    snapshots: list[dict[str, torch.Tensor]], r: float
) -> dict[str, torch.Tensor]:
    """Computes the expected simplex-crossover result from parent snapshots.

    This mirrors the documented formula independently of the implementation
    under test, so a bug in the implementation cannot be masked by reusing it.

    Args:
        snapshots: Per-parent snapshots (``snapshots[0]`` is the primary).
        r: The line-search blend factor.

    Returns:
        A mapping from parameter name to the expected child tensor.
    """

    primary = snapshots[0]
    others = snapshots[1:]

    expected: dict[str, torch.Tensor] = {}
    for name, primary_tensor in primary.items():
        mean_others = torch.stack([snap[name] for snap in others]).mean(dim=0)
        expected[name] = primary_tensor + r * (mean_others - primary_tensor)
    return expected


def _linear_module(kind: str, n_inputs: int, n_outputs: int) -> torch.nn.Module:
    """Builds a trainable linear encoder or decoder.

    Args:
        kind: Either ``"encoder"`` or ``"decoder"``.
        n_inputs: Input feature dimension.
        n_outputs: Output feature dimension.

    Returns:
        A ``LinearEncoder`` or ``LinearDecoder`` with random initial weights.
    """

    if kind == "encoder":
        return initialize_encoder(
            target="pennylane",
            encoding_str="linear",
            n_inputs=n_inputs,
            n_outputs=n_outputs,
        )
    return initialize_decoder(
        target="pennylane",
        decoding_str="linear",
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )


@pytest.mark.parametrize("r", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("n_parents", [2, 3])
@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_simplex_crossover_matches_formula(kind: str, n_parents: int, r: float) -> None:
    """The child equals ``primary + r * (mean(others) - primary)`` element-wise.

    Args:
        kind: Whether to cross encoders or decoders.
        n_parents: Number of parents (primary plus one or more others).
        r: The blend factor.
    """

    torch.manual_seed(0)
    modules = [_linear_module(kind, n_inputs=4, n_outputs=3) for _ in range(n_parents)]
    snapshots = [_snapshot(module) for module in modules]

    child = torch_simplex_crossover(modules, r)

    expected = _expected_simplex(snapshots, r)
    child_state = child.state_dict()
    assert set(child_state.keys()) == set(expected.keys())
    for name, expected_tensor in expected.items():
        assert torch.allclose(child_state[name], expected_tensor, atol=1e-6), name


def test_simplex_crossover_r_zero_returns_primary_weights() -> None:
    """With ``r = 0`` the child weights equal the primary's (no blending)."""

    torch.manual_seed(1)
    modules = [_linear_module("decoder", n_inputs=4, n_outputs=3) for _ in range(3)]
    primary_snapshot = _snapshot(modules[0])

    child = torch_simplex_crossover(modules, r=0.0)

    child_state = child.state_dict()
    for name, primary_tensor in primary_snapshot.items():
        assert torch.allclose(child_state[name], primary_tensor, atol=1e-6), name


def test_simplex_crossover_updates_all_rows_including_value_output() -> None:
    """Every output row is blended -- the regression guard for the no-op bug.

    A three-output decoder is used to mimic the RL "policy + value" decoder,
    where the last row is the state-value output. With the primary filled with
    zeros, the others with a constant, and ``r = 1.0``, a correctly-updated
    child must equal that constant on *every* row, including the value row.
    The pre-fix implementation left the child equal to the primary (all
    zeros), so this fails loudly if the write-back regresses.
    """

    decoders = [
        initialize_decoder(
            target="pennylane", decoding_str="linear", n_inputs=4, n_outputs=3
        )
        for _ in range(3)
    ]
    with torch.no_grad():
        decoders[0].layer.weight.fill_(0.0)
        decoders[0].layer.bias.fill_(0.0)
        decoders[1].layer.weight.fill_(2.0)
        decoders[1].layer.bias.fill_(2.0)
        decoders[2].layer.weight.fill_(2.0)
        decoders[2].layer.bias.fill_(2.0)

    child = torch_simplex_crossover(decoders, r=1.0)

    weight = child.layer.weight
    bias = child.layer.bias

    # not left at the primary's zeros anywhere...
    assert not torch.allclose(weight, torch.zeros_like(weight))
    # ...and specifically the trailing value row (index 2) is blended
    assert torch.allclose(weight[2], torch.full_like(weight[2], 2.0))
    assert torch.allclose(bias, torch.full_like(bias, 2.0))
    # the full tensor is the constant, i.e. all rows updated
    assert torch.allclose(weight, torch.full_like(weight, 2.0))


@pytest.mark.parametrize("r", [0.3, 0.5, 1.0])
def test_simplex_crossover_leaves_parents_unmodified(r: float) -> None:
    """Crossover must not mutate any parent module in place.

    Args:
        r: The blend factor.
    """

    torch.manual_seed(2)
    modules = [_linear_module("decoder", n_inputs=4, n_outputs=3) for _ in range(3)]
    snapshots = [_snapshot(module) for module in modules]

    torch_simplex_crossover(modules, r)

    for module, snapshot in zip(modules, snapshots):
        current = module.state_dict()
        for name, original in snapshot.items():
            assert torch.allclose(current[name], original, atol=1e-8), name


def test_simplex_crossover_child_is_independent_of_primary() -> None:
    """The child is a distinct module whose later edits don't touch the primary."""

    torch.manual_seed(3)
    modules = [_linear_module("encoder", n_inputs=4, n_outputs=3) for _ in range(2)]
    primary_snapshot = _snapshot(modules[0])

    child = torch_simplex_crossover(modules, r=0.5)
    assert child is not modules[0]

    with torch.no_grad():
        child.layer.weight.add_(1.0)

    # mutating the child must leave the primary untouched
    current_primary = modules[0].state_dict()
    for name, original in primary_snapshot.items():
        assert torch.allclose(current_primary[name], original, atol=1e-8), name


def _genome_with_coders(genome_number: int, encoder, decoder) -> CircuitGenome:
    """Builds a minimal genome carrying the given encoder and decoder.

    Args:
        genome_number: Unique genome identifier.
        encoder: The encoder to attach.
        decoder: The decoder to attach.

    Returns:
        A :class:`CircuitGenome` with ``encoder``/``decoder`` set.
    """

    genome = CircuitGenome(
        genome_number=genome_number,
        target="pennylane",
        input_qubits=expand_registers({"q": 2}),
    )
    genome.encoder = encoder
    genome.decoder = decoder
    return genome


def test_crossover_encoder_decoder_blends_trainable_modules() -> None:
    """``crossover_encoder_decoder`` runs simplex crossover on trainable coders."""

    torch.manual_seed(4)
    encoders = [_linear_module("encoder", n_inputs=4, n_outputs=3) for _ in range(3)]
    decoders = [_linear_module("decoder", n_inputs=4, n_outputs=3) for _ in range(3)]

    encoder_snapshots = [_snapshot(e) for e in encoders]
    decoder_snapshots = [_snapshot(d) for d in decoders]

    parents = [_genome_with_coders(i, encoders[i], decoders[i]) for i in range(3)]

    r = 0.5
    child_encoder, child_decoder = crossover_encoder_decoder(parents, r)

    expected_encoder = _expected_simplex(encoder_snapshots, r)
    for name, expected_tensor in expected_encoder.items():
        assert torch.allclose(
            child_encoder.state_dict()[name], expected_tensor, atol=1e-6
        ), name

    expected_decoder = _expected_simplex(decoder_snapshots, r)
    for name, expected_tensor in expected_decoder.items():
        assert torch.allclose(
            child_decoder.state_dict()[name], expected_tensor, atol=1e-6
        ), name


def test_crossover_encoder_decoder_copies_non_trainable_coders() -> None:
    """Non-trainable coders are copied from the primary rather than blended."""

    identity_encoders = [IdentityEncoder(n_inputs=4, n_outputs=4) for _ in range(3)]
    clipped_decoders = [ClippedDecoder(n_inputs=4, n_outputs=2) for _ in range(3)]
    parents = [
        _genome_with_coders(i, identity_encoders[i], clipped_decoders[i])
        for i in range(3)
    ]

    child_encoder, child_decoder = crossover_encoder_decoder(parents, r=0.5)

    # IdentityEncoder.copy()/ClippedDecoder.copy() return themselves (stateless),
    # so the child coders come from the primary parent, not the others.
    assert child_encoder is parents[0].encoder
    assert child_decoder is parents[0].decoder
