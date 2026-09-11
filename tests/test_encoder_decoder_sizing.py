"""Tests for encoder/decoder sizing validation.

Mis-sized configurations build incorrect networks, so they are rejected up
front (before any encoder/decoder is constructed) with a ``ValueError``:

* an identity encoder must have ``n_inputs == n_outputs`` (it passes its input
  through unchanged and cannot clip or resize);
* amplitude embedding must fit the encoder's values into ``2**n_input_qubits``
  amplitudes;
* an identity encoder feeding a non-amplitude mode must match the circuit's
  expected number of quantum inputs; and
* a clipped decoder cannot expand its input (``n_outputs <= n_inputs``).
"""

from __future__ import annotations

import pytest

from src.circuits.decoder import initialize_decoder
from src.circuits.encoder import initialize_encoder, validate_encoder_sizing


def test_identity_encoder_rejects_size_mismatch() -> None:
    """An identity encoder with ``n_inputs != n_outputs`` raises."""
    with pytest.raises(ValueError, match="IdentityEncoder requires"):
        initialize_encoder("pennylane", "identity", n_inputs=30, n_outputs=8)


def test_identity_encoder_accepts_matching_sizes() -> None:
    """An identity encoder with matching sizes is constructed."""
    encoder = initialize_encoder("pennylane", "identity", n_inputs=30, n_outputs=30)
    assert encoder.n_inputs == 30 and encoder.n_outputs == 30


def test_clipped_decoder_rejects_expansion() -> None:
    """A clipped decoder with ``n_outputs > n_inputs`` raises."""
    with pytest.raises(ValueError, match="ClippedDecoder requires"):
        initialize_decoder("pennylane", "clipped", n_inputs=4, n_outputs=8)


def test_clipped_decoder_accepts_reduction() -> None:
    """A clipped decoder that reduces (or preserves) size is constructed."""
    decoder = initialize_decoder("pennylane", "clipped", n_inputs=8, n_outputs=4)
    assert decoder.n_inputs == 8 and decoder.n_outputs == 4


def test_amplitude_register_too_small_rejected() -> None:
    """Amplitude embedding raises when the register cannot hold the input."""
    # 30 values cannot fit in a 4-qubit (2**4 = 16 amplitude) register.
    with pytest.raises(ValueError, match="amplitude encoding"):
        initialize_encoder(
            "pennylane",
            "identity",
            n_inputs=30,
            n_outputs=30,
            quantum_input_mode="amplitude",
            n_input_qubits=4,
        )


def test_amplitude_register_large_enough_accepted() -> None:
    """Amplitude embedding is accepted when the register is large enough."""
    # 30 values fit in an 8-qubit (2**8 = 256 amplitude) register.
    encoder = initialize_encoder(
        "pennylane",
        "identity",
        n_inputs=30,
        n_outputs=30,
        quantum_input_mode="amplitude",
        n_input_qubits=8,
    )
    assert encoder.n_inputs == 30


def test_identity_mismatch_with_quantum_inputs_rejected() -> None:
    """An identity encoder that does not match the circuit's inputs raises."""
    # u3 mode with 4 input qubits expects 3*4 = 12 quantum inputs, not 30.
    with pytest.raises(ValueError, match="quantum inputs"):
        initialize_encoder(
            "pennylane",
            "identity",
            n_inputs=30,
            n_outputs=30,
            quantum_input_mode="u3",
            n_input_qubits=4,
        )


def test_matching_identity_sizing_accepted() -> None:
    """A correctly sized identity encoder is constructed without error."""
    # ry mode with 4 input qubits expects 4 quantum inputs; feed exactly 4.
    encoder = initialize_encoder(
        "pennylane",
        "identity",
        n_inputs=4,
        n_outputs=4,
        quantum_input_mode="ry",
        n_input_qubits=4,
    )
    assert encoder.n_outputs == 4


def test_non_identity_encoder_not_size_constrained() -> None:
    """A linear encoder may legitimately map different input/output sizes."""
    # No exception: the linear encoder can resize 30 -> 12.
    validate_encoder_sizing(
        "linear", n_inputs=30, n_outputs=12, quantum_input_mode="u3", n_input_qubits=4
    )
