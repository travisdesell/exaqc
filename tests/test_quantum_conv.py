"""Tests for the EXAQC quantum convolution encoder and decoder."""

from __future__ import annotations

import copy

import pytest
import torch

from src.circuits.encoder import QuantumConvEncoder
from src.circuits.decoder import QuantumConvDecoder


@pytest.fixture
def cifar_conv_blocks():
    """Returns the two classical blocks preceding the quantum convolution."""
    return [
        {
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "batch_norm": True,
            "activation": "relu",
            "pool": {
                "type": "max",
                "kernel_size": 2,
                "stride": 2,
            },
        },
        {
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "batch_norm": True,
            "activation": "relu",
            "pool": {
                "type": "max",
                "kernel_size": 2,
                "stride": 2,
            },
        },
    ]


@pytest.fixture
def quantum_conv_encoder(cifar_conv_blocks):
    """Creates the CIFAR-10 quantum convolution encoder."""
    return QuantumConvEncoder(
        n_inputs=3 * 32 * 32,
        n_outputs=8,
        input_channels=3,
        input_height=32,
        input_width=32,
        conv_blocks=cifar_conv_blocks,
        compressed_channels=2,
        patch_size=2,
        patch_stride=2,
        output_activation="tanh",
    )


@pytest.fixture
def quantum_conv_decoder():
    """Creates the CIFAR-10 quantum convolution decoder."""
    return QuantumConvDecoder(
        n_inputs=8,
        n_outputs=10,
        feature_height=4,
        feature_width=4,
        out_channels=128,
        adaptive_pool_size=(2, 2),
        fully_connected_layers=(256, 128),
    )


def test_quantum_conv_encoder_output_shape(
    quantum_conv_encoder,
):
    """Encoder should convert each image into sixteen 8-value patches."""
    batch_size = 4

    inputs = torch.randn(
        batch_size,
        3,
        32,
        32,
    )

    outputs = quantum_conv_encoder(inputs)

    # Conv1 + pool:
    # 32x32 -> 16x16
    #
    # Conv2 + pool:
    # 16x16 -> 8x8
    #
    # 2x2 patches with stride 2:
    # 8x8 -> 4x4 = 16 patches/image
    #
    # Each patch:
    # 2 channels * 2 * 2 = 8 values.
    assert outputs.shape == (
        batch_size * 16,
        8,
    )


def test_quantum_conv_encoder_supports_batch_size_one(
    quantum_conv_encoder,
):
    """Encoder should preserve correct patch shape for one image."""
    inputs = torch.randn(
        1,
        3,
        32,
        32,
    )

    outputs = quantum_conv_encoder(inputs)

    assert outputs.shape == (16, 8)


def test_quantum_conv_encoder_output_is_finite(
    quantum_conv_encoder,
):
    """Encoder should not produce NaN or infinite values."""
    inputs = torch.randn(
        3,
        3,
        32,
        32,
    )

    outputs = quantum_conv_encoder(inputs)

    assert torch.isfinite(outputs).all()


def test_quantum_conv_encoder_tanh_bounds(
    quantum_conv_encoder,
):
    """Tanh encoding should restrict quantum inputs to [-1, 1]."""
    inputs = torch.randn(
        2,
        3,
        32,
        32,
    ) * 100.0

    outputs = quantum_conv_encoder(inputs)

    assert torch.all(outputs <= 1.0)
    assert torch.all(outputs >= -1.0)


def test_quantum_conv_encoder_rejects_wrong_image_shape(
    quantum_conv_encoder,
):
    """Encoder should reject image tensors with incorrect dimensions."""
    inputs = torch.randn(
        2,
        1,
        28,
        28,
    )

    with pytest.raises(
        ValueError,
        match="Expected image shape",
    ):
        quantum_conv_encoder(inputs)


def test_quantum_conv_encoder_rejects_non_image_input(
    quantum_conv_encoder,
):
    """Encoder should require a four-dimensional image tensor."""
    inputs = torch.randn(
        4,
        3072,
    )

    with pytest.raises(
        ValueError,
        match="expects",
    ):
        quantum_conv_encoder(inputs)


def test_quantum_conv_encoder_patch_size_matches_quantum_inputs(
    cifar_conv_blocks,
):
    """Patch feature count must match the quantum input dimension."""
    # compressed_channels=2 and patch_size=2 gives:
    #
    # 2 * 2 * 2 = 8
    #
    # Therefore n_outputs=6 is invalid.
    with pytest.raises(
        ValueError,
        match="quantum circuit expects",
    ):
        QuantumConvEncoder(
            n_inputs=3 * 32 * 32,
            n_outputs=6,
            input_channels=3,
            input_height=32,
            input_width=32,
            conv_blocks=cifar_conv_blocks,
            compressed_channels=2,
            patch_size=2,
            patch_stride=2,
        )


def test_quantum_conv_encoder_has_correct_compression_layer(
    quantum_conv_encoder,
):
    """The pre-quantum projection should compress 64 channels to 2."""
    layer = quantum_conv_encoder.channel_compression

    assert layer.in_channels == 64
    assert layer.out_channels == 2
    assert layer.kernel_size == (1, 1)


def test_quantum_conv_encoder_is_trainable(
    quantum_conv_encoder,
):
    """Classical projection parameters should receive gradients."""
    inputs = torch.randn(
        2,
        3,
        32,
        32,
    )

    outputs = quantum_conv_encoder(inputs)

    loss = outputs.sum()
    loss.backward()

    weight_gradient = (
        quantum_conv_encoder
        .channel_compression
        .weight
        .grad
    )

    assert weight_gradient is not None
    assert torch.isfinite(weight_gradient).all()


def test_quantum_conv_decoder_output_shape(
    quantum_conv_decoder,
):
    """Decoder should reconstruct patches and produce class logits."""
    batch_size = 4
    n_patches = 16

    quantum_outputs = torch.randn(
        batch_size * n_patches,
        8,
    )

    logits = quantum_conv_decoder(
        quantum_outputs,
        genome=None,
    )

    assert logits.shape == (
        batch_size,
        10,
    )


def test_quantum_conv_decoder_supports_batch_size_one(
    quantum_conv_decoder,
):
    """Decoder should support one original image."""
    quantum_outputs = torch.randn(
        16,
        8,
    )

    logits = quantum_conv_decoder(
        quantum_outputs,
        genome=None,
    )

    assert logits.shape == (1, 10)


def test_quantum_conv_decoder_rejects_invalid_patch_count(
    quantum_conv_decoder,
):
    """Patch count must be divisible by patches per image."""
    # Decoder expects:
    #
    # feature_height * feature_width
    # = 4 * 4
    # = 16 patches/image.
    #
    # 17 cannot correspond to a complete image batch.
    quantum_outputs = torch.randn(
        17,
        8,
    )

    with pytest.raises(
        ValueError,
        match="not divisible",
    ):
        quantum_conv_decoder(
            quantum_outputs,
            genome=None,
        )


def test_quantum_conv_decoder_has_correct_expansion_layer(
    quantum_conv_decoder,
):
    """Post-quantum projection should expand 8 channels to 128."""
    layer = quantum_conv_decoder.channel_expansion

    assert layer.in_channels == 8
    assert layer.out_channels == 128
    assert layer.kernel_size == (1, 1)


def test_quantum_conv_decoder_classifier_shape(
    quantum_conv_decoder,
):
    """Classifier should preserve the original 512->256->128->10 head."""
    linear_layers = [
        module
        for module
        in quantum_conv_decoder.classifier.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    assert len(linear_layers) == 3

    assert linear_layers[0].in_features == 512
    assert linear_layers[0].out_features == 256

    assert linear_layers[1].in_features == 256
    assert linear_layers[1].out_features == 128

    assert linear_layers[2].in_features == 128
    assert linear_layers[2].out_features == 10


def test_quantum_conv_decoder_is_trainable(
    quantum_conv_decoder,
):
    """Decoder parameters should receive gradients."""
    quantum_outputs = torch.randn(
        32,
        8,
        requires_grad=True,
    )

    logits = quantum_conv_decoder(
        quantum_outputs,
        genome=None,
    )

    loss = logits.sum()
    loss.backward()

    gradient = (
        quantum_conv_decoder
        .channel_expansion
        .weight
        .grad
    )

    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_quantum_conv_encoder_copy_is_independent(
    quantum_conv_encoder,
):
    """Copied encoders should not share parameter storage."""
    copied_encoder = quantum_conv_encoder.copy()

    original_parameter = next(
        quantum_conv_encoder.parameters()
    )

    copied_parameter = next(
        copied_encoder.parameters()
    )

    assert copied_encoder is not quantum_conv_encoder

    assert (
        original_parameter.data_ptr()
        != copied_parameter.data_ptr()
    )

    assert torch.equal(
        original_parameter,
        copied_parameter,
    )


def test_quantum_conv_decoder_copy_is_independent(
    quantum_conv_decoder,
):
    """Copied decoders should not share parameter storage."""
    copied_decoder = quantum_conv_decoder.copy()

    original_parameter = next(
        quantum_conv_decoder.parameters()
    )

    copied_parameter = next(
        copied_decoder.parameters()
    )

    assert copied_decoder is not quantum_conv_decoder

    assert (
        original_parameter.data_ptr()
        != copied_parameter.data_ptr()
    )

    assert torch.equal(
        original_parameter,
        copied_parameter,
    )


def test_quantum_conv_encoder_constructor_args(
    quantum_conv_encoder,
):
    """Encoder should expose sufficient reconstruction arguments."""
    args = quantum_conv_encoder.get_constructor_args()

    assert args["input_channels"] == 3
    assert args["input_height"] == 32
    assert args["input_width"] == 32

    assert args["compressed_channels"] == 2
    assert args["patch_size"] == 2
    assert args["patch_stride"] == 2

    assert args["n_outputs"] == 8

    assert len(args["conv_blocks"]) == 2


def test_quantum_conv_does_not_modify_input(
    quantum_conv_encoder,
):
    """Encoder forward pass should not mutate the input tensor."""
    inputs = torch.randn(
        2,
        3,
        32,
        32,
    )

    original = inputs.clone()

    _ = quantum_conv_encoder(inputs)

    assert torch.equal(
        inputs,
        original,
    )


def test_quantum_conv_end_to_end_shape(
    quantum_conv_encoder,
    quantum_conv_decoder,
):
    """Tests the complete quantum-convolution tensor interface.

    A real EXAQC circuit is intentionally not used here. The identity
    operation represents an 8-input/8-output quantum circuit so this test
    isolates the encoder/decoder tensor plumbing from PennyLane.
    """
    batch_size = 3

    inputs = torch.randn(
        batch_size,
        3,
        32,
        32,
    )

    # CNN + patch extraction:
    #
    # [B, 3, 32, 32]
    # ->
    # [B * 16, 8]
    quantum_inputs = quantum_conv_encoder(
        inputs
    )

    assert quantum_inputs.shape == (
        batch_size * 16,
        8,
    )

    # Mock an eight-qubit circuit with eight expectation-value outputs.
    #
    # The actual EXAQC circuit will replace this operation.
    quantum_outputs = quantum_inputs

    # Spatial reconstruction + original dense head.
    logits = quantum_conv_decoder(
        quantum_outputs,
        genome=None,
    )

    assert logits.shape == (
        batch_size,
        10,
    )