"""Tests for ``src.utils.draw_hybrid_model.draw_hybrid_model`` and stage ``describe_layers``.

``draw_hybrid_model`` composes a classical-style architecture diagram for a genome's
hybrid model: the encoder stages, the quantum input-encoding interface block,
the quantum circuit image itself, the output-readout interface block, and the
decoder stages, laid out left to right. Each encoder/decoder describes its own
layers via ``describe_layers()``.

These tests verify (1) that each encoder/decoder reports sensible ``LayerSpec``s
and (2) that ``draw_hybrid_model`` actually **writes** a valid PNG for a variety of
circuit complexities and encoder/decoder types on both the ``pennylane`` and
``qiskit`` targets -- both with and without an embedded circuit figure. Since
``draw_hybrid_model`` catches its own failures and merely logs a warning, asserting
the file exists and is a valid PNG proves the composition succeeded.

All artifacts are written into pytest's per-test ``tmp_path`` (auto-removed).
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend before anything imports pyplot.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.circuits.encoder import initialize_encoder  # noqa: E402
from src.circuits.decoder import initialize_decoder  # noqa: E402
from src.utils.draw_hybrid_model import draw_hybrid_model  # noqa: E402
from tests.supervised_trainer_test_utils import (  # noqa: E402
    build_classification_genome,
    COMPLEXITY_LEVELS_WITH_MULTI_PARAM,
)

#: Targets whose generated circuits are exercised.
TARGETS: tuple[str, ...] = ("pennylane", "qiskit")

#: Encoder/decoder combinations exercised by the drawing tests.
ENCODER_DECODER_PAIRS: tuple[tuple[str, str], ...] = (
    ("identity", "clipped"),
    ("linear", "linear"),
)

#: The first eight bytes of any PNG file (the PNG signature).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: A small CNN configuration used to exercise the convolutional describe_layers.
_CNN_CONFIG: dict = {
    "input_channels": 3,
    "input_height": 16,
    "input_width": 16,
    "conv_blocks": [
        {
            "out_channels": 8,
            "kernel_size": 3,
            "pool": {"type": "max", "kernel_size": 2},
        },
        {"out_channels": 16, "kernel_size": 3},
    ],
    "fully_connected_layers": [32],
}


def _assert_valid_png(path) -> None:
    """Asserts that ``path`` exists, is non-empty, and is a valid PNG.

    Args:
        path: Path to the file to check.
    """
    assert path.is_file(), f"expected a PNG at {path}"
    assert path.stat().st_size > 0
    with open(path, "rb") as handle:
        assert handle.read(len(_PNG_MAGIC)) == _PNG_MAGIC


# ---------------------------------------------------------------------
# describe_layers()
# ---------------------------------------------------------------------


def test_linear_encoder_describe_layers() -> None:
    """A linear encoder reports a single fully connected layer to ``n_outputs``."""
    encoder = initialize_encoder("pennylane", "linear", n_inputs=5, n_outputs=12)
    specs = encoder.describe_layers()
    assert [spec.kind for spec in specs] == ["fc"]
    assert specs[0].out_shape == (12,)


def test_identity_encoder_describe_layers() -> None:
    """An identity encoder reports a single pass-through block."""
    encoder = initialize_encoder("pennylane", "identity", n_inputs=6, n_outputs=6)
    specs = encoder.describe_layers()
    assert [spec.kind for spec in specs] == ["identity"]


def test_cnn_encoder_describe_layers() -> None:
    """A CNN encoder reports conv/pool/flatten/fc layers ending at ``n_outputs``."""
    n_inputs = _CNN_CONFIG["input_channels"] * 16 * 16
    encoder = initialize_encoder(
        "pennylane", "cnn", n_inputs=n_inputs, n_outputs=9, config=_CNN_CONFIG
    )
    specs = encoder.describe_layers()
    kinds = [spec.kind for spec in specs]
    # two conv blocks (first has a pool), then flatten, hidden fc, output fc
    assert kinds == ["conv", "pool", "conv", "flatten", "fc", "fc"]
    assert specs[-1].out_shape == (9,)

    # Phase 2: convolution/pooling layers report exact (channels, height, width)
    # transitions traced through the real modules, and the flatten reduces the
    # final 3-D feature map to a 1-D vector.
    conv1, pool1, conv2, flatten = specs[0], specs[1], specs[2], specs[3]
    assert conv1.in_shape == (3, 16, 16) and conv1.out_shape == (8, 16, 16)
    assert pool1.in_shape == (8, 16, 16) and pool1.out_shape == (8, 8, 8)
    assert conv2.in_shape == (8, 8, 8) and conv2.out_shape == (16, 8, 8)
    # the second conv has no explicit pool; the adaptive pool (4x4) feeds flatten
    assert flatten.in_shape == (16, 4, 4)
    assert flatten.out_shape == (16 * 4 * 4,)


def test_clipped_decoder_describe_layers() -> None:
    """A clipped decoder reports a single pass-through block to ``n_outputs``."""
    decoder = initialize_decoder("pennylane", "clipped", n_inputs=4, n_outputs=3)
    specs = decoder.describe_layers()
    assert [spec.kind for spec in specs] == ["passthrough"]
    assert specs[0].out_shape == (3,)


def test_linear_decoder_describe_layers() -> None:
    """A linear decoder reports a single fully connected layer to ``n_outputs``."""
    decoder = initialize_decoder("pennylane", "linear", n_inputs=4, n_outputs=3)
    specs = decoder.describe_layers()
    assert [spec.kind for spec in specs] == ["fc"]
    assert specs[0].out_shape == (3,)


# ---------------------------------------------------------------------
# draw_hybrid_model
# ---------------------------------------------------------------------


@pytest.mark.parametrize("complexity", COMPLEXITY_LEVELS_WITH_MULTI_PARAM)
@pytest.mark.parametrize("target", TARGETS)
def test_draw_hybrid_model_writes_valid_diagram(
    target: str, complexity: str, tmp_path
) -> None:
    """``draw_hybrid_model`` writes a valid composed diagram for each circuit/target.

    Uses a placeholder for the quantum circuit (``quantum_circuit_fig=None``) so
    the test does not depend on the target-specific circuit drawing; the
    embedding path is covered separately.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        complexity: Circuit complexity level to build.
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome, _ = build_classification_genome(
        genome_number=3,
        target=target,
        complexity=complexity,
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )
    genome.initialize_model()

    draw_hybrid_model(str(tmp_path), genome, "diagram.png", quantum_circuit_fig=None)

    _assert_valid_png(tmp_path / "diagram.png")


@pytest.mark.parametrize("encoder_name,decoder_name", ENCODER_DECODER_PAIRS)
@pytest.mark.parametrize("target", TARGETS)
def test_draw_hybrid_model_handles_encoder_decoder_types(
    target: str, encoder_name: str, decoder_name: str, tmp_path
) -> None:
    """``draw_hybrid_model`` composes a valid diagram for each encoder/decoder type.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        encoder_name: Encoder to build (``"identity"`` or ``"linear"``).
        decoder_name: Decoder to build (``"clipped"`` or ``"linear"``).
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome, _ = build_classification_genome(
        genome_number=4,
        target=target,
        complexity="shallow",
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        include_parametric=True,
    )
    genome.initialize_model()

    draw_hybrid_model(str(tmp_path), genome, "diagram.png", quantum_circuit_fig=None)

    _assert_valid_png(tmp_path / "diagram.png")


def test_draw_hybrid_model_embeds_quantum_circuit_figure(tmp_path) -> None:
    """``draw_hybrid_model`` embeds a provided circuit figure into the diagram.

    Passing a real matplotlib figure exercises the rasterize-and-``imshow``
    embedding path (as ``save_circuit`` does with the drawn quantum circuit).

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome, _ = build_classification_genome(
        genome_number=5,
        target="pennylane",
        complexity="shallow",
        encoder_name="linear",
        decoder_name="linear",
        include_parametric=True,
    )
    genome.initialize_model()

    circuit_figure = plt.figure(figsize=(3, 2))
    circuit_figure.add_subplot(111).plot([0, 1, 2], [0, 1, 0])

    draw_hybrid_model(
        str(tmp_path), genome, "embedded.png", quantum_circuit_fig=circuit_figure
    )
    plt.close(circuit_figure)

    _assert_valid_png(tmp_path / "embedded.png")


def test_draw_hybrid_model_with_cnn_encoder(tmp_path) -> None:
    """``draw_hybrid_model`` composes a valid diagram for a CNN encoder.

    The CNN encoder is attached directly (``build_classification_genome`` does
    not build CNN encoders), and no model initialization is needed because the
    circuit is a placeholder -- the diagram only reads ``describe_layers`` and
    the genome's hyperparameters.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """
    genome, _ = build_classification_genome(
        genome_number=6,
        target="pennylane",
        complexity="shallow",
        encoder_name="identity",
        decoder_name="clipped",
        include_parametric=True,
    )

    n_inputs = _CNN_CONFIG["input_channels"] * 16 * 16
    genome.encoder = initialize_encoder(
        "pennylane",
        "cnn",
        n_inputs=n_inputs,
        n_outputs=genome.n_quantum_inputs(),
        config=_CNN_CONFIG,
    )

    draw_hybrid_model(
        str(tmp_path), genome, "cnn_diagram.png", quantum_circuit_fig=None
    )

    _assert_valid_png(tmp_path / "cnn_diagram.png")


@pytest.mark.parametrize("target", TARGETS)
def test_save_circuit_writes_single_composed_diagram(
    target: str, tmp_path, monkeypatch
) -> None:
    """``save_circuit`` produces exactly one composed diagram embedding the circuit.

    Exercises the real integration: ``save_circuit`` draws the target-specific
    quantum circuit and hands it to ``draw_hybrid_model``, writing a single tagged
    PNG (the composed diagram) alongside the json/txt.

    Args:
        target: Either ``"pennylane"`` or ``"qiskit"``.
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: used to ``chdir`` into ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "artifacts"

    genome, _ = build_classification_genome(
        genome_number=8,
        target=target,
        complexity="deep",
        encoder_name="linear",
        decoder_name="linear",
        include_parametric=True,
    )
    genome.metadata = {
        "best_training_metrics": {"loss": 0.1, "mean_class_accuracy": {"mean": 0.9}},
        "best_validation_metrics": {"loss": 0.2, "mean_class_accuracy": {"mean": 0.8}},
    }
    genome.initialize_model()
    genome.save_circuit(insert_type="best", out_dir=str(out_dir))

    pngs = sorted(p for p in out_dir.iterdir() if p.suffix == ".png")
    assert len(pngs) == 1
    _assert_valid_png(pngs[0])
    assert pngs[0].name.startswith("best_genome_8_")
