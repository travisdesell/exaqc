"""Architecture-diagram compositor for a genome's hybrid model.

:func:`draw_hybrid_model` renders a classical-style architecture figure for a
:class:`~src.circuits.circuit.CircuitGenome`'s hybrid model -- the encoder
stages, the quantum input-encoding interface, the quantum circuit image itself
(drawn by ``save_circuit`` and passed in), the output-readout interface, and the
decoder stages -- laid out left to right. Each encoder/decoder describes itself
as a list of :class:`~src.circuits.layer_spec.LayerSpec`
(``describe_layers()``), keeping stage-specific knowledge in the stage classes
and out of this renderer.

This module was extracted from :mod:`src.utils.helpers` to keep the diagram
code (and its matplotlib dependency) separate from the general genome/training
utilities.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from loguru import logger

from src.circuits.layer_spec import LayerSpec

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome


# ---------------------------------------------------------------------
# Network-architecture diagram compositor
# ---------------------------------------------------------------------
#
# ``draw_hybrid_model`` renders a classical-style architecture figure for a genome's
# hybrid model -- the encoder stages, then the quantum input-encoding interface,
# the quantum circuit image itself (drawn by ``save_circuit`` and passed in),
# the output-readout interface, and finally the decoder stages -- laid out left
# to right. Each encoder/decoder describes itself as a list of ``LayerSpec``
# (``describe_layers()``), keeping stage-specific knowledge in the stage classes
# and out of this renderer.

#: Friendly labels for the quantum input-encoding interface block, keyed by
#: ``quantum_input_mode``.
_INPUT_MODE_LABELS: dict[str, str] = {
    "u3": "U3 Encoding",
    "rx": "RX Encoding",
    "ry": "RY Encoding",
    "rz": "RZ Encoding",
    "basis": "Basis Encoding",
    "amplitude": "Amplitude Embedding",
}

#: Friendly labels for the quantum output-readout interface block, keyed by
#: ``quantum_output_mode``.
_OUTPUT_MODE_LABELS: dict[str, str] = {
    "probs": "Probabilities",
    "expval": "Pauli-Z Expectation",
    "state": "Statevector",
}

#: Fill colors per layer kind for the block renderer.
_BLOCK_FILL: dict[str, str] = {
    "fc": "#e8eef5",
    "identity": "#ececec",
    "passthrough": "#ececec",
    "block": "#ececec",
    "flatten": "#e3e3e3",
    "conv": "#f3e7cf",
    "pool": "#e7d6f0",
    "encoding": "#d7ecef",
    "readout": "#d7ecef",
}

#: Human-readable labels for the stage-group brackets drawn beneath the figure.
_GROUP_LABELS: dict[str, str] = {
    "encoder": "Encoder",
    "quantum": "Quantum Circuit",
    "decoder": "Decoder",
}


def _encoding_layer_spec(genome: CircuitGenome) -> LayerSpec:
    """Builds the input-encoding interface block from the genome's input mode.

    Args:
        genome: The genome whose ``quantum_input_mode`` selects the label.

    Returns:
        A ``"encoding"`` :class:`LayerSpec` labelled for the input mode, with the
        quantum-input count as its shape when it can be computed.
    """
    hyperparameters = getattr(genome, "hyperparameters", {}) or {}
    mode = hyperparameters.get("quantum_input_mode", "u3")
    try:
        out_shape: tuple[int, ...] | None = (genome.n_quantum_inputs(),)
    except Exception:
        out_shape = None

    # Amplitude embedding is a reduction: it pads its input to 2**n_qubits
    # amplitudes and encodes them into n_qubits qubits. Spell that out rather
    # than showing only the padded amplitude-vector length.
    annotation: str | None = None
    if mode == "amplitude":
        try:
            n_qubits = len(genome.input_indexes)
            annotation = f"pad\n→ [{2 ** n_qubits}]\n→ [{n_qubits}] qubits"
            out_shape = None
        except Exception:
            annotation = None

    return LayerSpec(
        kind="encoding",
        label=_INPUT_MODE_LABELS.get(mode, str(mode)),
        out_shape=out_shape,
        annotation=annotation,
    )


def _readout_layer_spec(genome: CircuitGenome) -> LayerSpec:
    """Builds the output-readout interface block from the genome's output mode.

    Args:
        genome: The genome whose ``quantum_output_mode`` selects the label.

    Returns:
        A ``"readout"`` :class:`LayerSpec` labelled for the output mode, with the
        quantum-output count as its shape when it can be computed.
    """
    hyperparameters = getattr(genome, "hyperparameters", {}) or {}
    mode = hyperparameters.get("quantum_output_mode", "probs")
    try:
        out_shape: tuple[int, ...] | None = (genome.n_quantum_outputs(),)
    except Exception:
        out_shape = None
    return LayerSpec(
        kind="readout",
        label=_OUTPUT_MODE_LABELS.get(mode, str(mode)),
        out_shape=out_shape,
    )


def _figure_to_image(figure: Figure) -> np.ndarray:
    """Renders a matplotlib figure to an RGBA image array.

    Used to embed the pre-drawn quantum-circuit figure (from ``qml.draw_mpl`` or
    a qiskit ``circuit.draw``) inside the composed architecture figure via
    ``imshow``.

    Args:
        figure: The matplotlib figure to rasterize.

    Returns:
        An ``(H, W, 4)`` float RGBA image array.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    image = plt.imread(buffer)
    buffer.close()
    return image


#: Geometry (in axes fraction) shared by tensor and transform boxes so columns
#: line up: (x0, width, y0, height).
_BOX_GEOMETRY: tuple[float, float, float, float] = (0.30, 0.40, 0.10, 0.80)

#: Maximum number of nodes drawn in a tensor box before it is truncated to a
#: few top/bottom nodes plus a vertical ellipsis (large tensors stay legible).
_TENSOR_MAX_NODES = 9


def _shape_text(shape: tuple[int, ...] | None) -> str:
    """Formats a shape tuple as ``[d0, d1, ...]`` (or ``""`` for ``None``).

    Args:
        shape: The shape tuple to format, or ``None``.

    Returns:
        A bracketed, comma-joined shape string, or the empty string.
    """
    if shape is None:
        return ""
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def _draw_tensor_nodes(ax: plt.Axes, n_nodes: int | None) -> None:
    """Draws the column of open circles representing a tensor's units.

    Large tensors are truncated to a few top and bottom nodes plus a vertical
    ellipsis so a tensor with thousands of units stays legible (matching the
    reference VGG-style diagram). An unknown size draws the truncated pattern.

    Args:
        ax: The tensor-box axes (``0..1`` limits, axis off) to draw into.
        n_nodes: The (leading) number of units, or ``None`` when unknown.
    """
    _x0, _width, y0, height = _BOX_GEOMETRY
    pad = 0.06
    bottom, top = y0 + pad, y0 + height - pad

    node_style = {
        "marker": "o",
        "linestyle": "None",
        "markersize": 8,
        "markerfacecolor": "white",
        "markeredgecolor": "0.4",
        "markeredgewidth": 1.1,
    }

    if n_nodes is not None and n_nodes <= _TENSOR_MAX_NODES:
        ys = np.linspace(bottom, top, max(int(n_nodes), 1))
        ax.plot([0.5] * len(ys), ys, **node_style)
        return

    span = top - bottom
    top_ys = np.linspace(top - span * 0.30, top, 4)
    bottom_ys = np.linspace(bottom, bottom + span * 0.30, 3)
    ax.plot([0.5] * len(top_ys), top_ys, **node_style)
    ax.plot([0.5] * len(bottom_ys), bottom_ys, **node_style)
    ax.text(
        0.5,
        (bottom + top) / 2.0,
        "⋮",
        ha="center",
        va="center",
        fontsize=14,
        color="0.4",
    )


#: Face fill colors for a 3-D feature-map tensor block (front, top, right), the
#: darker side/top giving the block a shaded volumetric look.
_TENSOR3D_FRONT = "#cfe0ef"
_TENSOR3D_TOP = "#e2edf6"
_TENSOR3D_SIDE = "#a9c4de"


def _draw_tensor_box_3d(
    ax: plt.Axes,
    shape: tuple[int, ...],
    scale_ctx: tuple[float, float] | None,
    title: str | None = None,
) -> None:
    """Renders a 3-D ``(channels, height, width)`` feature-map tensor as a block.

    The block's front face is scaled by the map's spatial size and its depth by
    the channel count -- both relative to the largest feature map in the diagram
    -- so activations visibly shrink spatially and grow in depth as channels
    increase (matching the reference convolutional-architecture style).

    Args:
        ax: The column axes to draw into.
        shape: The ``(channels, height, width)`` tensor shape.
        scale_ctx: ``(max_spatial, max_channels)`` across the diagram's 3-D
            tensors, used to scale this block. ``None`` falls back to a fixed
            mid-size block.
        title: Optional heading drawn above the block.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title:
        ax.text(0.5, 1.03, title, ha="center", va="bottom", fontsize=9, color="teal")

    channels, map_h, map_w = int(shape[0]), int(shape[1]), int(shape[2])

    if scale_ctx is not None:
        max_spatial, max_channels = scale_ctx
        spatial_scale = max(map_h, map_w) / max_spatial if max_spatial else 1.0
        channel_scale = (
            np.log(channels + 1) / np.log(max_channels + 1) if max_channels > 1 else 1.0
        )
    else:
        spatial_scale, channel_scale = 0.7, 0.7

    spatial_scale = float(np.clip(spatial_scale, 0.35, 1.0))
    channel_scale = float(np.clip(channel_scale, 0.25, 1.0))

    # Front face sized by the spatial dims (preserving aspect), depth by
    # channels. Kept narrow horizontally so the diagram stays compact.
    longest = max(map_h, map_w)
    front_w = 0.30 * spatial_scale * (map_w / longest)
    front_h = 0.62 * spatial_scale * (map_h / longest)
    depth = 0.07 + 0.15 * channel_scale
    dx, dy = depth * 0.55, depth

    # Center the whole block (front + depth offset) in the axes.
    x0 = 0.5 - (front_w + dx) / 2.0
    y0 = 0.5 - (front_h + dy) / 2.0

    top_face = Polygon(
        [
            (x0, y0 + front_h),
            (x0 + front_w, y0 + front_h),
            (x0 + front_w + dx, y0 + front_h + dy),
            (x0 + dx, y0 + front_h + dy),
        ],
        closed=True,
        facecolor=_TENSOR3D_TOP,
        edgecolor="0.4",
        linewidth=1.0,
    )
    side_face = Polygon(
        [
            (x0 + front_w, y0),
            (x0 + front_w, y0 + front_h),
            (x0 + front_w + dx, y0 + front_h + dy),
            (x0 + front_w + dx, y0 + dy),
        ],
        closed=True,
        facecolor=_TENSOR3D_SIDE,
        edgecolor="0.4",
        linewidth=1.0,
    )
    front_face = Polygon(
        [
            (x0, y0),
            (x0 + front_w, y0),
            (x0 + front_w, y0 + front_h),
            (x0, y0 + front_h),
        ],
        closed=True,
        facecolor=_TENSOR3D_FRONT,
        edgecolor="0.4",
        linewidth=1.0,
    )
    # Draw the back-facing top/side first, then the front on top.
    ax.add_patch(top_face)
    ax.add_patch(side_face)
    ax.add_patch(front_face)

    ax.text(
        0.5, -0.04, _shape_text(shape), ha="center", va="top", fontsize=8, color="0.3"
    )


def _draw_tensor_box(
    ax: plt.Axes,
    shape: tuple[int, ...] | None,
    title: str | None = None,
    scale_ctx: tuple[float, float] | None = None,
) -> None:
    """Renders a data-tensor column.

    Tensor boxes represent the data flowing between transform layers (the input
    tensor, every intermediate activation, and the output tensor). A 3-D
    ``(channels, height, width)`` feature map is drawn as a shaded volumetric
    block; any other tensor is drawn as a bordered column of circles (one per
    unit, truncated when large). Both are annotated below with the tensor's
    dimensions.

    Args:
        ax: The column axes to draw into.
        shape: The tensor's per-sample shape (excluding batch), or ``None`` when
            not known.
        title: Optional heading drawn above the box (used to label the model's
            ``"Input"`` and ``"Output"`` tensors).
        scale_ctx: ``(max_spatial, max_channels)`` context for scaling 3-D
            feature-map blocks relative to each other.
    """
    if shape is not None and len(shape) == 3:
        _draw_tensor_box_3d(ax, shape, scale_ctx, title=title)
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title:
        ax.text(0.5, 1.03, title, ha="center", va="bottom", fontsize=9, color="teal")

    x0, width, y0, height = _BOX_GEOMETRY
    border = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="white",
        edgecolor="0.55",
        linewidth=1.2,
    )
    ax.add_patch(border)

    _draw_tensor_nodes(ax, shape[0] if shape else None)

    dim_text = _shape_text(shape)
    if dim_text:
        ax.text(0.5, -0.04, dim_text, ha="center", va="top", fontsize=8, color="0.3")


def _draw_block(ax: plt.Axes, kind: str, label: str) -> None:
    """Draws a narrow, tall rounded block with a 90-degree-rotated label.

    Every stage layer -- encoder/decoder layers and the quantum
    input-encoding/output-readout interfaces -- is drawn with this uniform
    narrow box and a vertical label. Boxes (rather than a column of one node per
    unit) keep the diagram compact and legible even when a layer has a very large
    number of units, and give Phase 2's convolution/pooling boxes a consistent
    style to build on. The unit counts are shown as the ``in -> out`` annotation
    beneath the box by :func:`_draw_layer_column`.

    Args:
        ax: The axes (with ``0..1`` limits and axis off) to draw into.
        kind: The layer kind, selecting the fill color.
        label: The text to render (rotated) inside the block.
    """
    x0, width, y0, height = _BOX_GEOMETRY

    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=_BLOCK_FILL.get(kind, "#ececec"),
        edgecolor="0.4",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        0.5,
        y0 + height / 2.0,
        label,
        ha="center",
        va="center",
        fontsize=8,
        rotation=90,
    )


def _transform_annotation(spec: LayerSpec) -> str:
    """Builds the multi-line annotation drawn beneath a transform box.

    For a convolution it states the feature-map connectivity (input maps ->
    output maps) and the spatial size change; for a pooling layer it states the
    spatial size change. Every other transform shows its ``in -> out`` shape
    transition stacked onto two lines.

    Args:
        spec: The transform :class:`LayerSpec`.

    Returns:
        The annotation text (possibly multi-line), or ``""`` when there is none.
    """
    if spec.annotation is not None:
        return spec.annotation

    in_shape, out_shape = spec.in_shape, spec.out_shape
    spatial = in_shape is not None and out_shape is not None and len(in_shape) == 3

    if spec.kind == "conv" and spatial:
        c_in, h_in, w_in = in_shape
        c_out, h_out, w_out = out_shape
        return f"{c_in} → {c_out} maps\n{h_in}×{w_in} → {h_out}×{w_out}"

    if spec.kind == "pool" and spatial:
        _c_in, h_in, w_in = in_shape
        _c_out, h_out, w_out = out_shape
        return f"{h_in}×{w_in} → {h_out}×{w_out}"

    return spec.io_text().replace(" → ", "\n→ ")


def _draw_transform_column(ax: plt.Axes, spec: LayerSpec) -> None:
    """Renders one transform layer (encoder/decoder/interface) into its column.

    Transform layers -- the operations that map one tensor to another (linear,
    convolution, pooling, flatten, the input-encoding and output-readout
    interfaces) -- are drawn as rotated-label boxes with NO circles, annotated
    below with their ``in -> out`` size transition. The tensors they map between
    are drawn as separate tensor columns by :func:`_draw_tensor_box`.

    Args:
        ax: The column axes to draw into.
        spec: The :class:`LayerSpec` describing the transform.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_block(ax, spec.kind, spec.label)

    annotation = _transform_annotation(spec)
    if annotation:
        ax.text(0.5, -0.04, annotation, ha="center", va="top", fontsize=7, color="0.3")


def _draw_circuit_column(ax: plt.Axes, circuit_figure: Figure | None) -> None:
    """Embeds the quantum-circuit image (or a placeholder) into its column axes.

    Args:
        ax: The (wider) circuit column axes.
        circuit_figure: The pre-drawn quantum-circuit figure, or ``None`` to draw
            a labelled placeholder block instead.
    """
    if circuit_figure is None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        _draw_block(ax, "block", "Quantum\nCircuit")
        return

    image = _figure_to_image(circuit_figure)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("0.4")


def _group_ranges(
    columns: list[Column],
) -> list[tuple[str, int, int]]:
    """Collapses the per-column group tags into contiguous index ranges.

    Args:
        columns: The ordered ``(group, item)`` columns.

    Returns:
        A list of ``(group, start_index, end_index)`` tuples, one per contiguous
        run of the same group.
    """
    ranges: list[tuple[str, int, int]] = []
    for index, (group, _item) in enumerate(columns):
        if ranges and ranges[-1][0] == group:
            label, start, _end = ranges[-1]
            ranges[-1] = (label, start, index)
        else:
            ranges.append((group, index, index))
    return ranges


def _draw_arrows(figure: Figure, axes: list[plt.Axes]) -> None:
    """Draws left-to-right arrows between consecutive column axes.

    Args:
        figure: The composed figure.
        axes: The per-column axes in left-to-right order.
    """
    for left_ax, right_ax in zip(axes, axes[1:]):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        y = (left_box.y0 + left_box.y1) / 2.0
        arrow = FancyArrowPatch(
            (left_box.x1, y),
            (right_box.x0, y),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            color="0.45",
            linewidth=1.2,
        )
        figure.add_artist(arrow)


def _draw_group_brackets(
    figure: Figure,
    axes: list[plt.Axes],
    columns: list[Column],
) -> None:
    """Draws labelled brackets under each stage group (encoder/quantum/decoder).

    Args:
        figure: The composed figure.
        axes: The per-column axes in left-to-right order.
        columns: The ordered ``(group, item)`` columns.
    """
    bracket_y = 0.12
    tick_height = 0.015
    for group, start, end in _group_ranges(columns):
        x0 = axes[start].get_position().x0
        x1 = axes[end].get_position().x1
        figure.add_artist(
            Line2D(
                [x0, x1],
                [bracket_y, bracket_y],
                transform=figure.transFigure,
                color="0.3",
                linewidth=1.2,
            )
        )
        for x in (x0, x1):
            figure.add_artist(
                Line2D(
                    [x, x],
                    [bracket_y, bracket_y + tick_height],
                    transform=figure.transFigure,
                    color="0.3",
                    linewidth=1.2,
                )
            )
        figure.text(
            (x0 + x1) / 2.0,
            bracket_y - 0.03,
            _GROUP_LABELS.get(group, group),
            ha="center",
            va="top",
            fontsize=11,
            color="teal",
        )


#: Grid width ratio of the embedded quantum-circuit column relative to the
#: narrow tensor/transform columns. Large so the circuit itself gets the most
#: room.
_CIRCUIT_WIDTH_RATIO = 6.0

#: Grid width ratio for every (narrow) tensor or transform column.
_STAGE_WIDTH_RATIO = 0.6

#: Grid width ratio for a 3-D feature-map tensor block (kept close to the other
#: slim columns so the diagram stays compact).
_TENSOR3D_WIDTH_RATIO = 0.85

#: Grid width ratio for convolution/pooling/flatten transforms, whose 3-D
#: annotation labels are a little wider than a plain box.
_SPATIAL_TRANSFORM_KINDS = frozenset({"conv", "pool", "flatten"})
_SPATIAL_TRANSFORM_WIDTH_RATIO = 0.75

#: A single diagram column: ``(group, (element_kind, payload))`` where
#: ``element_kind`` is ``"tensor"`` (payload is a shape tuple or ``None``),
#: ``"transform"`` (payload is a :class:`LayerSpec`), or ``"circuit"`` (payload
#: is ``None``).
Column = tuple[str, tuple[str, "tuple[int, ...] | LayerSpec | None"]]


def _column_width_ratio(element: tuple[str, object]) -> float:
    """Returns the grid width ratio for one diagram column.

    The embedded circuit column is by far the widest. 3-D feature-map tensors
    and the convolution/pooling/flatten transforms (with their longer 3-D
    transition labels) get a little more room than the other slim columns.

    Args:
        element: The column's ``(element_kind, payload)`` pair.

    Returns:
        The gridspec width ratio for the column.
    """
    element_kind, payload = element
    if element_kind == "circuit":
        return _CIRCUIT_WIDTH_RATIO
    if element_kind == "tensor" and payload is not None and len(payload) == 3:
        return _TENSOR3D_WIDTH_RATIO
    if (
        element_kind == "transform"
        and isinstance(payload, LayerSpec)
        and payload.kind in _SPATIAL_TRANSFORM_KINDS
    ):
        return _SPATIAL_TRANSFORM_WIDTH_RATIO
    return _STAGE_WIDTH_RATIO


def _build_columns(genome: CircuitGenome) -> list[Column]:
    """Builds the ordered tensor/transform columns for a genome's diagram.

    The diagram alternates data-tensor columns with the transforms that map
    between them: the input tensor, then each encoder transform followed by the
    tensor it produces, then the quantum region (input-encoding transform, the
    circuit, output-readout transform), then the readout tensor and each decoder
    transform followed by its output tensor. This guarantees an input tensor on
    the far left and an output tensor on the far right.

    Args:
        genome: The genome whose encoder, decoder, and hyperparameters drive the
            diagram.

    Returns:
        The ordered list of :data:`Column` entries.
    """
    columns: list[Column] = []

    encoder = getattr(genome, "encoder", None)
    decoder = getattr(genome, "decoder", None)
    encoder_layers = encoder.describe_layers() if encoder is not None else []

    # Input tensor: full image shape for a CNN encoder, else the first
    # transform's input (a flat feature vector).
    if encoder is not None and hasattr(encoder, "input_channels"):
        input_shape: tuple[int, ...] | None = (
            encoder.input_channels,
            encoder.input_height,
            encoder.input_width,
        )
    elif encoder_layers and encoder_layers[0].in_shape is not None:
        input_shape = encoder_layers[0].in_shape
    elif encoder is not None:
        input_shape = (encoder.n_inputs,)
    else:
        input_shape = None
    columns.append(("encoder", ("tensor", input_shape)))

    # Each encoder transform, followed by the tensor it produces -- except the
    # last encoder output tensor, which is omitted because it carries the same
    # information as the following quantum input-encoding box.
    for index, layer in enumerate(encoder_layers):
        columns.append(("encoder", ("transform", layer)))
        if index != len(encoder_layers) - 1:
            columns.append(("encoder", ("tensor", layer.out_shape)))

    # Quantum region: input-encoding transform, circuit, output-readout
    # transform (no classical tensor columns in between). The classical tensor
    # read out of the circuit is omitted too, since the output-readout box
    # already carries the same information as the decoder's input tensor.
    columns.append(("quantum", ("transform", _encoding_layer_spec(genome))))
    columns.append(("quantum", ("circuit", None)))
    columns.append(("quantum", ("transform", _readout_layer_spec(genome))))

    # Each decoder transform, followed by the tensor it produces (the last is
    # the model's output tensor).
    if decoder is not None:
        for layer in decoder.describe_layers():
            columns.append(("decoder", ("transform", layer)))
            columns.append(("decoder", ("tensor", layer.out_shape)))
    else:
        # With no decoder, still show the readout tensor as the output tensor.
        try:
            readout_shape: tuple[int, ...] | None = (genome.n_quantum_outputs(),)
        except Exception:
            readout_shape = None
        columns.append(("decoder", ("tensor", readout_shape)))

    return columns


def _tensor_scale_context(columns: list[Column]) -> tuple[float, float] | None:
    """Computes the ``(max_spatial, max_channels)`` scaling context for 3-D maps.

    Args:
        columns: The diagram columns.

    Returns:
        ``(max_spatial, max_channels)`` over all 3-D feature-map tensors, or
        ``None`` when there are none.
    """
    spatials: list[int] = []
    channels: list[int] = []
    for _group, (element_kind, payload) in columns:
        if element_kind == "tensor" and payload is not None and len(payload) == 3:
            channel, map_h, map_w = payload
            channels.append(int(channel))
            spatials.append(int(max(map_h, map_w)))
    if not spatials:
        return None
    return float(max(spatials)), float(max(channels))


def draw_hybrid_model(
    out_dir: str,
    genome: CircuitGenome,
    output_filename: str,
    quantum_circuit_fig: Figure | None = None,
) -> None:
    """Draws a classical-style architecture diagram for a genome's hybrid model.

    The composed figure alternates data-tensor columns with the transforms that
    map between them, left to right: the input tensor, each encoder transform and
    the tensor it produces, the quantum region (input-encoding transform, the
    embedded circuit, output-readout transform), and each decoder transform and
    its output tensor. The encoder's final output tensor and the circuit's
    readout tensor are omitted because the flanking input-encoding/output-readout
    boxes already carry the same sizes. Tensor columns are drawn as boxes of
    circles annotated with their dimensions; transform columns are drawn as
    rotated-label boxes annotated with their ``in -> out`` size transition.
    Convolutional feature-map tensors (shape ``(channels, height, width)``) are
    drawn as shaded 3-D blocks scaled by their spatial size and channel depth.
    Arrows connect the columns and labelled brackets group them into Encoder /
    Quantum Circuit / Decoder.

    Args:
        out_dir: Directory to write the image into.
        genome: The genome to visualize; its ``encoder``, ``decoder`` and
            ``hyperparameters`` drive the diagram.
        output_filename: File name (within ``out_dir``) for the saved PNG.
        quantum_circuit_fig: The pre-rendered quantum-circuit matplotlib figure
            (from ``save_circuit``) to embed. If ``None``, a placeholder block is
            drawn in its place.

    Returns:
        None. Writes the composed PNG on success, or logs a warning and returns
        without writing on failure (the diagram is best-effort).
    """
    try:
        columns = _build_columns(genome)

        width_ratios = [_column_width_ratio(element) for _group, element in columns]
        figure_width = sum(width_ratios) * 0.85 + 0.8

        figure = plt.figure(figsize=(figure_width, 5.2))
        grid = figure.add_gridspec(
            1,
            len(columns),
            width_ratios=width_ratios,
            left=0.03,
            right=0.97,
            top=0.86,
            bottom=0.24,
            # Enough spacing for a visible arrow shaft (not just the head)
            # between columns; within the fixed figure width this only slims the
            # columns slightly rather than widening the diagram.
            wspace=0.4,
        )

        # The first and last columns are always the model's input and output
        # tensors; label them as such.
        last_index = len(columns) - 1
        scale_ctx = _tensor_scale_context(columns)

        axes: list[plt.Axes] = []
        for index, (_group, (element_kind, payload)) in enumerate(columns):
            ax = figure.add_subplot(grid[0, index])
            axes.append(ax)
            if element_kind == "circuit":
                _draw_circuit_column(ax, quantum_circuit_fig)
            elif element_kind == "tensor":
                if index == 0:
                    title: str | None = "Input"
                elif index == last_index:
                    title = "Output"
                else:
                    title = None
                _draw_tensor_box(ax, payload, title=title, scale_ctx=scale_ctx)
            else:
                _draw_transform_column(ax, payload)

        _draw_arrows(figure, axes)
        _draw_group_brackets(figure, axes, columns)
        figure.suptitle(
            f"Genome {genome.genome_number} Architecture",
            y=0.95,
            fontsize=13,
        )

        figure.savefig(os.path.join(out_dir, output_filename), dpi=200)
        plt.close(figure)
    except Exception as error:
        # The architecture diagram is best-effort; degrade to a concise warning
        # instead of aborting save_circuit.
        logger.warning(
            "Skipping network architecture diagram for genome {}: {}",
            getattr(genome, "genome_number", "?"),
            error,
        )
