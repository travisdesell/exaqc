from __future__ import annotations

from dataclasses import dataclass

#: Rendering kinds understood by the network-architecture compositor
#: (:func:`src.utils.helpers.draw_network`). Each :class:`LayerSpec` a
#: stage produces carries one of these:
#:
#: * ``"fc"``         -- a fully connected layer, drawn as a column of nodes.
#: * ``"conv"``       -- a convolutional layer (Phase 1: a simple block).
#: * ``"pool"``       -- a pooling layer (Phase 1: a simple block).
#: * ``"flatten"``    -- a flatten/reshape step, drawn as a simple block.
#: * ``"identity"``   -- a pass-through that does not change its input.
#: * ``"passthrough"``-- a non-trainable reshaping/slicing step (e.g. clipping).
#: * ``"encoding"``   -- the classical->quantum input encoding interface block.
#: * ``"readout"``    -- the quantum->classical output readout interface block.
#: * ``"block"``      -- a generic labelled block (the base-class default).
LAYER_KINDS: tuple[str, ...] = (
    "fc",
    "conv",
    "pool",
    "flatten",
    "identity",
    "passthrough",
    "encoding",
    "readout",
    "block",
)


@dataclass(frozen=True)
class LayerSpec:
    """A drawable description of one stage layer in the architecture diagram.

    Encoders and decoders describe themselves as an ordered list of these via
    ``describe_layers()``; the compositor in :func:`src.utils.helpers.draw_network`
    turns each into a visual column. This keeps per-stage knowledge in the stage
    classes and out of the renderer.

    Attributes:
        kind: One of :data:`LAYER_KINDS`, selecting how the layer is drawn.
        label: Short human-readable name shown with the layer (e.g. ``"Linear"``
            or ``"Conv 3x3, 16ch"``).
        out_shape: The per-sample output shape of the layer (excluding the batch
            dimension), or ``None`` when a shape is not meaningful. For ``"fc"``
            layers ``out_shape[0]`` is the node count that drives how many nodes
            are drawn.
        in_shape: The per-sample input shape of the layer (excluding the batch
            dimension), or ``None`` when it is not known/meaningful. When set,
            the diagram annotates the layer with ``in -> out`` rather than just
            its output.
        annotation: Optional custom annotation text (may contain newlines) drawn
            beneath the layer in place of the shape-derived one. Used where the
            shape transition alone is misleading -- e.g. amplitude embedding,
            which pads its input to ``2**n_qubits`` amplitudes and encodes them
            into ``n_qubits`` qubits.
    """

    kind: str
    label: str
    out_shape: tuple[int, ...] | None = None
    in_shape: tuple[int, ...] | None = None
    annotation: str | None = None

    def __post_init__(self) -> None:
        """Validates ``kind``.

        Raises:
            ValueError: If ``kind`` is not one of :data:`LAYER_KINDS`.
        """
        if self.kind not in LAYER_KINDS:
            raise ValueError(
                f"Unknown LayerSpec kind {self.kind!r}; choices: {LAYER_KINDS}"
            )

    @staticmethod
    def _format_shape(shape: tuple[int, ...] | None) -> str:
        """Formats a shape tuple as ``[d0, d1, ...]`` (or ``""`` for ``None``).

        Args:
            shape: The shape tuple to format, or ``None``.

        Returns:
            A bracketed, comma-joined shape string, or the empty string.
        """
        if shape is None:
            return ""
        return "[" + ", ".join(str(dim) for dim in shape) + "]"

    def shape_text(self) -> str:
        """Returns a compact ``[d0, d1, ...]`` label for ``out_shape``.

        Returns:
            A bracketed, comma-joined shape string (e.g. ``"[4096]"``), or the
            empty string when ``out_shape`` is ``None``.
        """
        return self._format_shape(self.out_shape)

    def io_text(self) -> str:
        """Returns an ``in -> out`` shape annotation for the layer.

        Returns:
            ``"[in] -> [out]"`` when both shapes are known, ``"[out]"`` when only
            the output shape is known, or the empty string when neither is.
        """
        out_text = self._format_shape(self.out_shape)
        in_text = self._format_shape(self.in_shape)
        if in_text and out_text:
            return f"{in_text} → {out_text}"
        return out_text or in_text
