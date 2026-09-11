"""Renders a genome's architecture diagram and training plot to PNG bytes.

The search no longer draws an image for every genome it evaluates. Images are
rendered only when they are needed -- for the current-best genome files, and on
demand by the EXAQC dashboard -- from the same figure builders that
:meth:`~src.circuits.circuit.CircuitGenome.save_circuit` uses, so every image
looks the same wherever it was drawn.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.figure import Figure

from src.utils.draw_hybrid_model import build_hybrid_model_figure
from src.utils.training_plots import build_training_figure

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome

#: Resolution images are rendered at, matching the files ``save_circuit`` writes.
PNG_DPI = 200


def figure_to_png(figure: Figure, dpi: int = PNG_DPI) -> bytes:
    """Renders a matplotlib figure to PNG bytes and closes it.

    Args:
        figure: The figure to render.
        dpi: Resolution to render at.

    Returns:
        The PNG file contents.
    """

    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=dpi)
    finally:
        plt.close(figure)
    return buffer.getvalue()


def render_diagram_png(genome: CircuitGenome) -> bytes | None:
    """Renders a genome's architecture diagram, with its quantum circuit embedded.

    If the quantum circuit itself cannot be drawn, the diagram is still
    rendered with a placeholder in the circuit's place.

    Args:
        genome: The genome to draw; its model is generated if it has none yet.

    Returns:
        The PNG bytes, or ``None`` if the diagram could not be drawn (the
        failure is logged).
    """

    genome_number = getattr(genome, "genome_number", "?")

    circuit_figure: Figure | None = None
    try:
        circuit_figure = genome.draw_circuit_figure()
    except Exception as error:
        logger.warning(
            "Could not draw the quantum circuit of genome {}: {}", genome_number, error
        )

    try:
        return figure_to_png(
            build_hybrid_model_figure(genome, quantum_circuit_fig=circuit_figure)
        )
    except Exception as error:
        logger.warning(
            "Could not draw the architecture diagram of genome {}: {}",
            genome_number,
            error,
        )
        return None
    finally:
        if circuit_figure is not None:
            plt.close(circuit_figure)


def render_training_png(genome: CircuitGenome) -> bytes | None:
    """Renders a genome's per-epoch or per-episode training history.

    Args:
        genome: The trained genome whose metadata holds its training history.

    Returns:
        The PNG bytes, or ``None`` if the genome recorded no training metrics or
        the plot could not be drawn (a failure is logged).
    """

    try:
        figure = build_training_figure(genome)
    except Exception as error:
        logger.warning(
            "Could not draw the training plot of genome {}: {}",
            getattr(genome, "genome_number", "?"),
            error,
        )
        return None

    if figure is None:
        return None
    return figure_to_png(figure)
