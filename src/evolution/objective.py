from __future__ import annotations

import os
import platform
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from src.circuits.circuit import CircuitGenome


class Objective(ABC):

    @abstractmethod
    def __call__(self, genome: CircuitGenome):
        """
        Uses this objective function to evaluatate the provided genome. When completed this method
        should set the `fitness` attribute of the genome to a dictionary with key value pairs
        where the key is the name of the loss, and the value is the loss value.  This allows genomes
        to have multiple loss functions for multiple objectives.
        """
        pass


def evaluate_genome(
    objective: Objective | Callable[[CircuitGenome], None],
    genome: CircuitGenome,
    rank: int = 0,
) -> None:
    """Evaluates a genome and records how long it took and where it ran.

    This is the one place a search evaluates genomes -- the serial run and every
    MPI worker call it -- so every task records the same timing and placement
    without its objective or trainer having to. Timestamps are wall-clock seconds
    on the evaluating host, so they compare reliably only with other timestamps
    from the same host; ``evaluation_seconds`` is measured with a monotonic clock
    and is reliable everywhere.

    Args:
        objective: The objective that trains and evaluates the genome.
        genome: The genome to evaluate.
        rank: The MPI rank evaluating it (0 for a serial run).

    Returns:
        None. The objective sets the genome's fitness; this adds
        ``evaluation_started_at``, ``evaluation_finished_at`` and
        ``evaluation_seconds`` to its ``timing`` metadata, and sets
        ``evaluated_by`` to the evaluating ``rank``, ``host`` and ``pid``.
    """

    started_at = time.time()
    started = time.perf_counter()
    objective(genome)
    seconds = time.perf_counter() - started

    # written after the objective runs, so an objective that replaces the
    # metadata cannot lose them
    timing = genome.metadata.setdefault("timing", {})
    timing["evaluation_started_at"] = started_at
    timing["evaluation_finished_at"] = time.time()
    timing["evaluation_seconds"] = seconds
    genome.metadata["evaluated_by"] = {
        "rank": int(rank),
        "host": platform.node(),
        "pid": os.getpid(),
    }
