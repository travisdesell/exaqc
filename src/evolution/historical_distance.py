"""Structural compatibility distance on historical innovation IDs.

The distance is ``(c_E * E + c_D * D) / N`` with
``N = max(1, |H1|, |H2|)``. Excess genes are mismatched IDs greater than
``min(max(H1), max(H2))`` (an empty history has max 0). There is no
parameter/angle term.
"""

from __future__ import annotations

import math


def excess_disjoint(h1: list[int], h2: list[int]) -> tuple[int, int, int]:
    """Counts excess genes, disjoint genes, and the size normalizer.

    Inputs are treated as sets, so order and repeated IDs do not matter.

    Args:
        h1: Innovation IDs of the first genome.
        h2: Innovation IDs of the second genome.

    Returns:
        ``(E, D, N)`` where ``E`` is the excess count, ``D`` the disjoint
        count, and ``N`` the normalizer ``max(1, |H1|, |H2|)``.
    """

    left = set(h1)
    right = set(h2)
    mismatch = left ^ right
    boundary = min(max(left, default=0), max(right, default=0))
    excess = sum(1 for gene_id in mismatch if gene_id > boundary)
    disjoint = len(mismatch) - excess
    return excess, disjoint, max(1, len(left), len(right))


def historical_distance(
    h1: list[int],
    h2: list[int],
    excess_coefficient: float = 1.0,
    disjoint_coefficient: float = 1.0,
) -> float:
    """Returns the structural speciation distance.

    Args:
        h1: Innovation IDs of the first genome.
        h2: Innovation IDs of the second genome.
        excess_coefficient: Non-negative finite weight on excess genes.
        disjoint_coefficient: Non-negative finite weight on disjoint genes.

    Returns:
        ``(excess_coefficient * E + disjoint_coefficient * D) / N``.

    Raises:
        ValueError: If either coefficient is negative, NaN, or infinite.
    """

    for name, value in (
        ("excess_coefficient", excess_coefficient),
        ("disjoint_coefficient", disjoint_coefficient),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
    excess, disjoint, size = excess_disjoint(h1, h2)
    return (excess_coefficient * excess + disjoint_coefficient * disjoint) / size
