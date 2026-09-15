"""Unit tests for structural historical-innovation distance."""

from __future__ import annotations

import math

import pytest

from src.evolution.historical_distance import excess_disjoint, historical_distance


@pytest.mark.parametrize(
    ("h1", "h2", "expected"),
    [
        ([], [], (0, 0, 1)),
        ([1, 2, 5], [1, 2], (1, 0, 3)),
        ([1, 3, 5], [2, 4, 5], (0, 4, 3)),
        ([1, 2], [3, 4], (2, 2, 2)),
    ],
)
def test_excess_disjoint_table(
    h1: list[int], h2: list[int], expected: tuple[int, int, int]
) -> None:
    """The documented (E, D, N) table is exact."""

    assert excess_disjoint(h1, h2) == expected


def test_excess_disjoint_ignores_order_and_repeats() -> None:
    """Set normalization makes order and repeated IDs irrelevant."""

    assert excess_disjoint([5, 1, 2, 2], [2, 1]) == excess_disjoint([1, 2, 5], [1, 2])


def test_historical_distance_symmetry_and_identity() -> None:
    """Distance is symmetric and zero on identical histories."""

    h1 = [1, 3, 5]
    h2 = [2, 4, 5]
    assert historical_distance(h1, h1) == 0.0
    assert historical_distance(h1, h2) == historical_distance(h2, h1)
    assert historical_distance(h1, h2) == pytest.approx(4 / 3)


def test_unit_coefficients_match_table() -> None:
    """Unit coefficients recover the documented distances."""

    assert historical_distance([], []) == 0.0
    assert historical_distance([1, 2, 5], [1, 2]) == pytest.approx(1 / 3)
    assert historical_distance([1, 2], [3, 4]) == pytest.approx(2.0)


def test_join_is_strict() -> None:
    """Compatibility uses a strict less-than threshold."""

    distance = historical_distance([1, 2], [3, 4])
    assert not (distance < 2.0)
    assert distance < 2.0 + 1e-12


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_coefficients_raise(bad: float) -> None:
    """Negative, NaN, and infinite coefficients are rejected."""

    with pytest.raises(ValueError, match="excess_coefficient"):
        historical_distance([1], [2], excess_coefficient=bad)
    with pytest.raises(ValueError, match="disjoint_coefficient"):
        historical_distance([1], [2], disjoint_coefficient=bad)
