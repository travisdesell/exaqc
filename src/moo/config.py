from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from src.circuits.circuit import CircuitGenome


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    key: fitness dict key
    minimize: True if lower is better, False if higher is better (will be negated)
    """
    key: str
    minimize: bool = True


@dataclass
class MOOConfig:
    objectives: list[ObjectiveSpec]

    eps: float = 1e-12

    # parameter-guided preference score (soft tie-break only)
    preference_weights: Optional[dict[str, float]] = None

    # optional complexity as preference key "complexity"
    complexity_fn: Optional[Callable[[CircuitGenome], float]] = None

@dataclass
class MOORank:
    rank: int
    crowding: float = 0.0      # used by NSGA-II
    pref_score: float = 0.0    # optional guided tie-break
