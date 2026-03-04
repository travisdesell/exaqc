from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from src.circuits.circuit import CircuitGenome
from src.moo.config import MOOConfig, MOORank
from src.moo.utils import build_objective_matrix, fast_nondominated_sort


class MultiObjectiveSelector(ABC):
    """
    Abstract base for MOO survivor selection.
    """

    def __init__(self, cfg: MOOConfig):
        self.cfg = cfg

    def objective_matrix(self, population: Sequence[CircuitGenome]) -> np.ndarray:
        return build_objective_matrix(population, self.cfg)

    def fronts(self, population: Sequence[CircuitGenome]) -> list[list[int]]:
        F = self.objective_matrix(population)
        return fast_nondominated_sort(F, eps=self.cfg.eps)

    @abstractmethod
    def rank(self, population: Sequence[CircuitGenome]) -> dict[int, MOORank]:
        """
        Return idx->MOORank metadata for selection.
        """

    @abstractmethod
    def select_survivors(
        self,
        population: Sequence[CircuitGenome],
        max_population_size: int,
    ) -> list[CircuitGenome]:
        """
        Truncate population to max size using algorithm.
        """

    def steady_state_insert(
        self,
        population: list[CircuitGenome],
        child: CircuitGenome,
        max_population_size: int,
    ) -> list[CircuitGenome]:
        """
        Minimal-change steady-state update: select survivors from pop + child.
        """
        combined = list(population) + [child]
        return self.select_survivors(combined, max_population_size=max_population_size)
