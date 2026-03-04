from __future__ import annotations

from typing import Sequence
import numpy as np

from src.circuits.circuit import CircuitGenome
from src.moo.base import MultiObjectiveSelector
from src.moo.config import MOORank
from src.moo.utils import (
    fast_nondominated_sort, 
    normalize_objectives, 
    preference_score
)


def crowding_distance(F: np.ndarray, front: list[int], eps: float = 1e-12) -> dict[int, float]:
    m = F.shape[1]
    dist = {i: 0.0 for i in front}
    if len(front) == 0:
        return dist
    if len(front) == 1:
        dist[front[0]] = float("inf")
        return dist
    if len(front) == 2:
        dist[front[0]] = float("inf")
        dist[front[1]] = float("inf")
        return dist

    front_F = F[front, :]
    for j in range(m):
        order = np.argsort(front_F[:, j])
        f_sorted_idx = [front[k] for k in order.tolist()]

        dist[f_sorted_idx[0]] = float("inf")
        dist[f_sorted_idx[-1]] = float("inf")

        f_min = float(front_F[order[0], j])
        f_max = float(front_F[order[-1], j])
        denom = (f_max - f_min) + eps

        for k in range(1, len(front) - 1):
            prev_v = float(front_F[order[k - 1], j])
            next_v = float(front_F[order[k + 1], j])
            dist[f_sorted_idx[k]] += (next_v - prev_v) / denom

    return dist


class NSGA2Selector(MultiObjectiveSelector):
    """
    NSGA-II survivor selection (front rank + crowding distance).
    """

    def rank(self, population: Sequence[CircuitGenome]) -> dict[int, MOORank]:
        pop = list(population)
        F = self.objective_matrix(pop)
        fronts = fast_nondominated_sort(F, eps=self.cfg.eps)
        F_norm = normalize_objectives(F, eps=self.cfg.eps)

        ranks: dict[int, MOORank] = {}
        for r, front in enumerate(fronts):
            cd = crowding_distance(F, front, eps=self.cfg.eps)
            for i in front:
                ranks[i] = MOORank(rank=r, crowding=float(cd[i]), pref_score=0.0)

        if self.cfg.preference_weights:
            for i, g in enumerate(pop):
                ranks[i].pref_score = preference_score(g, cfg=self.cfg, F_norm_row=F_norm[i])

        return ranks

    def select_survivors(
        self,
        population: Sequence[CircuitGenome],
        max_population_size: int,
    ) -> list[CircuitGenome]:
        pop = list(population)
        if len(pop) <= max_population_size:
            return pop

        ranks = self.rank(pop)
        fronts = self.fronts(pop)

        def sort_key(i: int) -> tuple[int, float, float]:
            r = ranks[i]
            # rank asc, crowding desc, pref asc
            return (r.rank, -r.crowding, r.pref_score)

        survivors_idx: list[int] = []
        for front in fronts:
            if len(survivors_idx) + len(front) <= max_population_size:
                survivors_idx += front
            else:
                front_sorted = sorted(front, key=sort_key)
                survivors_idx += front_sorted[: (max_population_size - len(survivors_idx))]
                break

        return [pop[i] for i in survivors_idx]

