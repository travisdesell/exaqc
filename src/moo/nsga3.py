from __future__ import annotations

from typing import Sequence
import numpy as np

from src.circuits.circuit import CircuitGenome
from src.moo.config import MOOConfig, MOORank
from src.moo.base import MultiObjectiveSelector
from src.moo.utils import (
    fast_nondominated_sort,
    normalize_objectives,
    preference_score,
    
)


def das_dennis_reference_directions(m: int, H: int) -> np.ndarray:
    """
    Das–Dennis reference directions on simplex.
    """
    dirs: list[list[int]] = []

    def rec(left: int, k: int, prefix: list[int]) -> None:
        if k == m - 1:
            dirs.append(prefix + [left])
            return
        for i in range(left + 1):
            rec(left - i, k + 1, prefix + [i])

    rec(H, 0, [])
    W = np.asarray(dirs, dtype=np.float64) / float(H)
    return W


def associate_to_ref_dirs(Z: np.ndarray, ref_dirs: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Associate each z to nearest reference direction by perpendicular distance.
    """
    W = ref_dirs / (np.linalg.norm(ref_dirs, axis=1, keepdims=True) + eps)
    proj = Z @ W.T  # [N,R]
    proj_vecs = proj[:, :, None] * W[None, :, :]  # [N,R,m]
    diff = Z[:, None, :] - proj_vecs
    dists = np.linalg.norm(diff, axis=2)
    closest = np.argmin(dists, axis=1)
    min_dist = dists[np.arange(Z.shape[0]), closest]
    return closest, min_dist


def nsga3_niching_pick(
    *,
    chosen: list[int],
    last_front: list[int],
    F_norm: np.ndarray,
    ref_dirs: np.ndarray,
    n_to_pick: int,
    eps: float = 1e-12,
) -> list[int]:
    if n_to_pick <= 0:
        return []

    all_idx = chosen + last_front
    Z = F_norm[all_idx, :]
    assoc, dist = associate_to_ref_dirs(Z, ref_dirs, eps=eps)

    niche_count = np.zeros(ref_dirs.shape[0], dtype=np.int32)
    for k, idx in enumerate(all_idx):
        if idx in chosen:
            niche_count[assoc[k]] += 1

    cand_assoc: dict[int, int] = {}
    cand_dist: dict[int, float] = {}
    for k, idx in enumerate(all_idx):
        if idx in last_front:
            cand_assoc[idx] = int(assoc[k])
            cand_dist[idx] = float(dist[k])

    remaining = set(last_front)
    picked: list[int] = []

    for _ in range(n_to_pick):
        min_count = int(np.min(niche_count))
        niches = np.where(niche_count == min_count)[0]
        if niches.size == 0:
            break
        niche = int(niches[0])

        niche_cands = [i for i in remaining if cand_assoc[i] == niche]
        if not niche_cands:
            niche_count[niche] += 1
            continue

        niche_cands.sort(key=lambda i: cand_dist[i])
        pick = niche_cands[0]
        picked.append(pick)
        remaining.remove(pick)
        niche_count[niche] += 1

        if len(picked) >= n_to_pick:
            break

    if len(picked) < n_to_pick:
        rest = list(remaining)
        rest.sort(key=lambda i: cand_dist[i])
        picked += rest[: (n_to_pick - len(picked))]

    return picked


class NSGA3Selector(MultiObjectiveSelector):
    """
    NSGA-III survivor selection: fronts + reference-direction niching for the last front.
    """

    def __init__(self, cfg: MOOConfig, *, divisions: int = 12):
        super().__init__(cfg)
        self.divisions = divisions

    def rank(self, population: Sequence[CircuitGenome]) -> dict[int, MOORank]:
        """
        NSGA-III primarily uses ranks for fronts; selection uses niching.
        We still provide pref_score for optional parameter guidance.
        """
        pop = list(population)
        F = self.objective_matrix(pop)
        fronts = fast_nondominated_sort(F, eps=self.cfg.eps)
        F_norm = normalize_objectives(F, eps=self.cfg.eps)

        ranks: dict[int, MOORank] = {}
        for r, front in enumerate(fronts):
            for i in front:
                ranks[i] = MOORank(rank=r, crowding=0.0, pref_score=0.0)

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

        F = self.objective_matrix(pop)
        fronts = fast_nondominated_sort(F, eps=self.cfg.eps)
        F_norm = normalize_objectives(F, eps=self.cfg.eps)

        m = F.shape[1]
        ref_dirs = das_dennis_reference_directions(m=m, H=self.divisions)

        survivors_idx: list[int] = []
        for front in fronts:
            if len(survivors_idx) + len(front) <= max_population_size:
                survivors_idx += front
            else:
                n_to_pick = max_population_size - len(survivors_idx)

                # parameter-guided tweak: pre-order candidates in last front by pref_score
                if self.cfg.preference_weights:
                    ranks = self.rank(pop)
                    front = sorted(front, key=lambda i: ranks[i].pref_score)

                picked = nsga3_niching_pick(
                    chosen=survivors_idx,
                    last_front=front,
                    F_norm=F_norm,
                    ref_dirs=ref_dirs,
                    n_to_pick=n_to_pick,
                    eps=self.cfg.eps,
                )
                survivors_idx += picked
                break

        return [pop[i] for i in survivors_idx]