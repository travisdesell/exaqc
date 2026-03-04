from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np


try:
    from src.circuits.circuit import CircuitGenome
except Exception:
    CircuitGenome = Any


# ============================
# Specs / Config
# ============================

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
    objectives: List[ObjectiveSpec]

    eps: float = 1e-12

    # parameter-guided preference score (soft tie-break only)
    preference_weights: Optional[Dict[str, float]] = None

    # optional complexity as preference key "complexity"
    complexity_fn: Optional[Callable[[CircuitGenome], float]] = None


# ============================
# Core utilities
# ============================

def _get_metric(genome: CircuitGenome, key: str, default: float = float("inf")) -> float:
    fit = getattr(genome, "fitness", None) or {}
    v = fit.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def objective_vector(genome: CircuitGenome, cfg: MOOConfig) -> np.ndarray:
    """
    Always returns minimization vector.
    Max objectives are converted by negation.
    """
    vals: List[float] = []
    for obj in cfg.objectives:
        v = _get_metric(genome, obj.key)
        if not math.isfinite(v):
            v = float("inf")
        vals.append(v if obj.minimize else -v)
    return np.asarray(vals, dtype=np.float64)


def build_objective_matrix(pop: Sequence[CircuitGenome], cfg: MOOConfig) -> np.ndarray:
    return np.stack([objective_vector(g, cfg) for g in pop], axis=0)


def normalize_objectives(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mins = np.min(F, axis=0)
    maxs = np.max(F, axis=0)
    denom = (maxs - mins) + eps
    return (F - mins) / denom


def dominates(a: np.ndarray, b: np.ndarray, eps: float = 0.0) -> bool:
    le = np.all(a <= b + eps)
    lt = np.any(a < b - eps)
    return bool(le and lt)


def fast_nondominated_sort(F: np.ndarray, eps: float = 0.0) -> List[List[int]]:
    """
    Deb et al. fast non-dominated sorting.
    Returns fronts as lists of indices into F.
    """
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=np.int32)
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(F[p], F[q], eps=eps):
                S[p].append(q)
            elif dominates(F[q], F[p], eps=eps):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def preference_score(
    genome: CircuitGenome,
    *,
    cfg: MOOConfig,
    F_norm_row: np.ndarray,
) -> float:
    """
    Soft guided score used as a tie-breaker within same-front decisions.
    Lower is better.
    """
    if not cfg.preference_weights:
        return 0.0

    key_to_val: Dict[str, float] = {}
    for i, obj in enumerate(cfg.objectives):
        key_to_val[obj.key] = float(F_norm_row[i])

    if cfg.complexity_fn is not None:
        key_to_val["complexity"] = float(cfg.complexity_fn(genome))

    s = 0.0
    for k, w in cfg.preference_weights.items():
        v = key_to_val.get(k, None)
        if v is None:
            continue
        s += float(w) * float(v)
    return float(s)


# ============================
# Abstract interface
# ============================

@dataclass
class MOORank:
    rank: int
    crowding: float = 0.0      # used by NSGA-II
    pref_score: float = 0.0    # optional guided tie-break


class MultiObjectiveSelector(ABC):
    """
    Abstract base for MOO survivor selection.
    """

    def __init__(self, cfg: MOOConfig):
        self.cfg = cfg

    def objective_matrix(self, population: Sequence[CircuitGenome]) -> np.ndarray:
        return build_objective_matrix(population, self.cfg)

    def fronts(self, population: Sequence[CircuitGenome]) -> List[List[int]]:
        F = self.objective_matrix(population)
        return fast_nondominated_sort(F, eps=self.cfg.eps)

    @abstractmethod
    def rank(self, population: Sequence[CircuitGenome]) -> Dict[int, MOORank]:
        """
        Return idx->MOORank metadata for selection.
        """

    @abstractmethod
    def select_survivors(
        self,
        population: Sequence[CircuitGenome],
        max_population_size: int,
    ) -> List[CircuitGenome]:
        """
        Truncate population to max size using algorithm.
        """

    def steady_state_insert(
        self,
        population: List[CircuitGenome],
        child: CircuitGenome,
        max_population_size: int,
    ) -> List[CircuitGenome]:
        """
        Minimal-change steady-state update: select survivors from pop + child.
        """
        combined = list(population) + [child]
        return self.select_survivors(combined, max_population_size=max_population_size)


# ============================
# NSGA-II
# ============================

def crowding_distance(F: np.ndarray, front: List[int], eps: float = 1e-12) -> Dict[int, float]:
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

    def rank(self, population: Sequence[CircuitGenome]) -> Dict[int, MOORank]:
        pop = list(population)
        F = self.objective_matrix(pop)
        fronts = fast_nondominated_sort(F, eps=self.cfg.eps)
        F_norm = normalize_objectives(F, eps=self.cfg.eps)

        ranks: Dict[int, MOORank] = {}
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
    ) -> List[CircuitGenome]:
        pop = list(population)
        if len(pop) <= max_population_size:
            return pop

        ranks = self.rank(pop)
        fronts = self.fronts(pop)

        def sort_key(i: int) -> Tuple[int, float, float]:
            r = ranks[i]
            # rank asc, crowding desc, pref asc
            return (r.rank, -r.crowding, r.pref_score)

        survivors_idx: List[int] = []
        for front in fronts:
            if len(survivors_idx) + len(front) <= max_population_size:
                survivors_idx += front
            else:
                front_sorted = sorted(front, key=sort_key)
                survivors_idx += front_sorted[: (max_population_size - len(survivors_idx))]
                break

        return [pop[i] for i in survivors_idx]


# ============================
# NSGA-III
# ============================

def das_dennis_reference_directions(m: int, H: int) -> np.ndarray:
    """
    Das–Dennis reference directions on simplex.
    """
    dirs: List[List[int]] = []

    def rec(left: int, k: int, prefix: List[int]) -> None:
        if k == m - 1:
            dirs.append(prefix + [left])
            return
        for i in range(left + 1):
            rec(left - i, k + 1, prefix + [i])

    rec(H, 0, [])
    W = np.asarray(dirs, dtype=np.float64) / float(H)
    return W


def associate_to_ref_dirs(Z: np.ndarray, ref_dirs: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
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
    chosen: List[int],
    last_front: List[int],
    F_norm: np.ndarray,
    ref_dirs: np.ndarray,
    n_to_pick: int,
    eps: float = 1e-12,
) -> List[int]:
    if n_to_pick <= 0:
        return []

    all_idx = chosen + last_front
    Z = F_norm[all_idx, :]
    assoc, dist = associate_to_ref_dirs(Z, ref_dirs, eps=eps)

    niche_count = np.zeros(ref_dirs.shape[0], dtype=np.int32)
    for k, idx in enumerate(all_idx):
        if idx in chosen:
            niche_count[assoc[k]] += 1

    cand_assoc: Dict[int, int] = {}
    cand_dist: Dict[int, float] = {}
    for k, idx in enumerate(all_idx):
        if idx in last_front:
            cand_assoc[idx] = int(assoc[k])
            cand_dist[idx] = float(dist[k])

    remaining = set(last_front)
    picked: List[int] = []

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

    def rank(self, population: Sequence[CircuitGenome]) -> Dict[int, MOORank]:
        """
        NSGA-III primarily uses ranks for fronts; selection uses niching.
        We still provide pref_score for optional parameter guidance.
        """
        pop = list(population)
        F = self.objective_matrix(pop)
        fronts = fast_nondominated_sort(F, eps=self.cfg.eps)
        F_norm = normalize_objectives(F, eps=self.cfg.eps)

        ranks: Dict[int, MOORank] = {}
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
    ) -> List[CircuitGenome]:
        pop = list(population)
        if len(pop) <= max_population_size:
            return pop

        F = self.objective_matrix(pop)
        fronts = fast_nondominated_sort(F, eps=self.cfg.eps)
        F_norm = normalize_objectives(F, eps=self.cfg.eps)

        m = F.shape[1]
        ref_dirs = das_dennis_reference_directions(m=m, H=self.divisions)

        survivors_idx: List[int] = []
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