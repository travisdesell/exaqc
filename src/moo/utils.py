from __future__ import annotations

from typing import Sequence

import math
import numpy as np

from src.circuits.circuit import CircuitGenome
from src.moo.config import MOOConfig

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
    vals: list[float] = []
    for obj in cfg.objectives:
        v = _get_metric(genome, obj.key)
        if not math.isfinite(v):
            v = float("inf")
        vals.append(v if obj.minimize else -v)
    return np.asarray(vals, dtype=np.float64)


def build_objective_matrix(pop: Sequence[CircuitGenome], cfg: MOOConfig) -> np.ndarray:
    return np.stack([objective_vector(g, cfg) for g in pop], axis=0)


def normalize_objectives(f: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mins = np.min(f, axis=0)
    maxs = np.max(f, axis=0)
    denom = (maxs - mins) + eps
    return (f - mins) / denom


def dominates(a: np.ndarray, b: np.ndarray, eps: float = 0.0) -> bool:
    le = np.all(a <= b + eps)
    lt = np.any(a < b - eps)
    return bool(le and lt)


def fast_nondominated_sort(F: np.ndarray, eps: float = 0.0) -> list[list[int]]:
    """
    Deb et al. fast non-dominated sorting.
    Returns fronts as lists of indices into F.
    """
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=np.int32)
    fronts: list[list[int]] = [[]]

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
        Q: list[int] = []
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
    f_norm_row: np.ndarray,
) -> float:
    """
    Soft guided score used as a tie-breaker within same-front decisions.
    Lower is better.
    """
    if not cfg.preference_weights:
        return 0.0

    key_to_val: dict[str, float] = {}
    for i, obj in enumerate(cfg.objectives):
        key_to_val[obj.key] = float(f_norm_row[i])

    if cfg.complexity_fn is not None:
        key_to_val["complexity"] = float(cfg.complexity_fn(genome))

    s = 0.0
    for k, w in cfg.preference_weights.items():
        v = key_to_val.get(k, None)
        if v is None:
            continue
        s += float(w) * float(v)
    return float(s)