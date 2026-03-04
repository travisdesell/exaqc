from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Dict, Optional
import math
import numpy as np
from loguru import logger

# You can import CircuitGenome if available; otherwise keep it generic.
try:
    from src.circuits.circuit import CircuitGenome
except Exception:
    CircuitGenome = Any


# ============================
# Objective spec + config
# ============================

@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Defines a single objective extracted from genome.fitness.

    key:
        fitness dict key, e.g. "test_loss", "test_acc", "eval_return_mean", etc.
    minimize:
        True if lower is better (loss), False if higher is better (accuracy/return).
    """
    key: str
    minimize: bool = True


@dataclass
class MOOConfig:
    """
    Multi-objective optimization configuration.

    objectives:
        List of objective specs.
    algo:
        "nsga2" or "nsga3".
    eps:
        small epsilon for numerical stability.
    preference_weights:
        Optional weights to bias selection *after* Pareto ranking (parameter-guided).
        Example: {"test_loss": 1.0, "complexity": 0.1}
        Interpreted on normalized objective values.
    complexity_fn:
        Optional callable to compute a complexity scalar from a genome
        (e.g., number of gates, depth, enabled gates). Used as an extra objective or tie-breaker.
    """
    objectives: List[ObjectiveSpec]
    algo: str = "nsga2"
    eps: float = 1e-12

    # parameter-guided bias (soft)
    preference_weights: Optional[Dict[str, float]] = None

    # optional complexity as either objective or tie-break
    complexity_fn: Optional[Callable[[CircuitGenome], float]] = None

    # NSGA-III settings
    nsga3_divisions: int = 12     # Das-Dennis H divisions
    nsga3_use_extremes: bool = True


# ============================
# Helpers: objective vectors
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
    Convert genome fitness dict to a numeric objective vector that is ALWAYS minimization.
    If an objective is 'maximize', we convert to minimize by negating.
    """
    vals = []
    for obj in cfg.objectives:
        v = _get_metric(genome, obj.key)
        if not math.isfinite(v):
            # push invalid to worst
            v = float("inf")
        vals.append(v if obj.minimize else -v)

    # Optional: include complexity as an *extra* objective if provided AND not already included.
    if cfg.complexity_fn is not None:
        # Add it only if caller included a spec for it OR if they want implicit.
        # If you want explicit-only, remove this block.
        pass

    return np.asarray(vals, dtype=np.float64)


def build_objective_matrix(pop: Sequence[CircuitGenome], cfg: MOOConfig) -> np.ndarray:
    return np.stack([objective_vector(g, cfg) for g in pop], axis=0)


# ============================
# Pareto dominance + sorting
# ============================

def dominates(a: np.ndarray, b: np.ndarray, eps: float = 0.0) -> bool:
    """
    a dominates b (minimization) if a <= b in all objectives and a < b in at least one.
    """
    le = np.all(a <= b + eps)
    lt = np.any(a < b - eps)
    return bool(le and lt)


def fast_nondominated_sort(F: np.ndarray, eps: float = 0.0) -> List[List[int]]:
    """
    Deb et al. fast non-dominated sort.
    Returns list of fronts, each a list of indices into F.
    """
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=np.int32)
    rank = np.zeros(N, dtype=np.int32)

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
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    # last is empty
    if not fronts[-1]:
        fronts.pop()

    return fronts


# ============================
# NSGA-II crowding
# ============================

def crowding_distance(F: np.ndarray, front: List[int], eps: float = 1e-12) -> Dict[int, float]:
    """
    Compute crowding distance for individuals in a front.
    Returns dict idx->distance.
    """
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


# ============================
# Parameter-guided tie-break
# ============================

def _preference_score(genome: CircuitGenome, F_row: np.ndarray, cfg: MOOConfig, F_norm_row: np.ndarray) -> float:
    """
    Soft “parameter-guided” score used ONLY within the same front/tie situations.
    Lower is better (we treat it like a minimization scalar).
    """
    if not cfg.preference_weights:
        return 0.0

    # Build map objective_key -> normalized minimization value
    # objectives are in same order as cfg.objectives
    key_to_val = {}
    for i, obj in enumerate(cfg.objectives):
        key_to_val[obj.key] = float(F_norm_row[i])

    # Add optional complexity if user uses it as a preference key
    if cfg.complexity_fn is not None:
        key_to_val["complexity"] = float(cfg.complexity_fn(genome))

    s = 0.0
    for k, w in cfg.preference_weights.items():
        v = key_to_val.get(k, None)
        if v is None:
            continue
        s += float(w) * float(v)
    return s


def normalize_objectives(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Min-max normalize per objective.
    """
    mins = np.min(F, axis=0)
    maxs = np.max(F, axis=0)
    denom = (maxs - mins) + eps
    return (F - mins) / denom


# ============================
# NSGA-III reference directions
# ============================

def das_dennis_reference_directions(m: int, H: int) -> np.ndarray:
    """
    Generate Das–Dennis reference directions on the simplex:
        sum(w_i) = 1, w_i >= 0
    Returns array [n_dirs, m].
    """
    dirs = []

    def rec(left: int, k: int, prefix: List[int]):
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
    Associate each point z (normalized) to nearest reference direction by perpendicular distance.
    Returns (closest_ref_idx, distances).
    """
    # normalize ref dirs
    W = ref_dirs / (np.linalg.norm(ref_dirs, axis=1, keepdims=True) + eps)

    # For each z, compute distance to each ref direction:
    # d = ||z - (z·w) w||
    z_norm = Z
    proj = z_norm @ W.T  # [N, R]
    # Reconstruct projections
    proj_vecs = proj[:, :, None] * W[None, :, :]  # [N, R, m]
    diff = z_norm[:, None, :] - proj_vecs
    dists = np.linalg.norm(diff, axis=2)  # [N, R]

    closest = np.argmin(dists, axis=1)
    min_dist = dists[np.arange(Z.shape[0]), closest]
    return closest, min_dist


def nsga3_select_from_last_front(
    chosen: List[int],
    last_front: List[int],
    F_norm: np.ndarray,
    ref_dirs: np.ndarray,
    n_to_pick: int,
    eps: float = 1e-12,
) -> List[int]:
    """
    NSGA-III niching selection for the last partially-fill front.
    """
    if n_to_pick <= 0:
        return []

    # Associate all points (chosen + last_front) to ref dirs for niche counts
    all_idx = chosen + last_front
    Z = F_norm[all_idx, :]
    assoc, dist = associate_to_ref_dirs(Z, ref_dirs, eps=eps)

    # niche counts from already-chosen
    niche_count = np.zeros(ref_dirs.shape[0], dtype=np.int32)
    for i, idx in enumerate(all_idx):
        if idx in chosen:
            niche_count[assoc[i]] += 1

    # For candidates in last_front, record their association and distance
    cand_assoc = {}
    cand_dist = {}
    for i, idx in enumerate(all_idx):
        if idx in last_front:
            cand_assoc[idx] = int(assoc[i])
            cand_dist[idx] = float(dist[i])

    remaining = set(last_front)
    picked: List[int] = []

    for _ in range(n_to_pick):
        # pick niche with smallest count
        min_count = int(np.min(niche_count))
        niches = np.where(niche_count == min_count)[0]
        if niches.size == 0:
            break

        # choose one niche (deterministic: smallest id)
        niche = int(niches[0])

        # candidates in this niche
        niche_cands = [i for i in remaining if cand_assoc[i] == niche]
        if not niche_cands:
            # no candidate in that niche, mark it "unavailable" by bumping count
            niche_count[niche] += 1
            continue

        # if niche_count==0: pick closest; else pick any (closest is fine deterministic)
        niche_cands.sort(key=lambda i: cand_dist[i])
        pick = niche_cands[0]

        picked.append(pick)
        remaining.remove(pick)
        niche_count[niche] += 1

        if len(picked) >= n_to_pick:
            break

    # If still not enough, fill arbitrarily by distance
    if len(picked) < n_to_pick:
        rest = list(remaining)
        rest.sort(key=lambda i: cand_dist[i])
        picked += rest[: (n_to_pick - len(picked))]

    return picked


# ============================
# Core API: rank/sort/select
# ============================

@dataclass
class MOORank:
    rank: int
    crowding: float = 0.0
    pref_score: float = 0.0  # parameter-guided tie-breaker


def rank_population(population: Sequence[CircuitGenome], cfg: MOOConfig) -> Dict[int, MOORank]:
    """
    Compute Pareto front ranks + (NSGA-II crowding or NSGA-III association) metadata.
    Returns mapping idx -> MOORank.
    """
    F = build_objective_matrix(population, cfg)
    fronts = fast_nondominated_sort(F, eps=cfg.eps)
    F_norm = normalize_objectives(F, eps=cfg.eps)

    ranks: Dict[int, MOORank] = {}

    if cfg.algo.lower() == "nsga2":
        for r, front in enumerate(fronts):
            cd = crowding_distance(F, front, eps=cfg.eps)
            for i in front:
                ranks[i] = MOORank(rank=r, crowding=float(cd[i]))
        # preference score (soft)
        if cfg.preference_weights:
            for i, g in enumerate(population):
                ranks[i].pref_score = _preference_score(g, F[i], cfg, F_norm[i])
        return ranks

    if cfg.algo.lower() == "nsga3":
        # NSGA-III: we still compute front ranks here; actual selection uses ref dirs.
        for r, front in enumerate(fronts):
            for i in front:
                ranks[i] = MOORank(rank=r, crowding=0.0)
        if cfg.preference_weights:
            for i, g in enumerate(population):
                ranks[i].pref_score = _preference_score(g, F[i], cfg, F_norm[i])
        return ranks

    raise ValueError(f"Unknown MOO algo: {cfg.algo}")


def sort_key_nsga2(r: MOORank) -> Tuple[int, float, float]:
    """
    Lower rank is better. Higher crowding is better. Lower pref_score is better.
    We return key such that normal ascending sort works:
      (rank asc, crowding desc, pref_score asc) -> crowding desc achieved by negation.
    """
    return (r.rank, -r.crowding, r.pref_score)


def select_survivors(
    population: Sequence[CircuitGenome],
    *,
    cfg: MOOConfig,
    max_population_size: int,
) -> List[CircuitGenome]:
    """
    Select next population from a given population set (parents+offspring style).
    Works for both NSGA-II and NSGA-III.

    Returns a list of genomes (survivors) of length <= max_population_size.
    """
    pop = list(population)
    if len(pop) <= max_population_size:
        return pop

    F = build_objective_matrix(pop, cfg)
    fronts = fast_nondominated_sort(F, eps=cfg.eps)

    if cfg.algo.lower() == "nsga2":
        ranks = rank_population(pop, cfg)
        # Fill fronts, last front by crowding
        survivors: List[int] = []
        for front in fronts:
            if len(survivors) + len(front) <= max_population_size:
                survivors += front
            else:
                # sort this front by crowding (desc), preference (asc)
                front_sorted = sorted(front, key=lambda i: sort_key_nsga2(ranks[i]))
                survivors += front_sorted[: (max_population_size - len(survivors))]
                break
        return [pop[i] for i in survivors]

    if cfg.algo.lower() == "nsga3":
        # NSGA-III selection
        m = F.shape[1]
        ref_dirs = das_dennis_reference_directions(m=m, H=cfg.nsga3_divisions)
        F_norm = normalize_objectives(F, eps=cfg.eps)

        survivors: List[int] = []
        for front in fronts:
            if len(survivors) + len(front) <= max_population_size:
                survivors += front
            else:
                # Need niching on last front
                n_to_pick = max_population_size - len(survivors)

                # Parameter-guided tweak: pre-sort last front by pref_score if provided
                # so that within the same niche-distance ties you bias toward preferences.
                if cfg.preference_weights:
                    ranks = rank_population(pop, cfg)
                    front = sorted(front, key=lambda i: ranks[i].pref_score)

                picked = nsga3_select_from_last_front(
                    chosen=survivors,
                    last_front=front,
                    F_norm=F_norm,
                    ref_dirs=ref_dirs,
                    n_to_pick=n_to_pick,
                    eps=cfg.eps,
                )
                survivors += picked
                break

        return [pop[i] for i in survivors]

    raise ValueError(f"Unknown MOO algo: {cfg.algo}")


# ============================
# Convenience: steady-state insertion
# ============================

def steady_state_insert_moo(
    population: List[CircuitGenome],
    child: CircuitGenome,
    *,
    cfg: MOOConfig,
    max_population_size: int,
) -> List[CircuitGenome]:
    """
    Minimal-change steady-state update:
      new_pop = select_survivors(pop + [child], max_population_size)

    Returns the new population list.
    """
    combined = list(population) + [child]
    new_pop = select_survivors(combined, cfg=cfg, max_population_size=max_population_size)
    return new_pop