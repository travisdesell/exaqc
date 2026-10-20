"""NSGA-II style multi-objective population for EXAQC.

Drop-in ``PopulationStrategy``: genomes carry a ``fitness`` dict, and the
strategy ranks them on a list of keys (all minimised) by non-dominated
sorting with crowding distance (Deb et al. 2002).  Insertion is steady
state: the child joins the archive, the archive is re-ranked, and the
worst (highest rank, then lowest crowding) genome is dropped when the
archive exceeds ``max_population_size``.  Parents are chosen by binary
tournament on (rank, -crowding).

Optional constraints (``constraints={"task_loss": 0.05}``) turn a key
into a feasibility test: infeasible genomes are ranked after every
feasible one (by constraint violation), which is the standard
constrained-NSGA-II rule and is how a "the circuit must actually do the
task" requirement coexists with the hardware objectives.
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.population_strategy import PopulationStrategy


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_sort(points: List[Sequence[float]]) -> List[List[int]]:
    """Return fronts as lists of indices (front 0 = non-dominated)."""
    n = len(points)
    S = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(points[p], points[q]):
                S[p].append(q)
            elif _dominates(points[q], points[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(points: List[Sequence[float]], front: List[int]) -> Dict[int, float]:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    m = len(points[0])
    for k in range(m):
        order = sorted(front, key=lambda i: points[i][k])
        lo, hi = points[order[0]][k], points[order[-1]][k]
        dist[order[0]] = dist[order[-1]] = float("inf")
        span = hi - lo
        if span <= 0:
            continue
        for j in range(1, len(order) - 1):
            dist[order[j]] += (points[order[j + 1]][k] - points[order[j - 1]][k]) / span
    return dist


class ParetoPopulation(PopulationStrategy):
    def __init__(
        self,
        max_population_size: int,
        objective_keys: Sequence[str],
        constraints: Optional[Dict[str, float]] = None,
        out_dir: Optional[str] = None,
        log_every: int = 25,
    ):
        self.max_population_size = int(max_population_size)
        self.objective_keys = list(objective_keys)
        self.constraints = dict(constraints or {})
        self.out_dir = out_dir
        self.log_every = log_every
        self.population: List[CircuitGenome] = []
        self.rank: Dict[int, int] = {}
        self.crowd: Dict[int, float] = {}
        self.insertions = 0
        self.rejections = 0
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # PopulationStrategy interface -----------------------------------------
    def is_initializing(self) -> bool:
        return len(self.population) < self.max_population_size

    def get_best_genome(self) -> Optional[CircuitGenome]:
        """Best on the FIRST objective key among rank-0 genomes."""
        if not self.population:
            return None
        front = [g for g in self.population if self.rank.get(g.genome_number, 0) == 0]
        return min(front, key=lambda g: g.fitness[self.objective_keys[0]])

    def _tournament(self) -> CircuitGenome:
        a, b = random.sample(self.population, 2) if len(self.population) >= 2 else (self.population[0],) * 2
        ka = (self.rank[a.genome_number], -self.crowd[a.genome_number])
        kb = (self.rank[b.genome_number], -self.crowd[b.genome_number])
        return a if ka <= kb else b

    def get_parent(self, **kwargs) -> Tuple[Optional[CircuitGenome], dict]:
        if not self.population:
            return None, {}
        return self._tournament(), {}

    def get_parents(self, n_parents: int = 2, **kwargs) -> Tuple[Optional[List[CircuitGenome]], dict]:
        if len(self.population) < n_parents:
            return None, {}
        chosen: List[CircuitGenome] = []
        tries = 0
        while len(chosen) < n_parents and tries < 50 * n_parents:
            g = self._tournament()
            tries += 1
            if g not in chosen:
                chosen.append(g)
        if len(chosen) < n_parents:
            chosen = random.sample(self.population, n_parents)
        return chosen, {}

    # Ranking ---------------------------------------------------------------
    def _violation(self, g: CircuitGenome) -> float:
        v = 0.0
        for k, limit in self.constraints.items():
            v += max(0.0, float(g.fitness[k]) - float(limit))
        return v

    def _rerank(self) -> None:
        feasible = [g for g in self.population if self._violation(g) <= 0.0]
        infeasible = [g for g in self.population if self._violation(g) > 0.0]
        self.rank.clear()
        self.crowd.clear()
        n_fronts = 0
        if feasible:
            pts = [[float(g.fitness[k]) for k in self.objective_keys] for g in feasible]
            fronts = non_dominated_sort(pts)
            for r, front in enumerate(fronts):
                cd = crowding_distance(pts, front)
                for i in front:
                    self.rank[feasible[i].genome_number] = r
                    self.crowd[feasible[i].genome_number] = cd[i]
            n_fronts = len(fronts)
        # Infeasible: ranked after all feasible fronts, by violation.
        for g in sorted(infeasible, key=self._violation):
            self.rank[g.genome_number] = n_fronts + 1
            self.crowd[g.genome_number] = -self._violation(g)

    def insert_genome(self, genome: CircuitGenome, **kwargs) -> bool:
        if genome.fitness is None or any(k not in genome.fitness for k in self.objective_keys):
            logger.warning(f"genome {genome.genome_number} lacks objectives {self.objective_keys}; rejected")
            self.rejections += 1
            return False
        # Exact-duplicate objective vectors add nothing to the front -- but only
        # reject a duplicate when the incumbent is FEASIBLE and so actually on
        # the front. Rejecting infeasible duplicates starves the search whenever
        # the objectives are degenerate: with a noise model that assigns every
        # circuit the same failure probability (e.g. a noiseless or uniform
        # model), nearly every genome collides, the archive never accumulates
        # diversity, and no feasible circuit is ever found. That is an artifact
        # of this rule, not a property of the search problem.
        vec = tuple(float(genome.fitness[k]) for k in self.objective_keys)
        if self._violation(genome) <= 0.0:
            for g in self.population:
                if (self._violation(g) <= 0.0
                        and tuple(float(g.fitness[k]) for k in self.objective_keys) == vec):
                    self.rejections += 1
                    return False
        self.population.append(genome)
        self._rerank()
        inserted = True
        if len(self.population) > self.max_population_size:
            worst = max(self.population,
                        key=lambda g: (self.rank[g.genome_number], -self.crowd[g.genome_number]))
            inserted = worst is not genome
            self.population.remove(worst)
            self._rerank()
        self.insertions += int(inserted)
        if inserted and self.insertions % self.log_every == 0:
            self._log()
        return inserted

    # Reporting -------------------------------------------------------------
    def pareto_front(self) -> List[CircuitGenome]:
        return [g for g in self.population if self.rank.get(g.genome_number) == 0]

    def _log(self) -> None:
        front = self.pareto_front()
        logger.info(f"[pareto] inserted={self.insertions} archive={len(self.population)} "
                    f"front={len(front)} rejected={self.rejections}")
        for g in sorted(front, key=lambda g: [g.fitness[k] for k in self.objective_keys]):
            logger.info("   " + "  ".join(f"{k}={g.fitness[k]:.4g}" for k in self.objective_keys)
                        + f"  gates={[x.method_name for x in g.gates if x.enabled]}")
        if self.out_dir:
            self.save(os.path.join(self.out_dir, "pareto_front.json"))

    def save(self, path: str) -> None:
        rows = []
        for g in self.pareto_front():
            rows.append({
                "genome_number": g.genome_number,
                "fitness": {k: float(v) for k, v in g.fitness.items()},
                "gates": [(x.method_name, x.qubits, x.depth) for x in g.gates if x.enabled],
                "genome": g.to_dict(),
            })
        with open(path, "w") as f:
            json.dump({"objective_keys": self.objective_keys, "constraints": self.constraints,
                       "front": rows}, f, indent=2, default=str)
