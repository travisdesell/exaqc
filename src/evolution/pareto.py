"""Non-dominated sorting and crowding distance for steady-state NSGA-II.

The search inserts one genome at a time, so this is the ranking half of
NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002) without a generational
parent/offspring barrier. It is the ranking used by
:class:`~src.evolution.steady_state_nsga2.SteadyStateNSGA2`.

Two or three objectives are supported. Crowding distance stops separating
solutions once most of the population is mutually non-dominated, which is the
usual case at four or more objectives; that case is rejected rather than run
under a distance that no longer ranks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from src.circuits.circuit import CircuitGenome

Direction = Literal["min", "max"]


@dataclass(frozen=True)
class Objective:
    """One objective and the direction in which it improves.

    Args:
        name: Fitness key, or ``n_cnot`` for the decomposed two-qubit count.
        direction: ``max`` when larger values dominate, ``min`` when smaller
            values dominate.
    """

    name: str
    direction: Direction


#: Objectives the research probe selects on unless a run overrides them.
#: ``target_metric`` is the validation score every task already writes (higher
#: is better). ``n_cnot`` is the decomposed two-qubit cost the archive already
#: stores (lower is better).
DEFAULT_OBJECTIVES: tuple[Objective, ...] = (
    Objective("target_metric", "max"),
    Objective("n_cnot", "min"),
)


def parse_objective(text: str) -> Objective:
    """Parses one ``name:min`` or ``name:max`` objective.

    Args:
        text: The objective as written on the command line.

    Returns:
        The parsed objective.

    Raises:
        ValueError: If ``text`` is not ``name:min`` or ``name:max``.
    """

    name, separator, direction = text.partition(":")
    if separator != ":" or not name or direction not in ("min", "max"):
        raise ValueError(
            f"Objective {text!r} must be name:min or name:max "
            "(for example target_metric:max)."
        )
    if direction == "min":
        return Objective(name, "min")
    return Objective(name, "max")


def parse_objectives(texts: list[str] | None) -> tuple[Objective, ...]:
    """Parses the objective list a run selects on.

    Args:
        texts: ``name:direction`` strings. ``None`` selects
            :data:`DEFAULT_OBJECTIVES`.

    Returns:
        The objectives, in the order given.

    Raises:
        ValueError: If an entry is malformed, a name is repeated, or the count
            is outside the two-to-three range NSGA-II crowding can rank.
    """

    if texts is None:
        return DEFAULT_OBJECTIVES
    objectives = tuple(parse_objective(text) for text in texts)
    names = [objective.name for objective in objectives]
    if len(names) != len(set(names)):
        raise ValueError(f"Objectives repeat a name: {names}.")
    if not 2 <= len(objectives) <= 3:
        raise ValueError(
            "NSGA-II crowding ranks two or three objectives; "
            f"got {len(objectives)}. Four or more need NSGA-III, which "
            "this strategy does not run."
        )
    return objectives


def objective_value(genome: Any, objective: Objective) -> float:
    """Reads one objective off a genome.

    ``n_cnot`` is the decomposed count :func:`src.circuits.gate_complexity.gate_counts`
    writes into the archive, so selection and the summary column cannot drift.
    A genome with no gate list (a test double) may carry ``n_cnot`` in its
    fitness instead. Every other objective is a key of ``genome.fitness``.

    Args:
        genome: A genome or a test double with ``fitness`` and, for a real
            circuit, ``gates`` and ``target``.
        objective: The objective to read.

    Returns:
        The finite objective value.

    Raises:
        KeyError: If the genome does not carry the objective.
        ValueError: If the value is not a finite number.
    """

    if objective.name == "n_cnot" and getattr(genome, "gates", None) is not None:
        from src.circuits.gate_complexity import gate_counts

        value: Any = gate_counts(genome)["gates_cnot"]
    else:
        fitness = getattr(genome, "fitness", None) or {}
        if objective.name not in fitness:
            raise KeyError(
                f"Genome {getattr(genome, 'genome_number', None)} has no "
                f"{objective.name!r} objective. Fitness keys: {sorted(fitness)}."
            )
        value = fitness[objective.name]
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Objective {objective.name!r} must be a number, got {value!r}."
        ) from error
    if not math.isfinite(number):
        raise ValueError(
            f"Objective {objective.name!r} must be finite, got {number}."
        )
    return number


def dominates(left: Any, right: Any, objectives: tuple[Objective, ...]) -> bool:
    """Returns whether ``left`` Pareto-dominates ``right``.

    Dominance requires being no worse on every objective and strictly better
    on at least one. Equal vectors do not dominate each other.

    Args:
        left: The candidate dominator.
        right: The candidate dominated genome.
        objectives: Objectives and their directions.

    Returns:
        True when ``left`` dominates ``right``.
    """

    better = False
    for objective in objectives:
        left_value = objective_value(left, objective)
        right_value = objective_value(right, objective)
        if objective.direction == "min":
            if left_value > right_value:
                return False
            better = better or left_value < right_value
        else:
            if left_value < right_value:
                return False
            better = better or left_value > right_value
    return better


def non_dominated_fronts(
    genomes: list[Any], objectives: tuple[Objective, ...]
) -> list[list[Any]]:
    """Sorts genomes into successive non-dominated fronts.

    This is the fast non-dominated sort of NSGA-II. Front 0 is the Pareto
    front of ``genomes``. Each later front is the Pareto front of what remains.

    Args:
        genomes: Genomes to rank. The list is not modified.
        objectives: Objectives and their directions.

    Returns:
        Fronts from best to worst. Empty when ``genomes`` is empty.
    """

    if not genomes:
        return []

    dominated_by: dict[int, list[Any]] = {id(genome): [] for genome in genomes}
    domination_count: dict[int, int] = {id(genome): 0 for genome in genomes}
    first: list[Any] = []

    for genome in genomes:
        for other in genomes:
            if other is genome:
                continue
            if dominates(genome, other, objectives):
                dominated_by[id(genome)].append(other)
            elif dominates(other, genome, objectives):
                domination_count[id(genome)] += 1
        if domination_count[id(genome)] == 0:
            first.append(genome)

    fronts: list[list[Any]] = []
    current = first
    while current:
        fronts.append(current)
        nxt: list[Any] = []
        for genome in current:
            for other in dominated_by[id(genome)]:
                domination_count[id(other)] -= 1
                if domination_count[id(other)] == 0:
                    nxt.append(other)
        current = nxt
    return fronts


def crowding_distances(
    front: list[Any], objectives: tuple[Objective, ...]
) -> dict[int, float]:
    """Assigns NSGA-II crowding distance within one front.

    Endpoints of each objective get infinite distance, so truncation keeps the
    extremes. A constant objective adds nothing: every member is an endpoint
    of it. A front of two or fewer genomes is all endpoints.

    Args:
        front: One non-dominated front.
        objectives: Objectives and their directions. Direction does not affect
            distance; only the spread of values does.

    Returns:
        Crowding distance keyed by ``id(genome)``. Infinite distances are
        ``math.inf``.
    """

    distances = {id(genome): 0.0 for genome in front}
    if len(front) <= 2:
        return {identity: math.inf for identity in distances}

    for objective in objectives:
        ordered = sorted(front, key=lambda genome: objective_value(genome, objective))
        distances[id(ordered[0])] = math.inf
        distances[id(ordered[-1])] = math.inf
        span = objective_value(ordered[-1], objective) - objective_value(
            ordered[0], objective
        )
        if span == 0.0:
            continue
        for index in range(1, len(ordered) - 1):
            identity = id(ordered[index])
            if math.isinf(distances[identity]):
                continue
            previous = objective_value(ordered[index - 1], objective)
            following = objective_value(ordered[index + 1], objective)
            distances[identity] += (following - previous) / span
    return distances


def finite_crowding(distance: float) -> float | None:
    """Encodes a crowding distance for JSON metadata.

    Args:
        distance: A crowding distance, possibly infinite.

    Returns:
        ``None`` for an infinite distance, otherwise the distance itself.
    """

    if math.isinf(distance):
        return None
    return float(distance)


def stamp_fronts(genomes: list[Any], objectives: tuple[Objective, ...]) -> None:
    """Records Pareto rank and crowding on each genome's metadata.

    Infinite crowding is stored as ``None`` because genome JSON cannot hold
    an infinity. A missing rank means the genome is not in ``genomes``.

    Args:
        genomes: The population to stamp.
        objectives: Objectives the ranks were computed from.

    Returns:
        None. Sets ``pareto_rank`` and ``crowding_distance`` on each genome.
    """

    for rank, front in enumerate(non_dominated_fronts(genomes, objectives)):
        distances = crowding_distances(front, objectives)
        for genome in front:
            metadata: dict[str, Any] = genome.metadata
            metadata["pareto_rank"] = rank
            metadata["crowding_distance"] = finite_crowding(distances[id(genome)])


def scalar_elite(genomes: list[CircuitGenome]) -> CircuitGenome | None:
    """Returns the genome scalar selection would have kept.

    Scalar insertion sorts on ``fitness["loss"]`` ascending. That is the
    preference the published runs used, independent of the Pareto front.

    Args:
        genomes: Genomes that recorded a loss.

    Returns:
        The genome with the smallest loss, breaking ties toward the smaller
        genome number. ``None`` when ``genomes`` is empty or none of them
        recorded a finite loss.
    """

    ranked: list[tuple[float, int, CircuitGenome]] = []
    for genome in genomes:
        fitness = genome.fitness or {}
        loss = fitness.get("loss")
        if loss is None:
            continue
        try:
            number = float(loss)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            ranked.append((number, int(genome.genome_number), genome))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]
