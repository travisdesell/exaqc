# src/evolution/moo/objective_spec.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Specification for one objective.

    Args:
        name:
            Key in genome.fitness.

        sign:
            If -1.0, internally negate the objective so NSGA can
            treat everything as minimization.

        weight:
            Optional scaling factor.
    """

    name: str
    sign: float = 1.0      # +1=minimize, -1=maximize
    weight: float = 1.0

    def minimization_value(self, value: float) -> float:
        return self.sign * self.weight * float(value)