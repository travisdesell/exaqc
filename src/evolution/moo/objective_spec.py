"""Multi-objective fitness objective specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    """Describes one objective used during multi-objective optimization.

    EXAQC treats every objective internally as a minimization objective.
    The ``sign`` field controls whether the original objective should be
    minimized or maximized.

    For an objective that should be minimized, use::

        ObjectiveSpec(name="loss", sign=1.0)

    For an objective that should be maximized, use::

        ObjectiveSpec(name="accuracy", sign=-1.0)

    The raw value stored in ``CircuitGenome.fitness`` is never modified.
    The sign transformation is applied only when the objective is used
    by the multi-objective optimization algorithm.

    Args:
        name: Key used to retrieve the objective from ``genome.fitness``.
        sign: Multiplier used to convert the objective to minimization form.
            Use ``1.0`` for minimization and ``-1.0`` for maximization.
        weight: Optional positive scaling factor applied to the transformed
            objective.

    Raises:
        ValueError: If ``sign`` is not either ``1.0`` or ``-1.0``.
        ValueError: If ``weight`` is not greater than zero.
        ValueError: If ``name`` is empty.
    """

    name: str
    sign: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the objective specification."""
        if not self.name:
            raise ValueError("Objective name must not be empty.")

        if self.sign not in {-1.0, 1.0}:
            raise ValueError(
                "Objective sign must be 1.0 for minimization or "
                "-1.0 for maximization."
            )

        if self.weight <= 0.0:
            raise ValueError("Objective weight must be greater than zero.")

    def transform(self, value: float) -> float:
        """Transform a raw objective value into minimization form.

        Args:
            value: Raw objective value stored in ``genome.fitness``.

        Returns:
            Objective value transformed into minimization form.
        """
        return self.sign * self.weight * float(value)
