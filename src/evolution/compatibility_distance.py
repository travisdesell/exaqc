"""Three-component speciation compatibility dissimilarity.

``δ = c_s * d_s + c_q * d_q + c_c * d_c``. ``d_s`` is the Euclidean length of
the unmatched enabled-gate innovations, ``d_q`` the Euclidean length of the
shortest ``2π`` angle differences on enabled gates both genomes carry, and
``d_c`` the Euclidean distance between trainable encoder and decoder
parameters. Disabled gates do not enter. The sum is a compatibility
dissimilarity: tests cover identity, symmetry, and each component, and do not
treat it as a metric.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol


class CompatibleGenome(Protocol):
    """The genome fields the compatibility distance reads."""

    def get_historical_gate_innovations(self) -> list[int]:
        """Returns sorted unique innovation IDs of enabled gates."""

    def enabled_gate_parameters(self) -> dict[int, list[float]]:
        """Returns enabled-gate parameter values keyed by innovation."""

    def classical_parameters(self) -> list[float]:
        """Returns trainable encoder parameters, then decoder parameters."""


def _as_float(value: object, description: str) -> float:
    """Returns ``value`` as a finite float.

    Args:
        value: A real number or a scalar tensor.
        description: What ``value`` is, used in the error message.

    Returns:
        ``value`` as a finite float.

    Raises:
        ValueError: If ``value`` is not a finite real number.
    """

    if isinstance(value, bool):
        raise ValueError(f"{description} {value!r} is not a real number")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        detach = getattr(value, "detach", None)
        if not callable(detach):
            raise ValueError(f"{description} {value!r} is not a real number")
        tensor = detach()
        numel = getattr(tensor, "numel", None)
        if not callable(numel) or numel() != 1:
            raise ValueError(f"{description} {value!r} is not a real number")
        number = float(tensor.reshape(-1)[0].item())
    if not math.isfinite(number):
        raise ValueError(f"{description} {value!r} is not a finite real number")
    return number


def _require_coefficient(name: str, value: float) -> None:
    """Rejects a coefficient that is not a finite non-negative number.

    Args:
        name: The coefficient's name, used in the error message.
        value: The coefficient to check.

    Returns:
        None.

    Raises:
        ValueError: If ``value`` is not a finite number greater than or equal
            to zero.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


def angular_difference(theta_a: float, theta_b: float) -> float:
    """Returns the shortest wrapped difference of two angles.

    Args:
        theta_a: The first angle, in radians.
        theta_b: The second angle, in radians.

    Returns:
        The difference in ``[0, π]``,
        ``|((theta_a - theta_b + π) mod 2π) - π|``.

    Raises:
        ValueError: If either angle is not a finite real number.
    """

    delta = _as_float(theta_a, "angle") - _as_float(theta_b, "angle")
    return abs(((delta + math.pi) % (2.0 * math.pi)) - math.pi)


def structural_distance(ids_a: Sequence[int], ids_b: Sequence[int]) -> float:
    """Returns the structural distance between two enabled-innovation sets.

    Args:
        ids_a: Enabled-gate innovation IDs of the first genome.
        ids_b: Enabled-gate innovation IDs of the second genome.

    Returns:
        ``sqrt(|I_a △ I_b|)``. Order and repeated IDs do not matter.
    """

    return math.sqrt(len(set(ids_a) ^ set(ids_b)))


def quantum_parameter_distance(
    params_a: Mapping[int, Sequence[float]],
    params_b: Mapping[int, Sequence[float]],
    names_a: Mapping[int, Sequence[str]] | None = None,
    names_b: Mapping[int, Sequence[str]] | None = None,
) -> float:
    """Returns the quantum-parameter distance over matching enabled gates.

    Args:
        params_a: Parameter values of the first genome, keyed by innovation.
        params_b: Parameter values of the second genome, keyed by innovation.
        names_a: Parameter names parallel to ``params_a``. When both name maps
            are given, a matched innovation whose names differ is rejected.
        names_b: Parameter names parallel to ``params_b``.

    Returns:
        The Euclidean length of the shortest angle differences on innovations
        present in both maps. Parameter-free gates and an empty intersection
        contribute zero. Unmatched innovations are ignored.

    Raises:
        ValueError: If a matched innovation has different parameter names or
            different parameter counts, or a value is not a finite real number.
    """

    total = 0.0
    for innovation in set(params_a) & set(params_b):
        left = list(params_a[innovation])
        right = list(params_b[innovation])
        if names_a is not None and names_b is not None:
            left_names = list(names_a[innovation])
            right_names = list(names_b[innovation])
            if left_names != right_names:
                raise ValueError(
                    f"innovation {innovation} parameter names differ: "
                    f"{left_names} vs {right_names}"
                )
        if len(left) != len(right):
            raise ValueError(
                f"innovation {innovation} has {len(left)} parameters on one "
                f"genome and {len(right)} on the other"
            )
        for theta_a, theta_b in zip(left, right):
            total += angular_difference(theta_a, theta_b) ** 2
    return math.sqrt(total)


def classical_parameter_distance(
    weights_a: Sequence[float], weights_b: Sequence[float]
) -> float:
    """Returns the Euclidean distance between two classical parameter vectors.

    Args:
        weights_a: Trainable encoder parameters, then decoder parameters, of
            the first genome.
        weights_b: The same coordinates of the second genome.

    Returns:
        The Euclidean length of the coordinate-wise differences. Two empty
        vectors have distance zero.

    Raises:
        ValueError: If the vectors have different lengths, or a value is not a
            finite real number.
    """

    if len(weights_a) != len(weights_b):
        raise ValueError(
            "classical parameter counts differ: "
            f"{len(weights_a)} vs {len(weights_b)}"
        )
    total = 0.0
    for left, right in zip(weights_a, weights_b):
        delta = _as_float(left, "classical parameter") - _as_float(
            right, "classical parameter"
        )
        total += delta**2
    return math.sqrt(total)


def _parameter_names(genome: object) -> dict[int, list[str]] | None:
    """Returns enabled-gate parameter names when the genome provides them.

    Args:
        genome: A genome that may implement ``enabled_gate_parameter_names``.

    Returns:
        The name map, or ``None`` when the genome has no such method.
    """

    method = getattr(genome, "enabled_gate_parameter_names", None)
    if not callable(method):
        return None
    names: dict[int, list[str]] = method()
    return names


def compatibility_distance(
    genome_a: CompatibleGenome,
    genome_b: CompatibleGenome,
    structure_coefficient: float = 1.0,
    quantum_coefficient: float = 1.0,
    classical_coefficient: float = 1.0,
) -> float:
    """Returns the compatibility dissimilarity of two genomes.

    Args:
        genome_a: The first genome.
        genome_b: The second genome.
        structure_coefficient: Non-negative finite weight ``c_s`` on ``d_s``.
        quantum_coefficient: Non-negative finite weight ``c_q`` on ``d_q``.
        classical_coefficient: Non-negative finite weight ``c_c`` on ``d_c``.

    Returns:
        ``c_s * d_s + c_q * d_q + c_c * d_c``.

    Raises:
        ValueError: If a coefficient is not a finite non-negative number, or a
            component rejects the genomes' parameters.
    """

    _require_coefficient("structure_coefficient", structure_coefficient)
    _require_coefficient("quantum_coefficient", quantum_coefficient)
    _require_coefficient("classical_coefficient", classical_coefficient)
    structural = structural_distance(
        genome_a.get_historical_gate_innovations(),
        genome_b.get_historical_gate_innovations(),
    )
    quantum = quantum_parameter_distance(
        genome_a.enabled_gate_parameters(),
        genome_b.enabled_gate_parameters(),
        _parameter_names(genome_a),
        _parameter_names(genome_b),
    )
    classical = classical_parameter_distance(
        genome_a.classical_parameters(),
        genome_b.classical_parameters(),
    )
    return (
        float(structure_coefficient) * structural
        + float(quantum_coefficient) * quantum
        + float(classical_coefficient) * classical
    )
