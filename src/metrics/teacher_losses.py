"""Differentiable losses for the quantum-teacher imitation task.

Each quantity is written **once**, here, as a differentiable per-sample function
of a student's output and the teacher's target. Those same functions back both
the training loss (a scalar to minimize, via :data:`TEACHER_LOSSES`) and the
reported metrics (:mod:`src.metrics.teacher_metrics`), so a metric can never
drift away from the loss it is meant to describe.

Two of the measures share their core: the Bures angle is derived from the
classical fidelity rather than recomputing an overlap.

Distribution measures
    :func:`classical_fidelity`, :func:`bures_angle` and :func:`kl_divergence`
    treat both arguments as probability distributions over the readout basis
    states, which is what the ``"probs"`` quantum output mode produces. They
    take the absolute value and renormalize first, so raw outputs that are not
    perfectly normalized are handled gracefully. They are **not** meaningful for
    the ``"expval"`` readout, whose values live in ``[-1, 1]`` and do not form a
    distribution -- use :func:`mean_squared_error` there.

Sign conventions
    A *loss* is always something to minimize. Fidelity is a similarity in
    ``[0, 1]`` where ``1`` is a perfect match, so its loss is ``1 - fidelity``;
    the other three are already discrepancies and are minimized directly.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

#: Small constant guarding divisions and logarithms.
EPSILON: float = 1e-12

#: How far below 1 the angle *loss* caps its ``arccos`` argument, keeping the
#: gradient finite where a student matches its teacher. This puts a floor of
#: about ``sqrt(2 * eps)`` radians on the loss; the reported metric is exact.
ANGLE_GRADIENT_EPSILON: float = 1e-7


def _as_batch(tensor: Tensor) -> Tensor:
    """Views a single sample as a batch of one.

    Args:
        tensor: A tensor of shape ``[n_outputs]`` or ``[batch_size, n_outputs]``.

    Returns:
        A 2-D tensor of shape ``[batch_size, n_outputs]``.
    """

    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor


def _as_distributions(predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Coerces two output tensors into comparable probability distributions.

    Values are made non-negative and renormalized to sum to one along the last
    dimension, so outputs that are not perfectly normalized still compare
    sensibly.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs.

    Returns:
        A tuple of the normalized predictions and targets, both 2-D.

    Raises:
        ValueError: If the two tensors have different shapes.
    """

    predictions = _as_batch(predictions)
    targets = _as_batch(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape, received "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}."
        )

    predictions = predictions.abs()
    targets = targets.abs()

    predictions = predictions / (predictions.sum(dim=-1, keepdim=True) + EPSILON)
    targets = targets / (targets.sum(dim=-1, keepdim=True) + EPSILON)

    return predictions, targets


def classical_fidelity(predictions: Tensor, targets: Tensor) -> Tensor:
    """Computes the per-sample classical (Bhattacharyya) fidelity.

    For distributions ``q`` and ``p`` this is ``(sum_x sqrt(p_x q_x))^2``, which
    lies in ``[0, 1]`` and equals ``1`` only for identical distributions.

    Args:
        predictions: The student's outputs, ``[n_outputs]`` or
            ``[batch_size, n_outputs]``.
        targets: The teacher's outputs, same shape as ``predictions``.

    Returns:
        A ``[batch_size]`` tensor of fidelities.
    """

    predictions, targets = _as_distributions(predictions, targets)

    # Take the two square roots separately rather than sqrt(p * q). Teacher
    # targets are frequently exactly zero, and sqrt of a product that is zero
    # because of the *target* would hand autograd an inf * 0 = nan gradient for
    # the prediction. Factored this way a zero target contributes an exact zero
    # with no gradient path, and only the prediction needs a floor to keep the
    # derivative of sqrt finite. Adding an epsilon inside the product instead
    # would also inflate the overlap above 1 for sparse distributions.
    overlap = (
        torch.sqrt(predictions.clamp_min(EPSILON)) * torch.sqrt(targets.clamp_min(0.0))
    ).sum(dim=-1)

    return overlap**2


def bures_angle(
    predictions: Tensor,
    targets: Tensor,
    gradient_epsilon: float = 0.0,
) -> Tensor:
    """Computes the per-sample Bures angle between two distributions.

    This is ``arccos(sqrt(fidelity))``, a proper distance in ``[0, pi/2]`` that
    is ``0`` for identical distributions. It is reused from
    :func:`classical_fidelity` rather than recomputing the overlap.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs, same shape as ``predictions``.
        gradient_epsilon: How far below 1 to cap the ``arccos`` argument. The
            derivative of ``arccos`` diverges at 1 -- exactly where a student
            matches its teacher -- so training passes a small positive value to
            keep gradients finite, at the cost of reporting a correspondingly
            small angle instead of 0 for a perfect match. The default of ``0``
            is exact and is what the reported metric uses, since metrics are
            accumulated without gradients.

    Returns:
        A ``[batch_size]`` tensor of angles in radians.
    """

    fidelity = classical_fidelity(predictions, targets)
    overlap = torch.sqrt(fidelity.clamp_min(0.0))
    return torch.arccos(overlap.clamp(0.0, 1.0 - gradient_epsilon))


def kl_divergence(predictions: Tensor, targets: Tensor) -> Tensor:
    """Computes the per-sample KL divergence from target to prediction.

    This is ``KL(target || prediction) = sum_x p_x log(p_x / q_x)``, the penalty
    for describing the teacher's distribution using the student's. It is zero
    only when the two agree, and is asymmetric.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs, same shape as ``predictions``.

    Returns:
        A ``[batch_size]`` tensor of divergences in nats.
    """

    predictions, targets = _as_distributions(predictions, targets)

    predictions = predictions.clamp_min(EPSILON)
    targets = targets.clamp_min(EPSILON)

    return (targets * (targets.log() - predictions.log())).sum(dim=-1)


def mean_squared_error(predictions: Tensor, targets: Tensor) -> Tensor:
    """Computes the per-sample mean squared error across the outputs.

    Unlike the distribution measures this makes no assumption about the outputs,
    so it is the measure to use for the ``"expval"`` readout.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs, same shape as ``predictions``.

    Returns:
        A ``[batch_size]`` tensor of mean squared errors.

    Raises:
        ValueError: If the two tensors have different shapes.
    """

    predictions = _as_batch(predictions)
    targets = _as_batch(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape, received "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}."
        )

    return ((predictions - targets) ** 2).mean(dim=-1)


def fidelity_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Computes ``1 - fidelity``, averaged over the batch.

    Fidelity is a similarity, so it is turned into a loss to minimize.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs, same shape as ``predictions``.

    Returns:
        A scalar loss tensor.
    """

    return 1.0 - classical_fidelity(predictions, targets).mean()


def angle_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean Bures angle over the batch.

    Uses :data:`ANGLE_GRADIENT_EPSILON` so the gradient stays finite as the
    student converges onto the teacher.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs, same shape as ``predictions``.

    Returns:
        A scalar loss tensor.
    """

    return bures_angle(
        predictions, targets, gradient_epsilon=ANGLE_GRADIENT_EPSILON
    ).mean()


def kl_divergence_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean KL divergence over the batch.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs, same shape as ``predictions``.

    Returns:
        A scalar loss tensor.
    """

    return kl_divergence(predictions, targets).mean()


def mean_squared_error_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Computes the mean squared error over the batch.

    Args:
        predictions: The student's outputs.
        targets: The teacher's outputs, same shape as ``predictions``.

    Returns:
        A scalar loss tensor.
    """

    return mean_squared_error(predictions, targets).mean()


#: The per-sample measure behind each selectable loss, keyed by its command-line
#: name. The metrics in :mod:`src.metrics.teacher_metrics` report these same
#: functions, so a run's reported measure always matches the one it optimized.
TEACHER_MEASURES: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "fidelity": classical_fidelity,
    "angle": bures_angle,
    "kl": kl_divergence,
    "mse": mean_squared_error,
}

#: The scalar training loss for each selectable name.
TEACHER_LOSSES: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "fidelity": fidelity_loss,
    "angle": angle_loss,
    "kl": kl_divergence_loss,
    "mse": mean_squared_error_loss,
}

#: Loss names offered as command-line choices, in registry order.
TEACHER_LOSS_NAMES: tuple[str, ...] = tuple(TEACHER_LOSSES)


def get_teacher_loss(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    """Looks up a training loss function by its command-line name.

    Args:
        name: One of :data:`TEACHER_LOSS_NAMES`.

    Returns:
        The scalar loss function for that name.

    Raises:
        ValueError: If ``name`` is not a known loss.
    """

    if name not in TEACHER_LOSSES:
        raise ValueError(
            f"Unknown teacher loss {name!r}; choices: {list(TEACHER_LOSS_NAMES)}"
        )
    return TEACHER_LOSSES[name]
