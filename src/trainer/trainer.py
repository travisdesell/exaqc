"""
Quantum state training utilities.

This module provides a generic trainer for optimizing parameterized quantum
circuits whose outputs are quantum statevectors. The trainer supports multiple
loss functions (fidelity, angular distance, KL divergence, and observable MSE)
and integrates with PyTorch-based optimization pipelines.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, List, Tuple, Union
from src.utils.losses import (
    fidelity,
    loss_one_minus_fidelity,
    loss_state_angle,
    loss_kl_divergence,
    loss_obs_mse
)

import torch


LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "fidelity": loss_one_minus_fidelity,
    "angle": loss_state_angle,
    "kl": loss_kl_divergence,
    "mse": loss_obs_mse,
}


# ---------- Utilities ----------

def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    """Ensure a tensor is complex-valued.

    If the input tensor is real-valued, it is promoted to a complex tensor
    with zero imaginary component.

    Args:
        x: Input tensor.

    Returns:
        A complex-valued tensor with the same shape as ``x``.
    """
    if not torch.is_complex(x):
        return x.to(dtype=torch.complex128 if x.dtype == torch.float64 else torch.complex64)
    return x


def _normalize_state(psi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a quantum statevector.

    This function enforces numerical stability by ensuring the L2 norm
    of the statevector is 1.

    Args:
        psi: Complex-valued quantum statevector.
        eps: Small constant to prevent division by zero.

    Returns:
        A normalized quantum statevector.
    """
    norm = torch.linalg.norm(psi)
    return psi / (norm + eps)


@dataclass
class TrainStepLog:
    """Container for logging training progress.

    Attributes:
        step: Training step index.
        loss: Scalar loss value at this step.
        metric: Dictionary of logged metrics (e.g., fidelity).
    """
    step: int
    loss: float
    metric: Dict[str, float]


class QuantumStateTrainer:
    """Trainer for quantum models that output final quantum states.

    This class optimizes a parameterized quantum model so that its output
    statevector matches a target state under a specified loss function.

    The trainer assumes:
    - The model returns a quantum statevector (e.g., via ``qml.state()``).
    - Gradients are available through PyTorch autograd.

    Supported loss functions:
    - Fidelity loss (1 - |⟨ψ|φ⟩|²)
    - Angular (geodesic) state distance
    - KL divergence on probability distributions
    - Mean-squared error on observable expectations
    """

    def __init__(
        self,
        model_fn: Callable[..., torch.Tensor],
        params: Union[torch.nn.Parameter, List[torch.nn.Parameter]],
        target: Union[torch.Tensor, Callable[..., torch.Tensor]],
        loss_name: str = "fidelity",
        *,
        normalize_states: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the quantum state trainer.

        Args:
            model_fn: Callable mapping parameters (and optional inputs)
                to a quantum statevector.
            params: Trainable PyTorch parameter or list of parameters.
            target: Fixed target statevector or callable returning a target
                statevector.
            loss_name: Name of the loss function to use.
            normalize_states: Whether to normalize model and target states.
            device: Optional device to place parameters and tensors on.
            dtype: Optional dtype to cast parameters to.
            loss_kwargs: Optional keyword arguments for the loss function.

        Raises:
            ValueError: If ``loss_name`` is not recognized.
        """
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(
                f"Unknown loss_name '{loss_name}'. Choose from {list(LOSS_REGISTRY.keys())}"
            )

        self.model_fn = model_fn
        self.params = params if isinstance(params, list) else [params]
        self.target = target
        self.loss_fn = LOSS_REGISTRY[loss_name]
        self.loss_name = loss_name

        self.normalize_states = normalize_states
        self.loss_kwargs = loss_kwargs or {}

        self.device = device
        self.dtype = dtype

        if self.device is not None or self.dtype is not None:
            for p in self.params:
                with torch.no_grad():
                    if self.device is not None:
                        p.data = p.data.to(self.device)
                    if self.dtype is not None:
                        p.data = p.data.to(self.dtype)

    @torch.no_grad()
    def get_target_state(self, *model_args, **model_kwargs) -> torch.Tensor:
        """Retrieve the target quantum state.

        If the target is callable, it is invoked with the same arguments
        passed to the model. The resulting state may be normalized.

        Args:
            *model_args: Positional arguments forwarded to the target callable.
            **model_kwargs: Keyword arguments forwarded to the target callable.

        Returns:
            Complex-valued target quantum statevector.
        """
        phi = self.target(*model_args, **model_kwargs) if callable(self.target) else self.target
        phi = _ensure_complex(phi)
        if self.device is not None:
            phi = phi.to(self.device)
        if self.normalize_states:
            phi = _normalize_state(phi)
        return phi

    def forward_state(self, *model_args, **model_kwargs) -> torch.Tensor:
        """Compute the model output quantum state.

        Args:
            *model_args: Positional arguments passed to the model function.
            **model_kwargs: Keyword arguments passed to the model function.

        Returns:
            Complex-valued, optionally normalized model output statevector.
        """
        psi = (
            self.model_fn(*self.params, *model_args, **model_kwargs)
            if self._model_expects_params_first()
            else self.model_fn(*model_args, **model_kwargs)
        )

        psi = _ensure_complex(psi)
        if self.normalize_states:
            psi = _normalize_state(psi)
        return psi

    def _model_expects_params_first(self) -> bool:
        """Return whether ``model_fn`` expects parameters as its first argument.

        Returns:
            True if parameters are passed as the first argument.
        """
        return True

    def compute_loss(self, *model_args, **model_kwargs):
        """Compute loss and metrics for a single training case.

        Args:
            *model_args: Positional arguments for the model and target.
            **model_kwargs: Keyword arguments for the model and target.

        Returns:
            Tuple containing:
                - loss: Scalar loss tensor.
                - metrics: Dictionary with keys ``'loss'`` and ``'fidelity'``.
        """
        psi = self.forward_state(*model_args, **model_kwargs)
        phi = self.get_target_state(*model_args, **model_kwargs)

        loss = self.loss_fn(psi, phi, **self.loss_kwargs)
        fid = fidelity(psi, phi)

        metrics = {"fidelity": fid, "loss": loss}
        return loss, metrics

    def fit(
        self,
        *,
        steps: int = 200,
        lr: float = 0.1,
        optimizer_ctor=torch.optim.Adam,
        model_args=None,
        model_kwargs=None,
        model_args_list=None,
        grad_clip_norm=None,
        log_every: int = 25,
        callback=None,
    ):
        """Run the training loop.

        Supports averaging loss over multiple input configurations per
        optimization step, which is useful for learning quantum gates
        or input-conditional circuits.

        Args:
            steps: Number of optimization steps.
            lr: Learning rate.
            optimizer_ctor: PyTorch optimizer constructor.
            model_args: Single set of positional arguments (ignored if
                ``model_args_list`` is provided).
            model_kwargs: Keyword arguments passed to model and target.
            model_args_list: List of argument tuples averaged per step.
            grad_clip_norm: Optional gradient clipping norm.
            log_every: Logging frequency (in steps).
            callback: Optional callable invoked with a ``TrainStepLog``.

        Returns:
            A list of ``TrainStepLog`` objects recording training progress.
        """
        model_kwargs = model_kwargs or {}

        if model_args_list is None:
            model_args = model_args or ()
            model_args_list = [tuple(model_args)]
        else:
            model_args_list = [tuple(a) for a in model_args_list]

        opt = optimizer_ctor(self.params, lr=lr)
        logs = []

        for step in range(steps):
            opt.zero_grad()

            losses = []
            fidelities = []
            for args in model_args_list:
                L, metrics = self.compute_loss(*args, **model_kwargs)
                losses.append(L)
                fidelities.append(metrics["fidelity"])

            loss = torch.stack(losses).mean()
            fid = torch.stack(fidelities).mean()

            if not loss.requires_grad:
                break
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.params, grad_clip_norm)

            opt.step()

            if (step % log_every) == 0 or step == steps - 1:
                log = TrainStepLog(
                    step=step,
                    loss=float(loss.detach().cpu().item()),
                    metric={
                        "loss": float(loss.detach().cpu().item()),
                        "fidelity": float(fid.detach().cpu().item()),
                    },
                )
                logs.append(log)
                if callback is not None:
                    callback(log)

        return logs
