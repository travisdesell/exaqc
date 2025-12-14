from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, List, Tuple, Union
from utils.losses import *

import torch


LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "fidelity": loss_one_minus_fidelity,
    "angle": loss_state_angle,
    "kl": loss_kl_divergence,
    "mse": loss_obs_mse,
}


# ---------- Utilities ----------

def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        # Promote real -> complex with zero imaginary part
        return x.to(dtype=torch.complex128 if x.dtype == torch.float64 else torch.complex64)
    return x


def _normalize_state(psi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerical safety: ensure ||psi||=1."""
    norm = torch.linalg.norm(psi)
    return psi / (norm + eps)


@dataclass
class TrainStepLog:
    step: int
    loss: float
    metric: Dict[str, float]


class QuantumStateTrainer:
    """
    Train a model that outputs a final statevector |psi(theta, ...)> to match a target state |phi>.

    model_fn:
        Callable returning a complex torch tensor statevector (qml.state()).
        Signature examples:
            model_fn(params) -> state
            model_fn(params, x) -> state  (if you later want batching/inputs)

    target:
        Either a fixed torch tensor statevector OR a callable target_fn() -> statevector.

    loss_name:
        One of: 'fidelity' (default), 'angle', 'kl', 'mse'

    Notes:
    - Works best with PennyLane QNodes using interface="torch" and diff_method="backprop" (statevector sim).
    - For shot-based / hardware you typically use expectation-based losses instead of qml.state().
    """

    def __init__(
        self,
        model_fn: Callable[..., torch.Tensor],
        params: Union[torch.nn.Parameter, List[torch.nn.Parameter]],
        target: Union[torch.Tensor, Callable[[], torch.Tensor]],
        loss_name: str = "fidelity",
        *,
        normalize_states: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss_name '{loss_name}'. Choose from {list(LOSS_REGISTRY.keys())}")

        self.model_fn = model_fn
        self.params = params if isinstance(params, list) else [params]
        self.target = target
        self.loss_fn = LOSS_REGISTRY[loss_name]
        self.loss_name = loss_name

        self.normalize_states = normalize_states
        self.loss_kwargs = loss_kwargs or {}

        self.device = device
        self.dtype = dtype

        # Move parameters if requested
        if self.device is not None or self.dtype is not None:
            for p in self.params:
                with torch.no_grad():
                    if self.device is not None:
                        p.data = p.data.to(self.device)
                    if self.dtype is not None:
                        p.data = p.data.to(self.dtype)

    @torch.no_grad()
    def get_target_state(self, *model_args, **model_kwargs) -> torch.Tensor:
        # target can be a fixed tensor OR a callable that may depend on model_args
        if callable(self.target):
            phi = self.target(*model_args, **model_kwargs)
        else:
            phi = self.target

        phi = _ensure_complex(phi)
        if self.device is not None:
            phi = phi.to(self.device)
        if self.normalize_states:
            phi = _normalize_state(phi)
        return phi


    def forward_state(self, *model_args, **model_kwargs) -> torch.Tensor:
        psi = self.model_fn(*self.params, *model_args, **model_kwargs) \
              if self._model_expects_params_first() else self.model_fn(*model_args, **model_kwargs)

        # If your model_fn already closes over params, it can ignore the params list.
        psi = _ensure_complex(psi)
        if self.normalize_states:
            psi = _normalize_state(psi)
        return psi

    def _model_expects_params_first(self) -> bool:
        return True

    def compute_loss(self, *model_args, **model_kwargs):
        psi = self.forward_state(*model_args, **model_kwargs)
        phi = self.get_target_state(*model_args, **model_kwargs)

        L = self.loss_fn(psi, phi, **self.loss_kwargs)

        F = fidelity(psi, phi)
        metrics = {"fidelity": F, "loss": L}
        return L, metrics


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
        model_kwargs = model_kwargs or {}

        # If model_args_list is provided, we ignore model_args and iterate the list each step
        if model_args_list is None:
            model_args = model_args or ()
            model_args_list = [tuple(model_args)]
        else:
            model_args_list = [tuple(a) for a in model_args_list]

        opt = optimizer_ctor(self.params, lr=lr)
        logs = []

        for step in range(steps):
            opt.zero_grad()

            # average over training cases
            losses = []
            fidelities = []
            for args in model_args_list:
                L, metrics = self.compute_loss(*args, **model_kwargs)
                losses.append(L)
                fidelities.append(metrics["fidelity"])

            loss = torch.stack(losses).mean()
            fid = torch.stack(fidelities).mean()

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

