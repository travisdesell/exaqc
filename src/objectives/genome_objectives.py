from __future__ import annotations
from typing import TYPE_CHECKING

import math
import numpy as np
import torch
import copy

from typing import Optional, Iterable, Callable, Any
from loguru import logger

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome
from src.noise import BaseNoiseModel, PennyLaneNoiseModel

from src.utils.losses import (  # noqa: F401
    loss_ce,
    loss_kl_divergence,
    loss_one_minus_fidelity,
    loss_state_angle,
    ce_onehot_on_probs,
    class_avg_ce_onehot_on_probs,
    LOSS_REGISTRY,
)

from src.utils.helpers import (
    BalancedBatchSampler,
    torch_params_to_genome,
    genome_to_torch_params,
)

STATEVECTOR_LOSSES = {"fidelity", "angle", "kl", "ce"}


def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is represented as a complex-valued tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Complex tensor.
    """
    x = torch.as_tensor(x)
    if not torch.is_complex(x):
        x = x.to(torch.complex128 if x.dtype == torch.float64 else torch.complex64)
    return x


def _normalize_state(psi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a quantum state vector.

    Args:
        psi (torch.Tensor): State vector.
        eps (float, optional): Numerical stability constant.

    Returns:
        torch.Tensor: Normalized state vector.
    """
    return psi / (torch.linalg.norm(psi) + eps)


@torch.no_grad()
def compute_teacher_metrics(
    *,
    phi: torch.Tensor,
    psi: torch.Tensor,
    extra: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute fidelity-based teacher metrics.

    Computes fidelity loss, angle loss, and optionally readout cross-entropy
    between a target state and predicted state.

    Args:
        phi (torch.Tensor): Target quantum state.
        psi (torch.Tensor): Predicted quantum state.
        extra (dict[str, Any], optional): Optional dictionary containing readout
            information for additional metrics.

    Returns:
        dict[str, float]: Dictionary of computed metrics.
    """
    phi = _normalize_state(_ensure_complex(phi))
    psi = _normalize_state(_ensure_complex(psi))

    out = {}
    out["loss"] = float(loss_one_minus_fidelity(phi, psi).cpu().item())
    out["fidelity_loss"] = float(loss_one_minus_fidelity(phi, psi).cpu().item())
    out["angle_loss"] = float(loss_state_angle(phi, psi).cpu().item())

    if extra is not None and "readout_ce" in extra and extra["readout_ce"]:
        y = extra["y_onehot"]
        p = extra["readout_probs_fn"](psi)
        out["ce"] = float(ce_onehot_on_probs(p, y).cpu().item())

    return out


@torch.no_grad()
def _eval_teacher_split(
    data, genome, params, teacher_qnode, loss_fn: Optional[Callable] = None
):
    """Evaluate teacher-style losses on a dataset split.

    Args:
        data (Iterable): Input samples.
        genome (CircuitGenome): Circuit genome.
        params (dict): Torch parameters.
        teacher_qnode (Callable): Target QNode producing reference states.
        loss_fn (Callable, optional): Custom loss function.

    Returns:
        dict[str, float] | None: Aggregated evaluation metrics.
    """
    if data is None:
        return None
    losses = []
    fids = []
    angle_losses = []
    for item in data:
        x = item if not isinstance(item, (tuple, list)) else item[0]
        phi = torch.as_tensor(teacher_qnode(x))
        psi = genome.circuit(x, params)
        L = loss_one_minus_fidelity(phi, psi) if loss_fn is None else loss_fn(phi, psi)
        losses.append(L)
        fids.append(1.0 - L)
        angle_losses.append(loss_state_angle(phi, psi))
    return {
        "loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
        "fidelity": float(torch.stack(fids).mean().item()) if fids else 0.0,
        "angle_loss": (
            float(torch.stack(angle_losses).mean().item()) if angle_losses else 0.0
        ),
    }


@torch.no_grad()
def _eval_supervised_split(
    data,
    genome,
    params,
    n_classes,
    loss_fn: Optional[Callable] = None,
    class_counts: Optional[dict] = None,
    alpha=None,
    eps:float = 1e-08
):
    """Evaluate supervised classification metrics on a dataset split.

    Args:
        data (Iterable): Dataset samples `(x, y, class)`.
        genome (CircuitGenome): Circuit genome.
        params (dict): Torch parameters.
        n_classes (int): Number of output classes.
        loss_fn (Callable, optional): Custom loss function.
        class_counts (dict, optional): Per-class counts.
        alpha (np.array, optional): Per-class alpha values for balanced loss.

    Returns:
        dict[str, float] | None: Loss and accuracy metrics.
    """
    if data is None:
        return None
    losses = []
    probas = []
    y_onehots = []
    correct = 0
    total = 0
    for x, y, cls in data:
        probs = genome.circuit(x, params)
        probs = torch.as_tensor(probs, dtype=torch.float32)

        probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps)
        probs = probs.clamp_min(eps)

        probs = probs[:n_classes]
        probs = probs / (probs.sum() + 1e-12).clamp_min(eps)
        L = ce_onehot_on_probs(probs, y, alpha_per_class=alpha)
        losses.append(L)
        pred = int(torch.argmax(probs).item())
        true = int(torch.argmax(y).item())
        correct += int(pred == true)
        total += 1
        probas.append(probs)
        y_onehots.append(y)

    if loss_fn.__name__ != "class_avg_ce_onehot_on_probs":
        loss = float(torch.stack(losses).mean().item()) if losses else 0.0
    else:
        probs = torch.stack([p.to(torch.float32) for p in probas], dim=0)
        y_onehots = torch.stack([p.to(torch.float32) for p in y_onehots], dim=0)
        loss = float(loss_fn(probs, y_onehots))

    return {
        "loss": loss,
        "acc": float(correct / max(total, 1)),
    }


@torch.no_grad()
def eval_forward_only(
    genome: CircuitGenome,
    train_list: list,
    test_list: list = None,
    teacher_qnode=None,
    n_classes: int = 3,
    loss_fn: Optional[Callable] = None,
    class_counts: Optional[tuple] = None,
    alpha: Optional[torch.Tensor] = None,
):
    """Evaluate a genome without gradient updates.

    Supports both teacher (statevector) and supervised classification modes.

    Args:
        genome (CircuitGenome): Circuit genome.
        train_list (list): Training dataset.
        test_list (list, optional): Test dataset.
        teacher_qnode (Callable, optional): Teacher QNode.
        n_classes (int): Number of classes.
        loss_fn (Callable, optional): Loss function.
        class_counts (tuple, optional): Per-class counts.
        alpha (np.array, optional): Per-class alpha values for balanced loss.

    Returns:
        dict[str, float]: Evaluation metrics.
    """
    mode = "teacher" if teacher_qnode is not None else "supervised"
    params = genome_to_torch_params(genome)

    if mode == "teacher":
        tr = _eval_teacher_split(
            train_list, genome, params, teacher_qnode, loss_fn=loss_fn
        )
        te = (
            _eval_teacher_split(
                test_list, genome, params, teacher_qnode, loss_fn=loss_fn
            )
            if test_list is not None
            else None
        )
    else:
        tr = _eval_supervised_split(
            train_list,
            genome,
            params,
            n_classes,
            loss_fn=loss_fn,
            class_counts=class_counts[0],
            alpha=alpha,
        )
        te = (
            _eval_supervised_split(
                test_list,
                genome,
                params,
                n_classes,
                loss_fn=ce_onehot_on_probs,
                class_counts=class_counts[1],
                alpha=alpha,
            )
            if test_list is not None
            else None
        )
    out = {f"{k}": v for k, v in tr.items()}
    if te is not None:
        out.update({f"train_{k}": v for k, v in tr.items()})
        out.update({f"test_{k}": v for k, v in te.items()})
    logger.info(f"[End Step]: {out}")
    return out


def _train_with_pennylane(
    genome: CircuitGenome,
    train_data: Iterable[tuple[torch.Tensor, torch.Tensor]] = None,
    test_data: Iterable[tuple[torch.Tensor, torch.Tensor]] = None,
    epochs: int = 200,
    lr: float = 0.05,
    log_every: int = 1,
    n_classes: int = 3,
    batch_size: int = None,
    loss_name: str = "ce",
    shuffle_each_epoch: bool = True,
    # NEW:
    target_qnode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    encoding: str = "angle",
    noise_model: Optional[PennyLaneNoiseModel] = None,
):
    """Train a CircuitGenome using PennyLane-based differentiable execution.

    This routine supports two training modes:

    **1) Teacher / Statevector mode**
        - Uses a reference `target_qnode` producing a target quantum state
        - Optimizes state-based losses such as fidelity, angle, or KL divergence

    **2) Supervised classification mode**
        - Uses probabilistic readout from the circuit
        - Optimizes cross-entropy or related classification losses

    The function automatically configures the genome's PennyLane circuit
    (statevector vs probability output), performs mini-batch training using
    Adam, logs intermediate metrics, and writes trained parameters back
    into the genome.

    Args:
        genome (CircuitGenome): Quantum circuit genome to be trained.
        train_data (Iterable): Training dataset.
            - Teacher mode: iterable of inputs `x`
            - Supervised mode: iterable of `(x, y, class_id)`
        test_data (Iterable, optional): Optional test dataset.
        epochs (int): Number of optimization epochs.
        lr (float): Learning rate for Adam optimizer.
        log_every (int): Logging frequency (in epochs).
        n_classes (int): Number of output classes for supervised learning.
        batch_size (int, optional): Mini-batch size. If None, uses full dataset.
        loss_name (str): Name of loss function from `LOSS_REGISTRY`.
        shuffle_each_epoch (bool): Whether to reshuffle data each epoch.
        target_qnode (Callable, optional): Reference QNode producing target
            quantum states (required for teacher mode).
        encoding (str): The type of quantum encoding
        noise_model (Object, optional): The noise model to adopt

    Returns:
        None. The genome is updated in-place, and `genome.fitness` is set.
    """
    train_list = list(train_data)
    test_list = list(test_data) if test_data is not None else None

    best_params = None
    best_metrics = None
    logger.info(f"getting loss function for '{loss_name}'")
    loss_fn = LOSS_REGISTRY[loss_name]

    # For balanced loss setting Alpha from https://arxiv.org/pdf/1901.05555
    beta = (len(train_data) - 1) / len(train_data)
    alpha = (1.0 - beta) / (
        1.0 - np.power(beta, np.array(train_data.counts, dtype=np.float32))
    )

    # soft weighting
    # alpha = alpha / alpha.mean()
    # alpha = alpha ** 0.5

    alpha = torch.as_tensor(alpha / alpha.mean(), dtype=torch.float32)
    logger.info(f"Selected alphas: {alpha}")

    n = len(train_list)
    if batch_size is not None:
        batch_size = max(1, min(batch_size, n))

    sampler = BalancedBatchSampler(
        data=train_list, batch_size=batch_size, shuffle=shuffle_each_epoch
    )

    # Noise Model
    if noise_model is None:
        noise_model = PennyLaneNoiseModel.from_hyperparameters(genome.hyperparameters)

    # ----- choose output type -----
    use_state = (
        loss_name in {"fidelity", "angle", "kl"} and target_qnode
    )  # teacher-style losses

    if use_state:
        genome.generate_pennylane_circuit(
            input_mode=encoding,
            return_probs=False,
            measure_registers=False,
            noise_model=noise_model,
        )
        if target_qnode is None:
            raise ValueError(
                "target_qnode is required for teacher training with statevector losses."
            )
    else:
        genome.generate_pennylane_circuit(
            input_mode=encoding,
            return_probs=True,
            noise_model=noise_model,
        )

    torch_params = genome_to_torch_params(genome)

    # no params: forward-only eval
    if len(torch_params) == 0:
        metrics = eval_forward_only(
            genome,
            train_list,
            test_list,
            teacher_qnode=target_qnode,
            n_classes=n_classes,
            loss_fn=loss_fn,
            class_counts=(train_data.class_counts, test_data.class_counts),
            alpha=alpha,
        )
        genome.fitness = metrics
        return

    opt = torch.optim.Adam(torch_params.values(), lr=lr, weight_decay=0.00000)

    # --- state forward ---
    def forward_state(x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a normalized quantum state.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Normalized complex-valued statevector.
        """
        psi = genome.circuit(x, torch_params)
        psi = _normalize_state(_ensure_complex(psi))
        return psi

    # --- probs forward (your existing CE path) ---
    def forward_probs(x: torch.Tensor, eps:float = 1e-08) -> torch.Tensor:
        """Forward pass returning normalized class probabilities.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Probability vector of shape `[n_classes]`.
        """
        probs = genome.circuit(x, torch_params)
        probs = torch.as_tensor(probs, dtype=torch.float32)

        probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps)
        probs = probs.clamp_min(eps)

        probs = probs[:n_classes]
        probs = probs / (probs.sum() + 1e-12).clamp_min(eps)
        return probs

    # --- eval teacher loss/metrics ---
    @torch.no_grad()
    def eval_teacher(data_list):
        """Evaluate teacher-mode metrics on a dataset.

        Args:
            data_list (Iterable): Iterable of input samples.

        Returns:
            dict[str, float]: Mean loss, fidelity, and angle loss.
        """
        losses = []
        fidelities = []
        angle_losses = []
        for x in data_list:
            phi = _normalize_state(_ensure_complex(target_qnode(x)))
            psi = forward_state(x)
            fid_loss = loss_fn(phi, psi)
            losses.append(fid_loss)
            fidelities.append(1.0 - fid_loss)
            angle_losses.append(loss_state_angle(phi, psi))
        return {
            "loss": float(torch.stack(losses).mean().item()),
            "fidelity_loss": float(torch.stack(losses).mean().item()),
            "fidelity": float(torch.stack(fidelities).mean().item()),
            "angle_loss": float(torch.stack(angle_losses).mean().item()),
        }

    # --- eval supervised metrics ---
    @torch.no_grad()
    def eval_supervised(
        data_list,
        class_counts: Optional[dict],
        alpha=None,
    ):
        """Evaluate supervised classification metrics.

        Args:
            data_list (Iterable): Dataset samples `(x, y, class_id)`.
            class_counts (dict, optional): Per-class sample counts.
            alpha (np.array, optional): Per-class alpha values for balanced loss.

        Returns:
            dict[str, float]: Mean loss and classification accuracy.
        """
        losses = []
        probs = []
        y_onehots = []
        # per_class_pred = {}
        correct = 0
        total = 0
        for x, y, cls in data_list:
            # if cls not in per_class_pred:
            #     per_class_pred[cls] = 0
            p = forward_probs(x)
            eval_loss = ce_onehot_on_probs(
                p,
                y,
                alpha_per_class=alpha,
            )
            losses.append(eval_loss)
            pred = int(torch.argmax(p).item())
            true = int(torch.argmax(y).item())
            correct += int(pred == true)
            total += 1
            probs.append(p)
            y_onehots.append(y)
            # if pred == true:
            #     per_class_pred[cls] += 1

        # log = ""
        # for k, v in class_counts.items():
        #     log += f"[{k}] Accuracy: {per_class_pred[k]/v:.4f} ({per_class_pred[k]}/{v}) | "
        # logger.info(f"{log}")
        probs = torch.stack([p.to(torch.float32) for p in probs], dim=0)
        y_onehots = torch.stack([p.to(torch.float32) for p in y_onehots], dim=0)

        loss = (
            torch.stack(losses).mean()
            if loss_name != "per_class"
            else loss_fn(probs, y_onehots)
        )

        return {
            "loss": float(loss.item()),
            "acc": float(correct / max(total, 1)),
        }

    loss_global = float("inf")

    # ---- training loop ----
    for epoch in range(epochs):
        opt.zero_grad()

        batches_per_epoch = math.ceil(sampler.n_samples / sampler.batch_size)
        logger.debug(
            f"evaluating {batches_per_epoch} batches per epoch. n_samples: {sampler.n_samples}, "
            f"batch_size: {sampler.batch_size}"
        )

        for i in range(batches_per_epoch):

            batch = sampler.sample()

            losses = []
            probs = []
            y_onehots = []
            if use_state:
                for x in batch:
                    # teacher state: NO grad
                    with torch.no_grad():
                        phi = _normalize_state(_ensure_complex(target_qnode(x)))
                    psi = forward_state(x)
                    L = (
                        loss_one_minus_fidelity(phi, psi)
                        if loss_fn is None
                        else loss_fn(phi, psi)
                    )
                    losses.append(L)
            else:
                for x, y, _ in batch:
                    p = forward_probs(x)
                    if loss_name != "per_class":
                        L = (
                            ce_onehot_on_probs(p, y, alpha_per_class=alpha)
                            if loss_fn is None
                            else loss_fn(p, y, alpha_per_class=alpha)
                        )
                        losses.append(L)
                    probs.append(p)
                    y_onehots.append(y)

            probs = torch.stack([p.to(torch.float32) for p in probs], dim=0)
            y_onehots = torch.stack([p.to(torch.float32) for p in y_onehots], dim=0)

            loss = (
                torch.stack(losses).mean()
                if loss_name != "per_class"
                else loss_fn(probs, y_onehots)
            )

            if not loss.requires_grad:
                logger.warning(
                    "Loss has no grad path. Are parameters actually used inside the QNode?"
                )
                metrics = eval_forward_only(
                    genome,
                    train_list,
                    test_list,
                    teacher_qnode=target_qnode,
                    n_classes=n_classes,
                    loss_fn=loss_fn,
                    alpha=alpha,
                )
                genome.fitness = metrics
                return

            opt.zero_grad()
            loss.backward()

            # Sanitize NaN/Inf grads so they cannot persist into params.
            # clip_grad_norm_ does NOT fix NaN — its norm is NaN, so it scales
            # by NaN and leaves NaN gradients in place.
            nonfinite_grad = False
            for p in torch_params.values():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nonfinite_grad = True
                    p.grad = torch.where(
                        torch.isfinite(p.grad),
                        p.grad,
                        torch.zeros_like(p.grad),
                    )

            torch.nn.utils.clip_grad_norm_(list(torch_params.values()), max_norm=1.0)
            opt.step()

            # Belt-and-suspenders: if a param somehow became non-finite,
            # sanitize so NaNs cannot bake into the genome.
            for p in torch_params.values():
                if not torch.isfinite(p.data).all():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    nonfinite_grad = True

            if nonfinite_grad:
                logger.warning(
                    f"NaN/Inf gradients or params detected in genome "
                    f"{getattr(genome, 'genome_number', '?')} epoch {epoch}; sanitized."
                )

        if epoch % log_every == 0 or epoch == epochs - 1:
            if use_state:
                tr = eval_teacher(train_list)
                if test_list is not None:
                    te = eval_teacher(test_list)
                    logger.info(
                        f"[{epoch:04d}] fid_loss={tr['fidelity_loss']:.6f} angle_loss={tr['angle_loss']:.6f} "
                        f"| test_fid_loss={te['fidelity_loss']:.6f}"
                    )
                else:
                    logger.info(
                        f"[{epoch:04d}] fid_loss={tr['fidelity_loss']:.6f} angle_loss={tr['angle_loss']:.6f}"
                    )
            else:
                tr = eval_supervised(train_list, train_data.class_counts, alpha=alpha)
                if test_list is not None:
                    te = eval_supervised(test_list, test_data.class_counts, alpha=alpha)
                    logger.info(
                        f"[{epoch:04d}] loss={tr['loss']:.6f} acc={tr['acc']:.3f} | "
                        f"test_loss={te['loss']:.3f} test_acc={te['acc']:.3f}"
                    )
                else:
                    logger.info(
                        f"[{epoch:04d}] loss={tr['loss']:.6f} acc={tr['acc']:.3f}"
                    )

        torch_params_to_genome(genome, torch_params)

        # return final metrics
        metrics = eval_forward_only(
            genome,
            train_list,
            test_list,
            teacher_qnode=target_qnode,
            n_classes=n_classes,
            loss_fn=loss_fn,
            class_counts=(train_data.class_counts, test_data.class_counts),
            alpha=alpha,
        )

        if metrics["test_loss"] < loss_global:
            loss_global = metrics["test_loss"]
            best_params = copy.deepcopy(torch_params)
            best_metrics = copy.deepcopy(metrics)
            logger.info(f"Saved best model to genome:{genome.genome_number}")

    # Save best loss genome params and metrics
    if best_params is not None:
        torch_params_to_genome(genome, best_params)

    genome.fitness = best_metrics


# ---------- Public unified API ----------


def train_genome_objective(
    genome: CircuitGenome,
    *,
    dataset: list = None,
    teacher_qnode: Optional[Callable] = None,
    backend: str = "pennylane",
    loss: str = "fidelity",
    encoding: str = "angle",
    epochs: int = 200,
    lr: float = 0.05,
    n_classes: int = 3,
    log_every: int = 50,
    batch_size: int = None,
    noise_model: Optional[BaseNoiseModel] = None,
    qiskit_config: Optional[dict[str, Any]] = None,
) -> CircuitGenome:
    """
    Single entry point, backend-dispatched, minimal module hopping.
    """
    if qiskit_config is None:
        qiskit_config = {}

    if backend == "pennylane":
        train_data = dataset[0]
        test_data = dataset[1]
        _train_with_pennylane(
            genome,
            train_data=train_data,
            test_data=test_data,
            epochs=epochs,
            lr=lr,
            loss_name=loss,
            n_classes=n_classes,
            log_every=log_every,
            batch_size=batch_size,
            target_qnode=teacher_qnode,
            encoding=encoding,
            noise_model=noise_model,
        )
        return

    if backend == "qiskit":
        if dataset is None:
            raise ValueError("qiskit backend requires dataset.")

        # New EstimatorQNN-based path that mirrors _train_with_pennylane.
        # qiskit_config keys honored:
        #   gradient_method: "reverse" (default) | "param_shift"
        from src.objectives.qiskit_train import _train_with_qiskit

        gradient_method = qiskit_config.get("gradient_method", "reverse")

        train_data = dataset[0]
        test_data = dataset[1]
        _train_with_qiskit(
            genome,
            train_data=train_data,
            test_data=test_data,
            epochs=epochs,
            lr=lr,
            log_every=log_every,
            n_classes=n_classes,
            batch_size=batch_size,
            loss_name=loss,
            encoding=encoding,
            gradient_method=gradient_method,
            noise_model=noise_model,
        )
        return genome

    raise ValueError(f"Unknown backend: {backend}")
