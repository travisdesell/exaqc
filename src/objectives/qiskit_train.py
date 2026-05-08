"""Qiskit-side training that mirrors `_train_with_pennylane`.

Architecture:
    - Build a parametric ansatz from the genome via
      `genome.generate_qiskit_circuit_parametric()`.
    - Prepend a feature map (RY(pi*x) for angle encoding) using qiskit
      ParameterVector for the inputs.
    - Build joint-probability projector observables over the output qubits
      (so the QNN forward returns the same shape vector as PennyLane's
      `qml.probs(wires=output)`).
    - Wrap in `EstimatorQNN` with either `ReverseEstimatorGradient`
      (default, simulator backprop) or `ParamShiftEstimatorGradient`
      (hardware-ready param-shift).
    - Wrap that in `TorchConnector` so the Adam loop and NaN sanitizer
      block from the pennylane path are reused verbatim.
"""
from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import numpy as np
import torch
from loguru import logger

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import (
    ParamShiftEstimatorGradient,
    ReverseEstimatorGradient,
)
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from src.utils.helpers import (
    BalancedBatchSampler,
    genome_to_torch_params,
    torch_params_to_genome,
)
from src.utils.losses import (
    LOSS_REGISTRY,
    ce_onehot_on_probs,
)
from src.utils.qiskit_observables import output_projector_observables

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome


GRADIENT_METHODS = {"reverse", "param_shift"}


def _build_qnn(
    genome: "CircuitGenome",
    *,
    encoding: str,
    gradient_method: str,
    estimator,
) -> tuple[EstimatorQNN, list]:
    """Compose feature_map + ansatz, attach observables + gradient.

    Returns (qnn, ordered_weight_params) where ordered_weight_params is
    the qiskit Parameter list in canonical order so we can map TorchConnector
    weights back into genome gate parameters at the end.
    """
    if encoding != "angle":
        raise NotImplementedError(
            f"qiskit train currently only supports encoding='angle' (got '{encoding}')"
        )

    ansatz, weight_params = genome.generate_qiskit_circuit_parametric()
    n_total = ansatz.num_qubits
    input_indexes = list(genome.input_indexes)

    # Feature map: RY(pi * x_i) on input qubits, mirrors pennylane's angle path.
    x_params = ParameterVector("x", len(input_indexes))
    feature_map = QuantumCircuit(n_total)
    for i, q in enumerate(input_indexes):
        feature_map.ry(np.pi * x_params[i], q)

    full = QuantumCircuit(n_total)
    full.compose(feature_map, inplace=True)
    full.compose(ansatz, inplace=True)

    observables = output_projector_observables(n_total, list(genome.output_indexes))

    if gradient_method == "reverse":
        gradient = ReverseEstimatorGradient()
    elif gradient_method == "param_shift":
        gradient = ParamShiftEstimatorGradient(estimator=estimator)
    else:
        raise ValueError(
            f"gradient_method must be one of {GRADIENT_METHODS}, got {gradient_method!r}"
        )

    qnn = EstimatorQNN(
        circuit=full,
        estimator=estimator,
        input_params=list(x_params),
        weight_params=weight_params,
        observables=observables,
        gradient=gradient,
    )
    return qnn, weight_params


def _initial_torch_weights(
    genome: "CircuitGenome",
    weight_params: list,
) -> torch.Tensor:
    """Pack genome.parameters values into a torch tensor in `weight_params` order."""
    name_to_value: dict[str, float] = {}
    for gate in genome.gates:
        if not gate.enabled:
            continue
        for pname, value in gate.parameters.items():
            qparam = gate._get_qiskit_parameters()[pname]
            name_to_value[qparam.name] = float(value)

    init = torch.tensor(
        [name_to_value[p.name] for p in weight_params],
        dtype=torch.float64,
    )
    return init


def _write_back_weights(
    genome: "CircuitGenome",
    weight_params: list,
    trained: torch.Tensor,
) -> None:
    """Push trained weights back into genome.parameters."""
    name_to_value = {p.name: float(trained[i].detach()) for i, p in enumerate(weight_params)}
    for gate in genome.gates:
        if not gate.enabled:
            continue
        for pname in gate.parameters.keys():
            qparam = gate._get_qiskit_parameters()[pname]
            if qparam.name in name_to_value:
                gate.parameters[pname] = name_to_value[qparam.name]


def _train_with_qiskit(
    genome: "CircuitGenome",
    *,
    train_data: Iterable,
    test_data: Optional[Iterable] = None,
    epochs: int = 30,
    lr: float = 5e-4,
    log_every: int = 1,
    n_classes: int = 3,
    batch_size: Optional[int] = None,
    loss_name: str = "ce",
    encoding: str = "angle",
    gradient_method: str = "reverse",
    noise_model: Any = None,
    shuffle_each_epoch: bool = True,
) -> None:
    """Mirror of _train_with_pennylane on the qiskit/EstimatorQNN backend.

    Sets `genome.fitness` in-place and writes trained weights back into
    `genome.gates[*].parameters`.
    """
    train_list = list(train_data)
    test_list = list(test_data) if test_data is not None else None

    loss_fn = LOSS_REGISTRY[loss_name]

    beta = (len(train_list) - 1) / max(len(train_list), 1)
    alpha = (1.0 - beta) / (
        1.0 - np.power(beta, np.array(train_data.counts, dtype=np.float32))
    )
    alpha = torch.as_tensor(alpha / alpha.mean(), dtype=torch.float32)

    n = len(train_list)
    if batch_size is not None:
        batch_size = max(1, min(batch_size, n))

    sampler = BalancedBatchSampler(
        data=train_list, batch_size=batch_size, shuffle=shuffle_each_epoch
    )

    # Estimator selection: noiseless StatevectorEstimator if no noise;
    # AerEstimator with the qiskit noise model otherwise.
    if noise_model is not None:
        # ReverseEstimatorGradient does its own classical statevector tracking
        # and ignores the estimator -- so combining it with a noisy AerEstimator
        # would give noise-free gradients with noisy forward (semantically wrong).
        # Auto-switch to param_shift, which calls the estimator twice per param
        # and so respects the noise model.
        if gradient_method == "reverse":
            logger.warning(
                "noise_model provided but gradient_method='reverse'; auto-switching "
                "to 'param_shift' so gradients respect the noise. "
                "Pass gradient_method='param_shift' explicitly to silence this."
            )
            gradient_method = "param_shift"
        estimator = _build_aer_estimator(noise_model)
    else:
        estimator = StatevectorEstimator()

    qnn, weight_params = _build_qnn(
        genome,
        encoding=encoding,
        gradient_method=gradient_method,
        estimator=estimator,
    )

    if len(weight_params) == 0:
        # No-trainable-params: just a forward pass for fitness.
        logger.info(f"genome {genome.genome_number} has no trainable params; eval only")
        model = TorchConnector(qnn)
        metrics = _eval_with_qnn(model, train_list, test_list, n_classes, alpha, loss_fn, loss_name)
        genome.fitness = metrics
        return

    initial_weights = _initial_torch_weights(genome, weight_params)
    model = TorchConnector(qnn, initial_weights=initial_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    def forward_probs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # TorchConnector returns expvals of the projector observables, which
        # equal joint probabilities. Sanitize as pennylane path does.
        probs = model(x.to(torch.float64))
        probs = torch.as_tensor(probs, dtype=torch.float32)
        probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
        probs = probs[:n_classes]
        probs = probs / (probs.sum() + 1e-12).clamp_min(eps)
        return probs

    loss_global = float("inf")
    best_state = None
    best_metrics = None

    for epoch in range(epochs):
        batches_per_epoch = math.ceil(sampler.n_samples / sampler.batch_size)

        for _ in range(batches_per_epoch):
            batch = sampler.sample()

            losses = []
            probs_batch = []
            y_onehots = []
            for x, y, _ in batch:
                p = forward_probs(x)
                if loss_name != "per_class":
                    L = (
                        ce_onehot_on_probs(p, y, alpha_per_class=alpha)
                        if loss_fn is None
                        else loss_fn(p, y, alpha_per_class=alpha)
                    )
                    losses.append(L)
                probs_batch.append(p)
                y_onehots.append(y)

            probs_stack = torch.stack([p.to(torch.float32) for p in probs_batch], dim=0)
            y_stack = torch.stack([p.to(torch.float32) for p in y_onehots], dim=0)

            loss = (
                torch.stack(losses).mean()
                if loss_name != "per_class"
                else loss_fn(probs_stack, y_stack)
            )

            if not loss.requires_grad:
                logger.warning(f"loss has no grad path (genome {genome.genome_number})")
                metrics = _eval_with_qnn(model, train_list, test_list, n_classes, alpha, loss_fn, loss_name)
                genome.fitness = metrics
                return

            opt.zero_grad()
            loss.backward()

            # Same NaN sanitizer as pennylane path.
            nonfinite = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nonfinite = True
                    p.grad = torch.where(
                        torch.isfinite(p.grad),
                        p.grad,
                        torch.zeros_like(p.grad),
                    )

            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
            opt.step()

            for p in model.parameters():
                if not torch.isfinite(p.data).all():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    nonfinite = True

            if nonfinite:
                logger.warning(
                    f"NaN/Inf grads or weights in genome {genome.genome_number} "
                    f"epoch {epoch}; sanitized."
                )

        if epoch % log_every == 0 or epoch == epochs - 1:
            metrics = _eval_with_qnn(model, train_list, test_list, n_classes, alpha, loss_fn, loss_name)
            logger.info(
                f"[{epoch:04d}] qiskit/{gradient_method} "
                f"loss={metrics['train_loss']:.4f} acc={metrics['train_acc']:.3f} "
                f"| test_loss={metrics.get('test_loss', float('nan')):.4f} "
                f"test_acc={metrics.get('test_acc', float('nan')):.3f}"
            )
            if metrics.get("test_loss", float("inf")) < loss_global:
                loss_global = metrics["test_loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = copy.deepcopy(metrics)

    if best_state is not None:
        model.load_state_dict(best_state)
        genome.fitness = best_metrics
    else:
        genome.fitness = _eval_with_qnn(model, train_list, test_list, n_classes, alpha, loss_fn, loss_name)

    # Push trained weights into the genome's gate.parameters.
    trained_tensor = list(model.parameters())[0].detach()
    _write_back_weights(genome, weight_params, trained_tensor)


def _build_aer_estimator(noise_model):
    """Build an AerEstimator wrapping a qiskit noise model.

    Stage 4 will flesh this out; left as a hook so Stage 3 stays focused.
    """
    # Imported lazily so users without qiskit-aer don't pay the import cost.
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    qiskit_noise = (
        noise_model.to_qiskit_noise_model()
        if hasattr(noise_model, "to_qiskit_noise_model")
        else noise_model
    )

    options = {
        "backend_options": {
            "method": "density_matrix",
            "noise_model": qiskit_noise,
        },
        "default_precision": 0.0,  # exact density-matrix expectation, no shots
    }
    return AerEstimator(options=options)


@torch.no_grad()
def _eval_with_qnn(
    model: TorchConnector,
    train_list,
    test_list,
    n_classes: int,
    alpha,
    loss_fn,
    loss_name: str,
) -> dict[str, float]:
    eps = 1e-8

    def _split(data):
        if data is None:
            return None
        losses = []
        probs_all = []
        y_all = []
        correct = 0
        total = 0
        for x, y, _ in data:
            probs = model(x.to(torch.float64))
            probs = torch.as_tensor(probs, dtype=torch.float32)
            probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
            probs = probs[:n_classes]
            probs = probs / (probs.sum() + 1e-12).clamp_min(eps)
            L = ce_onehot_on_probs(probs, y, alpha_per_class=alpha)
            losses.append(L)
            pred = int(torch.argmax(probs).item())
            true = int(torch.argmax(y).item())
            correct += int(pred == true)
            total += 1
            probs_all.append(probs)
            y_all.append(y)

        if loss_name != "per_class":
            avg = float(torch.stack(losses).mean().item()) if losses else 0.0
        else:
            ps = torch.stack([p.to(torch.float32) for p in probs_all], dim=0)
            ys = torch.stack([p.to(torch.float32) for p in y_all], dim=0)
            avg = float(loss_fn(ps, ys))

        return {"loss": avg, "acc": float(correct / max(total, 1))}

    out = {}
    tr = _split(train_list)
    te = _split(test_list) if test_list is not None else None
    out["train_loss"] = tr["loss"]
    out["train_acc"] = tr["acc"]
    if te is not None:
        out["test_loss"] = te["loss"]
        out["test_acc"] = te["acc"]
    out["loss"] = tr["loss"]
    out["acc"] = tr["acc"]
    return out
