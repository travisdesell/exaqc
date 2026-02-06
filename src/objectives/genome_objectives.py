from __future__ import annotations

from typing import Optional, Iterable, Callable, Any
import torch
from loguru import logger
from src.circuits.circuit import CircuitGenome
from src.utils.losses import (
    loss_one_minus_fidelity,
    loss_state_angle,
    loss_kl_divergence,
    loss_obs_mse,
    loss_ce,
    ce_onehot_on_probs,
)

LOSS_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "fidelity": loss_one_minus_fidelity,
    "angle": loss_state_angle,
    "kl": loss_kl_divergence,
    "mse": loss_obs_mse,
    "ce": loss_ce,
}

STATEVECTOR_LOSSES = {"fidelity", "angle", "kl", "ce"}


def genome_to_torch_params(genome: CircuitGenome) -> dict[str, torch.nn.Parameter]:
    params: dict[str, torch.nn.Parameter] = {}
    for gate in genome.gates:
        if gate.enabled:
            for name, value in gate.parameters.items():
                key = f"{gate.innovation_number}:{name}"
                params[key] = torch.nn.Parameter(
                    torch.tensor(float(value), dtype=torch.float64)
                )
    return params


def _extract_param_value(v: torch.Tensor | float) -> float:
    """Convert a parameter value (Tensor or float) to float."""
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def torch_params_to_genome(
    genome: CircuitGenome, trained_params: dict[str, torch.Tensor] | dict[str, float]
):
    for gate in genome.gates:
        if gate.enabled:
            for name in gate.parameters.keys():
                key = f"{gate.innovation_number}:{name}"
                if key in trained_params:
                    gate.parameters[name] = _extract_param_value(trained_params[key])


def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if not torch.is_complex(x):
        # if PL returned real state by accident, promote
        x = x.to(torch.complex128 if x.dtype == torch.float64 else torch.complex64)
    return x


def _normalize_state(psi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return psi / (torch.linalg.norm(psi) + eps)


@torch.no_grad()
def compute_teacher_metrics(
    *,
    phi: torch.Tensor,  # target state
    psi: torch.Tensor,  # predicted state
    extra: dict[str, Any] | None = None,
) -> dict[str, float]:
    phi = _normalize_state(_ensure_complex(phi))
    psi = _normalize_state(_ensure_complex(psi))

    out = {}
    out["loss"] = float(loss_one_minus_fidelity(phi, psi).cpu().item())
    out["fidelity_loss"] = float(loss_one_minus_fidelity(phi, psi).cpu().item())
    out["angle_loss"] = float(loss_state_angle(phi, psi).cpu().item())

    # Optional extras if you want them:
    # - readout CE if you have onehot label in extra["y"]
    if extra is not None and "readout_ce" in extra and extra["readout_ce"]:
        # expects extra contains: n_qubits, readout_wires, n_classes, y_onehot
        y = extra["y_onehot"]
        p = extra["readout_probs_fn"](psi)  # returns [n_classes] probs
        out["ce"] = float(ce_onehot_on_probs(p, y).cpu().item())

    return out


@torch.no_grad()
def _eval_teacher_split(data, genome, params, teacher_qnode):
    """Evaluate teacher mode metrics on a data split."""
    if data is None:
        return None
    losses = []
    fids = []
    for item in data:
        x = item if not isinstance(item, (tuple, list)) else item[0]
        phi = torch.as_tensor(teacher_qnode(x))
        psi = genome.circuit(x, params)
        L = loss_one_minus_fidelity(phi, psi)
        losses.append(L)
        fids.append(1.0 - L)
    return {
        "loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
        "fidelity": float(torch.stack(fids).mean().item()) if fids else 0.0,
    }


@torch.no_grad()
def _eval_supervised_split(data, genome, params, n_classes):
    """Evaluate supervised mode metrics on a data split."""
    if data is None:
        return None
    losses = []
    correct = 0
    total = 0
    for x, y in data:
        probs = genome.circuit(x, params)
        probs = torch.as_tensor(probs, dtype=torch.float32)
        probs = probs[:n_classes]
        probs = probs / (probs.sum() + 1e-12)
        L = ce_onehot_on_probs(probs, y)
        losses.append(L)
        pred = int(torch.argmax(probs).item())
        true = int(torch.argmax(y).item())
        correct += int(pred == true)
        total += 1
    return {
        "loss": float(torch.stack(losses).mean().item()) if losses else 0.0,
        "acc": float(correct / max(total, 1)),
    }


@torch.no_grad()
def eval_forward_only(
    genome: CircuitGenome,
    train_list: list,
    test_list: list = None,
    teacher_qnode=None,
    n_classes: int = 3,
):
    mode = "teacher" if teacher_qnode is not None else "supervised"
    params = genome_to_torch_params(genome)

    if mode == "teacher":
        tr = _eval_teacher_split(train_list, genome, params, teacher_qnode)
        te = (
            _eval_teacher_split(test_list, genome, params, teacher_qnode)
            if test_list is not None
            else None
        )
    else:
        tr = _eval_supervised_split(train_list, genome, params, n_classes)
        te = (
            _eval_supervised_split(test_list, genome, params, n_classes)
            if test_list is not None
            else None
        )

    out = {f"{k}": v for k, v in tr.items()} if te is None else {f"{k}": v for k, v in te.items()}
    if te is not None:
        out.update({f"train_{k}": v for k, v in tr.items()})
        out.update({f"test_{k}": v for k, v in te.items()})
    return out


def _train_with_pennylane(
    genome: CircuitGenome,
    train_data: Iterable[tuple[torch.Tensor, torch.Tensor]] = None,
    test_data: Iterable[tuple[torch.Tensor, torch.Tensor]] = None,
    steps: int = 200,
    lr: float = 0.05,
    log_every: int = 25,
    n_classes: int = 3,
    batch_size: int = None,
    loss_name: str = "ce",
    shuffle_each_step: bool = True,
    # NEW:
    target_qnode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    train_list = list(train_data)
    test_list = list(test_data) if test_data is not None else None

    # ----- choose output type -----
    use_state = (
        loss_name in {"fidelity", "angle", "kl"} and target_qnode
    )  # teacher-style losses
    # (You can include "ce" here too if you want CE from state marginal,
    # but right now your CE expects probs, so keep CE on probs.)

    if use_state:
        genome.generate_pennylane_circuit(
            input_mode="angle", return_probs=False, measure_registers=False
        )
        if target_qnode is None:
            raise ValueError(
                "target_qnode is required for teacher training with statevector losses."
            )
    else:
        genome.generate_pennylane_circuit(input_mode="angle", return_probs=True)

    torch_params = genome_to_torch_params(genome)

    # no params: forward-only eval
    if len(torch_params) == 0:
        metrics = eval_forward_only(
            genome,
            train_list,
            test_list,
            teacher_qnode=target_qnode,
            n_classes=n_classes,
        )
        genome.fitness = metrics
        return metrics

    opt = torch.optim.Adam(torch_params.values(), lr=lr, weight_decay=0.0)

    n = len(train_list)
    if batch_size is not None:
        batch_size = max(1, min(batch_size, n))

    def sample_batch():
        if batch_size is None:
            return train_list
        if shuffle_each_step:
            idx = torch.randint(low=0, high=n, size=(batch_size,))
            return [train_list[i] for i in idx.tolist()]
        start = (step * batch_size) % n
        return [train_list[(start + i) % n] for i in range(batch_size)]

    # --- state forward ---
    def forward_state(x: torch.Tensor) -> torch.Tensor:
        psi = genome.circuit(x, torch_params)
        psi = _normalize_state(_ensure_complex(psi))
        return psi

    # --- probs forward (your existing CE path) ---
    def forward_probs(x: torch.Tensor) -> torch.Tensor:
        probs = genome.circuit(x, torch_params)
        probs = torch.as_tensor(probs, dtype=torch.float32)
        probs = probs[:n_classes]
        probs = probs / probs.sum()
        return probs

    # --- eval teacher loss/metrics ---
    @torch.no_grad()
    def eval_teacher(data_list):
        losses = []
        fidelities = []
        angle_losses = []
        for x in data_list:
            phi = _normalize_state(_ensure_complex(target_qnode(x)))
            psi = forward_state(x)
            fid_loss = loss_one_minus_fidelity(phi, psi)
            losses.append(fid_loss)
            fidelities.append(1.0 - fid_loss)
            angle_losses.append(loss_state_angle(phi, psi))
        return {
            "loss": float(torch.stack(losses).mean().item()),
            "fidelity_loss": float(torch.stack(losses).mean().item()),
            "fidelity": float(torch.stack(fidelities).mean().item()),
            "angle_loss": float(torch.stack(angle_losses).mean().item()),
        }

    # --- eval supervised metrics (your old path) ---
    @torch.no_grad()
    def eval_supervised(data_list):
        losses = []
        correct = 0
        total = 0
        for x, y in data_list:  #  y is one-hot
            p = forward_probs(x)
            losses.append(ce_onehot_on_probs(p, y))
            correct += int(torch.argmax(p).item() == torch.argmax(y).item())
            total += 1
        return {
            "loss": float(torch.stack(losses).mean().item()),
            "acc": float(correct / max(total, 1)),
        }

    # ---- training loop ----
    for step in range(steps):
        opt.zero_grad()
        batch = sample_batch()

        losses = []
        if use_state:
            for x in batch:
                # teacher state: NO grad
                with torch.no_grad():
                    phi = _normalize_state(_ensure_complex(target_qnode(x)))
                psi = forward_state(x)
                L = loss_one_minus_fidelity(phi, psi)
                losses.append(L)
        else:
            for x, y in batch:
                p = forward_probs(x)
                L = ce_onehot_on_probs(p, y)
                losses.append(L)

        loss = torch.stack(losses).mean()

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
            )
            genome.fitness = metrics
            return metrics

        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            if use_state:
                tr = eval_teacher(train_list)
                if test_list is not None:
                    te = eval_teacher(test_list)
                    logger.info(
                        f"[{step:04d}] fid_loss={tr['fidelity_loss']:.6f} angle_loss={tr['angle_loss']:.6f} | test_fid_loss={te['fidelity_loss']:.6f}"  # noqa
                    )
                else:
                    logger.info(
                        f"[{step:04d}] fid_loss={tr['fidelity_loss']:.6f} angle_loss={tr['angle_loss']:.6f}"
                    )
            else:
                tr = eval_supervised(train_list)
                if test_list is not None:
                    te = eval_supervised(test_list)
                    logger.info(
                        f"[{step:04d}] loss={tr['loss']:.6f} acc={tr['acc']:.3f} | test_acc={te['acc']:.3f}"
                    )
                else:
                    logger.info(
                        f"[{step:04d}] loss={tr['loss']:.6f} acc={tr['acc']:.3f}"
                    )

    torch_params_to_genome(genome, torch_params)

    # return final metrics
    metrics = eval_forward_only(
        genome,
        train_list,
        test_list,
        teacher_qnode=target_qnode,
        n_classes=n_classes,
    )
    genome.fitness = metrics

    return metrics


# ---------- Qiskit ML route (TorchConnector + output-bit loss) ----------


def _train_with_qiskit_ml_outputs(
    genome: CircuitGenome,
    *,
    dataset: Iterable[tuple[torch.Tensor, Any]],
    n_qubits: int,
    input_qubits: list[int],
    output_qubits: list[int],
    x_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    y_extractor: Optional[Callable[[Any], torch.Tensor]] = None,
    steps: int = 300,
    lr: float = 0.05,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> dict[str, float]:
    """
    Generic Qiskit ML + Torch training for genomes.

    Learns weights of a parametric circuit (from genome) using EstimatorQNN outputs:
      - features: x (derived from input_bits)
      - labels:   y (derived from target object)
      - outputs:  Z expvals on output_qubits -> converted to p(bit=1)

    The abstraction points are x_extractor and y_extractor.

    Requirements:
      - Gate.add_to_qiskit_circuit(... backend="qiskit_ml") creates/caches Parameters internally
      - dataset yields (input_bits, target_obj)
    """
    import numpy as np
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector

    try:
        from qiskit.primitives import StatevectorEstimator as PrimitiveEstimator
    except Exception as e:
        raise RuntimeError(
            "Need qiskit.primitives.Estimator for Qiskit ML backend."
        ) from e
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # Gradient classes live here in modern Qiskit
    from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

    if x_extractor is None:
        # Default: use raw bits on input_qubits as float vector
        def x_extractor(input_bits: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                [float(input_bits[q].item()) for q in input_qubits], dtype=torch.float32
            )

    if y_extractor is None:
        raise ValueError(
            "y_extractor is required (how to turn target into training labels)."
        )

    if loss_fn is None:
        # Default: MSE
        loss_fn = loss_obs_mse

    # ----- Feature map: RX(pi*x) on input qubits -----
    x_params = ParameterVector("x", len(input_qubits))
    feature_map = QuantumCircuit(n_qubits)
    for i, q in enumerate(input_qubits):
        feature_map.rx(np.pi * x_params[i], q)

    # ----- Ansatz from genome (parametric via Gate backend='qiskit_ml') -----
    qregs = {
        name: QuantumRegister(size, name=name)
        for name, size in genome.registers.items()
    }
    ansatz = QuantumCircuit(*qregs.values())

    genome.sort_gates()
    for gate in genome.gates:
        gate.add_to_qiskit_circuit(qregs, ansatz)

    # ansatz = genome.generate_qiskit_circuit(measure_registers=True)

    # compose
    full = QuantumCircuit(n_qubits)
    full.compose(feature_map, inplace=True)
    full.compose(ansatz, inplace=True)

    # ----- Stable weight order (Gate-owned Parameters) -----
    weight_params = []
    weight_keys = []
    genome.sort_gates()
    for gate in genome.gates:
        if not gate.enabled:
            continue
        qparams = gate._get_qiskit_parameters()  # {pname: Parameter}
        for pname in sorted(qparams.keys()):
            weight_params.append(qparams[pname])
            weight_keys.append(f"{gate.innovation_number}:{pname}")

    # ----- Observables: Z on output qubits -----
    observables = []
    for q in output_qubits:
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - q] = "Z"  # Qiskit: rightmost is qubit 0
        observables.append(SparsePauliOp("".join(pauli)))

    pm = generate_preset_pass_manager(optimization_level=1)  # good default
    estimator = PrimitiveEstimator()
    gradient = ParamShiftEstimatorGradient(estimator=estimator)

    qnn = EstimatorQNN(
        circuit=full,
        estimator=estimator,
        input_params=list(x_params),
        weight_params=weight_params,
        observables=observables,
        gradient=gradient,
        pass_manager=pm,
    )

    model = TorchConnector(qnn)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    # ----- Train -----
    for _ in range(steps):
        opt.zero_grad()
        losses = []
        for input_bits, target_obj in dataset:
            x = x_extractor(input_bits)  # shape [len(input_qubits)]
            y = y_extractor(target_obj)  # shape [len(output_qubits)] or compatible

            expvals = model(x)  # [-1,1]
            p1 = (1.0 - expvals) / 2.0  # prob(bit=1)
            losses.append(loss_fn(p1, y))

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

    # ----- Write weights back to genome -----
    trained_w = model.weight.detach().cpu().flatten()
    trained = {
        weight_keys[i]: float(trained_w[i].item()) for i in range(len(weight_keys))
    }

    for gate in genome.gates:
        for pname in gate.parameters.keys():
            k = f"{gate.innovation_number}:{pname}"
            if k in trained:
                gate.parameters[pname] = trained[k]

    # ----- Eval avg loss -----
    with torch.no_grad():
        eval_losses = []
        for input_bits, target_obj in dataset:
            x = x_extractor(input_bits)
            y = y_extractor(target_obj)
            expvals = model(x)
            p1 = (1.0 - expvals) / 2.0
            eval_losses.append(loss_fn(p1, y))
        avg_loss = float(torch.stack(eval_losses).mean().cpu().item())

    return {"loss": avg_loss}


# ---------- Public unified API ----------


def train_genome_objective(
    genome: CircuitGenome,
    *,
    target_state: Optional[torch.Tensor] = None,
    input_bits: Optional[torch.Tensor] = None,
    # dataset: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
    dataset: list = None,
    teacher_qnode: Optional[Callable] = None,
    backend: str = "pennylane",
    loss: str = "fidelity",
    steps: int = 200,
    lr: float = 0.05,
    n_classes: int = 3,
    log_every: int = 50,
    bath_size: int = None,
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
        metrics = _train_with_pennylane(
            genome,
            train_data=train_data,
            test_data=test_data,
            steps=steps,
            lr=lr,
            loss_name=loss,
            n_classes=n_classes,
            log_every=log_every,
            batch_size=bath_size,
            target_qnode=teacher_qnode,
        )
        genome.fitness = metrics
        # logger.debug(f"genome fitness: {genome.fitness}")
        return genome

    if backend == "qiskit":
        if dataset is None:
            raise ValueError("qiskit_ml requires dataset.")

        n_qubits = qiskit_config.get("n_qubits")
        input_qubits = qiskit_config.get("input_qubits")
        output_qubits = qiskit_config.get("output_qubits")
        x_extractor = qiskit_config.get("x_extractor")
        y_extractor = qiskit_config.get("y_extractor")
        loss_fn = LOSS_REGISTRY[loss]

        if n_qubits is None or input_qubits is None or output_qubits is None:
            raise ValueError(
                "qiskit_ml requires n_qubits, input_qubits, output_qubits in qiskit_config."
            )

        metrics = _train_with_qiskit_ml_outputs(
            genome,
            dataset=dataset,
            n_qubits=n_qubits,
            input_qubits=input_qubits,
            output_qubits=output_qubits,
            x_extractor=x_extractor,
            y_extractor=y_extractor,
            steps=steps,
            lr=lr,
            loss_fn=loss_fn,
        )
        genome.fitness = metrics
        return genome

    raise ValueError(f"Unknown backend: {backend}")
