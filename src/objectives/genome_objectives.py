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
    loss_readout_ce_from_state,
)

LOSS_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "fidelity": loss_one_minus_fidelity,
    "angle": loss_state_angle,
    "kl": loss_kl_divergence,
    "mse": loss_obs_mse,
    "ce": loss_readout_ce_from_state,
}

STATEVECTOR_LOSSES = {"fidelity", "angle", "kl", "ce"}

# ---------- Shared param IO ----------


def genome_to_torch_params(genome: CircuitGenome) -> dict[str, torch.nn.Parameter]:
    params: dict[str, torch.nn.Parameter] = {}
    for gate in genome.gates:
        for name, value in gate.parameters.items():
            # key = f"{gate.innovation_number}:{name}"
            key = name
            params[key] = torch.nn.Parameter(
                torch.tensor(float(value), dtype=torch.float64)
            )
    return params


def torch_params_to_genome(
    genome: CircuitGenome, trained_params: dict[str, torch.Tensor] | dict[str, float]
):
    for gate in genome.gates:
        for name in gate.parameters.keys():
            # key = f"{gate.innovation_number}:{name}"
            key = name
            if key in trained_params:
                v = trained_params[key]
                if isinstance(v, torch.Tensor):
                    gate.parameters[name] = float(v.detach().cpu().item())
                else:
                    gate.parameters[name] = float(v)


# ---------- Shared metric packing (optional) ----------


def _compute_state_metrics(
    phi: torch.Tensor, psi: torch.Tensor, loss_kwargs: dict = None
) -> dict[str, float]:
    psi = psi / torch.linalg.norm(psi)
    phi = phi / torch.linalg.norm(phi)

    # fid_loss = float(loss_one_minus_fidelity(phi, psi).detach().cpu().item())
    # angle = float(loss_state_angle(phi, psi).detach().cpu().item())

    # probs_psi = statevector_to_probs(psi)
    # probs_phi = statevector_to_probs(phi)

    # tv = float(loss_total_variation(probs_phi, probs_psi).detach().cpu().item())
    # kl = float(
    #     loss_kl_divergence(
    #         probs_phi.clamp_min(1e-12),
    #         probs_psi.clamp_min(1e-12),
    #     )
    #     .detach()
    #     .cpu()
    #     .item()
    # )

    ce_loss = loss_readout_ce_from_state(
        phi,
        psi,
        loss_kwargs["n_qubits"],
        loss_kwargs["readout_wires"],
        loss_kwargs["n_classes"],
    )

    return {
        # "fidelity_loss": fid_loss,
        # "angle_loss": angle,
        # "total_variation": tv,
        # "kl_divergence": kl,
        "ce": ce_loss,
    }


def ce_onehot_on_probs(
    probs: torch.Tensor, y_onehot: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    probs: shape [K], real, sums to 1 (we’ll enforce)
    y_onehot: shape [K], real one-hot
    """
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum()
    y_onehot = y_onehot.to(dtype=probs.dtype, device=probs.device)
    return -(y_onehot * torch.log(probs)).sum()


def eval_pennylane_forward_only(
    genome,
    data: Iterable[tuple[torch.Tensor, torch.Tensor]],
    parameters: dict[str, Any] = {},
    n_classes: int = 3,
) -> dict[str, float]:
    """
    data items: (x, y_onehot)
      x: float tensor (e.g., iris [4])
      y_onehot: float tensor [n_classes]
    Circuit must return probs on output wires.
    """
    # Ensure circuit exists and returns probs
    genome.generate_pennylane_circuit(input_mode="angle", return_probs=True)
    # circuit = genome.circuit

    # Empty params dict (since no trainable params)
    params: dict[str, torch.Tensor] = {}
    if len(parameters) > 0:
        params = parameters

    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data:
            probs = genome.circuit(x, params)
            probs = torch.as_tensor(probs, dtype=torch.float32)

            # if output has 2 qubits -> 4 probs; iris -> use first 3 and renorm
            probs = probs[:n_classes]
            probs = probs / probs.sum()

            L = ce_onehot_on_probs(probs, y)
            losses.append(L)

            pred = int(torch.argmax(probs).item())
            true = int(torch.argmax(y).item())
            correct += pred == true
            total += 1

    avg_loss = float(torch.stack(losses).mean().item()) if losses else 0.0
    acc = float(correct / max(total, 1))
    return {"loss": avg_loss, "acc": acc}


def _train_with_pennylane(
    genome,
    train_data: Iterable[tuple[torch.Tensor, torch.Tensor]] = None,
    test_data: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    *,
    steps: int = 200,
    lr: float = 0.05,
    log_every: int = 25,
    n_classes: int = 3,
    loss_name: str = "ce",
):
    """
    Expects dataset items: (x, y_onehot)
      x: torch.float32 shape [4] for iris
      y_onehot: torch.float32 shape [3]
    """
    # build qnode (returns probs on output wires)
    genome.generate_pennylane_circuit(input_mode="angle", return_probs=True)

    # params as torch.nn.Parameter dict
    torch_params = genome_to_torch_params(genome)
    if len(torch_params) == 0:
        # no params -> forward-only
        metrics = eval_pennylane_forward_only(genome, train_data, n_classes=3)
        # optionally also test metrics:
        # test_metrics = eval_pennylane_forward_only(genome, test_data, n_classes=3)
        genome.fitness = metrics
        return genome

    opt = torch.optim.Adam(torch_params.values(), lr=lr, weight_decay=0.0001)

    train_list = list(train_data)
    if test_data is not None:
        test_list = list(test_data)
    else:
        test_list = None

    def forward_probs(x: torch.Tensor) -> torch.Tensor:
        probs = genome.circuit(x, torch_params)  # shape [2**n_out]
        probs = torch.as_tensor(probs, dtype=torch.float32)
        # iris: use first n_classes bins (e.g. |00>,|01>,|10>) and renormalize
        probs = probs[:n_classes]
        probs = probs / probs.sum()
        return probs

    def accuracy(data_list) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_list:
                p = forward_probs(x)
                pred = int(torch.argmax(p).item())
                true = int(torch.argmax(y).item())
                correct += pred == true
                total += 1
        return correct / max(total, 1)

    # logger.debug(f"torch params before training: {torch_params}")

    # ---- training loop ----
    for step in range(steps):
        opt.zero_grad()

        losses = []
        for x, y in train_list:
            probs = forward_probs(x)
            L = ce_onehot_on_probs(probs, y)
            losses.append(L)

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        # logger.debug(f"torch params after training: {torch_params}")

        if step % log_every == 0 or step == steps - 1:
            train_acc = accuracy(train_list)
            if test_list is not None:
                test_acc = accuracy(test_list)
                logger.info(
                    f"[{step:04d}] loss={loss.item():.6f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
                )
            else:
                logger.info(
                    f"[{step:04d}] loss={loss.item():.6f} train_acc={train_acc:.3f}"
                )
            # test_metrics = eval_pennylane_forward_only(
            #     genome, test_data, parameters=torch_params, n_classes=3
            # )
            # genome.fitness = test_metrics

    # write params back
    torch_params_to_genome(genome, torch_params)
    return genome


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
        # from qiskit.primitives import Estimator as PrimitiveEstimator
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
    backend: str = "pennylane",
    loss: str = "fidelity",
    steps: int = 200,
    lr: float = 0.05,
    log_every: int = 50,
    qiskit_config: Optional[dict[str, Any]] = None,
) -> CircuitGenome:
    """
    Single entry point, backend-dispatched, minimal module hopping.
    """
    if qiskit_config is None:
        qiskit_config = {}

    if backend == "pennylane":
        # if target_state is None or input_bits is None:
        #     raise ValueError("PennyLane backend requires target_state and input_bits.")
        train_data = dataset[0]
        test_data = dataset[1]
        metrics = _train_with_pennylane(
            genome,
            train_data=train_data,
            test_data=test_data,
            steps=steps,
            lr=lr,
            loss_name=loss,
            log_every=log_every,
        )
        genome.fitness = metrics
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
