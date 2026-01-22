from __future__ import annotations

from typing import Optional, Iterable, Callable, Any
import torch

from src.circuits.circuit import CircuitGenome
from src.trainer import QuantumStateTrainer
from src.utils.losses import (
    loss_one_minus_fidelity,
    loss_state_angle,
    loss_total_variation,
    loss_kl_divergence,
    loss_obs_mse,
    statevector_to_probs,
)

LOSS_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "fidelity": loss_one_minus_fidelity,
    "angle": loss_state_angle,
    "kl": loss_kl_divergence,
    "mse": loss_obs_mse,
}

# ---------- Shared param IO ----------


def genome_to_torch_params(genome: CircuitGenome) -> dict[str, torch.nn.Parameter]:
    params: dict[str, torch.nn.Parameter] = {}
    for gate in genome.gates:
        for name, value in gate.parameters.items():
            key = f"{gate.innovation_number}:{name}"
            params[key] = torch.nn.Parameter(
                torch.tensor(float(value), dtype=torch.float64)
            )
    return params


def torch_params_to_genome(
    genome: CircuitGenome, trained_params: dict[str, torch.Tensor] | dict[str, float]
):
    for gate in genome.gates:
        for name in gate.parameters.keys():
            key = f"{gate.innovation_number}:{name}"
            if key in trained_params:
                v = trained_params[key]
                if isinstance(v, torch.Tensor):
                    gate.parameters[name] = float(v.detach().cpu().item())
                else:
                    gate.parameters[name] = float(v)


# ---------- Shared metric packing (optional) ----------


def _compute_state_metrics(phi: torch.Tensor, psi: torch.Tensor) -> dict[str, float]:
    psi = psi / torch.linalg.norm(psi)
    phi = phi / torch.linalg.norm(phi)

    fid_loss = float(loss_one_minus_fidelity(phi, psi).detach().cpu().item())
    angle = float(loss_state_angle(phi, psi).detach().cpu().item())

    probs_psi = statevector_to_probs(psi)
    probs_phi = statevector_to_probs(phi)

    tv = float(loss_total_variation(probs_phi, probs_psi).detach().cpu().item())
    kl = float(
        loss_kl_divergence(
            probs_phi.clamp_min(1e-12),
            probs_psi.clamp_min(1e-12),
        )
        .detach()
        .cpu()
        .item()
    )

    return {
        "fidelity_loss": fid_loss,
        "angle_loss": angle,
        "total_variation": tv,
        "kl_divergence": kl,
    }


# ---------- PennyLane route (statevector + fidelity etc.) ----------

# def _train_with_pennylane(
#     genome: CircuitGenome,
#     *,
#     target_state: Optional[torch.Tensor] = None,
#     input_bits: Optional[torch.Tensor] = None,
#     dataset: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
#     steps: int = 200,
#     lr: float = 0.05,
#     loss: str = "fidelity",
#     log_every: int = 50,
# ) -> dict[str, float]:
#     """
#     Uses genome.circuit (PennyLane QNode) directly; trains genome parameters with QuantumStateTrainer.
#     """
#     if dataset is not None:
#         cases = list(dataset)
#         if len(cases) == 0:
#             raise ValueError("dataset is empty")
#     else:
#         if target_state is None or input_bits is None:
#             raise ValueError("Provide either dataset or (input_bits, target_state).")
#         cases = [(input_bits, target_state)]

#     if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
#         genome.generate_pennylane_circuit()

#     torch_params = genome_to_torch_params(genome)
#     param_keys = list(torch_params.keys())
#     param_values = list(torch_params.values())

#     if len(torch_params) == 0:
#         # no params: just evaluate
#         qnode = genome.circuit
#         with torch.no_grad():
#             psi = qnode(input_bits, torch_params)
#         return _compute_state_metrics(target_state, psi)

#     # def model_fn(*flat_params, in_bits: torch.Tensor):
#     #     # convert flat list -> dict expected by qnode
#     #     p = dict(zip(param_keys, flat_params))
#     #     return genome.circuit(in_bits, p)

#     def model_fn(*flat_params_and_args):
#         *flat_params, in_bits = flat_params_and_args
#         p = dict(zip(param_keys, flat_params))
#         return genome.circuit(in_bits, p)


#     # Target as a callable of input_bits (trainer will call target(*model_args))
#     target_map = {tuple(ib.tolist()): ts for ib, ts in cases}

#     def target_callable(in_bits: torch.Tensor):
#         return target_map[tuple(in_bits.tolist())]

#     trainer = QuantumStateTrainer(
#         model_fn=model_fn,
#         params=param_values,
#         target=target_callable,
#         loss_name=loss,
#         normalize_states=True,
#     )

#     model_args_list = [(ib,) for ib, _ in cases]
#     trainer.fit(
#         steps=steps,
#         lr=lr,
#         log_every=log_every,
#         model_args_list=model_args_list,
#         )

#     # write back
#     torch_params_to_genome(genome, torch_params)

#     # final eval
#     with torch.no_grad():
#         psi = genome.circuit(input_bits, torch_params)
#     return _compute_state_metrics(target_state, psi)


def _train_with_pennylane(
    genome: CircuitGenome,
    *,
    target_state: Optional[torch.Tensor] = None,
    input_bits: Optional[torch.Tensor] = None,
    dataset: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
    steps: int = 200,
    lr: float = 0.05,
    loss: str = "fidelity",
    log_every: int = 50,
) -> dict[str, float]:
    """
    Uses genome.circuit (PennyLane QNode) directly; trains genome parameters with QuantumStateTrainer.

    Supports:
      - dataset mode: trains and evaluates averaged metrics across all cases
      - single-case mode: trains and evaluates on (input_bits, target_state)
      - zero-parameter circuits: evaluates only (no training)
    """
    # ---- Build cases ----
    if dataset is not None:
        cases = list(dataset)
        if len(cases) == 0:
            raise ValueError("dataset is empty")
    else:
        if target_state is None or input_bits is None:
            raise ValueError("Provide either dataset or (input_bits, target_state).")
        cases = [(input_bits, target_state)]

    # ---- Ensure circuit exists ----
    if getattr(genome, "circuit", None) is None or not callable(genome.circuit):
        # IMPORTANT: for statevector training, keep measure_registers=False in your CircuitGenome
        genome.generate_pennylane_circuit(measure_registers=(loss != "fidelity"))

    # ---- Extract params ----
    torch_params = genome_to_torch_params(genome)
    param_keys = list(torch_params.keys())
    param_values = list(torch_params.values())

    # ---- Helper: evaluate averaged metrics across cases ----
    def eval_avg_metrics() -> dict[str, float]:
        metrics_list = []
        with torch.no_grad():
            for ib, ts in cases:
                psi = genome.circuit(ib, torch_params)
                metrics_list.append(_compute_state_metrics(ts, psi))

        # average each metric key
        keys = metrics_list[0].keys()
        return {
            k: float(sum(m[k] for m in metrics_list) / len(metrics_list)) for k in keys
        }

    # ---- No trainable params: just evaluate ----
    if len(torch_params) == 0:
        return eval_avg_metrics()

    # ---- Model fn for Trainer (Option C compatible) ----
    def model_fn(*flat_params_and_args):
        # Trainer calls: model_fn(*params, *model_args)
        # model_args is (in_bits,)
        *flat_params, in_bits = flat_params_and_args
        p = dict(zip(param_keys, flat_params))
        return genome.circuit(in_bits, p)

    # ---- Target callable keyed by input bits ----
    target_map = {tuple(ib.tolist()): ts for ib, ts in cases}

    def target_callable(in_bits: torch.Tensor):
        return target_map[tuple(in_bits.tolist())]

    # ---- Train ----
    trainer = QuantumStateTrainer(
        model_fn=model_fn,
        params=param_values,
        target=target_callable,
        loss_name=loss,
        normalize_states=True,
    )

    model_args_list = [(ib,) for ib, _ in cases]
    trainer.fit(
        steps=steps,
        lr=lr,
        log_every=log_every,
        model_args_list=model_args_list,
    )

    # ---- Write trained params back into genome ----
    torch_params_to_genome(genome, torch_params)

    # ---- Final averaged evaluation ----
    return eval_avg_metrics()


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
        pass_manager = pm,
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

    return {"fidelity_loss": avg_loss}


# ---------- Public unified API ----------


def train_genome_objective(
    genome: CircuitGenome,
    *,
    target_state: Optional[torch.Tensor] = None,
    input_bits: Optional[torch.Tensor] = None,
    dataset: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None,
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
        metrics = _train_with_pennylane(
            genome,
            dataset=dataset,
            target_state=target_state,
            input_bits=input_bits,
            steps=steps,
            lr=lr,
            loss=loss,
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
