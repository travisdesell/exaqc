import pytest
import pennylane as qml
import torch
import math

from src.trainer import QuantumStateTrainer
from src.utils.losses import fidelity, objective_one_minus_fidelity_pl, qiskit_to_pl_state_forward

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

torch.set_default_dtype(torch.float64)


@pytest.fixture()
def circuits():
    """Fresh device + QNodes per test for isolation."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def model_state(theta, xbit: int):
        if xbit == 1:
            qml.PauliX(0)
        qml.RZ(theta[0], wires=0)
        qml.RY(theta[1], wires=0)
        qml.RZ(theta[2], wires=0)
        return qml.state()

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def target_state(xbit: int):
        if xbit == 1:
            qml.PauliX(0)
        qml.Hadamard(0)
        return qml.state()

    return model_state, target_state


def test_01_trainer_learns_hadamard_on_two_inputs(circuits):
    model_state, target_state = circuits

    torch.manual_seed(0)
    theta = torch.nn.Parameter(0.01 * torch.randn(3))

    trainer = QuantumStateTrainer(
        model_fn=lambda params, xbit: model_state(params, xbit),
        params=theta,
        target=lambda xbit: target_state(xbit),
        loss_name="fidelity",
    )

    trainer.fit(
        steps=250,
        lr=0.2,
        model_args_list=[(0,), (1,)],
        log_every=50,
    )

    with torch.no_grad():
        F0 = fidelity(model_state(theta, 0), target_state(0)).item()
        F1 = fidelity(model_state(theta, 1), target_state(1)).item()

    assert F0 >= 0.995, f"F(|0>) too low: {F0}"
    assert F1 >= 0.995, f"F(|1>) too low: {F1}"


def test_02_trainer_loss_is_differentiable(circuits):
    model_state, target_state = circuits

    theta = torch.nn.Parameter(0.01 * torch.randn(3))

    trainer = QuantumStateTrainer(
        model_fn=lambda params, xbit: model_state(params, xbit),
        params=theta,
        target=lambda xbit: target_state(xbit),
        loss_name="fidelity",
    )

    loss, metrics = trainer.compute_loss(0)
    assert loss.requires_grad, "Loss should require grad."

    loss.backward()

    assert theta.grad is not None, "theta.grad is None (no gradients)."
    assert theta.grad.shape == theta.shape
    assert torch.isfinite(theta.grad).all().item(), "Non-finite gradients found."
    assert "fidelity" in metrics
    assert torch.isfinite(metrics["fidelity"]).item(), "Non-finite fidelity metric."


def test_03_trainer_runs_and_logs(circuits):
    model_state, target_state = circuits

    torch.manual_seed(1)
    theta = torch.nn.Parameter(0.01 * torch.randn(3))

    trainer = QuantumStateTrainer(
        model_fn=lambda params, xbit: model_state(params, xbit),
        params=theta,
        target=lambda xbit: target_state(xbit),
        loss_name="fidelity",
    )

    logs = trainer.fit(
        steps=120,
        lr=0.2,
        model_args_list=[(0,)],
        log_every=20,
    )

    assert len(logs) >= 1

    last_fid = logs[-1].metric["fidelity"]
    assert last_fid >= 0.90, f"Final fidelity too low in smoke test: {last_fid}"


def test_04_train_parameterized_qiskit_circuit_as_hadamard_using_fidelity():
    """Train and test a parameterized Qiskit circuit to behave like a Hadamard gate.

    This test reuses the previously-declared helpers/loss style:
      - qiskit_to_pl_state_forward(...)
      - objective_one_minus_fidelity_pl(...)

    The circuit is built in Qiskit (a single U gate with 3 parameters), imported into
    PennyLane for execution/differentiation, trained with Torch/Adam, and evaluated
    via fidelity on the two basis inputs.

    Assertions:
      - Fidelity(H|0>, model(|0>)) > 0.999
      - Fidelity(H|1>, model(|1>)) > 0.999
    """
    torch.manual_seed(0)

    train_hparams = {"shots": None}

    # --- Targets: H|0> = |+>, H|1> = |-> (global phase irrelevant for fidelity) ---
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    target_plus = torch.tensor([inv_sqrt2 + 0.0j, inv_sqrt2 + 0.0j], dtype=torch.complex64)
    target_minus = torch.tensor([inv_sqrt2 + 0.0j, -inv_sqrt2 + 0.0j], dtype=torch.complex64)

    # --- Parameterized 1-qubit Qiskit circuit ---
    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lam")

    qc = QuantumCircuit(1)
    qc.u(theta, phi, lam, 0)

    # --- Trainable torch parameters (PennyLane binds by parameter name) ---
    t_theta = torch.randn((), dtype=torch.float32, requires_grad=True)
    t_phi = torch.randn((), dtype=torch.float32, requires_grad=True)
    t_lam = torch.randn((), dtype=torch.float32, requires_grad=True)

    opt = torch.optim.Adam([t_theta, t_phi, t_lam], 
                           weight_decay=1e-4, 
                           lr=0.2)

    # --- Training loop using the previously-defined fidelity objective helper ---
    for _step in range(250):
        opt.zero_grad(set_to_none=True)

        params = {theta: t_theta, phi: t_phi, lam: t_lam}

        loss0, _ = objective_one_minus_fidelity_pl(
            qc,
            target_state=target_plus,
            params=params,
            input_bits="0",
            train_hparams=train_hparams,
        )
        loss1, _ = objective_one_minus_fidelity_pl(
            qc,
            target_state=target_minus,
            params=params,
            input_bits="1",
            train_hparams=train_hparams,
        )

        loss = 0.5 * (loss0 + loss1)
        loss.backward()
        opt.step()

    # --- Evaluation (reuse forward helper to compute output states, then fidelity) ---
    # forward = qiskit_to_pl_state_forward(qc, input_bits="0", shots=None)
    with torch.no_grad():
        params_eval = {theta: t_theta, phi: t_phi, lam: t_lam}

        # Evaluate on |0>
        psi0 = qiskit_to_pl_state_forward(qc, input_bits="0", shots=None)(params_eval)
        fid0 = fidelity(target_plus, psi0).item()

        # Evaluate on |1>
        psi1 = qiskit_to_pl_state_forward(qc, input_bits="1", shots=None)(params_eval)
        fid1 = fidelity(target_minus, psi1).item()

    assert fid0 > 0.999, f"Fidelity on |0> too low: {fid0:.6f}"
    assert fid1 > 0.999, f"Fidelity on |1> too low: {fid1:.6f}"