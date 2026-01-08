import pytest
import pennylane as qml
import torch

from src.trainer import QuantumStateTrainer
from src.utils.losses import fidelity

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
