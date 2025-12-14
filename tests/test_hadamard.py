# Run:  python -m unittest -v test_hadamard.py


from src.trainer import QuantumStateTrainer
from utils.losses import fidelity

import unittest
import pennylane as qml
import torch

torch.set_default_dtype(torch.float64)


class TestQuantumStateTrainerHadamard(unittest.TestCase):
    def setUp(self):
        # Fresh device/QNodes per test for isolation
        self.dev = qml.device("default.qubit", wires=1)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def model_state(theta, xbit: int):
            # prepare |xbit>
            if xbit == 1:
                qml.PauliX(0)

            # universal 1q ansatz (up to global phase)
            qml.RZ(theta[0], wires=0)
            qml.RY(theta[1], wires=0)
            qml.RZ(theta[2], wires=0)
            return qml.state()

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def target_state(xbit: int):
            if xbit == 1:
                qml.PauliX(0)
            qml.Hadamard(0)
            return qml.state()

        self.model_state = model_state
        self.target_state = target_state

    def test_trainer_learns_hadamard_on_two_inputs(self):

        torch.manual_seed(0)
        theta = torch.nn.Parameter(0.01 * torch.randn(3))

        trainer = QuantumStateTrainer(
            model_fn=lambda params, xbit: self.model_state(params, xbit),
            params=theta,
            target=lambda xbit: self.target_state(xbit),   # <-- target depends on xbit
            loss_name="fidelity",
        )

        trainer.fit(
            steps=250,
            lr=0.2,
            model_args_list=[(0,), (1,)],                  # <-- train on both each step
            log_every=50,
        )

        with torch.no_grad():
            F0 = fidelity(self.model_state(theta, 0), self.target_state(0)).item()
            F1 = fidelity(self.model_state(theta, 1), self.target_state(1)).item()

        self.assertGreaterEqual(F0, 0.995, f"F(|0>) too low: {F0}")
        self.assertGreaterEqual(F1, 0.995, f"F(|1>) too low: {F1}")


    def test_trainer_loss_is_differentiable(self):
        theta = torch.nn.Parameter(0.01 * torch.randn(3))

        trainer = QuantumStateTrainer(
            model_fn=lambda params, xbit: self.model_state(params, xbit),
            params=theta,
            target=lambda xbit: self.target_state(xbit),  # <-- fix
            loss_name="fidelity",
        )

        loss, metrics = trainer.compute_loss(0)
        self.assertTrue(loss.requires_grad, "Loss should require grad.")
        loss.backward()

        self.assertIsNotNone(theta.grad, "theta.grad is None (no gradients).")
        self.assertEqual(theta.grad.shape, theta.shape)
        self.assertTrue(torch.isfinite(theta.grad).all().item(), "Non-finite gradients found.")
        self.assertIn("fidelity", metrics)
        self.assertTrue(torch.isfinite(metrics["fidelity"]).item(), "Non-finite fidelity metric.")

    def test_trainer_runs_and_logs(self):

        torch.manual_seed(1)
        theta = torch.nn.Parameter(0.01 * torch.randn(3))

        trainer = QuantumStateTrainer(
            model_fn=lambda params, xbit: self.model_state(params, xbit),
            params=theta,
            target=lambda xbit: self.target_state(xbit),
            loss_name="fidelity",
        )

        logs = trainer.fit(
            steps=120,
            lr=0.2,
            model_args_list=[(0,)],   # always train on |0>
            log_every=20,
        )

        self.assertGreaterEqual(len(logs), 1)

        # if you want last logged fidelity:
        last_fid = logs[-1].metric["fidelity"]
        self.assertGreaterEqual(last_fid, 0.90, f"Final fidelity too low in smoke test: {last_fid}")

