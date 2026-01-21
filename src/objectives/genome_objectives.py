"""
Objective functions for training CircuitGenome instances.

This module provides backend-agnostic utilities for:
  - Converting a CircuitGenome into a differentiable quantum model
  - Training circuit parameters using gradient-based optimization
  - Computing multiple quantum loss metrics
  - Writing trained parameters and fitness values back into the genome

Supported backends:
  - Qiskit circuits (via PennyLane conversion)
  - PennyLane-native circuits

"""

from __future__ import annotations

from typing import Dict, Any, Optional

import torch

# import pennylane as qml
from qiskit import QuantumCircuit

from src.circuits.circuit import CircuitGenome
from src.utils.losses import (
    loss_one_minus_fidelity,
    loss_state_angle,
    loss_total_variation,
    loss_kl_divergence,
    statevector_to_probs,
    qiskit_to_pl_state_forward,
)
from src.trainer import QuantumStateTrainer


def genome_to_torch_params(genome: CircuitGenome) -> Dict[str, torch.nn.Parameter]:
    """
    Convert genome gate parameters into trainable Torch parameters.

    Each parameter is uniquely keyed using the gate innovation number
    to avoid collisions between identical gate types.

    Args:
        genome: CircuitGenome containing parameterized gates.

    Returns:
        A dictionary mapping unique parameter identifiers to
        ``torch.nn.Parameter`` objects.
    """
    params = {}
    for gate in genome.gates:
        for name, value in gate.parameters.items():
            key = f"{gate.innovation_number}:{name}"
            params[key] = torch.nn.Parameter(
                torch.tensor(float(value), dtype=torch.float64)
            )
    return params


def torch_params_to_genome(
    genome: CircuitGenome,
    trained_params: Dict[str, torch.Tensor],
):
    """
    Write trained Torch parameter values back into the genome.

    This function mutates the genome in-place.

    Args:
        genome: CircuitGenome to update.
        trained_params: Dictionary of trained Torch parameters keyed
            by ``"<innovation_number>:<param_name>"``.
    """
    for gate in genome.gates:
        for name in gate.parameters:
            key = f"{gate.innovation_number}:{name}"
            if key in trained_params:
                gate.parameters[name] = float(trained_params[key].detach().cpu().item())


def build_forward_from_genome(
    genome: CircuitGenome, *, input_bits=None, target: str = "qiskit"
):
    """
    Build a differentiable forward model from a CircuitGenome.

    This function automatically selects the appropriate backend:
      - PennyLane-native circuits are executed directly
      - Qiskit circuits are converted to PennyLane via ``qml.from_qiskit``

    Args:
        genome: CircuitGenome defining the quantum circuit.
        input_bits: Optional computational basis input state.
        target: Str flag to denote whether the circuit is in qiskit or pennylane

    Returns:
        A callable ``forward(params) -> torch.Tensor`` that returns
        a complex-valued quantum statevector.
    """

    def _build_pl_forward(
        genome: CircuitGenome,
        *,
        input_bits: Optional[Any] = None,
    ):
        """Construct a forward function for PennyLane-native genomes.

        Args:
            genome: PennyLane-backed CircuitGenome.
            input_bits: Optional basis-state input (currently unused).

        Returns:
            A callable that executes the circuit and returns ``qml.state()``.
        """
        _, qnode_fn = genome.generate_pennylane_circuit()
        n_qubits = sum(genome.registers.values())

        if input_bits is None:
            input_bits = torch.zeros(n_qubits, dtype=torch.int64)

        def forward(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            return qnode_fn(input_bits, params)

        return forward

    def _build_qiskit_forward(
        genome: CircuitGenome,
        *,
        input_bits: Optional[Any] = None,
    ):
        """Construct a forward function for Qiskit-backed genomes.

        The Qiskit circuit is converted to a PennyLane QNode using
        ``qml.from_qiskit``.

        Args:
            genome: Qiskit-backed CircuitGenome.
            input_bits: Optional computational basis input state.

        Returns:
            A callable that executes the circuit and returns a statevector.
        """
        qc: QuantumCircuit = genome.generate_qiskit_circuit()

        forward = qiskit_to_pl_state_forward(
            qc,
            input_bits=input_bits,
            shots=None,
        )

        def wrapped_forward(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            return forward(params)

        return wrapped_forward

    if target == "pennylane":
        return _build_pl_forward(genome, input_bits=input_bits)

    if target == "qiskit":
        return _build_qiskit_forward(genome, input_bits=input_bits)

    raise ValueError(f"Unsupported genome backend: {target}")


def train_genome_objective(
    genome: CircuitGenome,
    *,
    target_state: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    input_bits=None,
    log_every: int = 50,
    target: str = "qiskit",
) -> CircuitGenome:
    """
    Train a genome using fidelity loss and record multiple fitness metrics.
    """

    # 1️⃣ Extract trainable parameters
    torch_params = genome_to_torch_params(genome)

    has_trainable_params = len(torch_params) > 0

    if has_trainable_params:

        # 2️⃣ Build forward model
        model_fn = build_forward_from_genome(
            genome,
            input_bits=input_bits,
            target=target,
        )

        param_keys = list(torch_params.keys())
        param_values = list(torch_params.values())

        def wrapped_model_fn(*flat_params):
            param_dict = dict(zip(param_keys, flat_params))
            return model_fn(param_dict)

        # 3️⃣ Trainer
        trainer = QuantumStateTrainer(
            model_fn=wrapped_model_fn,
            params=param_values,
            target=target_state,
            loss_name="fidelity",
            normalize_states=True,
        )

        # 4️⃣ Train
        logs = trainer.fit(  # noqa: F841
            steps=steps,
            lr=lr,
            log_every=log_every,
        )

        # 5️⃣ Final forward pass
        with torch.no_grad():
            psi = model_fn(torch_params)
            psi = psi / torch.linalg.norm(psi)
            phi = target_state / torch.linalg.norm(target_state)

        # 6️⃣ Compute all metrics
        fid = loss_one_minus_fidelity(phi, psi)
        angle = loss_state_angle(phi, psi)
        probs_psi = statevector_to_probs(psi)
        probs_phi = statevector_to_probs(phi)

        tv = loss_total_variation(probs_phi, probs_psi)
        kl = loss_kl_divergence(
            probs_phi.clamp_min(1e-12),
            probs_psi.clamp_min(1e-12),
        )

        # 7️⃣ Store fitness (losses only, as requested)
        genome.fitness = {
            "fidelity_loss": float(fid.item()),
            "angle_loss": float(angle.item()),
            "total_variation": float(tv.item()),
            "kl_divergence": float(kl.item()),
        }

        # 8️⃣ Write trained params back to genome
        torch_params_to_genome(genome, torch_params)

    return genome
