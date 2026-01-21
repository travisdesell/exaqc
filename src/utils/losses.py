from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import pennylane as qml
from qiskit import QuantumCircuit

InputState = Union[str, Sequence[int]]


def statevector_to_probs(psi: torch.Tensor) -> torch.Tensor:
    """Convert a quantum statevector to a probability distribution.

    Args:
        psi: Complex-valued statevector of shape ``(2**n_qubits,)``.

    Returns:
        Real-valued probability vector of shape ``(2**n_qubits,)``.
    """
    return psi.real**2 + psi.imag**2


def qiskit_to_pl_state_forward(
    qc: QuantumCircuit,
    *,
    n_qubits: Optional[int] = None,
    input_bits: Optional[InputState] = None,
    shots: Optional[int] = None,
):
    """
    Create a PennyLane-based forward-pass function for a Qiskit circuit.

    This function converts a Qiskit ``QuantumCircuit`` into a PennyLane
    QNode using ``qml.from_qiskit`` and returns a callable that performs
    a forward pass and outputs the final quantum statevector.

    The returned forward function supports:
      - Optional basis-state input preparation
      - Parameter binding via a dictionary
      - Analytic (shots=None) or stochastic execution

    Args:
        qc: A Qiskit ``QuantumCircuit`` defining the quantum model.
        n_qubits: Number of qubits in the circuit. If ``None``, inferred
            from ``qc.num_qubits``.
        input_bits: Optional computational-basis input state, specified
            as a bitstring (e.g. ``"010"``) or a sequence of integers
            (e.g. ``[0, 1, 0]``). If ``None``, the circuit is applied to
            ``|0...0⟩``.
        shots: Number of shots for execution. If ``None``, the forward
            pass is analytic and returns the exact statevector.

    Returns:
        A callable ``forward(params) -> torch.Tensor`` where:
          - ``params`` is a dictionary mapping Qiskit ``Parameter`` objects
            (or their names) to numeric values
          - the output is a complex-valued torch tensor of shape
            ``(2**n_qubits,)`` representing the final statevector
    """
    if n_qubits is None:
        n_qubits = qc.num_qubits

    # Import Qiskit circuit as a PennyLane quantum function
    qfunc = qml.from_qiskit(qc)

    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def forward(params: Dict[Any, Any]):
        """
        Execute the quantum circuit with the given parameters.

        Args:
            params: Dictionary mapping Qiskit parameters (or their names)
                to numerical values. Values may be torch tensors with
                ``requires_grad=True``.

        Returns:
            A complex-valued torch tensor representing the final
            quantum statevector.
        """
        if input_bits is not None:
            if isinstance(input_bits, str):
                bits = [int(b) for b in input_bits]
            else:
                bits = list(map(int, input_bits))

            qml.BasisState(
                torch.tensor(bits, dtype=torch.int64),
                wires=range(n_qubits),
            )

        # PennyLane binds parameters by name
        kwargs = {getattr(k, "name", str(k)): v for k, v in (params or {}).items()}

        qfunc(wires=range(n_qubits), **kwargs)
        return qml.state()

    return forward


def fidelity(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute the quantum state fidelity between two pure states.

    Fidelity is defined as:

        F(φ, ψ) = |⟨φ | ψ⟩|²

    This quantity is invariant to global phase and lies in [0, 1].

    Args:
        phi: Target complex-valued statevector of shape (2**n_qubits,).
        psi: Output (predicted) complex-valued statevector of shape (2**n_qubits,).

    Returns:
        A scalar tensor representing the fidelity between the two states.
    """
    phi = phi.to(dtype=psi.dtype, device=psi.device)
    overlap = torch.vdot(phi, psi)  # ⟨φ | ψ⟩
    return torch.abs(overlap) ** 2


def loss_one_minus_fidelity(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute a minimization loss based on state fidelity.

    This loss is defined as:

        L = 1 − |⟨φ | ψ⟩|²

    Minimizing this loss is equivalent to maximizing fidelity.

    Args:
        phi: Target complex-valued statevector of shape (2**n_qubits,).
        psi: Output (predicted) complex-valued statevector of shape (2**n_qubits,).

    Returns:
        A scalar tensor representing the fidelity-based loss.
    """
    return 1.0 - fidelity(phi, psi)


def objective_one_minus_fidelity_pl(
    qc: QuantumCircuit,
    *,
    target_state: torch.Tensor,
    params: Dict[Any, Any],
    input_bits: Optional[InputState] = None,
    train_hparams: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, QuantumCircuit]:
    """
    Computes a fidelity-based loss for a Qiskit circuit using PennyLane.

    This objective:
      1. Runs a forward pass of the Qiskit circuit in PennyLane
      2. Computes the fidelity with respect to a target quantum state
      3. Returns ``1 - fidelity`` as a minimization loss

    Args:
        qc: Qiskit ``QuantumCircuit`` defining the quantum model.
        target_state: Target complex-valued statevector tensor of shape
            ``(2**n_qubits,)``.
        params: Dictionary mapping Qiskit parameters to numerical values.
            Values may be torch tensors to enable gradient-based training.
        train_hparams: Dictionary of training hyperparameters.
        input_bits: Optional computational-basis input state for the
            forward pass.

    Returns:
        A tuple ``(loss, qc)`` where:
          - ``loss`` is a scalar torch tensor equal to
            ``1 - |⟨φ | ψ⟩|²``
          - ``qc`` is the original Qiskit circuit passed to the function
    """
    forward = qiskit_to_pl_state_forward(
        qc,
        input_bits=input_bits,
        shots=None,
    )

    psi = forward(params)

    phi = target_state.to(dtype=psi.dtype, device=psi.device)
    phi = phi / torch.linalg.norm(phi)

    loss = 1.0 - fidelity(phi, psi)
    return loss, qc


def loss_state_angle(
    phi: torch.Tensor,
    psi: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the geodesic (angular) distance between two quantum states.

    This loss corresponds to the Fubini–Study distance for pure states:

        L = arccos(|⟨φ | ψ⟩|)

    It is invariant to global phase and provides a smooth geometric
    notion of distance on the projective Hilbert space.

    Args:
        phi: Target complex-valued statevector of shape (2**n_qubits,).
        psi: Output (predicted) complex-valued statevector of shape (2**n_qubits,).
        eps: Small numerical constant to prevent invalid values for arccos.

    Returns:
        A scalar tensor representing the angular distance between states.
    """
    overlap_mag = torch.abs(torch.vdot(phi, psi)).clamp(max=1.0 - eps)
    return torch.arccos(overlap_mag)


def objective_state_angle_pl(
    qc: QuantumCircuit,
    *,
    target_state: torch.Tensor,
    params: Dict[Any, Any],
    input_bits: Optional[InputState] = None,
    eps: float = 1e-12,
    train_hparams: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, QuantumCircuit]:
    """
    Compute the Fubini–Study (angular) distance between quantum states.

    This objective performs a forward pass of the given Qiskit circuit
    using PennyLane and computes:

        L = arccos(|⟨φ | ψ⟩|)

    The loss is invariant to global phase and defines a geodesic
    distance on the projective Hilbert space.

    Args:
        qc: Qiskit ``QuantumCircuit`` defining the quantum model.
        target_state: Target complex-valued statevector tensor of shape
            ``(2**n_qubits,)``.
        params: Dictionary mapping Qiskit parameters to numerical values.
        input_bits: Optional computational-basis input state.
        eps: Small numerical constant to avoid invalid ``arccos`` inputs.
        train_hparams: Dictionary of training hyperparameters.

    Returns:
        A tuple ``(loss, qc)`` where:
          - ``loss`` is the angular distance between states
          - ``qc`` is the original Qiskit circuit
    """
    forward = qiskit_to_pl_state_forward(
        qc,
        input_bits=input_bits,
        shots=None,
    )

    psi = forward(params)

    phi = target_state.to(dtype=psi.dtype, device=psi.device)
    phi = phi / torch.linalg.norm(phi)

    overlap_mag = torch.abs(torch.vdot(phi, psi)).clamp(max=1.0 - eps)
    loss = torch.arccos(overlap_mag)

    return loss, qc


def loss_total_variation(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute total variation distance between two vectors.

    The total variation (TV) distance is defined as:

        TV(p, q) = 0.5 * Σ |pₓ − qₓ|

    Note:
        This loss is **only meaningful when `phi` and `psi` represent
        probability distributions**, not raw quantum statevectors.
        If statevectors are used, they should first be converted to
        probabilities via |ψ|².

    Args:
        phi: Target probability distribution tensor.
        psi: Output probability distribution tensor.

    Returns:
        A scalar tensor representing the total variation distance.
    """
    return 0.5 * torch.sum(torch.abs(psi - phi))


def objective_total_variation_pl(
    qc: QuantumCircuit,
    *,
    target_probs: torch.Tensor,
    params: Dict[Any, Any],
    input_bits: Optional[InputState] = None,
    eps: float = 1e-12,
    train_hparams: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, QuantumCircuit]:
    """
    Compute total variation distance between output and target distributions.

    The total variation distance is defined as:

        TV(p, q) = 0.5 * Σ |pₓ − qₓ|

    This objective assumes the circuit output is interpreted as a
    probability distribution derived from ``|ψ|²``.

    Args:
        qc: Qiskit ``QuantumCircuit`` defining the quantum model.
        target_probs: Target probability distribution tensor.
        params: Dictionary mapping Qiskit parameters to numerical values.
        input_bits: Optional computational-basis input state.
        eps: Small numerical constant for normalization safety.
        train_hparams: Dictionary of training hyperparameters.

    Returns:
        A tuple ``(loss, qc)`` where:
          - ``loss`` is the total variation distance
          - ``qc`` is the original Qiskit circuit
    """
    forward = qiskit_to_pl_state_forward(
        qc,
        input_bits=input_bits,
        shots=None,
    )

    psi = forward(params)
    p = statevector_to_probs(psi)
    p = p / (p.sum() + eps)

    q = target_probs.to(dtype=p.dtype, device=p.device)
    q = q / (q.sum() + eps)

    loss = 0.5 * torch.sum(torch.abs(p - q))
    return loss, qc


def loss_kl_divergence(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute the Kullback–Leibler (KL) divergence between two distributions.

    The KL divergence is defined as:

        KL(φ || ψ) = Σ φₓ · log(φₓ / ψₓ)

    Note:
        This loss assumes that `phi` and `psi` are **valid probability
        distributions** (non-negative and summing to 1). It should not
        be applied directly to complex-valued statevectors.

    Args:
        phi: Target probability distribution tensor.
        psi: Output probability distribution tensor.

    Returns:
        A scalar tensor representing the KL divergence.
    """
    return torch.sum(phi * (torch.log(phi) - torch.log(psi)))


def objective_kl_divergence_pl(
    qc: QuantumCircuit,
    *,
    target_probs: torch.Tensor,
    params: Dict[Any, Any],
    input_bits: Optional[InputState] = None,
    eps: float = 1e-12,
    train_hparams: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, QuantumCircuit]:
    """
    Compute the Kullback–Leibler divergence between target and output distributions.

    The KL divergence is defined as:

        KL(φ || ψ) = Σ φₓ · log(φₓ / ψₓ)

    Both distributions must be valid probability distributions.

    Args:
        qc: Qiskit ``QuantumCircuit`` defining the quantum model.
        target_probs: Target probability distribution tensor.
        params: Dictionary mapping Qiskit parameters to numerical values.
        input_bits: Optional computational-basis input state.
        eps: Small numerical constant for numerical stability.
        train_hparams: Dictionary of training hyperparameters.

    Returns:
        A tuple ``(loss, qc)`` where:
          - ``loss`` is the KL divergence
          - ``qc`` is the original Qiskit circuit
    """
    forward = qiskit_to_pl_state_forward(
        qc,
        input_bits=input_bits,
        shots=None,
    )

    psi = forward(params)
    p = statevector_to_probs(psi)
    p = p / (p.sum() + eps)

    q = target_probs.to(dtype=p.dtype, device=p.device)
    q = q / (q.sum() + eps)

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)

    loss = torch.sum(q * (torch.log(q) - torch.log(p)))
    return loss, qc


def loss_obs_mse(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Compute mean-squared error between observable outputs.

    This loss is typically used when `phi` and `psi` represent vectors
    of expectation values (e.g., ⟨Z⟩, ⟨ZZ⟩, etc.) rather than quantum states.

    Args:
        phi: Target observable values tensor.
        psi: Output observable values tensor.

    Returns:
        A scalar tensor representing the mean-squared error.
    """
    return torch.mean((psi - phi) ** 2)


def objective_obs_mse_pl(
    qc: QuantumCircuit,
    *,
    target_obs: torch.Tensor,
    params: Dict[Any, Any],
    forward_fn=None,
    train_hparams: Optional[Dict[str, Any]] = None,
    input_bits: Optional[InputState] = None,
) -> Tuple[torch.Tensor, QuantumCircuit]:
    """
    Compute mean-squared error between observable outputs.

    If ``forward_fn`` is not provided, this function builds a PennyLane QNode
    internally from the Qiskit circuit and the observables specified in
    ``train_hparams["observables"]``.

    Args:
        qc: Qiskit ``QuantumCircuit`` defining the quantum model.
        target_obs: Target observable values tensor (shape must match forward output).
        params: Dictionary mapping Qiskit parameters to numerical values.
        forward_fn: Optional callable performing the forward pass and returning
            a tensor of observable expectation values.
        train_hparams: Dictionary of training hyperparameters. If ``forward_fn`` is None,
            this must include:
              - "observables": list of PennyLane observables to measure
            Optional keys:
              - "shots": Optional[int]
              - "diff_method": str, e.g. "parameter-shift" (default) or "backprop"
        input_bits: Optional computational-basis input state for the forward pass.

    Returns:
        A tuple ``(loss, qc)`` where:
          - ``loss`` is the mean-squared error
          - ``qc`` is the original Qiskit circuit
    """
    train_hparams = train_hparams or {}

    if forward_fn is None:
        observables = train_hparams.get("observables", None)
        if observables is None:
            raise ValueError(
                "objective_obs_mse_pl requires either forward_fn or train_hparams['observables']."
            )

        shots = train_hparams.get("shots", None)
        diff_method = train_hparams.get("diff_method", "parameter-shift")

        n_qubits = qc.num_qubits
        qfunc = qml.from_qiskit(qc)
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def forward_fn(local_params: Dict[Any, Any]):
            if input_bits is not None:
                if isinstance(input_bits, str):
                    bits = [int(b) for b in input_bits]
                else:
                    bits = list(map(int, input_bits))
                qml.BasisState(
                    torch.tensor(bits, dtype=torch.int64), wires=range(n_qubits)
                )

            kwargs = {
                getattr(k, "name", str(k)): v for k, v in (local_params or {}).items()
            }
            qfunc(wires=range(n_qubits), **kwargs)

            return [qml.expval(obs) for obs in observables]

    psi_obs = forward_fn(params)
    loss = loss_obs_mse(target_obs, psi_obs)
    return loss, qc
