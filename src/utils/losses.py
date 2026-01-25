from __future__ import annotations

from typing import Sequence, Union

import torch

InputState = Union[str, Sequence[int]]


def statevector_to_probs(psi: torch.Tensor) -> torch.Tensor:
    """Convert a quantum statevector to a probability distribution.

    Args:
        psi: Complex-valued statevector of shape ``(2**n_qubits,)``.

    Returns:
        Real-valued probability vector of shape ``(2**n_qubits,)``.
    """
    return psi.real**2 + psi.imag**2


def marginal_probs_from_statevector(
    psi: torch.Tensor,
    *,
    n_qubits: int,
    readout_wires: list[int],
) -> torch.Tensor:
    """
    psi: complex tensor shape [2**n_qubits]
    Returns probs over readout_wires only, shape [2**len(readout_wires)].

    Assumes wire index 0 is the *first* qubit in BasisState(wires=range(n_qubits)),
    i.e. consistent with your PennyLane wiring.
    """
    probs_full = (psi.conj() * psi).real.to(
        dtype=torch.float32
    )  # |amp|^2, shape [2**n_qubits]
    probs_full = probs_full.reshape([2] * n_qubits)

    # sum out non-readout axes
    keep = sorted(readout_wires)
    sum_axes = [ax for ax in range(n_qubits) if ax not in keep]
    probs_marg = probs_full
    for ax in reversed(sum_axes):
        probs_marg = probs_marg.sum(dim=ax)

    # Now probs_marg has shape [2]*len(keep), flatten
    return probs_marg.reshape(-1)


def fidelity(phi: torch.Tensor, psi: torch.Tensor, eps:float=1e-12) -> torch.Tensor:
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
    phi = phi / (torch.linalg.norm(phi) + eps)
    psi = psi / (torch.linalg.norm(psi) + eps)
    overlap = torch.vdot(phi, psi)  # ⟨φ | ψ⟩
    return torch.abs(overlap) ** 2


def loss_one_minus_fidelity(
    phi: torch.Tensor, psi: torch.Tensor, **kwargs
) -> torch.Tensor:
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


def loss_state_angle(
    phi: torch.Tensor, psi: torch.Tensor, eps: float = 1e-12, **kwargs
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
    phi = phi / (torch.linalg.norm(phi) + eps)
    psi = psi / (torch.linalg.norm(psi) + eps)
    phi = phi.to(dtype=psi.dtype, device=psi.device)
    overlap_mag = torch.abs(torch.vdot(phi, psi)).clamp(max=1.0 - eps)
    return torch.arccos(overlap_mag)


def loss_total_variation(
    phi: torch.Tensor, psi: torch.Tensor, eps:float = 1e-12, **kwargs
) -> torch.Tensor:
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
    phi = phi / (torch.linalg.norm(phi) + eps)
    psi = psi / (torch.linalg.norm(psi) + eps)
    phi = phi.to(dtype=psi.dtype, device=psi.device)
    return 0.5 * torch.sum(torch.abs(psi - phi))


def loss_kl_divergence(phi: torch.Tensor, psi: torch.Tensor, eps:float=1e-12, **kwargs) -> torch.Tensor:
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
    phi = phi / (torch.linalg.norm(phi) + eps)
    psi = psi / (torch.linalg.norm(psi) + eps)
    phi = phi.to(dtype=psi.dtype, device=psi.device)
    return torch.sum(phi * (torch.log(phi) - torch.log(psi)))


def loss_obs_mse(phi: torch.Tensor, psi: torch.Tensor, **kwargs) -> torch.Tensor:
    """Compute mean-squared error between observable outputs.

    This loss is typically used when `phi` and `psi` represent vectors
    of expectation values (e.g., ⟨Z⟩, ⟨ZZ⟩, etc.) rather than quantum states.

    Args:
        phi: Target observable values tensor.
        psi: Output observable values tensor.

    Returns:
        A scalar tensor representing the mean-squared error.
    """
    phi = phi.to(dtype=psi.dtype, device=psi.device)
    return torch.mean((psi - phi) ** 2)


def loss_ce(
    phi: torch.Tensor,
    psi: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute mean-squared error between observable outputs.

    This loss is typically used when `phi` and `psi` represent vectors
    of expectation values (e.g., ⟨Z⟩, ⟨ZZ⟩, etc.) rather than quantum states.

    Args:
        phi: Target observable values tensor.
        psi: Output observable values tensor.

    Returns:
        A scalar tensor representing the mean-squared error.
    """
    phi = phi.to(dtype=torch.float32)
    psi = psi.to(dtype=torch.float32)
    phi = phi / (phi.sum() + eps)
    psi = psi / (psi.sum() + eps)
    phi = phi.to(dtype=psi.dtype, device=psi.device)
    return -(phi * torch.log(psi.clamp_min(eps))).mean()


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
    return -(y_onehot * torch.log(probs)).mean()


def loss_readout_ce_from_state(
    phi: torch.Tensor,
    psi: torch.Tensor,
    n_qubits: int,
    readout_wires: list[int] = None,
    n_classes: int = 3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute mean-squared error between observable outputs.

    This loss is typically used when `phi` and `psi` represent vectors
    of expectation values (e.g., ⟨Z⟩, ⟨ZZ⟩, etc.) rather than quantum states.

    Args:
        phi: Target observable values tensor.
        psi: Output observable values tensor.
        n_qubits: The number of qubits
        n_classes: The number of output classes
        readout_wires: The list of wires to be measure

    Returns:
        A scalar tensor representing the mean-squared error.
    """
    if readout_wires is None:
        raise ValueError("Need to specify the read_out wires")
    psi = marginal_probs_from_statevector(
        psi, n_qubits=n_qubits, readout_wires=readout_wires
    )  # [4]
    psi = psi[:n_classes]
    return loss_ce(phi, psi, eps=eps)
