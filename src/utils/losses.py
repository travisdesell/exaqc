import torch


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
