import torch

def fidelity(phi, psi):
    '''
    Fidelity Loss between psi and phi -> |<psi|phi>|^2
    
    :param phi: target complex statevector of shape (2**n_qubits,)
    :param psi: output complex statevector of shape (2**n_qubits,)
    '''
    overlap = torch.vdot(phi, psi)               # <psi|phi>
    return torch.abs(overlap) ** 2               # |<psi|phi>|^2

def loss_one_minus_fidelity(phi, psi):
    return 1.0 - fidelity(phi, psi)

def loss_state_angle(phi, psi, eps=1e-12):
    '''
    geodesic loss (phase-invariant) arccos(∣⟨ψ∣ϕ⟩∣)
    
    :param phi: target complex statevector of shape (2**n_qubits,)
    :param psi: output complex statevector of shape (2**n_qubits,)
    :param eps: Error Margin
    '''
    loss = torch.abs(torch.vdot(phi, psi)).clamp(max=1.0-eps)
    return torch.arccos(loss)

def loss_total_variation(phi, psi):
    '''
    Docstring for loss_total_variation TV(p,q)=0.5 * ∑​|px-qx|
    
    :param phi: target complex statevector of shape (2**n_qubits,)
    :param psi: output complex statevector of shape (2**n_qubits,)
    '''
    return 0.5 * torch.sum(torch.abs(psi - phi))

def loss_kl_divergence(phi, psi):
    '''
    Docstring for loss_kl_divergence KL(p||q)=∑pxlog(qx/px)
    
    :param phi: target complex statevector of shape (2**n_qubits,)
    :param psi: output complex statevector of shape (2**n_qubits,)
    '''
    return torch.sum(psi * (torch.log(psi) - torch.log(phi)))

def loss_obs_mse(phi, psi):
    '''
    Docstring for loss_obs_mse
    
    :param phi: target complex statevector of shape (2**n_qubits,)
    :param psi: output complex statevector of shape (2**n_qubits,)
    '''
    return torch.mean((psi - phi) ** 2)