import pennylane as qml
import torch

def loss_fidelity(psi, phi):
    '''
    Fidelity Loss between psi and phi -> |<psi|phi>|^2
    
    :param psi: target complex statevector of shape (2**n_qubits,)
    :param phi: output complex statevector of shape (2**n_qubits,)
    '''
    overlap = torch.vdot(psi, phi)               # <psi|phi>
    return torch.abs(overlap) ** 2               # |<psi|phi>|^2

def loss_state_angle(psi, phi, eps=1e-12):
    '''
    geodesic loss (phase-invariant) arccos(∣⟨ψ∣ϕ⟩∣)
    
    :param psi: target complex statevector of shape (2**n_qubits,)
    :param phi: output complex statevector of shape (2**n_qubits,)
    :param eps: Error Margin
    '''
    loss = torch.abs(torch.vdot(psi, phi)).clamp(max=1.0-eps)
    return torch.arccos(loss)

def loss_total_variation(psi, phi):
    '''
    Docstring for loss_total_variation TV(p,q)=0.5 * ∑​|px-qx|
    
    :param psi: target complex statevector of shape (2**n_qubits,)
    :param phi: output complex statevector of shape (2**n_qubits,)
    '''
    return 0.5 * torch.sum(torch.abs(psi - phi))

def loss_kl_divergence(psi, phi):
    '''
    Docstring for loss_kl_divergence KL(p||q)=∑pxlog(qx/px)
    
    :param psi: target complex statevector of shape (2**n_qubits,)
    :param phi: output complex statevector of shape (2**n_qubits,)
    '''
    return torch.sum(psi * (torch.log(psi) - torch.log(phi)))

def loss_obs_mse(psi, phi):
    '''
    Docstring for loss_obs_mse
    
    :param psi: target complex statevector of shape (2**n_qubits,)
    :param phi: output complex statevector of shape (2**n_qubits,)
    '''
    return torch.mean((psi - phi) ** 2)