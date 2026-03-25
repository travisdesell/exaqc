import torch
import numpy as np

def NIG_NLL(y, mu, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)
    nll = 0.5 * torch.log(torch.tensor(np.pi) / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll

def NIG_Reg(y, mu, v, alpha, beta, reduce=True):
    error = y - mu
    evi = 2 * v + alpha + 1/beta
    reg = error * evi

    return torch.mean(reg) if reduce else reg