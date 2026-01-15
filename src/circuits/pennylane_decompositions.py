import pennylane as qml
import numpy as np

# ============================================================
# Helper: multi-controlled X using ancilla-free decomposition
# ============================================================

def mcx(wires):
    """
    Multi-controlled X using recursive decomposition.
    wires = [c1, c2, ..., ck, target]
    """
    *controls, target = wires
    if len(controls) == 1:
        qml.CNOT(wires=[controls[0], target])
    elif len(controls) == 2:
        qml.Toffoli(wires=[controls[0], controls[1], target])
    else:
        # Decompose recursively
        qml.Toffoli(wires=[controls[-2], controls[-1], target])
        mcx(controls[:-1] + [target])
        qml.Toffoli(wires=[controls[-2], controls[-1], target])
        mcx(controls[:-1] + [target])


# ============================================================
# CSX (Controlled sqrt(X))
# ============================================================

def csx(control, target):
    qml.Hadamard(wires=target)
    qml.CRX(np.pi / 2, wires=[control, target])
    qml.Hadamard(wires=target)


# ============================================================
# CS (Controlled S)
# ============================================================

def cs(control, target):
    qml.ControlledPhaseShift(np.pi / 2, wires=[control, target])


# ============================================================
# CSDG (Controlled S-dagger)
# ============================================================

def csdg(control, target):
    qml.ControlledPhaseShift(-np.pi / 2, wires=[control, target])


# ============================================================
# DCX (Double CNOT)
# ============================================================

def dcx(q1, q2):
    qml.CNOT(wires=[q1, q2])
    qml.CNOT(wires=[q2, q1])


# ============================================================
# ECR (Echoed Cross Resonance)
# ============================================================

def ecr(q1, q2):
    qml.Hadamard(wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RZ(np.pi / 2, wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.Hadamard(wires=q2)


# ============================================================
# RCCX (Margolus gate)
# ============================================================

def rccx(c1, c2, target):
    qml.Hadamard(wires=target)
    qml.T(wires=target)
    qml.CNOT(wires=[c2, target])
    qml.T.adjoint()(wires=target)
    qml.CNOT(wires=[c1, target])
    qml.T(wires=target)
    qml.CNOT(wires=[c2, target])
    qml.T.adjoint()(wires=target)
    qml.Hadamard(wires=target)


# ============================================================
# RCCCX (3-controlled Margolus)
# ============================================================

def rcccx(c1, c2, c3, target):
    qml.Hadamard(wires=target)
    qml.T(wires=target)
    mcx([c1, c2, c3, target])
    qml.T.adjoint()(wires=target)
    qml.Hadamard(wires=target)


# ============================================================
# RZX gate
# ============================================================

def rzx(theta, q1, q2):
    qml.Hadamard(wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RZ(theta, wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.Hadamard(wires=q2)


# ============================================================
# MS (Mølmer–Sørensen)
# ============================================================

def ms(theta, q1, q2):
    qml.RX(np.pi / 2, wires=q1)
    qml.RX(np.pi / 2, wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RZ(theta, wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RX(-np.pi / 2, wires=q1)
    qml.RX(-np.pi / 2, wires=q2)


# ============================================================
# CU (general controlled U)
# ============================================================

def cu(theta, phi, delta, gamma, control, target):
    qml.RZ((phi + delta) / 2, wires=target)
    qml.RY(theta / 2, wires=target)
    qml.CNOT(wires=[control, target])
    qml.RY(-theta / 2, wires=target)
    qml.RZ(-(phi + delta) / 2, wires=target)
    qml.CNOT(wires=[control, target])
    qml.RZ(delta, wires=target)
    qml.RZ(gamma, wires=control)


# ============================================================
# MCP (multi-controlled phase)
# ============================================================

def mcp(phi, controls, target):
    for c in controls:
        qml.CNOT(wires=[c, target])
    qml.RZ(phi, wires=target)
    for c in reversed(controls):
        qml.CNOT(wires=[c, target])


# ============================================================
# MCRX / MCRY / MCRZ
# ============================================================

def mcrx(phi, controls, target):
    for c in controls:
        qml.CNOT(wires=[c, target])
    qml.RX(phi, wires=target)
    for c in reversed(controls):
        qml.CNOT(wires=[c, target])


def mcry(phi, controls, target):
    for c in controls:
        qml.CNOT(wires=[c, target])
    qml.RY(phi, wires=target)
    for c in reversed(controls):
        qml.CNOT(wires=[c, target])


def mcrz(phi, controls, target):
    for c in controls:
        qml.CNOT(wires=[c, target])
    qml.RZ(phi, wires=target)
    for c in reversed(controls):
        qml.CNOT(wires=[c, target])


# ============================================================
# SX and SXDG
# ============================================================

def sx(qubit):
    qml.RX(np.pi / 2, wires=qubit)


def sxdg(qubit):
    qml.RX(-np.pi / 2, wires=qubit)
