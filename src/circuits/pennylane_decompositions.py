# import pennylane as qml
# import numpy as np

# # =========================
# # MCX (multi-controlled X)
# # =========================
# def mcx(controls, target):
#     if len(controls) == 1:
#         qml.CNOT(wires=[controls[0], target])
#     elif len(controls) == 2:
#         qml.Toffoli(wires=[controls[0], controls[1], target])
#     else:
#         # recursive decomposition
#         qml.Toffoli(wires=[controls[-2], controls[-1], target])
#         mcx(controls[:-1] + [target], target)
#         qml.Toffoli(wires=[controls[-2], controls[-1], target])
#         mcx(controls[:-1] + [target], target)

# # =========================
# # CSX (controlled sqrt X)
# # =========================
# def csx(control, target):
#     qml.Hadamard(wires=target)
#     qml.CRX(np.pi / 2, wires=[control, target])
#     qml.Hadamard(wires=target)

# # =========================
# # CS (controlled S)
# # =========================
# def cs(control, target):
#     qml.ControlledPhaseShift(np.pi / 2, wires=[control, target])

# # =========================
# # CSDG (controlled S-dagger)
# # =========================
# def csdg(control, target):
#     qml.ControlledPhaseShift(-np.pi / 2, wires=[control, target])

# # =========================
# # DCX (Double CNOT)
# # =========================
# def dcx(q1, q2):
#     qml.CNOT(wires=[q1, q2])
#     qml.CNOT(wires=[q2, q1])

# # =========================
# # ECR (Echoed Cross Resonace)
# # =========================
# def ecr(q1, q2):
#     qml.Hadamard(wires=q2)
#     qml.CNOT(wires=[q1, q2])
#     qml.RZ(np.pi/2, wires=q2)
#     qml.CNOT(wires=[q1, q2])
#     qml.Hadamard(wires=q2)

# # =========================
# # RCCX (Margolus)
# # =========================
# def rccx(c1, c2, target):
#     qml.Hadamard(wires=target)
#     qml.T(wires=target)
#     qml.CNOT(wires=[c2, target])
#     qml.T.adjoint()(wires=target)
#     qml.CNOT(wires=[c1, target])
#     qml.T(wires=target)
#     qml.CNOT(wires=[c2, target])
#     qml.T.adjoint()(wires=target)
#     qml.Hadamard(wires=target)

# # =========================
# # RCCCX (3-control Margolus)
# # =========================
# def rcccx(c1, c2, c3, target):
#     qml.Hadamard(wires=target)
#     qml.T(wires=target)
#     qml.Toffoli(wires=[c1, c2, c3])
#     qml.T.adjoint()(wires=target)
#     qml.Hadamard(wires=target)

# # =========================
# # RZX gate
# # =========================
# def rzx(theta, q1, q2):
#     qml.Hadamard(wires=q2)
#     qml.CNOT(wires=[q1, q2])
#     qml.RZ(theta, wires=q2)
#     qml.CNOT(wires=[q1, q2])
#     qml.Hadamard(wires=q2)

# # =========================
# # MS (Molmer-Sorensen)
# # =========================
# def ms(theta, q1, q2):
#     qml.RX(np.pi/2, wires=q1)
#     qml.RX(np.pi/2, wires=q2)
#     qml.CNOT(wires=[q1, q2])
#     qml.RZ(theta, wires=q2)
#     qml.CNOT(wires=[q1, q2])
#     qml.RX(-np.pi/2, wires=q1)
#     qml.RX(-np.pi/2, wires=q2)


import pennylane as qml
import numpy as np

# ============================================================
# Multi-controlled X using recursive decomposition
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
        # Recursive decomposition
        qml.Toffoli(wires=[controls[-2], controls[-1], target])
        mcx(controls[:-1] + [target])
        qml.Toffoli(wires=[controls[-2], controls[-1], target])
        mcx(controls[:-1] + [target])

# ============================================================
# Controlled S / S-dagger / sqrt(X)
# ============================================================
def cs(control_qubit, target_qubit):
    qml.ControlledPhaseShift(np.pi / 2, wires=[control_qubit, target_qubit])

def csdg(control_qubit, target_qubit):
    qml.ControlledPhaseShift(-np.pi / 2, wires=[control_qubit, target_qubit])

def csx(control_qubit, target_qubit):
    qml.Hadamard(wires=target_qubit)
    qml.CRX(np.pi / 2, wires=[control_qubit, target_qubit])
    qml.Hadamard(wires=target_qubit)

# ============================================================
# Double CNOT / Echoed Cross-Resonance
# ============================================================
def dcx(qubit1, qubit2):
    qml.CNOT(wires=[qubit1, qubit2])
    qml.CNOT(wires=[qubit2, qubit1])

def ecr(qubit1, qubit2):
    qml.Hadamard(wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.RZ(np.pi / 2, wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.Hadamard(wires=qubit2)

# ============================================================
# Simplified Toffoli variants
# ============================================================
def rccx(control_qubit1, control_qubit2, target_qubit):
    qml.Hadamard(wires=target_qubit)
    qml.T(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.T.adjoint()(wires=target_qubit)
    qml.CNOT(wires=[control_qubit1, target_qubit])
    qml.T(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.T.adjoint()(wires=target_qubit)
    qml.Hadamard(wires=target_qubit)

def rcccx(control_qubit1, control_qubit2, control_qubit3, target_qubit):
    qml.Hadamard(wires=target_qubit)
    qml.T(wires=target_qubit)
    mcx([control_qubit1, control_qubit2, control_qubit3, target_qubit])
    qml.T.adjoint()(wires=target_qubit)
    qml.Hadamard(wires=target_qubit)

# ============================================================
# Multi-controlled phase / rotations
# ============================================================
def mcp(phi, control_qubits, target_qubit):
    for c in control_qubits:
        qml.CNOT(wires=[c, target_qubit])
    qml.RZ(phi, wires=target_qubit)
    for c in reversed(control_qubits):
        qml.CNOT(wires=[c, target_qubit])

def mcrx(theta, control_qubits, target_qubit):
    for c in control_qubits:
        qml.CNOT(wires=[c, target_qubit])
    qml.RX(theta, wires=target_qubit)
    for c in reversed(control_qubits):
        qml.CNOT(wires=[c, target_qubit])

def mcry(theta, control_qubits, target_qubit):
    for c in control_qubits:
        qml.CNOT(wires=[c, target_qubit])
    qml.RY(theta, wires=target_qubit)
    for c in reversed(control_qubits):
        qml.CNOT(wires=[c, target_qubit])

def mcrz(theta, control_qubits, target_qubit):
    for c in control_qubits:
        qml.CNOT(wires=[c, target_qubit])
    qml.RZ(theta, wires=target_qubit)
    for c in reversed(control_qubits):
        qml.CNOT(wires=[c, target_qubit])

# ============================================================
# CU (general controlled-U)
# ============================================================
def cu(theta, phi, lam, gamma, control_qubit, target_qubit):
    qml.RZ((phi + lam)/2, wires=target_qubit)
    qml.RY(theta/2, wires=target_qubit)
    qml.CNOT(wires=[control_qubit, target_qubit])
    qml.RY(-theta/2, wires=target_qubit)
    qml.RZ(-(phi + lam)/2, wires=target_qubit)
    qml.CNOT(wires=[control_qubit, target_qubit])
    qml.RZ(lam, wires=target_qubit)
    qml.RZ(gamma, wires=control_qubit)

# ============================================================
# RZX gate
# ============================================================
def rzx(theta, qubit1, qubit2):
    qml.Hadamard(wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.RZ(theta, wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.Hadamard(wires=qubit2)

# ============================================================
# MS (Mølmer–Sørensen)
# ============================================================
def ms(theta, qubits):
    q1, q2 = qubits
    qml.RX(np.pi / 2, wires=q1)
    qml.RX(np.pi / 2, wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RZ(theta, wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RX(-np.pi / 2, wires=q1)
    qml.RX(-np.pi / 2, wires=q2)

# ============================================================
# Sqrt X gates
# ============================================================
def sx(qubit):
    qml.RX(np.pi / 2, wires=qubit)

def sxdg(qubit):
    qml.RX(-np.pi / 2, wires=qubit)

# ============================================================
# S-dagger and T-dagger
# ============================================================
def sdg(qubit):
    qml.S.adjoint()(wires=qubit)

def tdg(qubit):
    qml.T.adjoint()(wires=qubit)

# ============================================================
# R / RV (parameterized single-qubit rotations)
# ============================================================
def r(theta, phi, qubit):
    qml.Rot(theta, phi, 0.0, wires=qubit)  # approximate single-qubit R gate

def rv(vx, vy, vz, qubit):
    qml.Rot(vx, vy, vz, wires=qubit)


# ============================================================
# R / RV (parameterized multi-qubit rotations)
# ============================================================
def rxx(theta, q1, q2):
    """Decomposition of RXX gate into native PennyLane operations"""
    qml.CNOT(wires=[q1, q2])
    qml.RX(theta, wires=q2)
    qml.CNOT(wires=[q1, q2])

def ryy(theta, q1, q2):
    """Decomposition of RYY(theta) into native PennyLane operations"""
    # Step 1: H on both qubits
    qml.H(wires=q1)
    qml.H(wires=q2)
    
    # Step 2: S gates on both qubits
    qml.S(wires=q1)
    qml.S(wires=q2)
    
    # Step 3: CNOT, RY, CNOT
    qml.CNOT(wires=[q1, q2])
    qml.RY(theta, wires=q2)
    qml.CNOT(wires=[q1, q2])
    
    # Step 4: undo S and H
    qml.S(wires=q1).adjoint()
    qml.S(wires=q2).adjoint()
    qml.H(wires=q1)
    qml.H(wires=q2)


def rzz(theta, q1, q2):
    """Decomposition of RZZ gate into native PennyLane operations"""
    qml.CNOT(wires=[q1, q2])
    qml.RZ(theta, wires=q2)
    qml.CNOT(wires=[q1, q2])
