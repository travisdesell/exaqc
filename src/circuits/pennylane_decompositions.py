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
    qml.ctrl(qml.SX(wires=target_qubit), control=control_qubit)


# ============================================================
# Double CNOT / Echoed Cross-Resonance
# ============================================================
def dcx(qubit1, qubit2):
    qml.CNOT(wires=[qubit1, qubit2])
    qml.CNOT(wires=[qubit2, qubit1])


def ecr(qubit1, qubit2):
    qml.Hadamard(wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.RZ(np.pi / 4, wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.Hadamard(wires=qubit2)

    qml.PauliX(wires=qubit1)

    qml.Hadamard(wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.RZ(-np.pi / 4, wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])
    qml.Hadamard(wires=qubit2)


# ============================================================
# Simplified Toffoli variants
# ============================================================
def rccx(control_qubit1, control_qubit2, target_qubit):
    qml.Hadamard(wires=target_qubit)
    qml.T(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.T(wires=target_qubit).adjoint()
    qml.CNOT(wires=[control_qubit1, target_qubit])
    qml.T(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.T(wires=target_qubit).adjoint()
    qml.Hadamard(wires=target_qubit)


def rcccx(control_qubit1, control_qubit2, control_qubit3, target_qubit):
    qml.Hadamard(wires=target_qubit)
    qml.CNOT(wires=[control_qubit3, target_qubit])
    qml.adjoint(qml.T)(wires=target_qubit)
    qml.CNOT(wires=[control_qubit1, target_qubit])
    qml.T(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.adjoint(qml.T)(wires=target_qubit)
    qml.CNOT(wires=[control_qubit1, target_qubit])
    qml.T(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.adjoint(qml.T)(wires=target_qubit)
    qml.CNOT(wires=[control_qubit3, target_qubit])
    qml.T(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.adjoint(qml.T)(wires=target_qubit)
    qml.CNOT(wires=[control_qubit2, target_qubit])
    qml.Hadamard(wires=target_qubit)


# ============================================================
# Multi-controlled phase / rotations
# ============================================================
def mcp(phi, control_qubits, target_qubit):
    qml.ctrl(qml.PhaseShift(phi, wires=target_qubit), control=control_qubits)


def mcrx(theta, control_qubits, target_qubit):
    qml.ctrl(qml.RX(theta, wires=target_qubit), control=control_qubits)


def mcry(theta, control_qubits, target_qubit):
    qml.ctrl(qml.RY(theta, wires=target_qubit), control=control_qubits)


def mcrz(theta, control_qubits, target_qubit):
    qml.ctrl(qml.RZ(theta, wires=target_qubit), control=control_qubits)


# ============================================================
# CU (general controlled-U)
# ============================================================
def cu(theta, phi, lam, gamma, control_qubit, target_qubit):
    # qml.RZ((phi + lam) / 2, wires=target_qubit)
    # qml.RY(theta / 2, wires=target_qubit)
    # qml.CNOT(wires=[control_qubit, target_qubit])
    # qml.RY(-theta / 2, wires=target_qubit)
    # qml.RZ(-(phi + lam) / 2, wires=target_qubit)
    # qml.CNOT(wires=[control_qubit, target_qubit])
    # qml.RZ(lam, wires=target_qubit)
    # qml.RZ(gamma, wires=control_qubit)

    qml.ControlledPhaseShift(gamma, wires=[control_qubit, target_qubit])
    qml.ctrl(qml.U3(theta, phi, lam, wires=target_qubit), control=control_qubit)


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
    qml.SX(wires=qubit).adjoint()


# ============================================================
# S-dagger and T-dagger
# ============================================================
def sdg(qubit):
    qml.S(wires=qubit).adjoint()


def tdg(qubit):
    qml.T(wires=qubit).adjoint()


# ============================================================
# R / RV (parameterized single-qubit rotations)
# ============================================================
def r(theta, phi, qubit):
    qml.Rot(phi, theta, -phi, wires=qubit)  # approximate single-qubit R gate


def rv(vx, vy, vz, qubit):
    qml.Rot(vx, vy, vz, wires=qubit)
