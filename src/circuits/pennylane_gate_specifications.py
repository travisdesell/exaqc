from src.circuits.gate_specifications import GateSpecification, GateSpecifications

pennylane_gate_specifications = GateSpecifications(target="pennylane")

# ======================================================
# Multi-controlled / multi-qubit gates
# ======================================================

pennylane_gate_specifications["ccx"] = GateSpecification(
    name="Toffoli",
    qubits=["wires"],
)

pennylane_gate_specifications["ccz"] = GateSpecification(
    name="Symmetric",
    qubits=["wires"],
)

pennylane_gate_specifications["ch"] = GateSpecification(
    name="Controlled Hadamard",
    qubits=["wires"],
)

pennylane_gate_specifications["cswap"] = GateSpecification(
    name="Controlled SWAP (Fredkin)",
    qubits=["wires"],
)

pennylane_gate_specifications["cx"] = GateSpecification(
    name="Controlled X",
    qubits=["wires"],
)

pennylane_gate_specifications["cy"] = GateSpecification(
    name="Controlled Y",
    qubits=["wires"],
)

pennylane_gate_specifications["cz"] = GateSpecification(
    name="Controlled Z",
    qubits=["wires"],
)

# ======================================================
# Controlled phase / rotations
# ======================================================

pennylane_gate_specifications["cp"] = GateSpecification(
    name="Controlled Phase",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["crx"] = GateSpecification(
    name="Controlled RX",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["cry"] = GateSpecification(
    name="Controlled RY",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["crz"] = GateSpecification(
    name="Controlled RZ",
    parameters=["phi"],
    qubits=["wires"],
)

# ======================================================
# Controlled S / SX family (no native PL ops → decompose)
# ======================================================

pennylane_gate_specifications["cs"] = GateSpecification(
    needs_validation=True,
    name="Controlled S",
    qubits=["wires"],
)

pennylane_gate_specifications["csdg"] = GateSpecification(
    needs_validation=True,
    name="Controlled S^dagger",
    qubits=["wires"],
)

pennylane_gate_specifications["csx"] = GateSpecification(
    needs_validation=True,
    name="Controlled sqrt X",
    qubits=["wires"],
)

# ======================================================
# CU (general controlled U) – no native op
# ======================================================

pennylane_gate_specifications["cu"] = GateSpecification(
    needs_validation=True,
    name="Controlled U",
    parameters=["theta", "phi", "delta", "gamma"],
    qubits=["wires"],
)

# ======================================================
# Double / echoed / special two-qubit gates
# ======================================================

pennylane_gate_specifications["dcx"] = GateSpecification(
    needs_validation=True,
    name="Double CNOT",
    qubits=["wires"],
)

pennylane_gate_specifications["ecr"] = GateSpecification(
    needs_validation=True,
    name="Echoed Cross-Resonance",
    qubits=["wires"],
)

pennylane_gate_specifications["iswap"] = GateSpecification(
    name="iSWAP",
    qubits=["wires"],
)

pennylane_gate_specifications["swap"] = GateSpecification(
    name="SWAP",
    qubits=["wires"],
)

# ======================================================
# Multi-controlled family (explicitly flagged)
# ======================================================

pennylane_gate_specifications["mcp"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled Phase",
    parameters=["delta"],
    qubits=["wires"],
)

pennylane_gate_specifications["mcrx"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled X Rotation",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["mcry"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled Y Rotation",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["mcrz"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled Z Rotation",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["mcx"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled X",
    qubits=["wires"],
)

# ======================================================
# Mølmer–Sørensen, Pauli, etc.
# ======================================================

pennylane_gate_specifications["ms"] = GateSpecification(
    needs_validation=True,
    name="Mølmer–Sørensen",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["pauli"] = GateSpecification(
    needs_validation=True,
    name="Pauli",
    parameters=["pauli_string"],
    qubits=["wires"],
)

# ======================================================
# Single-qubit standard gates
# ======================================================

pennylane_gate_specifications["h"] = GateSpecification(
    name="Hadamard",
    qubits=["wires"],
)

pennylane_gate_specifications["id"] = GateSpecification(
    name="Identity",
    qubits=["wires"],
)

pennylane_gate_specifications["x"] = GateSpecification(
    name="X",
    qubits=["wires"],
)

pennylane_gate_specifications["y"] = GateSpecification(
    name="Y",
    qubits=["wires"],
)

pennylane_gate_specifications["z"] = GateSpecification(
    name="Z",
    qubits=["wires"],
)

pennylane_gate_specifications["s"] = GateSpecification(
    name="S",
    qubits=["wires"],
)

pennylane_gate_specifications["sdg"] = GateSpecification(
    name="S-adjoint",
    qubits=["wires"],
)

pennylane_gate_specifications["t"] = GateSpecification(
    name="T",
    qubits=["wires"],
)

pennylane_gate_specifications["tdg"] = GateSpecification(
    name="T-adjoint",
    qubits=["wires"],
)

pennylane_gate_specifications["sx"] = GateSpecification(
    needs_validation=True,
    name="sqrt X",
    qubits=["wires"],
)

pennylane_gate_specifications["sxdg"] = GateSpecification(
    needs_validation=True,
    name="inverse sqrt X",
    qubits=["wires"],
)

# ======================================================
# Parametric single-qubit rotations
# Pennylane uses: RX(phi), RY(phi), RZ(phi)
# ======================================================

pennylane_gate_specifications["rx"] = GateSpecification(
    name="RX",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["ry"] = GateSpecification(
    name="RY",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["rz"] = GateSpecification(
    name="RZ",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["r"] = GateSpecification(
    needs_validation=True,
    name="R",
    parameters=["theta", "phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["rv"] = GateSpecification(
    needs_validation=True,
    name="RV",
    parameters=["vx", "vy", "vz"],
    qubits=["wires"],
)

# ======================================================
# Two-qubit parametric interactions
# ======================================================

pennylane_gate_specifications["rxx"] = GateSpecification(
    name="RXX",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["ryy"] = GateSpecification(
    name="RYY",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["rzz"] = GateSpecification(
    name="RZZ",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["rzx"] = GateSpecification(
    needs_validation=True,
    name="RZX",
    parameters=["phi"],
    qubits=["wires"],
)

# ======================================================
# Phase / U family
# Pennylane:
#   U1(phi)
#   U2(phi, delta)
#   U3(theta, phi, delta)
# ======================================================

pennylane_gate_specifications["p"] = GateSpecification(
    name="PhaseShift",
    parameters=["phi"],
    qubits=["wires"],
)

pennylane_gate_specifications["u"] = GateSpecification(
    name="U3",
    parameters=["theta", "phi", "delta"],
    qubits=["wires"],
)

# ======================================================
# Simplified Toffoli variants (no native ops)
# ======================================================

pennylane_gate_specifications["rccx"] = GateSpecification(
    needs_validation=True,
    name="Simplified Toffoli (Margolus)",
    qubits=["wires"],
)

pennylane_gate_specifications["rcccx"] = GateSpecification(
    needs_validation=True,
    name="Simplified 3-Controlled Toffoli",
    qubits=["wires"],
)
