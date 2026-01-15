from src.circuits.gate_specifications import GateSpecification, GateSpecifications

pennylane_gate_specifications = GateSpecifications(target="pennylane")

# ======================================================
# Multi-controlled / controlled gates (native PL)
# ======================================================
pennylane_gate_specifications["ccx"] = GateSpecification(
    name="Toffoli",
    qubits=["control_qubit1", "control_qubit2", "target_qubit"],
    pennylane_op="Toffoli",
)

pennylane_gate_specifications["ccz"] = GateSpecification(
    name="CCZ",
    qubits=["control_qubit1", "control_qubit2", "target_qubit"],
    pennylane_op="CCZ",
)

pennylane_gate_specifications["ch"] = GateSpecification(
    name="Controlled Hadamard",
    qubits=["control_qubit", "target_qubit"],
    pennylane_op="CH",
)

pennylane_gate_specifications["cx"] = GateSpecification(
    name="Controlled X",
    qubits=["control_qubit", "target_qubit"],
    pennylane_op="CNOT",
)

pennylane_gate_specifications["cy"] = GateSpecification(
    name="Controlled Y",
    qubits=["control_qubit", "target_qubit"],
    pennylane_op="CY",
)

pennylane_gate_specifications["cz"] = GateSpecification(
    name="Controlled Z",
    qubits=["control_qubit", "target_qubit"],
    pennylane_op="CZ",
)

pennylane_gate_specifications["cswap"] = GateSpecification(
    name="Controlled SWAP (Fredkin)",
    qubits=["control_qubit", "target_qubit1", "target_qubit2"],
    pennylane_op="CSWAP",
)

# ======================================================
# Controlled parametric gates (native PL)
# ======================================================
pennylane_gate_specifications["cp"] = GateSpecification(
    name="Controlled Phase",
    qubits=["control_qubit", "target_qubit"],
    parameters=["phi"],
    pennylane_op="ControlledPhaseShift",
)

pennylane_gate_specifications["crx"] = GateSpecification(
    name="Controlled RX",
    qubits=["control_qubit", "target_qubit"],
    parameters=["phi"],
    pennylane_op="CRX",
)

pennylane_gate_specifications["cry"] = GateSpecification(
    name="Controlled RY",
    qubits=["control_qubit", "target_qubit"],
    parameters=["phi"],
    pennylane_op="CRY",
)

pennylane_gate_specifications["crz"] = GateSpecification(
    name="Controlled RZ",
    qubits=["control_qubit", "target_qubit"],
    parameters=["phi"],
    pennylane_op="CRZ",
)

# ======================================================
# Controlled / single-qubit decompositions (needs_validation)
# ======================================================
pennylane_gate_specifications["cs"] = GateSpecification(
    name="Controlled S",
    qubits=["control_qubit", "target_qubit"],
    needs_validation=True,
)

pennylane_gate_specifications["csdg"] = GateSpecification(
    name="Controlled S^dagger",
    qubits=["control_qubit", "target_qubit"],
    needs_validation=True,
)

pennylane_gate_specifications["csx"] = GateSpecification(
    name="Controlled sqrt X",
    qubits=["control_qubit", "target_qubit"],
    needs_validation=True,
)

pennylane_gate_specifications["cu"] = GateSpecification(
    name="Controlled U",
    qubits=["control_qubit", "target_qubit"],
    parameters=["theta", "phi", "lam", "gamma"],
    needs_validation=True,
)

pennylane_gate_specifications["dcx"] = GateSpecification(
    name="Double CNOT",
    qubits=["qubit1", "qubit2"],
    needs_validation=True,
)

pennylane_gate_specifications["ecr"] = GateSpecification(
    name="Echoed Cross-Resonance",
    qubits=["qubit1", "qubit2"],
    needs_validation=True,
)

pennylane_gate_specifications["rccx"] = GateSpecification(
    name="Simplified Toffoli (Margolus)",
    qubits=["control_qubit1", "control_qubit2", "target_qubit"],
    needs_validation=True,
)

pennylane_gate_specifications["rcccx"] = GateSpecification(
    name="Simplified 3-Controlled Toffoli",
    qubits=["control_qubit1", "control_qubit2", "control_qubit3", "target_qubit"],
    needs_validation=True,
)

# ======================================================
# Multi-controlled variadic gates
# ======================================================
pennylane_gate_specifications["mcx"] = GateSpecification(
    name="Multi-Controlled X",
    qubits=["control_qubits...", "target_qubit"],
    needs_validation=True,
)

pennylane_gate_specifications["mcp"] = GateSpecification(
    name="Multi-Controlled Phase",
    qubits=["control_qubits...", "target_qubit"],
    parameters=["theta"],
    needs_validation=True,
)

pennylane_gate_specifications["mcrx"] = GateSpecification(
    name="Multi-Controlled X Rotation",
    qubits=["control_qubits...", "target_qubit"],
    parameters=["theta"],
    needs_validation=True,
)

pennylane_gate_specifications["mcry"] = GateSpecification(
    name="Multi-Controlled Y Rotation",
    qubits=["control_qubits...", "target_qubit"],
    parameters=["theta"],
    needs_validation=True,
)

pennylane_gate_specifications["mcrz"] = GateSpecification(
    name="Multi-Controlled Z Rotation",
    qubits=["control_qubits...", "target_qubit"],
    parameters=["theta"],
    needs_validation=True,
)

# ======================================================
# Single-qubit standard gates
# ======================================================
pennylane_gate_specifications["h"] = GateSpecification(
    name="Hadamard",
    qubits=["qubit"],
    pennylane_op="Hadamard",
)

pennylane_gate_specifications["id"] = GateSpecification(
    name="Identity",
    qubits=["qubit"],
    pennylane_op="Identity",
)

pennylane_gate_specifications["x"] = GateSpecification(
    name="X",
    qubits=["qubit"],
    pennylane_op="PauliX",
)

pennylane_gate_specifications["y"] = GateSpecification(
    name="Y",
    qubits=["qubit"],
    pennylane_op="PauliY",
)

pennylane_gate_specifications["z"] = GateSpecification(
    name="Z",
    qubits=["qubit"],
    pennylane_op="PauliZ",
)

pennylane_gate_specifications["s"] = GateSpecification(
    name="S",
    qubits=["qubit"],
    pennylane_op="S",
)

pennylane_gate_specifications["sdg"] = GateSpecification(
    name="S-adjoint",
    qubits=["qubit"],
    needs_validation=True,  # use decomposition
)

pennylane_gate_specifications["t"] = GateSpecification(
    name="T",
    qubits=["qubit"],
    pennylane_op="T",
)

pennylane_gate_specifications["tdg"] = GateSpecification(
    name="T-adjoint",
    qubits=["qubit"],
    needs_validation=True,  # use decomposition
)

pennylane_gate_specifications["sx"] = GateSpecification(
    name="sqrt X",
    qubits=["qubit"],
    needs_validation=True,
)

pennylane_gate_specifications["sxdg"] = GateSpecification(
    name="inverse sqrt X",
    qubits=["qubit"],
    needs_validation=True,
)

# ======================================================
# Parametric single-qubit rotations
# ======================================================
pennylane_gate_specifications["rx"] = GateSpecification(
    name="RX",
    qubits=["qubit"],
    parameters=["phi"],
    pennylane_op="RX",
)

pennylane_gate_specifications["ry"] = GateSpecification(
    name="RY",
    qubits=["qubit"],
    parameters=["phi"],
    pennylane_op="RY",
)

pennylane_gate_specifications["rz"] = GateSpecification(
    name="RZ",
    qubits=["qubit"],
    parameters=["phi"],
    pennylane_op="RZ",
)

pennylane_gate_specifications["r"] = GateSpecification(
    name="R",
    qubits=["qubit"],
    parameters=["theta", "phi"],
    needs_validation=True,
)

pennylane_gate_specifications["rv"] = GateSpecification(
    name="RV",
    qubits=["qubit"],
    parameters=["vx", "vy", "vz"],
    needs_validation=True,
)

# ======================================================
# Two-qubit parametric gates
# ======================================================
pennylane_gate_specifications["rxx"] = GateSpecification(
    name="RXX",
    qubits=["qubit1", "qubit2"],
    parameters=["theta"],
)

pennylane_gate_specifications["ryy"] = GateSpecification(
    name="RYY",
    qubits=["qubit1", "qubit2"],
    parameters=["theta"],
    needs_validation=True,
)

pennylane_gate_specifications["rzz"] = GateSpecification(
    name="RZZ",
    qubits=["qubit1", "qubit2"],
    parameters=["theta"],
    pennylane_op="MultiRZ",
)

pennylane_gate_specifications["rzx"] = GateSpecification(
    name="RZX",
    qubits=["qubit1", "qubit2"],
    parameters=["theta"],
    needs_validation=True,
)

# ======================================================
# iSWAP and SWAP
# ======================================================
pennylane_gate_specifications["iswap"] = GateSpecification(
    name="iSWAP",
    qubits=["qubit1", "qubit2"],
    pennylane_op="ISWAP",
)

pennylane_gate_specifications["swap"] = GateSpecification(
    name="SWAP",
    qubits=["qubit1", "qubit2"],
    pennylane_op="SWAP",
)

# ======================================================
# Phase / U family
# ======================================================
pennylane_gate_specifications["p"] = GateSpecification(
    name="Phase",
    qubits=["qubit"],
    parameters=["phi"],
    pennylane_op="PhaseShift",
)

pennylane_gate_specifications["u"] = GateSpecification(
    name="U",
    qubits=["qubit"],
    parameters=["theta", "phi", "delta"],
    pennylane_op="U3",
)
