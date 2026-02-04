from src.circuits.gate_specifications import GateSpecification, GateSpecifications

qiskit_gate_specifications = GateSpecifications(target="qiskit")
qiskit_gate_specifications["ccx"] = GateSpecification(
    name="Toffoli", qubits=["control_qubit1", "control_qubit2", "target_qubit"]
)

qiskit_gate_specifications["ccz"] = GateSpecification(
    name="Symmetric",
    qubits=["control_qubit1", "control_qubit2", "target_qubit"],
)

qiskit_gate_specifications["ch"] = GateSpecification(
    name="Controlled Hadamard",
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cp"] = GateSpecification(
    name="Controlled Phase",
    parameters=["theta"],
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["crx"] = GateSpecification(
    name="Controlled RX",
    parameters=["theta"],
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cry"] = GateSpecification(
    name="Controlled RY",
    parameters=["theta"],
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["crz"] = GateSpecification(
    name="Controlled RZ",
    parameters=["theta"],
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cs"] = GateSpecification(
    name="Controlled S",
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["csdg"] = GateSpecification(
    name="Controlled S^dagger",
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cswap"] = GateSpecification(
    name="Controlled SWAP (Fredkin)",
    qubits=["control_qubit", "target_qubit1", "target_qubit2"],
)

qiskit_gate_specifications["csx"] = GateSpecification(
    name="Controlled sqrt X",
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cu"] = GateSpecification(
    name="Controlled U",
    parameters=["theta", "phi", "lam", "gamma"],
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cx"] = GateSpecification(
    name="Controlled X",
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cy"] = GateSpecification(
    name="Controlled Y",
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["cz"] = GateSpecification(
    name="Controlled Z",
    qubits=["control_qubit", "target_qubit"],
)

qiskit_gate_specifications["dcx"] = GateSpecification(
    name="Double CNOT",
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["ecr"] = GateSpecification(
    name="Echoed Cross-Resonance",
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["h"] = GateSpecification(
    name="Hadamard",
    qubits=["qubit"],
)

qiskit_gate_specifications["id"] = GateSpecification(
    name="Identity",
    qubits=["qubit"],
    needs_validation=True, # doesn't need validation but doesnt make sense to use
)

qiskit_gate_specifications["iswap"] = GateSpecification(
    name="iSWAP",
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["mcp"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled Phase",
    parameters=["lam"],
    qubits=["control_qubits...", "target_qubit"],
)

qiskit_gate_specifications["mcrx"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled X Rotation",
    parameters=["theta"],
    qubits=["q_controls...", "q_target"],
)

qiskit_gate_specifications["mcry"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled Y Rotation",
    parameters=["theta"],
    qubits=["q_controls...", "q_target"],
)

qiskit_gate_specifications["mcrz"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled Z Rotation",
    parameters=["theta"],
    qubits=["q_controls...", "q_target"],
)

qiskit_gate_specifications["mcx"] = GateSpecification(
    needs_validation=True,
    name="Multi-Controlled X",
    parameters=["theta"],
    qubits=["control_qubits...", "target_qubit"],
)

qiskit_gate_specifications["ms"] = GateSpecification(
    needs_validation=True,
    name="Mølmer–Sørensen",
    parameters=["theta"],
    qubits=["qubits..."],
)

qiskit_gate_specifications["p"] = GateSpecification(
    name="Phase",
    parameters=["theta"],
    qubits=["qubit"],
)

qiskit_gate_specifications["pauli"] = GateSpecification(
    needs_validation=True,
    name="Pauli",
    parameters=["pauli_string"],
    qubits=["qubits..."],
)

qiskit_gate_specifications["r"] = GateSpecification(
    name="R",
    parameters=["theta", "phi"],
    qubits=["qubit"],
)

qiskit_gate_specifications["rcccx"] = GateSpecification(
    name="Simplified 3-Controlled Toffoli",
    qubits=["control_qubit1", "control_qubit2", "control_qubit3", "target_qubit"],
)

qiskit_gate_specifications["rccx"] = GateSpecification(
    name="Simplified Toffoli (Margolus)",
    qubits=["control_qubit1", "control_qubit2", "target_qubit"],
)

qiskit_gate_specifications["rv"] = GateSpecification(
    name="RV",
    parameters=["vx", "vy", "vz"],
    qubits=["qubit"],
)

qiskit_gate_specifications["rx"] = GateSpecification(
    name="RX",
    parameters=["theta"],
    qubits=["qubit"],
)

qiskit_gate_specifications["rxx"] = GateSpecification(
    name="RXX",
    parameters=["theta"],
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["ry"] = GateSpecification(
    name="RY",
    parameters=["theta"],
    qubits=["qubit"],
)

qiskit_gate_specifications["ryy"] = GateSpecification(
    name="RYY",
    parameters=["theta"],
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["rz"] = GateSpecification(
    name="RZ",
    parameters=["phi"],
    qubits=["qubit"],
)

qiskit_gate_specifications["rzx"] = GateSpecification(
    name="RZX",
    parameters=["theta"],
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["rzz"] = GateSpecification(
    name="RZZ",
    parameters=["theta"],
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["s"] = GateSpecification(
    name="S",
    qubits=["qubit"],
)

qiskit_gate_specifications["sdg"] = GateSpecification(
    name="S-adjoint",
    qubits=["qubit"],
)

qiskit_gate_specifications["swap"] = GateSpecification(
    name="SWAP",
    qubits=["qubit1", "qubit2"],
)

qiskit_gate_specifications["sx"] = GateSpecification(
    name="sqrt X",
    qubits=["qubit"],
)

qiskit_gate_specifications["sxdg"] = GateSpecification(
    name="inverse sqrt X",
    qubits=["qubit"],
)

qiskit_gate_specifications["t"] = GateSpecification(
    name="T",
    qubits=["qubit"],
)

qiskit_gate_specifications["tdg"] = GateSpecification(
    name="T-adjoint",
    qubits=["qubit"],
)

qiskit_gate_specifications["u"] = GateSpecification(
    name="U",
    parameters=["theta", "phi", "lam"],
    qubits=["qubit"],
)

qiskit_gate_specifications["x"] = GateSpecification(
    name="X",
    qubits=["qubit"],
)

qiskit_gate_specifications["y"] = GateSpecification(
    name="Y",
    qubits=["qubit"],
)

qiskit_gate_specifications["z"] = GateSpecification(
    name="z",
    qubits=["qubit"],
)


"""
print("\\begin{table}[h!]")
print("\\begin{tabular}{llll}")
print("\\hline")
print("\\textbf{Gate} & \\textbf{Method} & \\textbf{Qubits} & \\textbf{Parameters} \\\\")
print("\\hline")

for method_name, gate in sorted(qiskit_gate_specifications.items()):
    if not gate.needs_validation:
        name = gate.name.replace('^dagger', '\\textsuperscript{\\textdagger}')
        params = ''
        if gate.parameters is not None:
            params = ', '.join(gate.parameters)

        qubits = ', '.join(gate.qubits)
        qubits = qubits.replace('_', '\\_')

        print(f"{name} & {method_name} & {qubits} & {params} \\\\")
        print("\\hline")

print("\\hline")
print("\\end{tabular}")
print("\\caption{Available Qiskit gates, their qubits and parameters (if parameterized).}")
print("\\label{tab:qiskit_gates}")
print("\\end{table}")
"""
