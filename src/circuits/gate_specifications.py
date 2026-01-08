'''
The following dict provides information about all the possible gates that can
be applied to a QuantumCircuit.

The key for an entry is the method name to be used on the QuantumCircuit object.
 * 'name' for an entry is a long form name for the gate.
 * if 'parameters' is present, it specifies the parameter names taken by the gate
    method. These need to be in the same order that the gate method accepts these
    arguments (which typically come before any qubit arguments).
 * 'qubits' specfies the qubit argument names. These need to be in the same order
    as the game method that accepts them (and they typically come after any
    qubit arguments).
'''

gate_specifications = {
    'ccx': {
        'name' : 'Toffoli',
        'qubits' : ['control_qubit1', 'control_qubit2', 'target_qubit'],
    },

    'ccz': {
        'name' : 'Symmetric',
        'qubits' : ['control_qubit1', 'control_qubit2', 'target_qubit'],
    },

    'ch': {
        'name' : 'Controlled Hadamard',
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cp': {
        'name' : 'Controlled Phase',
        'parameters' : ['theta'],
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'crx': {
        'name' : 'Controlled RX',
        'parameters' : ['theta'],
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cry': {
        'name' : 'Controlled RY',
        'parameters' : ['theta'],
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'crz': {
        'name' : 'Controlled RZ',
        'parameters' : ['theta'],
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cs': {
        'name' : 'Controlled S',
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'csdg': {
        'name' : 'Controlled S^dagger',
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cswap': {
        'name' : 'Controlled SWAP (Fredkin)',
        'qubits' : ['control_qubit', 'target_qubit1', 'target_qubit2'],
    },

    'csx': {
        'name' : 'Controlled sqrt X',
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cu': {
        'name' : 'Controlled U',
        'parameters' : ['theta', 'phi', 'lam', 'gamma'],
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cx': {
        'name' : 'Controlled X',
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cy': {
        'name' : 'Controlled Y',
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'cz': {
        'name' : 'Controlled Z',
        'qubits' : ['control_qubit', 'target_qubit'],
    },

    'dcx': {
        'name' : 'Double CNOT',
        'qubits' : ['qubit1', 'qubit2'],
    },

    'ecr': {
        'name' : 'Echoed Cross-Resonance',
        'qubits' : ['qubit1', 'qubit2'],
    },

    'h': {
        'name' : 'Hadamard',
        'qubits' : ['qubit'],
    },

    'id': {
        'name' : 'Identity',
        'qubits' : ['qubit'],
    },

    'iswap': {
        'name' : 'iSWAP',
        'qubits' : ['qubit1', 'qubit2'],
    },

    'mcp': {
        'needs_validation' : True,
        'name' : 'Multi-Controlled Phase',
        'parameters' : ['lam'],
        'qubits' : ['control_qubits...', 'target_qubit'],
    },

    'mcrx': {
        'needs_validation' : True,
        'name' : 'Multi-Controlled X Rotation',
        'parameters' : ['theta'],
        'qubits' : ['q_controls...', 'q_target'],
    },

    'mcry': {
        'needs_validation' : True,
        'name' : 'Multi-Controlled Y Rotation',
        'parameters' : ['theta'],
        'qubits' : ['q_controls...', 'q_target'],
    },

    'mcrz': {
        'needs_validation' : True,
        'name' : 'Multi-Controlled Z Rotation',
        'parameters' : ['theta'],
        'qubits' : ['q_controls...', 'q_target'],
    },

    'mcx': {
        'needs_validation' : True,
        'name' : 'Multi-Controlled X',
        'parameters' : ['theta'],
        'qubits' : ['control_qubits...', 'target_qubit'],
    },

    'ms': {
        'needs_validation' : True,
        'name' : 'Mølmer–Sørensen',
        'parameters' : ['theta'],
        'qubits' : ['qubits...'],
    },

    'p': {
        'name' : 'Phase',
        'parameters' : ['theta'],
        'qubits' : ['qubit'],
    },

    'pauli': {
        'needs_validation' : True,
        'name' : 'Pauli',
        'parameters' : ['pauli_string'],
        'qubits' : ['qubits...'],
    },

    'r': {
        'name' : 'R',
        'parameters' : ['theta', 'phi'],
        'qubits' : ['qubit'],
    },

    'rcccx': {
        'name' : 'Simplified 3-Controlled Toffoli',
        'qubits' : ['control_qubit1', 'control_qubit2', 'control_qubit3', 'target_qubit'],
    },

    'rccx': {
        'name' : 'Simplified Toffoli (Margolus)',
        'qubits' : ['control_qubit1', 'control_qubit2', 'target_qubit'],
    },

    'rv': {
        'name' : 'RV',
        'parameters' : ['vx', 'vy', 'vz'],
        'qubits' : ['qubit'],
    },

    'rx': {
        'name' : 'RX',
        'parameters' : ['theta'],
        'qubits' : ['qubit'],
    },

    'rxx': {
        'name' : 'RXX',
        'parameters' : ['theta'],
        'qubits' : ['qubit1', 'qubit2'],
    },

    'ry': {
        'name' : 'RY',
        'parameters' : ['theta'],
        'qubits' : ['qubit'],
    },

    'ryy': {
        'name' : 'RYY',
        'parameters' : ['theta'],
        'qubits' : ['qubit1', 'qubit2'],
    },

    'rz': {
        'name' : 'RZ',
        'parameters' : ['phi'],
        'qubits' : ['qubit'],
    },

    'rzx': {
        'name' : 'RZX',
        'parameters' : ['theta'],
        'qubits' : ['qubit1', 'qubit2'],
    },

    'rzz': {
        'name' : 'RZZ',
        'parameters' : ['theta'],
        'qubits' : ['qubit1', 'qubit2'],
    },

    's': {
        'name' : 'S',
        'qubits' : ['qubit'],
    },

    'sdg': {
        'name' : 'S-adjoint',
        'qubits' : ['qubit'],
    },

    'swap': {
        'name' : 'SWAP',
        'qubits' : ['qubit1', 'qubit2'],
    },

    'sx': {
        'name' : 'sqrt X',
        'qubits' : ['qubit'],
    },

    'sxdg': {
        'name' : 'inverse sqrt X',
        'qubits' : ['qubit'],
    },

    't': {
        'name' : 'T',
        'qubits' : ['qubit'],
    },

    'tdg': {
        'name' : 'T-adjoint',
        'qubits' : ['qubit'],
    },

    'u': {
        'name' : 'U',
        'parameters' : ['theta', 'phi', 'lam'],
        'qubits' : ['qubit'],
    },

    'x': {
        'name' : 'X',
        'qubits' : ['qubit'],
    },

    'y': {
        'name' : 'Y',
        'qubits' : ['qubit'],
    },

    'z': {
        'name' : 'z',
        'qubits' : ['qubit'],
    },

}