

gate_specifications = {
    'ccx': {
        'name' : 'Toffoli',
        'n_qubits' : 3,
    },

    'ccz': {
        'name' : 'Symmetric',
        'n_qubits' : 3,
    },

    'ch': {
        'name' : 'Controlled Hadamard',
        'n_qubits' : 2,
    },

    'cp': {
        'name' : 'Controlled Phase',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    'crx': {
        'name' : 'Controlled RX',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    'cry': {
        'name' : 'Controlled RY',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    'crz': {
        'name' : 'Controlled RZ',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    'cs': {
        'name' : 'Controlled S',
        'n_qubits' : 2,
    },

    'csdg': {
        'name' : 'Controlled S^dagger',
        'n_qubits' : 2,
    },

    'cswap': {
        'name' : 'Controlled SWAP (Fredkin)',
        'n_qubits' : 3,
    },

    'csx': {
        'name' : 'Controlled sqrt X',
        'n_qubits' : 2,
    },

    'cu': {
        'name' : 'Controlled U',
        'n_qubits' : 2,
        'parameters' : ['theta', 'phi', 'lam', 'gamma'],
    },

    'cx': {
        'name' : 'Controlled X',
        'n_qubits' : 2,
    },

    'cy': {
        'name' : 'Controlled Y',
        'n_qubits' : 2,
    },

    'cz': {
        'name' : 'Controlled Z',
        'n_qubits' : 2,
    },

    'dcx': {
        'name' : 'Double CNOT',
        'n_qubits' : 2,
    },

    'ecr': {
        'name' : 'Echoed Cross-Resonance',
        'n_qubits' : 2,
    },

    'h': {
        'name' : 'Hadamard',
        'n_qubits' : 1,
    },

    'id': {
        'name' : 'Identity',
        'n_qubits' : 1,
    },

    'iswap': {
        'name' : 'iSWAP',
        'n_qubits' : 2,
    },

    'mcp': {
        'needs_validaton' : True,
        'name' : 'Multi-Controlled Phase',
        'n_qubits' : 0,
        'parameters' : ['lam'],
    },

    'mcrx': {
        'needs_validaton' : True,
        'name' : 'Multi-Controlled X Rotation',
        'n_qubits' : 0,
        'parameters' : ['theta'],
    },

    'mcry': {
        'needs_validaton' : True,
        'name' : 'Multi-Controlled Y Rotation',
        'n_qubits' : 0,
        'parameters' : ['theta'],
    },

    'mcrz': {
        'needs_validaton' : True,
        'name' : 'Multi-Controlled Z Rotation',
        'n_qubits' : 0,
        'parameters' : ['theta'],
    },

    'mcx': {
        'needs_validaton' : True,
        'name' : 'Multi-Controlled Z Rotation',
        'n_qubits' : 0,
        'parameters' : ['theta'],
    },

    'ms': {
        'needs_validaton' : True,
        'name' : 'Mølmer–Sørensen',
        'n_qubits' : 0,
    },

    'p': {
        'name' : 'Phase',
        'n_qubits' : 1,
        'parameters' : ['theta'],
    },

    'pauli': {
        'needs_validaton' : True,
        'name' : 'Pauli',
        'n_qubits' : 1,
        'parameters' : ['pauli_string'],
    },

    'r': {
        'name' : 'R',
        'n_qubits' : 1,
        'parameters' : ['theta', 'phi'],
    },

    'rcccx': {
        'name' : 'Simplified 3-Controlled Toffoli',
        'n_qubits' : 4,
    },

    'rccx': {
        'name' : 'Simplified Toffoli (Margolus)',
        'n_qubits' : 3,
    },

    'rv': {
        'name' : 'RV',
        'n_qubits' : 1,
        'parameters' : ['vx', 'vy', 'vz'],
    },

    'rx': {
        'name' : 'RX',
        'n_qubits' : 1,
        'parameters' : ['theta'],
    },

    'rxx': {
        'name' : 'RXX',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    'ry': {
        'name' : 'RY',
        'n_qubits' : 1,
        'parameters' : ['theta'],
    },

    'ryy': {
        'name' : 'RYY',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    'rz': {
        'name' : 'RZ',
        'n_qubits' : 1,
        'parameters' : ['phi'],
    },

    'rzx': {
        'name' : 'RZX',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    'rzz': {
        'name' : 'RZZ',
        'n_qubits' : 2,
        'parameters' : ['theta'],
    },

    's': {
        'name' : 'S',
        'n_qubits' : 1,
    },

    'sdg': {
        'name' : 'S-adjoint',
        'n_qubits' : 1,
    },

    'swap': {
        'name' : 'SWAP',
        'n_qubits' : 2,
    },

    'sx': {
        'name' : 'sqrt X',
        'n_qubits' : 1,
    },

    'sxdg': {
        'name' : 'inverse sqrt X',
        'n_qubits' : 1,
    },

    't': {
        'name' : 'T',
        'n_qubits' : 1,
    },

    'tdg': {
        'name' : 'T-adjoint',
        'n_qubits' : 1,
    },

    'u': {
        'name' : 'U',
        'n_qubits' : 1,
        'parameters' : ['theta', 'phi', 'lam'],
    },

    'x': {
        'name' : 'X',
        'n_qubits' : 1,
    },

    'y': {
        'name' : 'Y',
        'n_qubits' : 1,
    },

    'z': {
        'name' : 'z',
        'n_qubits' : 1,
    },

}
