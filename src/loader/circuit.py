from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import yaml
import torch


@dataclass
class LoadedModel:
    """Bundle returned by the loader.

    Attributes:
        framework: "pennylane" or "qiskit".
        config: Parsed YAML config.
        params: Dict of torch Parameters (for trainable params).
        build: Callable that returns a circuit object:
            - For PennyLane: returns a QNode callable
            - For Qiskit: returns a QuantumCircuit
        bind: Optional callable used for parameter binding (Qiskit use-case).
    """
    framework: str
    config: Dict[str, Any]
    params: Dict[str, Union[torch.nn.Parameter, torch.Tensor]]
    build: Callable[[], Any]
    bind: Optional[Callable[[Any, Dict[str, Any]], Any]] = None


# -----------------------
# YAML parsing
# -----------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return cfg


# -----------------------
# Parameter handling
# -----------------------

def _init_tensor(init: str, shape: Tuple[int, ...], init_cfg: Dict[str, Any]) -> torch.Tensor:
    """Create an initial tensor according to init scheme."""
    if init == "zeros":
        return torch.zeros(shape, dtype=torch.float64)
    if init == "ones":
        return torch.ones(shape, dtype=torch.float64)
    if init == "normal":
        scale = float(init_cfg.get("scale", 0.01))
        return scale * torch.randn(shape, dtype=torch.float64)
    if init == "uniform":
        lo, hi = init_cfg.get("range", [-0.1, 0.1])
        lo, hi = float(lo), float(hi)
        return (hi - lo) * torch.rand(shape, dtype=torch.float64) + lo
    raise ValueError(f"Unknown init scheme '{init}'.")


def build_torch_parameters(param_cfg: Dict[str, Any]) -> Dict[str, Union[torch.nn.Parameter, torch.Tensor]]:
    """Build torch parameters from YAML 'parameters' section.

    Expected schema:
      parameters:
        theta:
          shape: [6]
          init: normal|uniform|zeros|ones
          scale: 0.01           # for normal
          range: [-0.1, 0.1]    # for uniform
          trainable: true

    Returns:
        Dict mapping parameter name -> torch.nn.Parameter (if trainable) else torch.Tensor.
    """
    params: Dict[str, Union[torch.nn.Parameter, torch.Tensor]] = {}
    for name, spec in (param_cfg or {}).items():
        shape_list = spec.get("shape", [])
        if not isinstance(shape_list, list):
            raise ValueError(f"parameters.{name}.shape must be a list, got {type(shape_list)}")
        shape = tuple(int(x) for x in shape_list) if shape_list else tuple()
        init = str(spec.get("init", "normal"))
        trainable = bool(spec.get("trainable", True))

        t = _init_tensor(init, shape, spec)
        params[name] = torch.nn.Parameter(t) if trainable else t
    return params


# -----------------------
# Expression resolution: "theta[3]" -> tensor element
# -----------------------

def resolve_param(expr: Any, params: Dict[str, Any]) -> Any:
    """Resolve YAML param expressions to concrete python objects.

    Supported:
      - numeric literals (int/float)
      - string like "theta" (whole tensor)
      - string like "theta[3]" or "theta[1,2]" for indexing

    Returns:
        Tensor / scalar suitable for the backend.
    """
    if isinstance(expr, (int, float)):
        return float(expr)

    if not isinstance(expr, str):
        raise ValueError(f"Unsupported param expression type: {type(expr)} ({expr})")

    s = expr.strip()
    if "[" not in s:
        if s not in params:
            raise KeyError(f"Unknown parameter '{s}'")
        return params[s]

    base, idx = s.split("[", 1)
    base = base.strip()
    idx = idx.rstrip("]").strip()
    if base not in params:
        raise KeyError(f"Unknown parameter '{base}'")

    tensor = params[base]
    if isinstance(tensor, torch.nn.Parameter):
        tensor = tensor  # ok
    # parse indices
    parts = [p.strip() for p in idx.split(",") if p.strip()]
    indices = tuple(int(p) for p in parts)
    return tensor[indices] if len(indices) > 1 else tensor[indices[0]]


# -----------------------
# PennyLane builder
# -----------------------

_PL_GATE_MAP = {
    "H": "Hadamard",
    "Hadamard": "Hadamard",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "CNOT": "CNOT",
    "CZ": "CZ",
    "SWAP": "SWAP",
    "Toffoli": "Toffoli",
}

def build_pennylane_qnode(cfg: Dict[str, Any], params: Dict[str, Any]):
    """Build a PennyLane QNode from ML-friendly YAML."""
    import pennylane as qml

    backend = cfg["quantum_model"]["backend"]
    register = cfg["quantum_model"]["register"]
    circuit_spec = cfg["quantum_model"]["circuit"]
    meas = cfg["quantum_model"].get("measurement", {})

    n_qubits = int(register["num_qubits"])
    device_name = backend.get("device", "default.qubit")
    shots = backend.get("shots", None)

    dev = qml.device(device_name, wires=n_qubits, shots=shots)

    def _apply_gate(g: Dict[str, Any]):
        gate = str(g["gate"])
        wires = g.get("wires", [])
        if wires == "all":
            wires = list(range(n_qubits))
        if not isinstance(wires, list):
            raise ValueError(f"Gate wires must be list or 'all', got {wires}")

        op_name = _PL_GATE_MAP.get(gate, gate)
        if not hasattr(qml, op_name):
            raise ValueError(f"PennyLane does not have gate/op '{op_name}' (from '{gate}')")

        op = getattr(qml, op_name)

        if "param" in g:
            angle = resolve_param(g["param"], params)
            op(angle, wires=wires)
        elif "angle" in g:
            op(float(g["angle"]), wires=wires)
        else:
            op(wires=wires)

    # Measurement
    meas_type = meas.get("type", "state")
    meas_wires = meas.get("wires", "all")
    if meas_wires == "all":
        meas_wires = list(range(n_qubits))

    observables = meas.get("observables", [])

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode():
        for g in circuit_spec:
            _apply_gate(g)

        if meas_type == "state":
            return qml.state()
        if meas_type == "probabilities":
            return qml.probs(wires=meas_wires)
        if meas_type == "samples":
            return qml.sample(wires=meas_wires)
        if meas_type == "expectation":
            # Return a list of expvals (stack in torch later if needed)
            out = []
            for obs in observables:
                # allow "PauliZ@0" or dict {"type":"PauliZ","wires":[0]}
                if isinstance(obs, str) and "@" in obs:
                    obs_name, w = obs.split("@", 1)
                    w = int(w)
                    obs_op = getattr(qml, obs_name)
                    out.append(qml.expval(obs_op(wires=w)))
                elif isinstance(obs, dict):
                    obs_name = obs["type"]
                    w = obs["wires"]
                    if isinstance(w, list) and len(w) == 1:
                        w = w[0]
                    obs_op = getattr(qml, obs_name)
                    out.append(qml.expval(obs_op(wires=w)))
                else:
                    raise ValueError(f"Unsupported observable format: {obs}")
            return out

        raise ValueError(f"Unknown measurement.type '{meas_type}'")

    return qnode


# -----------------------
# Qiskit builder
# -----------------------

_QK_GATE_SIMPLE = {"H", "X", "Y", "Z", "CX", "CNOT", "CZ", "SWAP"}
_QK_GATE_PARAM = {"RX", "RY", "RZ"}

def build_qiskit_circuit(cfg: Dict[str, Any], params: Dict[str, Any]):
    """Build a Qiskit QuantumCircuit from ML-friendly YAML.

    Returns:
        qc: QuantumCircuit
        bind_fn: function(qc, values_dict) -> bound circuit
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    register = cfg["quantum_model"]["register"]
    circuit_spec = cfg["quantum_model"]["circuit"]
    meas = cfg["quantum_model"].get("measurement", {})

    n_qubits = int(register["num_qubits"])
    qc = QuantumCircuit(n_qubits)

    # Create Qiskit ParameterVectors for trainable params
    qparams: Dict[str, Any] = {}
    for name, t in params.items():
        # infer length
        if isinstance(t, torch.nn.Parameter):
            shape = tuple(t.shape)
        else:
            shape = tuple(getattr(t, "shape", ()))
        if len(shape) == 0:
            qparams[name] = ParameterVector(name, 1)
        elif len(shape) == 1:
            qparams[name] = ParameterVector(name, shape[0])
        else:
            raise ValueError(
                f"Qiskit loader supports only scalar/vector params for now. '{name}' has shape {shape}."
            )

    def _resolve_qiskit_param(expr: Any):
        if isinstance(expr, (int, float)):
            return float(expr)
        if not isinstance(expr, str):
            raise ValueError(f"Unsupported param expression {expr} for Qiskit.")
        s = expr.strip()
        if "[" not in s:
            pv = qparams.get(s)
            if pv is None:
                raise KeyError(f"Unknown parameter '{s}' in Qiskit param table.")
            return pv[0]  # scalar
        base, idx = s.split("[", 1)
        base = base.strip()
        idx = idx.rstrip("]").strip()
        pv = qparams.get(base)
        if pv is None:
            raise KeyError(f"Unknown parameter '{base}'")
        i = int(idx)
        return pv[i]

    def _apply_gate(g: Dict[str, Any]):
        gate = str(g["gate"])
        wires = g.get("wires", [])
        if wires == "all":
            wires = list(range(n_qubits))

        # normalize name
        if gate == "CNOT":
            gate = "CX"

        if gate in _QK_GATE_SIMPLE:
            if gate == "H":
                qc.h(wires[0])
            elif gate == "X":
                qc.x(wires[0])
            elif gate == "Y":
                qc.y(wires[0])
            elif gate == "Z":
                qc.z(wires[0])
            elif gate in {"CX", "CNOT"}:
                qc.cx(wires[0], wires[1])
            elif gate == "CZ":
                qc.cz(wires[0], wires[1])
            elif gate == "SWAP":
                qc.swap(wires[0], wires[1])
            else:
                raise ValueError(f"Unsupported simple Qiskit gate '{gate}'")
            return

        if gate in _QK_GATE_PARAM:
            angle = _resolve_qiskit_param(g.get("param", g.get("angle")))
            if gate == "RX":
                qc.rx(angle, wires[0])
            elif gate == "RY":
                qc.ry(angle, wires[0])
            elif gate == "RZ":
                qc.rz(angle, wires[0])
            return

        if gate == "Toffoli":
            qc.ccx(wires[0], wires[1], wires[2])
            return

        raise ValueError(f"Unsupported Qiskit gate '{gate}'")

    for g in circuit_spec:
        _apply_gate(g)

    # Measurement (classical bits)
    meas_type = meas.get("type", None)
    if meas_type in {"samples", "probabilities"}:
        wires = meas.get("wires", "all")
        if wires == "all":
            wires = list(range(n_qubits))
        qc.measure_all() if wires == list(range(n_qubits)) else qc.measure(wires, wires)

    def bind_fn(qc_in: Any, torch_params: Dict[str, Any]):
        """Bind Qiskit parameters using current torch parameter values.

        Args:
            qc_in: QuantumCircuit with Parameters.
            torch_params: dict of torch tensors/Parameters.

        Returns:
            A new QuantumCircuit with parameters assigned.
        """
        assign_map = {}
        for name, pv in qparams.items():
            t = torch_params[name]
            t_val = t.detach().cpu().numpy() if isinstance(t, torch.nn.Parameter) else t
            t_val = t_val.reshape(-1)
            for i in range(len(pv)):
                assign_map[pv[i]] = float(t_val[i])
        return qc_in.assign_parameters(assign_map, inplace=False)

    return qc, bind_fn


# -----------------------
# Public loader API
# -----------------------

def load_quantum_model_from_yaml(path: str) -> LoadedModel:
    """Load ML-friendly YAML and return a circuit builder (PennyLane or Qiskit).

    YAML schema expected (ML-friendly version):
      quantum_model:
        backend:
          framework: pennylane|qiskit
          device: default.qubit|qiskit.aer|...
          shots: null|1024
        register:
          num_qubits: N
        parameters:
          theta: {shape: [K], init: normal, scale: 0.01, trainable: true}
        circuit:
          - gate: RY
            wires: [0]
            param: theta[0]
          - gate: CNOT
            wires: [0, 1]
        measurement:
          type: state|probabilities|expectation|samples
          wires: all|[...]
          observables: [PauliZ@0, PauliZ@1]

    Returns:
        LoadedModel containing params and a builder function.
    """
    cfg = load_yaml(path)
    qm = cfg.get("quantum_model")
    if not isinstance(qm, dict):
        raise ValueError("YAML must contain a top-level 'quantum_model' mapping.")

    backend = qm.get("backend", {})
    framework = str(backend.get("framework", "pennylane")).lower()

    params = build_torch_parameters(qm.get("parameters", {}))

    if framework == "pennylane":
        def build():
            return build_pennylane_qnode(cfg, params)
        return LoadedModel(framework="pennylane", config=cfg, params=params, build=build)

    if framework == "qiskit":
        def build():
            qc, bind_fn = build_qiskit_circuit(cfg, params)
            return qc
        # expose binder for convenience
        qc, bind_fn = build_qiskit_circuit(cfg, params)
        return LoadedModel(framework="qiskit", config=cfg, params=params, build=lambda: qc, bind=bind_fn)

    raise ValueError(f"Unknown backend.framework '{framework}'. Use 'pennylane' or 'qiskit'.")
