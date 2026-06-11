# tests/test_epoch_strategy.py
from __future__ import annotations

from math import floor
from unittest.mock import patch

from src.circuits.pennylane_gate_specifications import pennylane_gate_specifications
from src.circuits.registers import expand_registers
from src.evolution.exaqc import EXAQC
from src.evolution.steady_state_population import SteadyStatePopulation


def _make_exaqc(**kwargs):
    def compare(_g1, _g2):
        return 0

    defaults = dict(
        gate_specifications=pennylane_gate_specifications,
        population=SteadyStatePopulation(
            max_population_size=5, compare=compare, out_dir=None
        ),
        objective=lambda g: None,
        hyperparameters={"epochs": 30, "learning_rate": 0.01},
        mutation_strategy=["uniform", "1", "2"],
        input_qubits=expand_registers({"input": 2, "output": 1}),
    )
    defaults.update(kwargs)
    return EXAQC(**defaults)


def test_epoch_strategy_const():
    ex = _make_exaqc(epoch_strategy="const", bp_min=5, bp_max=20)
    assert ex.get_hyperparameters()["epochs"] == 20


def test_epoch_strategy_scaled():
    ex = _make_exaqc(
        epoch_strategy="scaled", slope=1.0, exponent=1.0, bp_min=0, bp_max=100
    )
    ex.genome_number = 10
    expected = min(floor(1.0 * 10) + 0, 100)
    assert ex.get_hyperparameters()["epochs"] == expected


def test_epoch_strategy_rand():
    ex = _make_exaqc(epoch_strategy="rand", bp_min=5, bp_max=10)
    with patch("src.evolution.exaqc.random.randint", return_value=7):
        assert ex.get_hyperparameters()["epochs"] == 7
