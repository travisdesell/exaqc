from __future__ import annotations

import math
import pytest
from src.circuits.circuit import CircuitGenome
from src.circuits.registers import expand_registers
from src.objectives.genome_objectives import (
    train_genome_objective,
)
from src.quantum_datasets import (
    IrisDataset,
    WineDataset,
    SeedsDataset,
    BreastCancerDataset,
)

target_backend = "pennylane"


def make_dummy_genome(genome_number: int, input_qubits: int, out_qubits: int):
    input_qubits = expand_registers({"input": input_qubits})
    output_qubits = expand_registers({"output": out_qubits})

    genomes = [
        lambda: CircuitGenome(
            genome_number=genome_number,
            target=target_backend,
            input_qubits=input_qubits.copy(),
            output_qubits=output_qubits.copy(),
        ),
    ]

    last_err = None
    g = None
    for genome in genomes:
        try:
            g = genome()
            break
        except Exception as e:
            last_err = e

    if g is None:
        pytest.fail(
            "Could not construct CircuitGenome. "
            f"Update make_dummy_genome(). Last error: {last_err}"
        )

    return g


DATASETS = [
    ("iris", IrisDataset, 4, 3),
    ("wine", WineDataset, 13, 3),
    ("seeds", SeedsDataset, 7, 3),
    ("breast_cancer", BreastCancerDataset, 30, 2),
]

LOSSES = ["ce", "bce", "focal"]


# -------------------------------------------------
# Test
# -------------------------------------------------
@pytest.mark.parametrize("ds_name, ds_cls, input_size, n_classes", DATASETS)
@pytest.mark.parametrize("loss_name", LOSSES)
def test_classification_train_one_epoch(
    ds_name, ds_cls, input_size, n_classes, loss_name
):
    """
    Runs 1 optimization epoch using train_genome_objective
    and checks that:
        - no crash occurs
        - fitness dict is populated
        - metrics are finite
    """

    train_ds = ds_cls(split="train")
    test_ds = ds_cls(split="test")

    out_qubits = int(math.ceil(math.log(n_classes, 2))) if n_classes > 1 else 1

    genome = make_dummy_genome(
        genome_number=0,
        input_qubits=min(input_size, 15),
        out_qubits=out_qubits,
    )

    genome.generate_pennylane_circuit(return_probs=True, input_mode="angle")

    train_genome_objective(
        genome,
        dataset=[train_ds, test_ds],
        backend="pennylane",
        loss=loss_name,
        epochs=1,
        lr=1e-3,
        n_classes=n_classes,
        log_every=1,
        batch_size=3,
    )

    assert hasattr(genome, "fitness"), "Genome must have fitness after training."
    assert isinstance(genome.fitness, dict)

    # Expected keys
    for key in ["train_loss", "test_loss"]:
        assert key in genome.fitness
        assert math.isfinite(float(genome.fitness[key]))

    # Accuracy only present for CE-like losses
    if "train_acc" in genome.fitness:
        assert 0.0 <= genome.fitness["train_acc"] <= 1.0

    if "test_acc" in genome.fitness:
        assert 0.0 <= genome.fitness["test_acc"] <= 1.0
