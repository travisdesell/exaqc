"""Continue training a single evolved genome loaded from its JSON.

The evolutionary search writes every genome it evaluates as JSON (see
``CircuitGenome.to_dict``). This entry point loads one of those files back and
trains it further -- useful for taking the best genome of a search and giving it
a longer, more careful training run than the search itself could afford.

A genome file is self-describing. Besides its gates, qubit layout, target
framework and classical stages, EXAQC stamps every genome it generates with the
``task`` it was evolved for (``classification``, ``teacher`` or
``reinforcement_learning``) and the ``task_target`` it was run against (the
dataset, teacher circuit or environment name). Refining therefore needs nothing
but the file::

    python3 -m src.examples.refine_genome --genome best_genome.json

The hyperparameters recorded in the file are reused unchanged, so a refinement
run reproduces the original training setup unless something is explicitly
overridden with ``--set``, e.g. ``--set epochs=200``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any

from loguru import logger

import torch

from src.circuits.circuit import CircuitGenome
from src.datasets.teacher_loaders import get_teacher_dataloaders
from src.evolution.objective import Objective
from src.examples.classification import ClassificationObjective, load_data
from src.examples.reinforcement_learning import (
    ReinforcementLearningObjective,
    build_trainer,
    make_environment,
)
from src.examples.teacher import TeacherObjective
from src.metrics.mean_class_accuracy import MeanClassAccuracy

#: Defaults for the dataset-loading options that ``load_data`` reads but a
#: genome file does not record. Only the batch size comes from the genome.
_CLASSIFICATION_DATA_DEFAULTS: dict[str, Any] = {
    "data_dir": "data",
    "validation_fraction": 0.1,
    "training_samples": None,
    "validation_samples": None,
    "download_dataset": True,
    "num_workers": 0,
    "pin_memory": False,
    "normalization": "minmax",
    "seed": 0,
}

#: Number of teacher-labelled samples regenerated per split when refining a
#: teacher genome. The teacher is deterministic, so this only sets how much
#: fresh data the refinement sees.
_TEACHER_SAMPLES = 64


def load_genome(path: str) -> CircuitGenome:
    """Loads a genome from a JSON file written by the search.

    Args:
        path: Path to the genome's JSON file.

    Returns:
        The deserialized :class:`CircuitGenome`.

    Raises:
        ValueError: If the file does not contain a serialized genome, or does
            not record which task it was evolved for.
    """

    with open(path, "r", encoding="utf-8") as genome_file:
        serialized = json.load(genome_file)

    if not isinstance(serialized, dict) or "gates" not in serialized:
        raise ValueError(
            f"{path!r} does not look like a genome file (no 'gates' entry)."
        )

    genome = CircuitGenome.from_dict(serialized)

    if not genome.task or not genome.task_target:
        raise ValueError(
            f"{path!r} does not record the task it was evolved for "
            f"(task={genome.task!r}, task_target={genome.task_target!r}). It was "
            "probably written before those were recorded; re-run the search to "
            "produce a refinable genome."
        )

    return genome


def apply_overrides(genome: CircuitGenome, overrides: list[str]) -> None:
    """Applies ``key=value`` hyperparameter overrides to a genome, in place.

    Each value is coerced to the type of the value already stored under that
    key, so ``--set epochs=200`` stays an ``int`` and ``--set learning_rate=1e-3``
    stays a ``float``. A key the genome does not already carry is rejected
    rather than silently added, since that is almost always a typo.

    Args:
        genome: The genome whose hyperparameters are updated.
        overrides: Raw ``key=value`` strings from the command line.

    Returns:
        None. Mutates ``genome.hyperparameters``.

    Raises:
        ValueError: If an override is malformed, names an unknown
            hyperparameter, or cannot be coerced to the existing type.
    """

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override {override!r} is not in key=value form (e.g. epochs=200)."
            )

        key, _, raw_value = override.partition("=")
        key = key.strip()

        if key not in genome.hyperparameters:
            raise ValueError(
                f"Genome has no hyperparameter {key!r}; it carries: "
                f"{sorted(genome.hyperparameters)}"
            )

        current = genome.hyperparameters[key]

        try:
            if isinstance(current, bool):
                value: Any = raw_value.strip().lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                value = int(raw_value)
            elif isinstance(current, float):
                value = float(raw_value)
            else:
                value = raw_value
        except ValueError as error:
            raise ValueError(
                f"Could not read {raw_value!r} as a {type(current).__name__} for "
                f"hyperparameter {key!r}."
            ) from error

        logger.info("overriding hyperparameter {}: {} -> {}", key, current, value)
        genome.hyperparameters[key] = value


def build_classification_objective(
    genome: CircuitGenome,
    device: str | None,
) -> Objective:
    """Rebuilds the classification objective a genome was evolved under.

    Args:
        genome: The genome being refined. Its ``task_target`` names the dataset
            and its ``batch_size`` hyperparameter sizes the loaders.
        device: PyTorch device to train on.

    Returns:
        A :class:`ClassificationObjective` over the genome's own dataset.

    Raises:
        ValueError: If the genome's classical stages are not sized for the
            dataset it names, which means the file is inconsistent.
    """

    data_arguments = SimpleNamespace(
        dataset=genome.task_target,
        batch_size=int(genome.hyperparameters.get("batch_size", 1)),
        validation_batch_size=None,
        **_CLASSIFICATION_DATA_DEFAULTS,
    )
    training_loader, validation_loader = load_data(data_arguments)

    # The genome's classical stages were sized for this dataset when it was
    # evolved; a mismatch means the file is inconsistent, and would otherwise
    # surface as a shape error deep inside the first forward pass.
    if (
        genome.encoder is not None
        and genome.encoder.n_inputs != training_loader.n_features
    ):
        raise ValueError(
            f"Genome's encoder expects {genome.encoder.n_inputs} input features "
            f"but dataset {genome.task_target!r} has {training_loader.n_features}."
        )

    return ClassificationObjective(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        training_loss_function=torch.nn.CrossEntropyLoss(
            weight=training_loader.label_weights,
            reduction="mean",
        ),
        validation_loss_function=torch.nn.CrossEntropyLoss(
            weight=validation_loader.label_weights,
            reduction="mean",
        ),
        metrics={"mean_class_accuracy": MeanClassAccuracy(training_loader.n_labels)},
        device=device,
    )


def build_teacher_objective(
    genome: CircuitGenome,
    device: str | None,
) -> Objective:
    """Rebuilds the teacher-imitation objective a genome was evolved under.

    Everything needed is on the genome: ``task_target`` names the teacher
    circuit, and the wire layout, target framework and quantum modes are read
    back off the genome itself.

    Args:
        genome: The genome being refined.
        device: PyTorch device to train on.

    Returns:
        A :class:`TeacherObjective` over the regenerated teacher dataset.
    """

    input_wires = [index for _, index in genome.input_qubits]
    output_wires = [index for _, index in genome.output_qubits]

    training_loader, validation_loader = get_teacher_dataloaders(
        teacher_name=genome.task_target,
        input_wires=input_wires,
        output_wires=output_wires,
        target=genome.target,
        quantum_input_mode=genome.hyperparameters["quantum_input_mode"],
        quantum_output_mode=genome.hyperparameters["quantum_output_mode"],
        n_training_samples=_TEACHER_SAMPLES,
        n_validation_samples=_TEACHER_SAMPLES,
        batch_size=int(genome.hyperparameters.get("batch_size", 1)),
    )

    # 'mse' is the only measure valid for a non-distribution readout.
    loss_name = (
        "fidelity"
        if genome.hyperparameters["quantum_output_mode"] == "probs"
        else "mse"
    )

    return TeacherObjective(
        training_dataloader=training_loader,
        validation_dataloader=validation_loader,
        loss_name=loss_name,
        device=device,
    )


def build_reinforcement_learning_objective(
    genome: CircuitGenome,
    device: str | None,
) -> Objective:
    """Rebuilds the reinforcement-learning objective a genome was evolved under.

    ``task_target`` names the environment and the ``algo`` hyperparameter names
    the algorithm, so both come off the genome.

    Args:
        genome: The genome being refined.
        device: PyTorch device to train on (unused; the RL trainers read their
            configuration from the genome's hyperparameters).

    Returns:
        A :class:`ReinforcementLearningObjective` over the genome's environment.

    Raises:
        ValueError: If the genome does not record an algorithm, or the algorithm
            cannot drive the environment's action space.
    """

    algorithm = genome.hyperparameters.get("algo")
    if not algorithm:
        raise ValueError(
            "Genome does not record which algorithm it was trained with (no "
            "'algo' hyperparameter), so it cannot be refined."
        )

    environment = make_environment(genome.task_target)
    trainer = build_trainer(algorithm)

    if environment.continuous and not trainer.supports_continuous:
        raise ValueError(
            f"Algorithm {algorithm!r} cannot drive the continuous environment "
            f"{genome.task_target!r}."
        )

    return ReinforcementLearningObjective(environment=environment, trainer=trainer)


#: How to rebuild each task's objective, keyed by the genome's ``task``.
OBJECTIVE_BUILDERS = {
    "classification": build_classification_objective,
    "teacher": build_teacher_objective,
    "reinforcement_learning": build_reinforcement_learning_objective,
}


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for refining a saved genome.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Continue training a single evolved genome loaded from its JSON "
            "file. The genome records the task and target it was evolved for, "
            "so nothing else needs to be specified."
        )
    )

    parser.add_argument(
        "--genome",
        type=str,
        required=True,
        help="Path to the genome JSON file written by the evolutionary search.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Directory to write the refined genome (and its diagram) into.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one of the genome's own hyperparameters, e.g. "
            "--set epochs=200. May be given repeatedly. Without this the "
            "hyperparameters stored in the genome file are used unchanged."
        ),
    )
    parser.add_argument(
        "--save_circuit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write the refined genome's architecture diagram.",
    )
    parser.add_argument(
        "--save_training_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write a training-history plot beside the refined diagram.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to train on, e.g. 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    return parser


def main() -> None:
    """Loads a genome, trains it further, and writes the refined result."""

    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    logger.add(os.path.join(args.out_dir, "refine.log"))

    try:
        genome = load_genome(args.genome)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if genome.task not in OBJECTIVE_BUILDERS:
        parser.error(
            f"Genome records an unknown task {genome.task!r}; expected one of "
            f"{sorted(OBJECTIVE_BUILDERS)}."
        )

    logger.info(
        "Refining {} genome {} on {!r} ({} target, {} gates) from {}",
        genome.task,
        genome.genome_number,
        genome.task_target,
        genome.target,
        len(genome.gates),
        args.genome,
    )
    logger.info("starting fitness: {}", genome.fitness)
    logger.info("hyperparameters from the genome file: {}", genome.hyperparameters)

    try:
        apply_overrides(genome, args.overrides)
        objective = OBJECTIVE_BUILDERS[genome.task](genome, args.device)
    except ValueError as error:
        parser.error(str(error))

    starting_fitness = dict(genome.fitness) if genome.fitness else None

    objective(genome)

    logger.info("refined fitness: {}", genome.fitness)
    if starting_fitness:
        for key, refined in genome.fitness.items():
            if key in starting_fitness:
                logger.info(
                    "  {}: {:.6f} -> {:.6f}", key, starting_fitness[key], refined
                )

    refined_path = os.path.join(
        args.out_dir, f"refined_genome_{genome.genome_number}.json"
    )
    with open(refined_path, "w", encoding="utf-8") as genome_file:
        json.dump(genome.to_dict(), genome_file, indent=2)
    logger.info("wrote refined genome to {}", refined_path)

    if args.save_circuit:
        genome.save_circuit(
            insert_type="refined",
            out_dir=args.out_dir,
            save_training_plot=args.save_training_plot,
        )


if __name__ == "__main__":
    main()
