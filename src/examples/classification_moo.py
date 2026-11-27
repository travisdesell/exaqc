"""Run multi-objective EXAQC classification experiments.

This entry point runs the same search as ``src.examples.classification`` --
same datasets, encoders, decoders, trainer flags and archive -- but replaces the
single-objective population strategies with NSGA-II or NSGA-III, so the search
keeps a Pareto front over two or more objectives instead of a single ranking.

Every genome's fitness holds the standard classification keys (``loss`` and
``target_metric``, the mean class accuracy) plus the structural keys
``n_enabled_gates`` (the number of enabled quantum gates) and ``n_parameters``
(the number of parameters on those gates). Under device-card noise
(``--noise_device_card``) it also holds the card's simulation-free hardware
keys ``p_fail`` (the chance that any gate or wait errs), ``total_time_ns`` (the
circuit's duration) and ``n_entanglers`` (its ``cx`` count, routing included).
Under any noise it holds ``noise_gap`` and ``noise_gap_accuracy``: how much
worse the trained genome's validation loss and accuracy are with the noise
than without it. Any of these can be chosen as an objective with ``--objectives NAME:SIGN``,
where a sign of ``1`` minimizes and ``-1`` maximizes. The default minimizes
loss and the number of enabled gates.

Example::

    python3 -m src.examples.classification_moo \\
        --dataset iris \\
        --input_qubits 4 \\
        --output_qubits 2 \\
        --number_genomes 200 \\
        --objectives loss:1 n_enabled_gates:1 \\
        --out_dir results/iris_nsga2 \\
        nsga2 --max_population_size 30
"""

from __future__ import annotations

import argparse

from src.circuits.circuit import CircuitGenome
from src.evolution.moo.base_population import MultiObjectivePopulationBase
from src.evolution.moo.nsga2 import NSGA2
from src.evolution.moo.nsga3 import NSGA3
from src.evolution.moo.objective_spec import ObjectiveSpec
from src.examples import classification
from src.noise.device_card import HARDWARE_OBJECTIVES
from src.noise.pennylane_noise_model import PennyLaneNoiseModel
from src.objectives.classification_objective import ClassificationObjective
from src.utils.restart import RestartState

#: Fitness keys computed from a genome's structure rather than from training.
STRUCTURAL_OBJECTIVES: tuple[str, ...] = ("n_enabled_gates", "n_parameters")

#: How much the noise costs a trained genome: validation loss with noise minus
#: without, and validation accuracy without noise minus with (both lower is
#: better). Recorded only for genomes run with noise.
NOISE_GAP_OBJECTIVES: tuple[str, ...] = ("noise_gap", "noise_gap_accuracy")

#: Every fitness key a multi-objective classification genome records, and so
#: every name ``--objectives`` accepts. The hardware objectives
#: (:data:`~src.noise.device_card.HARDWARE_OBJECTIVES`) are recorded only
#: under ``--noise_device_card``, the noise gap only under any noise.
AVAILABLE_OBJECTIVES: tuple[str, ...] = (
    "loss",
    "target_metric",
    *STRUCTURAL_OBJECTIVES,
    *HARDWARE_OBJECTIVES,
    *NOISE_GAP_OBJECTIVES,
)

#: The objectives used when ``--objectives`` is not given.
DEFAULT_OBJECTIVES: list[str] = ["loss:1", "n_enabled_gates:1"]

#: The multi-objective population sub-commands and the class each builds.
POPULATION_CLASSES: dict[str, type[MultiObjectivePopulationBase]] = {
    "nsga2": NSGA2,
    "nsga3": NSGA3,
}


def structural_fitness(genome: CircuitGenome) -> dict[str, float]:
    """Measures the size of a genome's quantum circuit.

    Only enabled gates count: a disabled gate is carried by the genome but is
    not part of the circuit it builds. Encoder and decoder weights are not
    counted.

    Args:
        genome: The genome to measure.

    Returns:
        ``n_enabled_gates``, the number of enabled gates, and ``n_parameters``,
        the number of parameters those gates hold.
    """

    enabled = [gate for gate in genome.gates if gate.enabled]
    return {
        "n_enabled_gates": float(len(enabled)),
        "n_parameters": float(sum(len(gate.parameters) for gate in enabled)),
    }


def hardware_fitness(genome: CircuitGenome) -> dict[str, float]:
    """The device card's predicted cost of running a genome, when it has one.

    Computed from the genome's translation into card gates, without any
    simulation (see :func:`src.noise.device_card.hardware_metrics`).

    Args:
        genome: The genome, whose ``noise_*`` hyperparameters say whether it
            runs under device-card noise.

    Returns:
        ``p_fail``, ``total_time_ns`` and ``n_entanglers`` under the genome's
        card, or nothing when it has no device-card noise.
    """

    model = PennyLaneNoiseModel.from_hyperparameters(genome.hyperparameters)
    if model.noise_type != "device_card":
        return {}
    return model.hardware_metrics(genome)


class MultiObjectiveClassificationObjective(ClassificationObjective):
    """Classification objective that also records the circuit's size and cost.

    Trains and scores a genome exactly as :class:`ClassificationObjective`
    does, then adds :func:`structural_fitness`'s keys, and under device-card
    noise :func:`hardware_fitness`'s, so the population can trade accuracy off
    against circuit size, predicted failure rate and duration.
    """

    def __call__(self, genome: CircuitGenome) -> None:
        """Trains a genome and assigns its classification and structural fitness.

        Args:
            genome: Genome to train and evaluate.

        Returns:
            None. Sets ``genome.fitness`` (and
            ``genome.metadata["initial_fitness"]``, when the trainer recorded
            the inherited weights' metrics) to the classification fitness plus
            ``n_enabled_gates`` and ``n_parameters``, under device-card
            noise ``p_fail``, ``total_time_ns`` and ``n_entanglers``, and
            under any noise ``noise_gap`` and ``noise_gap_accuracy`` (see
            :meth:`noise_gap_fitness`).
        """

        super().__call__(genome)

        structure = {
            **structural_fitness(genome),
            **hardware_fitness(genome),
            **self.noise_gap_fitness(genome),
        }
        genome.fitness.update(structure)
        if genome.metadata.get("initial_fitness") is not None:
            # the inherited weights were only scored with noise, so their
            # noise gap is unknown
            genome.metadata["initial_fitness"].update(
                {k: v for k, v in structure.items() if k not in NOISE_GAP_OBJECTIVES}
            )

    def noise_gap_fitness(self, genome: CircuitGenome) -> dict[str, float]:
        """Measures how much a trained noisy genome's noise actually costs it.

        The genome's best weights are rebuilt without any noise and scored on
        the validation set once more; the gap is taken against the validation
        metrics the noisy training recorded for the same weights. One extra
        noiseless validation pass per genome.

        Args:
            genome: A genome this objective has just trained.

        Returns:
            ``noise_gap`` (validation loss with noise minus without; lower is
            better, near 0 when the noise barely matters) and
            ``noise_gap_accuracy`` (validation mean class accuracy without
            noise minus with; lower is better), or nothing when the genome runs
            without noise.
        """

        if not PennyLaneNoiseModel.from_hyperparameters(
            genome.hyperparameters
        ).is_noisy():
            return {}
        trainer = self.trainer
        # the same genome, best weights included, built without its noise
        clean = CircuitGenome.from_dict(genome.to_dict())
        clean.initialize_model(noise_model=PennyLaneNoiseModel())
        clean.hybrid_model.to(trainer.device)
        noiseless = trainer.get_metrics(
            clean,
            dataloader=trainer.validation_dataloader,
            loss_function=trainer.validation_loss_function,
        )
        noisy = genome.metadata["best_validation_metrics"]
        return {
            "noise_gap": float(noisy["loss"]) - float(noiseless["loss"]),
            "noise_gap_accuracy": float(noiseless["mean_class_accuracy"]["mean"])
            - float(noisy["mean_class_accuracy"]["mean"]),
        }


def add_population_strategies(parser: argparse.ArgumentParser) -> None:
    """Adds the multi-objective population sub-commands to a parser.

    Args:
        parser: The parser to add the ``population_strategy`` sub-commands to.

    Returns:
        None. Mutates ``parser`` by adding a required ``population_strategy``
        sub-command with ``nsga2`` and ``nsga3`` choices, each carrying that
        strategy's own arguments.
    """

    populations = parser.add_subparsers(
        dest="population_strategy",
        required=True,
        help="Specify the multi-objective algorithm that manages the population.",
    )
    NSGA2.initialize_parser(
        populations.add_parser("nsga2", help="Use a steady-state NSGA-II population.")
    )
    NSGA3.initialize_parser(
        populations.add_parser("nsga3", help="Use a steady-state NSGA-III population.")
    )


def build_parser() -> argparse.ArgumentParser:
    """Creates the multi-objective classification command-line parser.

    Returns:
        The classification parser (see
        :func:`src.examples.classification.build_parser`) with the
        multi-objective population sub-commands and ``--objectives``.
    """

    parser = classification.build_parser(
        add_population_strategies=add_population_strategies
    )
    parser.description = (
        "Run multi-objective EXAQC classification on UCI and image datasets with "
        "NSGA-II or NSGA-III."
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=DEFAULT_OBJECTIVES,
        metavar="NAME:SIGN",
        help=(
            "Objectives to optimize, as NAME:SIGN with sign 1 to minimize or -1 to "
            f"maximize (a bare NAME minimizes). Names: {', '.join(AVAILABLE_OBJECTIVES)} "
            f"({', '.join(HARDWARE_OBJECTIVES)} need --noise_device_card; "
            f"{', '.join(NOISE_GAP_OBJECTIVES)} need any noise). "
            "At least two are required. Like -ms and -ps, it takes every value up to "
            "the next flag, so do not put it directly before the nsga2/nsga3 sub-command."
        ),
    )
    return parser


def parse_objectives(specifications: list[str]) -> list[ObjectiveSpec]:
    """Turns ``--objectives`` values into objective specifications.

    Args:
        specifications: Each objective as ``NAME:SIGN`` or a bare ``NAME``.

    Returns:
        One :class:`ObjectiveSpec` per specification, in the order given.

    Raises:
        ValueError: If a specification is malformed, names an unknown or
            repeated objective, has a sign other than ``1`` or ``-1``, or fewer
            than two objectives are given.
    """

    objectives: list[ObjectiveSpec] = []
    for specification in specifications:
        name, _, sign_text = specification.partition(":")
        if not name:
            raise ValueError(f"invalid objective {specification!r}: it has no name.")
        if name not in AVAILABLE_OBJECTIVES:
            raise ValueError(
                f"unknown objective {name!r}; choose from {', '.join(AVAILABLE_OBJECTIVES)}."
            )
        if any(objective.name == name for objective in objectives):
            raise ValueError(f"objective {name!r} is given more than once.")
        try:
            sign = float(sign_text) if sign_text else 1.0
        except ValueError as error:
            raise ValueError(
                f"invalid sign {sign_text!r} for objective {name!r}; use 1 or -1."
            ) from error
        if sign not in (1.0, -1.0):
            raise ValueError(
                f"invalid sign {sign_text!r} for objective {name!r}; use 1 or -1."
            )
        objectives.append(ObjectiveSpec(name=name, sign=sign))

    if len(objectives) < 2:
        raise ValueError("multi-objective optimization needs at least two objectives.")

    return objectives


def build_population(
    args: argparse.Namespace, restart_state: RestartState | None
) -> MultiObjectivePopulationBase:
    """Builds the run's NSGA-II or NSGA-III population.

    Args:
        args: Parsed (possibly restarted) arguments carrying the population
            sub-command, its flags, ``--objectives`` and ``--seed``.
        restart_state: The stopped run being continued, or ``None`` for a new
            run.

    Returns:
        The population, restored from ``restart_state`` when given.
    """

    population_class = POPULATION_CLASSES[args.population_strategy]
    keywords: dict[str, int | None] = {}
    if population_class is NSGA3:
        keywords["reference_divisions"] = args.reference_divisions

    population = population_class(
        max_population_size=args.max_population_size,
        objectives=parse_objectives(args.objectives),
        tournament_size=args.tournament_size,
        seed=args.seed,
        **keywords,
    )

    if restart_state is not None:
        population.restore(restart_state)

    return population


def main() -> None:
    """Runs a multi-objective classification experiment.

    Returns:
        None. Runs the search to completion.
    """

    parser = build_parser()
    args = parser.parse_args()

    try:
        objectives = parse_objectives(args.objectives)
    except ValueError as error:
        parser.error(str(error))

    needs_card = [o.name for o in objectives if o.name in HARDWARE_OBJECTIVES]
    if needs_card and args.noise_device_card is None:
        parser.error(
            f"{', '.join(needs_card)} come from a device card; give --noise_device_card."
        )
    needs_noise = [o.name for o in objectives if o.name in NOISE_GAP_OBJECTIVES]
    noiseless = (
        args.noise_device_card is None
        and args.noise_calibration is None
        and args.noise_type == "none"
    )
    if needs_noise and noiseless:
        parser.error(
            f"{', '.join(needs_noise)} compare a noisy genome with its noiseless self; "
            "give a noise setting (e.g. --noise_device_card or --noise_type)."
        )

    classification.run(
        parser,
        args,
        build_population=build_population,
        objective_class=MultiObjectiveClassificationObjective,
    )


if __name__ == "__main__":
    main()
