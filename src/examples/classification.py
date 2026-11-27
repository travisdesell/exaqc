from __future__ import annotations

import argparse
import os
import sys

import json
from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.circuits.decoder import initialize_decoder
from src.circuits.encoder import initialize_encoder
from src.circuits.gate_specifications import GateSpecifications
from src.datasets.classification_loaders import CLASSIFICATION_DATASETS
from src.evolution.exaqc import EXAQC
from src.evolution.master_worker import run_evolution
from src.evolution.population_strategy import PopulationStrategy
from src.objectives.classification_objective import (
    ClassificationObjective,
    compare,
    load_data,
)
from src.utils import restart
from src.utils.restart import RestartState
from src.metrics.mean_class_accuracy import MeanClassAccuracy
from src.metrics.metric import Metric
from src.trainer.supervised_trainer import SupervisedTrainer
from src.utils.genome_archive import GenomeArchive
from src.noise.pennylane_noise_model import (
    PennyLaneNoiseModel,
    noise_hyperparameters_from_args,
)


def build_parser(
    add_population_strategies: Callable[
        [argparse.ArgumentParser], None
    ] = PopulationStrategy.initialize_parser,
) -> argparse.ArgumentParser:
    """Creates the classification command-line parser.

    Args:
        add_population_strategies: Adds the population-strategy sub-commands to
            the parser. Defaults to the shared ``steady_state`` / ``islands``
            sub-commands; ``src.examples.classification_moo`` passes its
            multi-objective ones instead.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=("Run batched EXAQC classification on UCI and image datasets.")
    )
    parser.add_argument(
        "--dataset",
        choices=CLASSIFICATION_DATASETS,
        required=True,
        help="Dataset to evolve classifiers on (a UCI tabular or image dataset).",
    )

    # The evolutionary search's own flags -- mutation/parent strategies,
    # crossover rates and the genome budget -- are owned by EXAQC so every entry
    # point stays in sync.
    EXAQC.initialize_parser(parser)

    # Where and how the run's outputs are written (--out_dir,
    # --shared_file_system) is owned by GenomeArchive.
    GenomeArchive.initialize_parser(parser)

    # The choice of population strategy (and each strategy's own flags) is owned
    # by PopulationStrategy.
    add_population_strategies(parser)

    # The supervised-training flags (epochs, learning rate, weight decay,
    # improvement cutoff, batch size) are owned by SupervisedTrainer so the
    # classification and teacher entry points stay in sync.
    SupervisedTrainer.initialize_parser(parser)

    # The backend (--target) and optional gate-set restriction (--use_only) are
    # owned by GateSpecifications.
    GateSpecifications.initialize_parser(parser)

    # The circuit-genome flags (qubit counts, quantum input/output modes,
    # quantum dropout, encoder/decoder) are owned by CircuitGenome so every
    # entry point stays in sync.
    CircuitGenome.initialize_parser(parser)

    # The simulated-noise flags (--noise_type, --noise_p_1q, ...,
    # --noise_calibration, --noise_qubits) are owned by the noise model; all
    # default to no noise.
    PennyLaneNoiseModel.initialize_parser(parser)

    parser.add_argument(
        "--encoder_config",
        type=str,
        default="configs",
        help="Path to a JSON file of encoder configuration options (e.g. CNN activation, batch norm).",
    )

    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=None,
        help="Batch size for validation/testing; defaults to --batch_size when unset.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory where datasets are stored (and downloaded to).",
    )

    parser.add_argument(
        "--download_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the dataset if it is not already present in --data_dir.",
    )

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction held out for validation: of an image dataset's training split, or "
            "of a UCI dataset's rows when --test_fraction is given (a UCI dataset without "
            "a test set always splits 80/20)."
        ),
    )

    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.0,
        help=(
            "UCI datasets only: fraction of the rows set aside first as a test set the "
            "search never sees (e.g. 0.1 with --validation_fraction 0.1 gives an "
            "80/10/10 split). Each genome's best weights are scored on it after training "
            "and recorded as test_loss / test_accuracy; selection never uses them. 0 for "
            "no test set."
        ),
    )

    parser.add_argument(
        "--training_samples",
        type=int,
        default=None,
        help="Cap the training set to this many samples (use all when unset).",
    )

    parser.add_argument(
        "--validation_samples",
        type=int,
        default=None,
        help="Cap the validation set to this many samples (use all when unset).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "PyTorch device to use for training, e.g. "
            "'cpu', 'cuda', or 'cuda:0'. "
            "Defaults to CUDA when available."
        ),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes used by the PyTorch DataLoaders.",
    )

    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pinned (page-locked) host memory in the DataLoaders for faster GPU transfers.",
    )

    parser.add_argument(
        "--cnn_channels",
        type=int,
        nargs=2,
        default=[16, 32],
        help="Output channel counts for the two CNN encoder convolution layers (image datasets).",
    )

    parser.add_argument(
        "--cnn_pooled_size",
        type=int,
        default=4,
        help="Spatial size (height = width) the CNN encoder pools its feature maps down to.",
    )

    parser.add_argument(
        "--cnn_dropout",
        type=float,
        default=0.0,
        help="Dropout rate applied inside the CNN encoder.",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "zscore", "minmax"],
        default="minmax",
        help="Feature normalization applied to tabular datasets before encoding.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible dataset splits and evolution.",
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    return parser


def load_encoder_config(
    path: str | None,
) -> dict:
    """Loads an encoder configuration file.

    Args:
        path: JSON configuration path or ``None``.

    Returns:
        Parsed encoder configuration.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    if path is None:
        return {}

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Encoder configuration must contain a JSON object.")

    return config


def default_population(
    args: argparse.Namespace, restart_state: RestartState | None
) -> PopulationStrategy:
    """Builds the shared single-objective population strategy for a run.

    Args:
        args: Parsed arguments carrying the population sub-command and flags.
        restart_state: The stopped run being continued, or ``None`` for a new
            run.

    Returns:
        The population strategy, restored from ``restart_state`` when given.
    """

    if restart_state is None:
        return PopulationStrategy.from_args(args, compare)
    return restart.restored_strategy(restart_state, args, compare)


def main() -> None:
    """Runs a classification experiment."""
    parser = build_parser()
    run(parser, parser.parse_args())


def run(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    build_population: Callable[
        [argparse.Namespace, RestartState | None], PopulationStrategy
    ] = default_population,
    objective_class: type[ClassificationObjective] = ClassificationObjective,
) -> None:
    """Runs a classification search from parsed arguments.

    Shared by this entry point and ``src.examples.classification_moo``, which
    differ only in the population strategy and the objective that scores each
    genome.

    Args:
        parser: The parser ``args`` came from, used to report invalid
            arguments.
        args: The parsed command-line arguments.
        build_population: Builds the run's population strategy from the
            (possibly restarted) arguments and the restart state, if any.
        objective_class: The objective trained and scored genomes with; its
            constructor takes the same arguments as
            :class:`ClassificationObjective`.

    Returns:
        None. Runs the search to completion (see :func:`run_evolution`).
    """

    # Decided before anything is built, and on every rank: a restarted run takes
    # its configuration from the run it continues, so the objective a worker
    # evaluates with matches the search the master restores.
    args, restart_state, run_for = restart.prepare(args, parser.error)

    # The output directory is created by GenomeArchive.from_args (on the serial
    # run or MPI master); loguru creates the run.log parent directory as needed
    # when the file sink is added.
    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)
    if args.save_run_log:
        logger.add(os.path.join(args.out_dir, "run.log"), level=args.logging_level)

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("Using PyTorch device: {}", device)

    # Checked on every rank before any work, so bad noise flags stop the run at once.
    try:
        noise_hyperparameters = noise_hyperparameters_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    logger.info("Noise settings: {}", noise_hyperparameters)

    try:
        training_loader, validation_loader = load_data(args)
    except ValueError as error:
        parser.error(str(error))

    # if training_loader.is_image and args.encoding != "cnn":
    #     parser.error("Image datasets require --encoding cnn in this implementation.")
    if not training_loader.is_image and args.encoding == "cnn":
        parser.error("CNN encoding is only valid for image datasets.")

    metrics: dict[str, Metric] = {
        "mean_class_accuracy": MeanClassAccuracy(training_loader.n_labels)
    }
    objective = objective_class(
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
        metrics=metrics,
        device=args.device,
        # held out from the search: scored and recorded only (--test_fraction)
        test_dataloader=getattr(training_loader, "test_loader", None),
    )

    def build_exaqc() -> EXAQC:
        """Builds the EXAQC search machinery for this run.

        This constructs the encoder/decoder, population strategy, gate set and
        the :class:`EXAQC` object. It is only invoked on the serial run and the
        MPI master (via :func:`run_evolution`), so worker ranks never build any
        of it.

        Returns:
            The fully-configured :class:`EXAQC` search wrapping ``objective``.
        """

        if args.encoding == "identity":
            # The identity encoder passes its input straight through, so its
            # output size must equal its input size (the raw feature count) --
            # it does not resize or clip to the qubit count.
            n_encoder_outputs = training_loader.n_features
        else:
            n_encoder_outputs = args.input_qubits
            if args.quantum_input_mode == "u3":
                n_encoder_outputs *= 3

        n_decoder_inputs = args.output_qubits
        if args.quantum_output_mode == "probs":
            n_decoder_inputs = 2**args.output_qubits

        encoder_config = None
        if training_loader.is_image and args.encoding == "cnn":
            channels, height, width = training_loader.input_shape

            encoder_config = load_encoder_config(args.encoder_config)

            encoder_config.update(
                {
                    "input_channels": channels,
                    "input_height": height,
                    "input_width": width,
                    "hidden_channels": args.cnn_channels,
                    "pooled_size": args.cnn_pooled_size,
                    "dropout": args.cnn_dropout,
                }
            )

        initial_encoder = initialize_encoder(
            target=args.target,
            encoding_str=args.encoding,
            n_inputs=training_loader.n_features,
            n_outputs=n_encoder_outputs,
            config=encoder_config,
            quantum_input_mode=args.quantum_input_mode,
            n_input_qubits=args.input_qubits,
        )
        initial_decoder = initialize_decoder(
            target=args.target,
            decoding_str=args.decoding,
            n_inputs=n_decoder_inputs,
            n_outputs=training_loader.n_labels,
        )

        hyperparameters = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "improvement_cutoff": args.improvement_cutoff,
            "batch_size": args.batch_size,
            "quantum_input_mode": args.quantum_input_mode,
            "quantum_output_mode": args.quantum_output_mode,
            "quantum_dropout": args.quantum_dropout,
            "quantum_dropout_type": args.quantum_dropout_type,
            "quantum_dropout_rate": args.quantum_dropout_rate,
            # every genome carries the noise settings; they describe no noise
            # unless noise flags were given
            **noise_hyperparameters,
        }

        # The gate set and population strategy are built from `args` by their own
        # factories (which every entry point shares); only the task-specific
        # encoder/decoder, hyperparameters and register layout are computed here.
        population = build_population(args, restart_state)

        search = EXAQC(
            gate_specifications=GateSpecifications.from_args(args),
            population=population,
            archive=GenomeArchive.from_args(args, restarting=restart_state is not None),
            objective=objective,
            initial_encoder=initial_encoder,
            initial_decoder=initial_decoder,
            hyperparameters=hyperparameters,
            mutation_strategy=args.mutation_strategy,
            parent_strategy=args.parent_strategy,
            binary_crossover_rate=args.binary_crossover_rate,
            n_ary_crossover_rate=args.n_ary_crossover_rate,
            exponential_crossover_rate=args.exponential_crossover_rate,
            input_registers={"input": args.input_qubits},
            output_registers={"input": args.output_qubits},
            task="classification",
            task_target=args.dataset,
            restarting=restart_state is not None,
        )

        if restart_state is not None:
            restart.resume(search, restart_state, args)

        return search

    # Every rank builds the objective (workers evaluate genomes with it); only
    # the serial run and the MPI master build the EXAQC search, via build_exaqc.
    run_evolution(
        objective=objective,
        build_exaqc=build_exaqc,
        run_for=run_for,
        timing_report_every=args.timing_report_every,
    )


if __name__ == "__main__":
    main()
