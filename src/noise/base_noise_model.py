from __future__ import annotations

import argparse
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional, Self, get_args

NoiseType = Literal[
    "none",
    "depolarizing",
    "bit_flip",
    "phase_flip",
    "amplitude_damping",
    "phase_damping",
    "thermal_relaxation",
    "mixed",
    "ibm_backend",
]


def _optional_float(value: Any) -> float | None:
    """Converts a hyperparameter value to a float, keeping ``None`` as ``None``.

    Args:
        value: The stored value, such as ``0.001``, ``"0.001"`` or ``None``.

    Returns:
        The value as a float, or ``None`` if it was ``None``.

    Raises:
        ValueError: If the value is not ``None`` and cannot be read as a number.
    """
    return None if value is None else float(value)


def _as_bool(value: Any) -> bool:
    """Converts a hyperparameter value to a bool, reading text values correctly.

    Plain ``bool()`` treats any non-empty text as true, so ``bool("False")`` is
    ``True``; this reads ``"true"``/``"false"``, ``"yes"``/``"no"``, ``"on"``/``"off"``
    and ``"1"``/``"0"`` (any case) by their meaning.

    Args:
        value: The stored value, such as ``True``, ``0`` or ``"False"``.

    Returns:
        The value as a bool.

    Raises:
        ValueError: If the value is text that is not one of the recognized words.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "on", "1"):
            return True
        if text in ("false", "no", "off", "0"):
            return False
        raise ValueError(f"Cannot read {value!r} as true or false.")
    return bool(value)


@dataclass
class BaseNoiseModel(ABC):
    """Abstract backend-agnostic quantum noise model.

    This class defines a unified interface for representing quantum noise
    across multiple quantum software backends such as PennyLane and Qiskit.
    Subclasses are responsible for implementing backend-specific logic for
    constructing and applying noise processes.

    The noise model itself remains separate from the genome representation,
    allowing the same evolved circuit to be evaluated under multiple hardware
    or simulation noise conditions.

    Attributes:
        noise_type:
            Type of noise channel to apply.

        p:
            Default noise probability used when backend-specific one- or
            two-qubit probabilities are not provided.

        p_1q:
            Optional probability for single-qubit noise channels.

        p_2q:
            Optional probability for two-qubit noise channels.

        gamma:
            Damping parameter used for amplitude damping, phase damping,
            or thermal relaxation models.

        apply_after_input_encoding:
            Whether noise should be inserted immediately after classical
            data encoding operations.

        apply_after_gates:
            Whether noise should be inserted after each evolved quantum gate.

        apply_before_measurement:
            Whether noise should be inserted immediately before measurement.
    """

    noise_type: NoiseType = "none"

    p: float = 0.0
    p_1q: Optional[float] = None
    p_2q: Optional[float] = None
    gamma: float = 0.0

    t1: float = 50e-6
    t2: float = 70e-6
    gate_time_1q: float = 50e-9
    gate_time_2q: float = 300e-9
    excited_state_population: float = 0.0

    apply_after_input_encoding: bool = False
    apply_after_gates: bool = True
    apply_before_measurement: bool = False

    # Per-qubit values for the physical qubits named by ``--noise_qubits``,
    # taken from the ``--noise_calibration`` file (wire ``i`` is the ``i``-th
    # listed qubit); ``None`` means the single hand-set fields above apply to
    # every wire. Read by ``_values_for_wires``; saved on each genome as
    # ``noise_per_qubit``.
    per_qubit: dict[str, Any] | None = None

    # Where the per-qubit values came from: the chip's name, the
    # calibration's date and the file's path; None for hand-set noise.
    ibm_backend_name: str | None = None
    calibration_date: str | None = None
    calibration_file: str | None = None

    # How chip noise translates each genome into the chip's own gates
    # (src.noise.compiled_schedule): the chip's gates, read from the file,
    # and the translator's optimization level and seed. Saved on each genome,
    # so final scoring and the real-chip run translate the same way.
    chip_gates: list[str] | None = None
    transpile_level: int = 2
    transpile_seed: int = 7

    def is_noisy(self) -> bool:
        """Determine whether the model applies any non-trivial noise.

        Returns:
            True if the configured noise type is not ``"none"``,
            otherwise False.
        """
        return self.noise_type != "none"

    def probability_for_n_wires(self, n_wires: int) -> float:
        """Return the appropriate noise probability for an operation.

        Single-qubit operations preferentially use ``p_1q`` if specified,
        while multi-qubit operations preferentially use ``p_2q``.
        If no specialized probability is available, the default ``p`` value is used.

        Args:
            n_wires:
                Number of qubits/wires involved in the operation.

        Returns:
            Noise probability associated with the operation.
        """
        if n_wires <= 1:
            return self.p_1q if self.p_1q is not None else self.p

        return self.p_2q if self.p_2q is not None else self.p

    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier for the noise model.

        Subclasses should return identifiers such as:

        - ``"pennylane"``
        - ``"qiskit"``

        Returns:
            Name of the backend associated with the noise model.
        """
        raise NotImplementedError

    @classmethod
    def from_hyperparameters(cls, hp: dict[str, Any]) -> Self:
        """Construct a noise model from hyperparameter dictionaries.

        This helper maps genome or experiment hyperparameters into a
        backend-specific noise model configuration. Missing keys take the
        field defaults, so a genome with no ``noise_*`` keys gets no noise.

        Expected hyperparameter keys include:

        - ``noise_type``
        - ``noise_p``
        - ``noise_p_1q``
        - ``noise_p_2q``
        - ``noise_gamma``
        - ``noise_t1``, ``noise_t2``
        - ``noise_gate_time_1q``, ``noise_gate_time_2q``
        - ``noise_excited_state_population``
        - ``noise_after_encoding``
        - ``noise_after_gates``
        - ``noise_before_measurement``
        - ``noise_per_qubit``, ``noise_chip``, ``noise_calibration_date``,
          ``noise_calibration`` (chip noise only; absent in older genomes,
          which then get None)

        Args:
            hp:
                Dictionary containing experiment or genome hyperparameters.

        Returns:
            An instance of the calling class (e.g. ``PennyLaneNoiseModel``)
            configured from the provided hyperparameters.

        Raises:
            ValueError: If a numeric setting cannot be read as a number, or a
                switch is text that is not a recognized true/false word.
        """
        return cls(
            noise_type=hp.get("noise_type", "none"),
            p=float(hp.get("noise_p", 0.0)),
            p_1q=_optional_float(hp.get("noise_p_1q", None)),
            p_2q=_optional_float(hp.get("noise_p_2q", None)),
            gamma=float(hp.get("noise_gamma", 0.0)),
            t1=float(hp.get("noise_t1", 50e-6)),
            t2=float(hp.get("noise_t2", 70e-6)),
            gate_time_1q=float(hp.get("noise_gate_time_1q", 50e-9)),
            gate_time_2q=float(hp.get("noise_gate_time_2q", 300e-9)),
            excited_state_population=float(
                hp.get("noise_excited_state_population", 0.0)
            ),
            apply_after_input_encoding=_as_bool(hp.get("noise_after_encoding", False)),
            apply_after_gates=_as_bool(hp.get("noise_after_gates", True)),
            apply_before_measurement=_as_bool(
                hp.get("noise_before_measurement", False)
            ),
            per_qubit=copy.deepcopy(hp.get("noise_per_qubit", None)),
            ibm_backend_name=hp.get("noise_chip", None),
            calibration_date=hp.get("noise_calibration_date", None),
            calibration_file=hp.get("noise_calibration", None),
            chip_gates=copy.deepcopy(hp.get("noise_chip_gates", None)),
            transpile_level=int(hp.get("noise_transpile_level", 2)),
            transpile_seed=int(hp.get("noise_transpile_seed", 7)),
        )

    def to_hyperparameters(self) -> dict[str, Any]:
        """Returns this model's settings as genome ``noise_*`` hyperparameters.

        The inverse of :meth:`from_hyperparameters`: produces the keys a genome
        carries (``noise_type``, ``noise_p_1q``, ``noise_t1``, ...) so that a
        noise model built any way (from flags, or from an IBM chip) travels
        with each genome to MPI workers and into the run's archive. Passing
        the result back to :meth:`from_hyperparameters` rebuilds an equal
        model. For chip noise it also carries the per-qubit values
        (``noise_per_qubit``) and their source (``noise_chip``,
        ``noise_calibration_date``, ``noise_calibration`` for the file's
        path), so every genome records exactly which
        chip, day, qubits and values it was scored with, and the circuit
        builder, which rebuilds the model from these keys, gets them back.

        Returns:
            The ``noise_*`` hyperparameter dict describing this model; every
            value is a plain ``str``, ``float``, ``bool``, ``None``, or (for
            ``noise_per_qubit``) a dict of lists and numbers, so it
            serializes to JSON.
        """
        return {
            "noise_type": self.noise_type,
            "noise_p": float(self.p),
            "noise_p_1q": None if self.p_1q is None else float(self.p_1q),
            "noise_p_2q": None if self.p_2q is None else float(self.p_2q),
            "noise_gamma": float(self.gamma),
            "noise_t1": float(self.t1),
            "noise_t2": float(self.t2),
            "noise_gate_time_1q": float(self.gate_time_1q),
            "noise_gate_time_2q": float(self.gate_time_2q),
            "noise_excited_state_population": float(self.excited_state_population),
            "noise_after_encoding": bool(self.apply_after_input_encoding),
            "noise_after_gates": bool(self.apply_after_gates),
            "noise_before_measurement": bool(self.apply_before_measurement),
            "noise_per_qubit": copy.deepcopy(self.per_qubit),
            "noise_chip": self.ibm_backend_name,
            "noise_calibration_date": self.calibration_date,
            "noise_calibration": self.calibration_file,
            "noise_chip_gates": copy.deepcopy(self.chip_gates),
            "noise_transpile_level": int(self.transpile_level),
            "noise_transpile_seed": int(self.transpile_seed),
        }

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds the noise command-line flags to a parser.

        The flags mirror the settings fields, plus one real-chip source that
        replaces the hand-set rates: ``--noise_calibration`` names a stored
        IBM calibration file (from ``src.examples.collect_calibration``) and
        ``--noise_qubits`` names the physical qubits the circuit wires map
        to; the two go together. Every default equals the field default, so
        a command that passes no noise flag gets no noise and behaves exactly
        as before.

        Args:
            parser: The parser to add the noise flags to.

        Returns:
            None. Mutates ``parser`` by adding a "noise" argument group with
            ``--noise_type``, ``--noise_p``, ``--noise_p_1q``,
            ``--noise_p_2q``, ``--noise_gamma``, ``--noise_t1``,
            ``--noise_t2``, ``--noise_gate_time_1q``, ``--noise_gate_time_2q``,
            ``--noise_excited_state_population``, ``--noise_after_encoding``,
            ``--noise_after_gates``, ``--noise_before_measurement``,
            ``--noise_calibration`` and ``--noise_qubits``.
        """
        group = parser.add_argument_group(
            "noise",
            "Simulated hardware noise added while circuits are trained and scored "
            "(PennyLane and Qiskit targets). Off unless --noise_type or "
            "--noise_calibration is given.",
        )
        group.add_argument(
            "--noise_type",
            choices=list(get_args(NoiseType)),
            default="none",
            help=(
                "Kind of noise to add after gates; 'none' disables noise. "
                "'ibm_backend' requires --noise_calibration and --noise_qubits."
            ),
        )
        group.add_argument(
            "--noise_p",
            type=float,
            default=0.0,
            help="Error probability used for any gate whose size has no specific rate below.",
        )
        group.add_argument(
            "--noise_p_1q",
            type=float,
            default=None,
            help="Error probability after one-qubit gates (defaults to --noise_p).",
        )
        group.add_argument(
            "--noise_p_2q",
            type=float,
            default=None,
            help="Error probability after gates on two or more qubits (defaults to --noise_p).",
        )
        group.add_argument(
            "--noise_gamma",
            type=float,
            default=0.0,
            help="Strength of the amplitude/phase damping and 'mixed' noise types.",
        )
        group.add_argument(
            "--noise_t1",
            type=float,
            default=50e-6,
            help="Qubit energy-relaxation time T1 in seconds, for thermal relaxation.",
        )
        group.add_argument(
            "--noise_t2",
            type=float,
            default=70e-6,
            help="Qubit dephasing time T2 in seconds, for thermal relaxation.",
        )
        group.add_argument(
            "--noise_gate_time_1q",
            type=float,
            default=50e-9,
            help="Duration of a one-qubit gate in seconds, for thermal relaxation.",
        )
        group.add_argument(
            "--noise_gate_time_2q",
            type=float,
            default=300e-9,
            help="Duration of a two-qubit gate in seconds, for thermal relaxation.",
        )
        group.add_argument(
            "--noise_excited_state_population",
            type=float,
            default=0.0,
            help="Thermal excited-state population (0 to 1), for thermal relaxation.",
        )
        group.add_argument(
            "--noise_after_encoding",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Also add the hand-set noise right after the input encoding. Chip noise "
                "(--noise_calibration) always adds it there."
            ),
        )
        group.add_argument(
            "--noise_after_gates",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Add noise after each evolved gate.",
        )
        group.add_argument(
            "--noise_before_measurement",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Also add the hand-set noise just before measurement. Chip noise "
                "(--noise_calibration) always adds each qubit's readout error there."
            ),
        )
        group.add_argument(
            "--noise_calibration",
            type=str,
            default=None,
            help=(
                "A stored IBM calibration file written by src.examples.collect_calibration, "
                "as its path (data/calibration/<chip>/<chip>_<date>.json) or as <chip>:<date> "
                "(e.g. ibm_fez:2026-11-06), which uses that day's stored file or, if it is "
                "missing, fetches it once from IBM with the saved Qiskit account. Each "
                "wire's rates and timings come from its own qubit in that file, replacing "
                "the hand-set values above. Requires --noise_qubits. Uses the calibrated "
                "'ibm_backend' channel; any other --noise_type is refused."
            ),
        )
        group.add_argument(
            "--noise_qubits",
            type=str,
            default=None,
            help=(
                "Comma-separated physical qubits of the calibration file that circuit "
                "wires 0, 1, ... map to, e.g. 10,11,12,13; consecutive qubits must be "
                "wired neighbours. Requires --noise_calibration."
            ),
        )
        group.add_argument(
            "--noise_transpile_level",
            type=int,
            choices=[0, 1, 2, 3],
            default=2,
            help=(
                "Chip noise: how hard Qiskit's translator shrinks each genome when "
                "turning it into the chip's gates (0 to 3). Use the same level as "
                "the real-chip run."
            ),
        )
        group.add_argument(
            "--noise_transpile_seed",
            type=int,
            default=7,
            help="Chip noise: seed for the translator's random choices, for repeatable runs.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Self:
        """Builds a noise model from parsed noise flags.

        Two sources, chosen by flag: the hand-set values, or one stored IBM
        calibration file when ``--noise_calibration`` and ``--noise_qubits``
        are given, read per qubit for the listed physical qubits
        (:mod:`src.noise.chip_noise`). The three placement switches always
        come from the flags.

        Args:
            args: Parsed arguments carrying the flags from
                :meth:`initialize_parser`.

        Returns:
            The configured noise model (``noise_type="none"`` when no noise
            was requested).

        Raises:
            ValueError: If a probability, ``--noise_gamma`` or
                ``--noise_excited_state_population`` is outside 0 to 1, if a
                time is not positive, if only one
                of ``--noise_calibration`` and ``--noise_qubits`` is given,
                if ``--noise_qubits`` is not a comma-separated list of
                distinct non-negative integers, if ``--noise_type
                ibm_backend`` is given without ``--noise_calibration``, or if
                ``--noise_calibration`` is given with a hand-set noise type;
                also if the calibration file is missing or the chosen qubits
                cannot be used
                (:func:`src.noise.chip_noise.noise_hyperparameters_from_calibration`).
        """
        placement = {
            "noise_after_encoding": args.noise_after_encoding,
            "noise_after_gates": args.noise_after_gates,
            "noise_before_measurement": args.noise_before_measurement,
            # chip-noise translation settings; recorded for every mode, used by chip noise
            "noise_transpile_level": getattr(args, "noise_transpile_level", 2),
            "noise_transpile_seed": getattr(args, "noise_transpile_seed", 7),
        }
        noise_calibration = getattr(args, "noise_calibration", None)
        noise_qubits_text = getattr(args, "noise_qubits", None)

        physical_qubits: list[int] | None = None
        if noise_qubits_text is not None:
            if noise_calibration is None:
                raise ValueError("--noise_qubits requires --noise_calibration.")
            try:
                physical_qubits = [int(q) for q in noise_qubits_text.split(",")]
            except ValueError:
                raise ValueError(
                    f"--noise_qubits must be comma-separated integers, got {noise_qubits_text!r}."
                )
            if any(q < 0 for q in physical_qubits) or len(set(physical_qubits)) != len(
                physical_qubits
            ):
                raise ValueError(
                    f"--noise_qubits must be distinct non-negative qubits, got {noise_qubits_text!r}."
                )

        # The two noise modes must not mix: 'ibm_backend' takes its numbers
        # from a calibration file, every other type from the hand-set flags.
        if noise_calibration is None and args.noise_type == "ibm_backend":
            raise ValueError(
                "--noise_type ibm_backend needs --noise_calibration and --noise_qubits; "
                "without them it would run on the hand-set default values."
            )
        if noise_calibration is not None and args.noise_type not in (
            "none",
            "ibm_backend",
        ):
            raise ValueError(
                f"--noise_calibration only works with --noise_type ibm_backend, got "
                f"{args.noise_type!r}; the hand-set types do not use the calibration values."
            )

        if noise_calibration is not None:
            if physical_qubits is None:
                raise ValueError("--noise_calibration requires --noise_qubits.")
            # imported here so building a model from hand-set flags never loads Qiskit
            from src.noise.chip_noise import noise_hyperparameters_from_calibration

            noise_type = args.noise_type if args.noise_type != "none" else "ibm_backend"
            hp = noise_hyperparameters_from_calibration(
                noise_calibration, physical_qubits, noise_type=noise_type
            )
        else:
            hp = {
                "noise_type": args.noise_type,
                "noise_p": args.noise_p,
                "noise_p_1q": args.noise_p_1q,
                "noise_p_2q": args.noise_p_2q,
                "noise_gamma": args.noise_gamma,
                "noise_t1": args.noise_t1,
                "noise_t2": args.noise_t2,
                "noise_gate_time_1q": args.noise_gate_time_1q,
                "noise_gate_time_2q": args.noise_gate_time_2q,
                "noise_excited_state_population": args.noise_excited_state_population,
            }
        hp.update(placement)

        model = cls.from_hyperparameters(hp)

        for name in ("p", "p_1q", "p_2q", "gamma", "excited_state_population"):
            value = getattr(model, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"noise {name} must be between 0 and 1, got {value}.")
        for name in ("t1", "t2", "gate_time_1q", "gate_time_2q"):
            if getattr(model, name) <= 0.0:
                raise ValueError(
                    f"noise {name} must be positive, got {getattr(model, name)}."
                )

        return model
