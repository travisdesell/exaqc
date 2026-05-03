from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


NoiseType = Literal[
    "none",
    "depolarizing",
    "bit_flip",
    "phase_flip",
    "amplitude_damping",
    "phase_damping",
    "thermal_relaxation",
    "mixed",
]


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

    apply_after_input_encoding: bool = False
    apply_after_gates: bool = True
    apply_before_measurement: bool = False

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
        while multi-qubit operations preferentially use ``p_2q``. If no
        specialized probability is available, the default ``p`` value is used.

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
    def from_hyperparameters(cls, hp: dict):
        """Construct a noise model from hyperparameter dictionaries.

        This helper maps genome or experiment hyperparameters into a
        backend-specific noise model configuration.

        Expected hyperparameter keys include:

        - ``noise_type``
        - ``noise_p``
        - ``noise_p_1q``
        - ``noise_p_2q``
        - ``noise_gamma``
        - ``noise_after_encoding``
        - ``noise_after_gates``
        - ``noise_before_measurement``

        Args:
            hp:
                Dictionary containing experiment or genome hyperparameters.

        Returns:
            Instantiated subclass of ``BaseNoiseModel`` configured using
            the provided hyperparameters.
        """
        return cls(
            noise_type=hp.get("noise_type", "none"),
            p=float(hp.get("noise_p", 0.0)),
            p_1q=hp.get("noise_p_1q", None),
            p_2q=hp.get("noise_p_2q", None),
            gamma=float(hp.get("noise_gamma", 0.0)),
            apply_after_input_encoding=bool(
                hp.get("noise_after_encoding", False)
            ),
            apply_after_gates=bool(
                hp.get("noise_after_gates", True)
            ),
            apply_before_measurement=bool(
                hp.get("noise_before_measurement", False)
            ),
        )