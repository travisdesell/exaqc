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
    """Backend-agnostic base class for quantum noise models.

    Subclasses should implement backend-specific noise insertion/construction.
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
        return self.noise_type != "none"

    def probability_for_n_wires(self, n_wires: int) -> float:
        if n_wires <= 1:
            return self.p_1q if self.p_1q is not None else self.p
        return self.p_2q if self.p_2q is not None else self.p

    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier."""
        raise NotImplementedError

    @classmethod
    def from_hyperparameters(cls, hp: dict):
        """Build a noise model from genome/objective hyperparameters."""
        return cls(
            noise_type=hp.get("noise_type", "none"),
            p=float(hp.get("noise_p", 0.0)),
            p_1q=hp.get("noise_p_1q", None),
            p_2q=hp.get("noise_p_2q", None),
            gamma=float(hp.get("noise_gamma", 0.0)),
            apply_after_input_encoding=bool(hp.get("noise_after_encoding", False)),
            apply_after_gates=bool(hp.get("noise_after_gates", True)),
            apply_before_measurement=bool(hp.get("noise_before_measurement", False)),
        )
