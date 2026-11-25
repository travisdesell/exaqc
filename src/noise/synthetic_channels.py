"""Synthetic noise channels: hand-set values, no chip data.

Each channel here takes its strength from values set on the command line
(``--noise_type`` with ``--noise_p``, ``--noise_gamma``, ``--noise_t1`` and
the like), never from a calibration file. They are used for the parameter
sweeps of RQ1, which show how fast accuracy falls as one kind of noise grows,
and as simple reference conditions next to the chip noise in
:mod:`src.noise.chip_channels`.

Every channel has the same shape: after a gate, insert one PennyLane channel
on each wire the gate touched. :meth:`PennyLaneNoiseModel.apply_to_wires`
picks which one runs from ``noise_type``.
"""

from __future__ import annotations

import pennylane as qml

from src.noise.base_noise_model import BaseNoiseModel


class SyntheticChannels(BaseNoiseModel):
    """Noise channels driven by hand-set values (the synthetic family).

    Mixed into :class:`src.noise.pennylane_noise_model.PennyLaneNoiseModel`;
    not used on its own. None of these read ``per_qubit``: every wire gets
    the same single value.
    """

    def _apply_depolarizing(self, wires: list[int]) -> None:
        """Apply depolarizing noise channels.

        Not used by the chip path (``--noise_calibration`` selects
        ``_apply_ibm_backend``). Kept for the hand-set synthetic sweeps
        (``--noise_type depolarizing``): RQ1 measures the noiseless-to-noisy
        accuracy gap under parameter sweeps as well as under a chip
        calibration, and sweeping one clean error rate shows how fast accuracy
        falls as noise grows, which no single calibration file can. Run at the
        strength of a chip's measured gate errors, it also shows whether a
        chip result depends only on the total error rate or on the chip's
        noise structure (decay, per-qubit values, readout).

        A multi-qubit gate is approximated by an independent single-qubit
        depolarizing channel on each of its wires, each using the two-qubit
        probability, rather than one joint two-qubit depolarizing channel.

        Args:
            wires:
                Target wires for the depolarizing channel.

        Returns:
            None. Inserts one ``DepolarizingChannel`` per wire into the QNode
            being built.
        """
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.DepolarizingChannel(
                p,
                wires=w,
            )

    def _apply_bit_flip(self, wires: list[int]) -> None:
        """Apply bit-flip noise channels.

        With probability ``p`` each wire's 0 and 1 swap. Real chips report no
        separate bit-flip rate for gates; their bit flips happen at readout,
        which the chip family models with each qubit's own readout error
        (``apply_readout_error``).

        Args:
            wires:
                Target wires for the bit-flip channel.

        Returns:
            None. Inserts one ``BitFlip`` per wire into the QNode being built.
        """
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.BitFlip(
                p,
                wires=w,
            )

    def _apply_phase_flip(self, wires: list[int]) -> None:
        """Apply phase-flip noise channels.

        Args:
            wires:
                Target wires for the phase-flip channel.

        Returns:
            None. Inserts one ``PhaseFlip`` per wire into the QNode being
            built.
        """
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.PhaseFlip(
                p,
                wires=w,
            )

    def _apply_amplitude_damping(self, wires: list[int]) -> None:
        """Apply amplitude damping channels.

        Args:
            wires:
                Target wires for amplitude damping.

        Returns:
            None. Inserts one ``AmplitudeDamping`` with strength ``gamma`` per
            wire into the QNode being built.
        """
        for w in wires:
            qml.AmplitudeDamping(
                self.gamma,
                wires=w,
            )

    def _apply_phase_damping(self, wires: list[int]) -> None:
        """Apply phase damping channels.

        Args:
            wires:
                Target wires for phase damping.

        Returns:
            None. Inserts one ``PhaseDamping`` with strength ``gamma`` per wire
            into the QNode being built.
        """
        for w in wires:
            qml.PhaseDamping(
                self.gamma,
                wires=w,
            )

    def _apply_mixed(self, wires: list[int]) -> None:
        """Apply a mixed noise model.

        The mixed model combines depolarizing noise and amplitude damping.

        Args:
            wires:
                Target wires for the mixed noise process.

        Returns:
            None. Inserts a ``DepolarizingChannel`` then an
            ``AmplitudeDamping`` on each wire into the QNode being built.
        """
        p = self.probability_for_n_wires(len(wires))
        for w in wires:
            qml.DepolarizingChannel(
                p,
                wires=w,
            )

            qml.AmplitudeDamping(
                self.gamma,
                wires=w,
            )

    def _apply_thermal_relaxation(self, wires: list[int]) -> None:
        """Apply PennyLane thermal relaxation error to target wires.

        Models real physics (decay towards 0 and loss of phase during the
        gate), but belongs to the synthetic family because T1, T2 and the gate
        durations are the single hand-set fields, not a chip's per-qubit
        values. The chip family applies the same channel inside
        ``_apply_ibm_backend``.

        Args:
            wires:
                Target wires for the thermal relaxation channel.

        Returns:
            None. Inserts one ``ThermalRelaxationError`` per wire into the
            QNode being built, or nothing if a timescale is not positive.
        """
        n_wires = len(wires)
        gate_time = self.gate_time_1q if n_wires <= 1 else self.gate_time_2q

        t1 = float(self.t1)
        t2 = float(self.t2)
        tg = float(gate_time)
        pe = float(self.excited_state_population)

        # Non-positive timescales would produce non-CP Kraus ops (NaN at measurement).
        # Skip noise rather than emit NaN.
        if t1 <= 0.0 or t2 <= 0.0 or tg <= 0.0:
            return

        # PennyLane's ThermalRelaxationError requires t2 <= 2*t1; averaged IBM
        # stats can violate this even when each qubit individually satisfies it.
        if t2 > 2.0 * t1:
            t2 = 2.0 * t1

        # Gate time must be strictly less than relaxation timescales for the
        # channel to remain physical.
        max_tg = 0.999 * min(t1, t2)
        if tg >= max_tg:
            tg = max_tg

        if pe < 0.0:
            pe = 0.0
        elif pe > 1.0:
            pe = 1.0

        for w in wires:
            qml.ThermalRelaxationError(pe, t1, t2, tg, wires=w)
