"""Accumulates how long named phases of a loop take, for timing reports.

Used by the MPI master (waiting on workers, generating, sending, deserializing
and inserting genomes) and by :meth:`EXAQC.insert_genome
<src.evolution.exaqc.EXAQC.insert_genome>` (the parts of an insertion), so a
run can report whether the master keeps up with its workers. Timers are
cumulative; a report over an interval takes the difference from a copy saved at
the start of it (see :meth:`PhaseTimer.since`).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class PhaseTimer:
    """Total seconds and occurrence counts for each named phase.

    Attributes:
        seconds: Total time spent in each phase.
        counts: How many times each phase was timed.
    """

    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, phase: str, seconds: float) -> None:
        """Records one occurrence of a phase.

        Args:
            phase: The phase's name.
            seconds: How long this occurrence took.

        Returns:
            None. Adds to ``seconds[phase]`` and ``counts[phase]``.
        """

        self.seconds[phase] = self.seconds.get(phase, 0.0) + seconds
        self.counts[phase] = self.counts.get(phase, 0) + 1

    @contextmanager
    def time(self, phase: str) -> Iterator[None]:
        """Times the enclosed block as one occurrence of a phase.

        The time is recorded even if the block raises.

        Args:
            phase: The phase's name.

        Yields:
            None, once, while the block runs.
        """

        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, time.perf_counter() - started)

    def copy(self) -> PhaseTimer:
        """Returns an independent copy of the totals so far.

        Returns:
            A new timer with the same seconds and counts.
        """

        return PhaseTimer(seconds=dict(self.seconds), counts=dict(self.counts))

    def since(self, earlier: PhaseTimer) -> PhaseTimer:
        """Returns what this timer accumulated after an earlier copy of it.

        Args:
            earlier: A copy of this timer taken earlier (see :meth:`copy`).

        Returns:
            A new timer holding only the time and counts added since
            ``earlier``; phases with no new occurrences are left out.
        """

        difference = PhaseTimer()
        for phase, seconds in self.seconds.items():
            count = self.counts.get(phase, 0) - earlier.counts.get(phase, 0)
            if count > 0:
                difference.seconds[phase] = seconds - earlier.seconds.get(phase, 0.0)
                difference.counts[phase] = count
        return difference

    def total(self, phases: Iterable[str] | None = None) -> float:
        """Sums the time spent in the given phases.

        Args:
            phases: The phases to sum, or ``None`` for every phase.

        Returns:
            The total seconds (0 for phases never timed).
        """

        names = self.seconds.keys() if phases is None else phases
        return sum(self.seconds.get(phase, 0.0) for phase in names)

    def describe(self, phases: Iterable[str] | None = None) -> str:
        """Formats each phase's mean duration and count for a log line.

        Args:
            phases: The phases to describe, in order, or ``None`` for every
                phase in the order first timed. Phases never timed are skipped.

        Returns:
            A comma-separated string such as ``"generate 4.10 ms (x1000)"``, or
            ``"none"`` when nothing was timed.
        """

        names = self.seconds.keys() if phases is None else phases
        parts = [
            f"{phase} {1000.0 * self.seconds[phase] / self.counts[phase]:.2f} ms (x{self.counts[phase]})"
            for phase in names
            if self.counts.get(phase, 0) > 0
        ]
        return ", ".join(parts) if parts else "none"
