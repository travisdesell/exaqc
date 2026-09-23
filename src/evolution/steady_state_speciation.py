"""Speciation as a third EXAQC population strategy.

Species membership is decided at insertion by structural distance on all
gate innovation IDs (join iff distance is at most the threshold). Species
are examined in cyclic order from a rotating assignment cursor. Parents are
drawn round-robin within species, with a small inter-species rate. Capacity
is global. The matching-gene angle term is reserved for later investigation
(``neat_c3`` must stay 0.0). This strategy does not subclass or modify islands.
"""

from __future__ import annotations

import argparse
import random

from collections.abc import Callable
from functools import cmp_to_key
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.circuits.circuit import CircuitGenome
from src.evolution.historical_distance import historical_distance
from src.evolution.population_strategy import PopulationStrategy, mark_discarded

if TYPE_CHECKING:
    from src.utils.restart import RestartState


class Species:
    """A slim species: id, members, and the latest surviving representative.

    Args:
        species_id: Stable integer identifier (never reused).
        genome: The first member; it is also the initial representative.
    """

    def __init__(self, species_id: int, genome: CircuitGenome) -> None:
        self.species_id = species_id
        self.genomes: list[CircuitGenome] = [genome]
        self.latest_genome: CircuitGenome | None = genome

    def recompute_latest(self) -> None:
        """Sets ``latest_genome`` to the member with the greatest genome number.

        Returns:
            None. Mutates ``latest_genome`` (``None`` if the species is empty).
        """

        if not self.genomes:
            self.latest_genome = None
            return
        self.latest_genome = max(self.genomes, key=lambda genome: genome.genome_number)


class SteadyStateSpeciation(PopulationStrategy):
    """Steady-state population partitioned by speciation distance."""

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds this population strategy's command-line arguments to a parser.

        Args:
            parser: The ``steady_state_speciation`` sub-parser.

        Returns:
            None. Mutates ``parser`` by adding this strategy's flags.
        """

        parser.add_argument(
            "--max_population_size",
            type=int,
            default=30,
            help="Maximum number of genomes retained across all species.",
        )
        parser.add_argument(
            "--species_threshold",
            type=float,
            default=0.6,
            help="Join a species iff structural distance is at most this threshold.",
        )
        parser.add_argument(
            "--neat_c1",
            type=float,
            default=1.0,
            help="Coefficient on excess genes in the structural distance.",
        )
        parser.add_argument(
            "--neat_c2",
            type=float,
            default=1.0,
            help="Coefficient on disjoint genes in the structural distance.",
        )
        parser.add_argument(
            "--neat_c3",
            type=float,
            default=0.0,
            help="Coefficient for the matching-gene angle term; reserved for later investigation (must be 0.0).",
        )
        parser.add_argument(
            "--inter_species_parent_rate",
            type=float,
            default=0.1,
            help="Fraction of multi-parent requests that mix two species.",
        )

    def __init__(
        self,
        max_population_size: int,
        compare: Callable[[CircuitGenome, CircuitGenome], int],
        species_threshold: float = 0.6,
        neat_c1: float = 1.0,
        neat_c2: float = 1.0,
        neat_c3: float = 0.0,
        inter_species_parent_rate: float = 0.1,
        rng_seed: int | None = None,
    ) -> None:
        """Creates a speciation population with global capacity.

        Disk output is owned by the search's ``GenomeArchive``; this strategy
        only selects, ranks, and partitions genomes.

        Args:
            max_population_size: Maximum retained genomes across all species.
            compare: Ordering used for fitness (negative means the first genome
                is better).
            species_threshold: Join when structural distance is at most this value.
            neat_c1: Excess-gene coefficient.
            neat_c2: Disjoint-gene coefficient.
            neat_c3: Angle-term coefficient; must be 0.0 until that term is investigated.
            inter_species_parent_rate: Probability a multi-parent request mixes
                species, when that path is eligible.
            rng_seed: Seed for shuffle and parent draws; ``None`` is unseeded.

        Raises:
            ValueError: If a constructor bound in the strategy contract is
                violated (including a non-zero ``neat_c3``).
        """

        if neat_c3 != 0.0:
            raise ValueError(
                "neat_c3 must be 0.0; the matching-gene angle term is reserved for later investigation"
            )
        if max_population_size < 1:
            raise ValueError("max_population_size must be >= 1")
        if species_threshold <= 0.0:
            raise ValueError("species_threshold must be > 0")
        if neat_c1 < 0.0 or neat_c2 < 0.0:
            raise ValueError("neat_c1 and neat_c2 must be >= 0")
        if not 0.0 <= inter_species_parent_rate <= 1.0:
            raise ValueError("inter_species_parent_rate must be in [0, 1]")

        self.max_population_size = max_population_size
        self.compare = compare
        self.species_threshold = species_threshold
        self.neat_c1 = neat_c1
        self.neat_c2 = neat_c2
        self.inter_species_parent_rate = inter_species_parent_rate
        self.rng = random.Random(rng_seed)
        self.insertions = 0
        self.species_list: list[Species] = []
        self.next_species_id = 0
        self.generation_species = 0
        self.assignment_species = 0

    def restore(self, state: "RestartState") -> None:
        """Takes back the species a stopped run held, so its search continues.

        Each genome goes back to the species recorded in its ``species_id``
        metadata. Species ids are never reused, so ``next_species_id`` resumes
        after the highest restored id. The breeding and assignment cursors
        start over (the same spirit as islands resetting ``current_island``).

        Args:
            state: The stopped run's state (see :mod:`src.utils.restart`).

        Returns:
            None. Rebuilds ``species_list``, restores the insertion count, and
            resets the breeding and assignment cursors.
        """

        held: dict[int, list[CircuitGenome]] = {}
        for genome in state.population:
            species_id = genome.metadata.get("species_id")
            if species_id is None:
                continue
            held.setdefault(int(species_id), []).append(genome)

        self.species_list = []
        for species_id in sorted(held):
            members = held[species_id]
            species = Species(species_id, members[0])
            species.genomes = list(members)
            species.recompute_latest()
            self.species_list.append(species)

        self.next_species_id = (
            max((species.species_id for species in self.species_list), default=-1) + 1
        )
        self.insertions = state.inserted_genomes
        self.generation_species = 0
        self.assignment_species = 0

        logger.info(
            "restored {} species holding {} genomes after {} insertions",
            len(self.species_list),
            len(self._all_genomes()),
            self.insertions,
        )

    def run_info(self) -> dict[str, Any]:
        """Describes this speciation run's fixed configuration for the archive.

        Returns:
            A ``speciation`` dict with the capacity, join threshold, distance
            coefficients and inter-species parent rate used for the run.
        """

        return {
            "speciation": {
                "max_population_size": self.max_population_size,
                "species_threshold": self.species_threshold,
                "neat_c1": self.neat_c1,
                "neat_c2": self.neat_c2,
                "inter_species_parent_rate": self.inter_species_parent_rate,
            }
        }

    def is_initializing(self) -> bool:
        """Returns whether the retained population is still below capacity.

        Returns:
            True if fewer than ``max_population_size`` genomes are retained.
        """

        return len(self._all_genomes()) < self.max_population_size

    def get_best_genome(self) -> CircuitGenome | None:
        """Returns the globally best retained genome, or ``None`` if empty.

        Returns:
            The first genome under ``compare``, or ``None``.
        """

        genomes = self._all_genomes()
        if not genomes:
            return None
        return min(genomes, key=cmp_to_key(self.compare))

    def get_population(self) -> list[CircuitGenome]:
        """Returns every retained genome, best first.

        Returns:
            A new list sorted by ``compare``. Modifying it does not affect the
            strategy.
        """

        return sorted(self._all_genomes(), key=cmp_to_key(self.compare))

    def get_parent(
        self, **kwargs: object
    ) -> tuple[CircuitGenome | None, dict[str, object] | None]:
        """Returns one parent from the next non-empty species.

        Args:
            **kwargs: Unused; accepted for the ``PopulationStrategy`` contract.

        Returns:
            A genome and mutation metadata (``crossover_type`` only), or
            ``(None, None)`` if empty. Species membership is stamped on insert,
            not here.
        """

        source = self._find_species(1)
        if source is None:
            return None, None
        self._advance_past(source)
        return self.rng.choice(source.genomes), {"crossover_type": "mutation"}

    def get_parents(
        self, n_parents: int = 2, **kwargs: object
    ) -> tuple[list[CircuitGenome] | None, dict[str, object] | None]:
        """Returns ``n_parents`` unique genomes, sorted by fitness.

        Args:
            n_parents: Number of parents requested.
            **kwargs: Unused; accepted for the ``PopulationStrategy`` contract.

        Returns:
            Sorted parents and intra/inter metadata (``crossover_type`` only),
            or ``(None, None)``. Species membership is stamped on insert.

        Raises:
            ValueError: If ``n_parents`` is less than 1.
        """

        if n_parents < 1:
            raise ValueError("n_parents must be >= 1")
        intra = self._find_species(n_parents)
        source = intra if intra is not None else self._find_species(1)
        if source is None:
            return None, None
        self._advance_past(source)
        total = len(self._all_genomes())
        inter_ok = (
            len(self.species_list) >= 2
            and total >= n_parents
            and self.rng.random() < self.inter_species_parent_rate
        )
        if inter_ok:
            return self._inter_parents(source, n_parents)
        if intra is None:
            return None, None
        parents = self.rng.sample(intra.genomes, n_parents)
        parents.sort(key=cmp_to_key(self.compare))
        return parents, {"crossover_type": "intra"}

    def insert_genome(self, genome: CircuitGenome, **kwargs: object) -> bool:
        """Assigns ``genome`` to a species and enforces global capacity.

        A genome whose historical signature matches one already held replaces
        it only if its fitness is better; otherwise it is discarded. An
        inserted genome that falls past capacity is discarded straight away.
        Every evaluated genome is recorded (``True``), discarded ones included,
        matching the steady-state contract.

        ``current_genome_number`` in ``kwargs`` is accepted and ignored.

        Args:
            genome: Evaluated genome to consider for retention.
            **kwargs: Extra insert options; unused.

        Returns:
            True: every genome is recorded as evaluated, whether it was kept or
            discarded.
        """

        self.insertions += 1
        existing = self._genome_with_signature(genome.get_historical_gate_signature())
        if existing is not None:
            if self.compare(existing, genome) > 0:
                species = self._replace_duplicate(existing, genome)
                self._finish(genome, True, "duplicate_replaced", 0.0, species)
                return True
            mark_discarded(genome, "duplicate_of_better", existing.genome_number)
            self._finish(genome, False, "discarded", None, None)
            return True

        species, action, distance = self._assign(genome)
        retained = True
        if len(self._all_genomes()) > self.max_population_size:
            retained = self._evict(genome)
            if not retained:
                action = "discarded"
                kept = self.get_population()
                lost_to = kept[-1].genome_number if kept else None
                mark_discarded(genome, "worse_than_population", lost_to)
        self._finish(genome, retained, action, distance, species)
        return True

    def _all_genomes(self) -> list[CircuitGenome]:
        """Returns every retained genome.

        Returns:
            Concatenation of species member lists.
        """

        return [genome for species in self.species_list for genome in species.genomes]

    def _find_species(self, min_size: int) -> Species | None:
        """Finds the next round-robin species with at least ``min_size`` members.

        Args:
            min_size: Minimum member count.

        Returns:
            The first matching species from the cursor, or ``None``.
        """

        count = len(self.species_list)
        for offset in range(count):
            species = self.species_list[(self.generation_species + offset) % count]
            if len(species.genomes) >= min_size:
                return species
        return None

    def _advance_past(self, species: Species) -> None:
        """Moves the round-robin cursor to the slot after ``species``.

        Args:
            species: The species that supplied parents for this request.

        Returns:
            None. Mutates ``generation_species``.
        """

        self.generation_species = (self.species_list.index(species) + 1) % len(
            self.species_list
        )

    def _species_of(self, genome: CircuitGenome) -> Species | None:
        """Returns the species that currently holds ``genome``.

        Args:
            genome: A retained or recently removed genome.

        Returns:
            The containing species, or ``None``.
        """

        for species in self.species_list:
            if genome in species.genomes:
                return species
        return None

    def _drop_species_at(self, index: int) -> None:
        """Deletes the species at ``index`` and repairs the round-robin cursors.

        Args:
            index: List position of the species to remove.

        Returns:
            None. Mutates ``species_list``, ``generation_species``, and
            ``assignment_species``.
        """

        del self.species_list[index]
        if not self.species_list:
            self.generation_species = 0
            self.assignment_species = 0
            return
        if index < self.generation_species:
            self.generation_species -= 1
        if index < self.assignment_species:
            self.assignment_species -= 1
        length = len(self.species_list)
        self.generation_species %= length
        self.assignment_species %= length

    def _remove_member(self, genome: CircuitGenome) -> None:
        """Removes ``genome`` and drops its species if emptied.

        Args:
            genome: Member to remove.

        Returns:
            None. Mutates species membership.
        """

        for index, species in enumerate(self.species_list):
            if genome in species.genomes:
                species.genomes.remove(genome)
                if not species.genomes:
                    self._drop_species_at(index)
                else:
                    species.recompute_latest()
                return

    def _genome_with_signature(
        self, signature: frozenset[tuple[int, bool]]
    ) -> CircuitGenome | None:
        """Finds a retained genome with the given historical signature.

        Args:
            signature: ``(innovation, enabled)`` pairs.

        Returns:
            The first matching retained genome, or ``None``.
        """

        for genome in self._all_genomes():
            if genome.get_historical_gate_signature() == signature:
                return genome
        return None

    def _replace_duplicate(
        self, existing: CircuitGenome, genome: CircuitGenome
    ) -> Species:
        """Replaces ``existing`` with ``genome`` in the same species.

        Args:
            existing: Worse duplicate already in the population.
            genome: Better candidate with the same historical signature.

        Returns:
            The species that now holds ``genome``.
        """

        species = self._species_of(existing)
        assert species is not None
        species.genomes.remove(existing)
        species.genomes.append(genome)
        species.recompute_latest()
        genome.metadata["species_id"] = species.species_id
        return species

    def _assign(self, genome: CircuitGenome) -> tuple[Species, str, float | None]:
        """Joins the first compatible species in cyclic order, or creates one.

        Species are examined from ``assignment_species`` onward. After the
        attempt, that cursor advances by one slot so no species keeps a
        permanent first-look advantage. Join uses a non-strict threshold.

        Args:
            genome: Candidate that is not a historical duplicate.

        Returns:
            The species, ``joined`` or ``created``, and the accepted distance
            (``None`` when a species is created).
        """

        if not self.species_list:
            return self._new_species(genome), "created", None
        history = genome.get_historical_gate_innovations()
        count = len(self.species_list)
        start = self.assignment_species % count
        for offset in range(count):
            species = self.species_list[(start + offset) % count]
            distance = historical_distance(
                species.latest_genome.get_historical_gate_innovations(),
                history,
                self.neat_c1,
                self.neat_c2,
            )
            if distance <= self.species_threshold:
                species.genomes.append(genome)
                species.recompute_latest()
                genome.metadata["species_id"] = species.species_id
                self.assignment_species = (start + 1) % len(self.species_list)
                return species, "joined", distance
        created = self._new_species(genome)
        self.assignment_species = (start + 1) % len(self.species_list)
        return created, "created", None

    def _new_species(self, genome: CircuitGenome) -> Species:
        """Creates a species whose first member is ``genome``.

        Args:
            genome: First member.

        Returns:
            The new species, already appended to ``species_list``.
        """

        species = Species(self.next_species_id, genome)
        self.next_species_id += 1
        self.species_list.append(species)
        genome.metadata["species_id"] = species.species_id
        return species

    def _eviction_compare(self, left: CircuitGenome, right: CircuitGenome) -> int:
        """Orders genomes best-first; equal fitness prefers the newer genome.

        Args:
            left: First genome.
            right: Second genome.

        Returns:
            Negative if ``left`` should sort before ``right``.
        """

        ranked = self.compare(left, right)
        if ranked != 0:
            return ranked
        return right.genome_number - left.genome_number

    def _evict(self, candidate: CircuitGenome) -> bool:
        """Removes one genome so the population fits capacity.

        Prefers the worst non-singleton member. If every species is a
        singleton, removes the global worst.

        Args:
            candidate: Genome that was just assigned.

        Returns:
            True if ``candidate`` is still retained.
        """

        ordered = sorted(self._all_genomes(), key=cmp_to_key(self._eviction_compare))
        victim = ordered[-1]
        for genome in reversed(ordered):
            species = self._species_of(genome)
            if species is not None and len(species.genomes) > 1:
                victim = genome
                break
        self._remove_member(victim)
        return victim is not candidate

    def _inter_parents(
        self, source: Species, n_parents: int
    ) -> tuple[list[CircuitGenome], dict[str, object]]:
        """Builds an inter-species parent set of size ``n_parents``.

        Args:
            source: Focal species supplying the first parent.
            n_parents: Exact number of unique parents to return.

        Returns:
            Fitness-sorted parents and inter-species metadata.
        """

        first = self.rng.choice(source.genomes)
        other = self.rng.choice(
            [species for species in self.species_list if species is not source]
        )
        other_best = min(other.genomes, key=cmp_to_key(self.compare))
        chosen: list[CircuitGenome] = [first]
        if other_best is not first:
            chosen.append(other_best)
        if len(chosen) < n_parents:
            pools = [source.genomes + other.genomes, self._all_genomes()]
            seen = {id(genome) for genome in chosen}
            for pool in pools:
                mixed = list(pool)
                self.rng.shuffle(mixed)
                for genome in mixed:
                    if id(genome) in seen:
                        continue
                    chosen.append(genome)
                    seen.add(id(genome))
                    if len(chosen) == n_parents:
                        break
                if len(chosen) == n_parents:
                    break
        chosen.sort(key=cmp_to_key(self.compare))
        return chosen, {"crossover_type": "inter"}

    def _finish(
        self,
        genome: CircuitGenome,
        retained: bool,
        action: str,
        distance: float | None,
        species: Species | None,
    ) -> None:
        """Sets insert metadata for retained genomes and logs the outcome.

        Discarded genomes already carry ``insert_type`` / ``discard_reason``
        from :func:`~src.evolution.population_strategy.mark_discarded` when
        that path applied; this method still stamps retained genomes and
        writes the structured speciation log.

        Args:
            genome: Candidate considered for insertion.
            retained: Whether ``genome`` is in the population after this call.
            action: ``joined``, ``created``, ``duplicate_replaced``, or
                ``discarded``.
            distance: Accepted representative distance, or ``None``.
            species: Species involved, if any.

        Returns:
            None. Mutates ``genome.metadata`` for retained genomes and logs.
        """

        population = self.get_population()
        if retained:
            genome.metadata["insert_type"] = "inserted"
            best = self.get_best_genome()
            if best is not None and genome.genome_number == best.genome_number:
                genome.metadata["insert_type"] = "global_best"
                logger.success(
                    f"🎯 New best genome {genome.genome_number} "
                    f"[insertion {self.insertions}] Population found new GLOBAL best genome "
                    f"with fitness: {genome.fitness}"
                )
        elif "insert_type" not in genome.metadata:
            genome.metadata["insert_type"] = "discarded"

        species_id = (
            species.species_id
            if species is not None
            else genome.metadata.get("species_id", "none")
        )
        species_size = len(species.genomes) if species is not None else 0
        logger.info(
            "speciation action={} genome={} species={} distance={} "
            "species_size={} population_size={} species_count={}".format(
                action,
                genome.genome_number,
                species_id if species_id is not None else "none",
                "na" if distance is None else distance,
                species_size,
                len(population),
                len(self.species_list),
            )
        )
