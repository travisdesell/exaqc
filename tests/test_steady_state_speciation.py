"""Unit tests for historical-innovation speciation."""

from __future__ import annotations

import argparse

import pytest

from loguru import logger

from src.evolution.steady_state_speciation import Species, SteadyStateSpeciation


class MockGenome:
    """Minimal genome with historical innovations and a loss fitness.

    Args:
        genome_number: Unique identity used as the latest-representative key.
        innovations: Innovation IDs on this genome.
        loss: Fitness loss (lower is better).
        enabled: Optional per-innovation enable flags; defaults to all True.
        target_metric: Optional target metric (unused by the strategy; kept for
            callers that still stamp fitness dicts).
    """

    def __init__(
        self,
        genome_number: int,
        innovations: list[int],
        loss: float,
        enabled: list[bool] | None = None,
        target_metric: float | None = None,
    ) -> None:
        self.genome_number = genome_number
        self.innovations = list(innovations)
        self.enabled = (
            [True] * len(self.innovations) if enabled is None else list(enabled)
        )
        self.fitness: dict[str, float] = {"loss": loss}
        if target_metric is not None:
            self.fitness["target_metric"] = target_metric
        self.metadata: dict[str, object] = {}

    def get_historical_gate_innovations(self) -> list[int]:
        """Returns sorted unique innovation IDs.

        Returns:
            Sorted unique IDs.
        """

        return sorted(set(self.innovations))

    def get_historical_gate_signature(self) -> frozenset[tuple[int, bool]]:
        """Returns historical duplicate identity.

        Returns:
            Pairs of innovation number and enabled flag.
        """

        return frozenset(zip(self.innovations, self.enabled))


def compatible(
    genome_number: int, loss: float, mask: int, target_metric: float | None = None
) -> MockGenome:
    """Returns a genome that joins species ``{1,2,3}`` with a unique signature.

    Args:
        genome_number: Unique identity.
        loss: Fitness loss.
        mask: Enable bits for innovations 2 and 3 (innovation 1 stays enabled).
        target_metric: Optional target metric.

    Returns:
        A mock genome with ``H = {1, 2, 3}``.
    """

    return MockGenome(
        genome_number,
        [1, 2, 3],
        loss,
        enabled=[True, bool(mask & 1), bool(mask & 2)],
        target_metric=target_metric,
    )


def distant(genome_number: int, loss: float, pair: tuple[int, int]) -> MockGenome:
    """Returns a genome whose history is far from ``{1, 2, 3}``.

    Args:
        genome_number: Unique identity.
        loss: Fitness loss.
        pair: Two innovation IDs used as the history.

    Returns:
        A mock genome with that two-gene history.
    """

    return MockGenome(genome_number, [pair[0], pair[1]], loss)


def compare(left: MockGenome, right: MockGenome) -> int:
    """Orders mock genomes by increasing loss.

    Args:
        left: First genome.
        right: Second genome.

    Returns:
        Negative if ``left`` is better.
    """

    return left.fitness["loss"] - right.fitness["loss"]


def make_pop(
    max_population_size: int = 10,
    species_threshold: float = 0.6,
    inter_species_parent_rate: float = 0.0,
    rng_seed: int = 0,
    **kwargs: object,
) -> SteadyStateSpeciation:
    """Builds a speciation strategy for unit tests.

    Args:
        max_population_size: Global capacity.
        species_threshold: Join threshold.
        inter_species_parent_rate: Inter-species parent probability.
        rng_seed: RNG seed.
        **kwargs: Extra constructor overrides.

    Returns:
        A ``SteadyStateSpeciation`` with no disk side effects.
    """

    return SteadyStateSpeciation(
        max_population_size=max_population_size,
        compare=compare,
        species_threshold=species_threshold,
        inter_species_parent_rate=inter_species_parent_rate,
        rng_seed=rng_seed,
        **kwargs,
    )


def test_latest_representative_tracks_genome_number() -> None:
    """The representative is the surviving member with the greatest number."""

    pop = make_pop()
    genomes = [
        compatible(2, 0.5, 0),
        compatible(9, 0.4, 1),
        compatible(5, 0.3, 2),
    ]
    for genome in genomes:
        assert pop.insert_genome(genome)
    species = pop.species_list[0]
    assert species.latest_genome.genome_number == 9
    pop._remove_member(genomes[1])
    assert species.latest_genome.genome_number == 5
    pop._remove_member(genomes[0])
    pop._remove_member(genomes[2])
    assert pop.species_list == []


def test_assignment_joins_compatible_and_creates_otherwise() -> None:
    """Identical histories join; fully disjoint histories form a new species."""

    pop = make_pop()
    assert pop.insert_genome(compatible(1, 0.4, 0))
    assert pop.insert_genome(compatible(2, 0.3, 1))
    assert len(pop.species_list) == 1
    assert pop.insert_genome(distant(3, 0.2, (10, 20)))
    assert len(pop.species_list) == 2


def test_singleton_protection_and_all_singleton_fallback() -> None:
    """Eviction prefers a non-singleton; all-singletons drop the global worst."""

    pop = make_pop(max_population_size=3)
    a1 = compatible(1, 0.1, 0)
    a2 = compatible(2, 0.2, 1)
    singleton = distant(3, 0.9, (10, 20))
    for genome in (a1, a2, singleton):
        pop.insert_genome(genome)
    extra = compatible(4, 0.15, 2)
    assert pop.insert_genome(extra)
    retained = {genome.genome_number for genome in pop._all_genomes()}
    assert 3 in retained
    assert 2 not in retained
    assert len(pop._all_genomes()) == 3

    singles = make_pop(max_population_size=3)
    for genome_number, innov in ((1, [1]), (2, [10, 20]), (3, [30, 40])):
        singles.insert_genome(MockGenome(genome_number, innov, float(genome_number)))
    worst_new = MockGenome(4, [50, 60], 9.0)
    # Capacity discard is still recorded so EXAQC can archive it.
    assert singles.insert_genome(worst_new)
    assert worst_new.metadata["insert_type"] == "discarded"
    assert {g.genome_number for g in singles._all_genomes()} == {1, 2, 3}


def test_capacity_one_keeps_global_best() -> None:
    """Capacity 1 retains exactly the better genome."""

    pop = make_pop(max_population_size=1)
    first = compatible(1, 0.5, 0)
    second = compatible(2, 0.1, 1)
    assert pop.insert_genome(first)
    assert pop.insert_genome(second)
    assert pop.get_best_genome().genome_number == 2
    worse = compatible(3, 0.9, 2)
    assert pop.insert_genome(worse)
    assert worse.metadata["insert_type"] == "discarded"
    assert pop.get_best_genome().genome_number == 2


def test_duplicate_replace_stays_in_same_species() -> None:
    """A better historical duplicate replaces in place; a worse one is discarded."""

    pop = make_pop()
    original = MockGenome(1, [1, 2], 0.5, enabled=[True, False])
    pop.insert_genome(original)
    species_id = original.metadata["species_id"]
    worse = MockGenome(2, [1, 2], 0.9, enabled=[True, False])
    assert not pop.insert_genome(worse)
    assert worse.metadata["insert_type"] == "discarded"
    better = MockGenome(3, [1, 2], 0.1, enabled=[True, False])
    assert pop.insert_genome(better)
    assert better.metadata["species_id"] == species_id
    assert pop.species_list[0].latest_genome.genome_number == 3
    assert original not in pop._all_genomes()


def test_parents_intra_inter_unique_and_unavailable() -> None:
    """Parent selection covers intra, inter, n>2 uniqueness, and empty requests."""

    intra = make_pop(inter_species_parent_rate=1.0, rng_seed=1)
    intra.insert_genome(compatible(1, 0.2, 0))
    intra.insert_genome(compatible(2, 0.3, 1))
    parents, metadata = intra.get_parents(2)
    assert metadata["crossover_type"] == "intra"
    assert len({id(genome) for genome in parents}) == 2
    assert parents[0].fitness["loss"] <= parents[1].fitness["loss"]

    mixed = make_pop(inter_species_parent_rate=1.0, rng_seed=1)
    for genome in (
        compatible(1, 0.1, 0),
        compatible(2, 0.2, 1),
        distant(3, 0.3, (10, 20)),
        MockGenome(4, [10, 20], 0.4, enabled=[True, False]),
        MockGenome(5, [10, 20], 0.5, enabled=[False, True]),
    ):
        mixed.insert_genome(genome)
    parents, metadata = mixed.get_parents(2)
    assert metadata["crossover_type"] == "inter"
    assert len(parents) == 2
    three, three_meta = mixed.get_parents(3)
    assert three_meta["crossover_type"] == "inter"
    assert len({id(genome) for genome in three}) == 3

    forced_intra = make_pop(inter_species_parent_rate=0.0, rng_seed=1)
    for genome in (
        compatible(1, 0.1, 0),
        compatible(2, 0.2, 1),
        distant(3, 0.3, (10, 20)),
        MockGenome(4, [10, 20], 0.4, enabled=[True, False]),
    ):
        forced_intra.insert_genome(genome)
    _, metadata = forced_intra.get_parents(2)
    assert metadata["crossover_type"] == "intra"

    tiny = make_pop(inter_species_parent_rate=0.0)
    tiny.insert_genome(compatible(1, 0.1, 0))
    tiny.insert_genome(distant(2, 0.2, (10, 20)))
    assert tiny.get_parents(2) == (None, None)
    fragmented = make_pop(inter_species_parent_rate=1.0, rng_seed=0)
    fragmented.insert_genome(compatible(1, 0.1, 0))
    fragmented.insert_genome(distant(2, 0.2, (10, 20)))
    parents, metadata = fragmented.get_parents(2)
    assert metadata["crossover_type"] == "inter"
    assert {genome.genome_number for genome in parents} == {1, 2}

    with pytest.raises(ValueError, match="n_parents"):
        tiny.get_parents(0)


def test_round_robin_cursor_and_species_deletion() -> None:
    """The breeding cursor rotates and is repaired when species disappear."""

    pop = make_pop(inter_species_parent_rate=0.0, rng_seed=0)
    pop.insert_genome(compatible(1, 0.1, 0))
    pop.insert_genome(distant(2, 0.2, (10, 20)))
    pop.insert_genome(distant(3, 0.3, (30, 40)))
    first, _ = pop.get_parent()
    second, _ = pop.get_parent()
    third, _ = pop.get_parent()
    assert {first.genome_number, second.genome_number, third.genome_number} == {1, 2, 3}

    pop.generation_species = 2
    pop._drop_species_at(0)
    assert pop.generation_species == 1
    pop.generation_species = 1
    deleted_id = pop.species_list[1].species_id
    pop._drop_species_at(1)
    assert deleted_id not in {species.species_id for species in pop.species_list}
    created = distant(9, 0.9, (90, 91))
    pop.insert_genome(created)
    assert created.metadata["species_id"] != deleted_id


def test_get_population_is_fitness_sorted() -> None:
    """``get_population`` is ordered by ``compare``, not species-list order."""

    pop = make_pop()
    worse = compatible(1, 0.9, 0)
    better = distant(2, 0.1, (10, 11))
    pop.insert_genome(worse)
    pop.insert_genome(better)
    population = pop.get_population()
    assert population[0] is better
    losses = [genome.fitness["loss"] for genome in population]
    assert losses == sorted(losses)
    population.append(compatible(3, 0.5, 2))
    assert len(pop._all_genomes()) == 2


def test_logging_one_outcome_record() -> None:
    """Each insert emits one structured speciation log line."""

    messages: list[str] = []
    handler_id = logger.add(lambda record: messages.append(record.record["message"]))
    try:
        pop = make_pop()
        pop.insert_genome(compatible(1, 0.4, 0))
        pop.insert_genome(compatible(2, 0.1, 1))
        pop.insert_genome(compatible(3, 0.9, 0))
    finally:
        logger.remove(handler_id)
    outcome = [line for line in messages if line.startswith("speciation action=")]
    assert len(outcome) == 3
    assert "action=created" in outcome[0]
    assert "action=joined" in outcome[1]
    assert "action=discarded" in outcome[2]
    for line in outcome:
        for field in (
            "genome=",
            "species=",
            "distance=",
            "species_size=",
            "population_size=",
            "species_count=",
        ):
            assert field in line


def test_logging_global_best() -> None:
    """New loss champions emit the same grep keys as steady state."""

    messages: list[str] = []
    handler_id = logger.add(lambda record: messages.append(record.record["message"]))
    try:
        pop = make_pop()
        pop.insert_genome(compatible(1, 0.5, 0))
        pop.insert_genome(compatible(2, 0.2, 1))
        pop.insert_genome(compatible(3, 0.3, 0))
    finally:
        logger.remove(handler_id)

    global_best = [line for line in messages if "GLOBAL best" in line]
    assert len(global_best) == 2
    assert "genome 1" in global_best[0]
    assert "genome 2" in global_best[1]
    assert not any("GLOBAL best" in line and "genome 3" in line for line in messages)


def test_constructor_and_parser_bounds() -> None:
    """Parser defaults match the contract and invalid constructors raise."""

    parser = argparse.ArgumentParser()
    SteadyStateSpeciation.initialize_parser(parser)
    args = parser.parse_args([])
    assert args.species_threshold == 0.6
    assert args.inter_species_parent_rate == 0.1
    assert args.neat_c3 == 0.0
    assert args.max_population_size == 30

    with pytest.raises(ValueError, match="angle distance is not implemented"):
        make_pop(neat_c3=0.4)
    with pytest.raises(ValueError, match="max_population_size"):
        make_pop(max_population_size=0)
    with pytest.raises(ValueError, match="species_threshold"):
        make_pop(species_threshold=0.0)
    with pytest.raises(ValueError, match="inter_species_parent_rate"):
        make_pop(inter_species_parent_rate=1.5)


def test_is_initializing_and_get_parent_empty() -> None:
    """Initialization follows global capacity; an empty population has no parent."""

    pop = make_pop(max_population_size=2)
    assert pop.is_initializing()
    assert pop.get_parent() == (None, None)
    pop.insert_genome(compatible(1, 0.2, 0))
    assert pop.is_initializing()
    pop.insert_genome(compatible(2, 0.1, 1))
    assert not pop.is_initializing()
    parent, metadata = pop.get_parent()
    assert parent is not None
    assert metadata["crossover_type"] == "mutation"


def test_three_hundred_inserts_keep_invariants() -> None:
    """A 300-genome insert stream never exceeds capacity or leaves stale reps."""

    pop = make_pop(
        max_population_size=30,
        inter_species_parent_rate=0.1,
        rng_seed=7,
    )
    for index in range(1, 301):
        if index % 7 == 0:
            genome = MockGenome(
                index,
                [100 + (index % 11), 200 + (index % 5)],
                index / 300.0,
                enabled=[True, index % 2 == 0],
            )
        else:
            genome = MockGenome(
                index,
                [1, 2, 3, 4 + (index % 5)],
                index / 300.0,
                enabled=[True, True, bool(index & 1), bool(index & 2)],
            )
        pop.insert_genome(genome)
        retained = pop._all_genomes()
        assert len(retained) <= 30
        for species in pop.species_list:
            assert species.latest_genome in species.genomes
            assert species.latest_genome.genome_number == max(
                member.genome_number for member in species.genomes
            )
            assert len(species.genomes) >= 1
        parent, metadata = pop.get_parent()
        assert parent is not None
        assert metadata["crossover_type"] == "mutation"
        if index >= 5:
            pair, _ = pop.get_parents(2)
            if pair is not None:
                assert len(pair) == 2
                assert len({id(member) for member in pair}) == 2
            five, _ = pop.get_parents(5)
            if five is not None:
                assert len(five) == 5
                assert len({id(member) for member in five}) == 5
    assert not pop.is_initializing()
    assert pop.get_best_genome() is not None


def test_species_created_with_first_member() -> None:
    """A new species uses the first genome as its representative."""

    species = Species(0, MockGenome(4, [1], 0.5))
    assert species.latest_genome.genome_number == 4
    species.genomes.clear()
    species.recompute_latest()
    assert species.latest_genome is None
