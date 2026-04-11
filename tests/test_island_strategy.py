from __future__ import annotations

from src.evolution.steady_state_islands import Island, SteadyStateIslands


class MockGenome:
    def __init__(self, fitness: float, genome_number: int, metadata: dict[str, any]):
        """
        A fake genome to test populations with.

        Args:
            fitness: is the genome's fitness value.
            metadata: is the genome's metadata which will either be
                empty or contain a target island id.
        """
        self.fitness = {
            "train_loss": fitness,
            "train_acc": fitness,
            "test_loss": fitness,
            "test_acc": fitness,
        }
        self.genome_number = genome_number
        self.metadata = metadata

    def has_same_gates(self, other: MockGenome) -> bool:
        """
        A mock method to handle the has same gates check, for these
        tests we're going to assume none of the genomes are the same so
        this will always return False.

        Returns:
            False, always
        """
        return False


def compare(mock1: MockGenome, mock2: MockGenome) -> int:
    """
    Used to sort genomes by mock genomes by fitness.

    Returns: 0 if the two genomes have equivalent fitnesses, a ngeative value if genome1 should be
        sorted before genome2, and a positive value if genome2 should be sorted before genome1
    """

    # this will return 0 if the fitnesses are the same, negative if genome1 should be before
    # genome2 (genome1's fitness would be lower), and positive if genome2 should be before
    # genome1 (genome2's fitness would be lower)
    return mock1.fitness["test_acc"] - mock2.fitness["test_acc"]


def test_island_repopulation():
    """
    Tests to make sure global best genomes are always inserted into repopulating
    islands and also tests that genomes are discarded appropriately for repopulating
    islands when their genome number is from before the repopulation event.
    """

    island = Island(id=0, max_size=5, compare=compare)
    island.repopulation_genome_number = 100

    # test adding a non-global best genome with a genome number
    # lower than the repopuation number (should not be inserted)
    m = MockGenome(0.5, 10, {})
    island.insert_genome(m)

    assert len(island.population) == 0
    assert m.metadata["insert_type"] == "discarded"

    m = MockGenome(0.25, 15, {"insert_type": ""})
    island.insert_genome(m)
    assert len(island.population) == 0
    assert m.metadata["insert_type"] == "discarded"

    # test adding a global best genome with a genome number
    # lower than the repopuation number (should be inserted)

    m = MockGenome(0.5, 10, {"insert_type": "global_best"})
    island.insert_genome(m)
    assert len(island.population) == 1
    assert m.metadata["insert_type"] == "global_best"

    # test adding a genome with a genome number greater than the
    # repopulation number (should be inserted)

    m = MockGenome(0.55, 110, {"insert_type": "inserted"})
    island.insert_genome(m)
    assert len(island.population) == 2
    assert m.metadata["insert_type"] == "inserted"

    m = MockGenome(0.5, 115, {})
    island.insert_genome(m)
    assert len(island.population) == 3
    assert m.metadata["insert_type"] == "inserted"

    # test the insert_type metadata is properly set for
    # a new local best genome
    m = MockGenome(0.15, 120, {})
    island.insert_genome(m)
    assert len(island.population) == 4
    assert m.metadata["insert_type"] == "local_best"

    # test the insert_type metadata is properly set for
    # a genome being inserted to the island
    m = MockGenome(0.65, 125, {})
    island.insert_genome(m)
    assert len(island.population) == 5
    assert m.metadata["insert_type"] == "inserted"

    # test the insert_type metadata is properly set for
    # a genome being discarded from a full island
    m = MockGenome(0.95, 130, {})
    island.insert_genome(m)
    assert len(island.population) == 5
    assert m.metadata["insert_type"] == "discarded"


def test_island_insertion():
    """
    Tests inserting mock genomes into a steady state island population to
    ensure islands are being inserted correctly.
    """

    population = SteadyStateIslands(
        n_islands=3,
        max_island_size=5,
        genomes_before_extinction=10,
        islands_to_extinct=1,
        compare=compare,
        out_dir=None,
    )

    # test adding in genomes with no metadata, these should fill
    # the islands with the least number of genomes first
    for i in range(3):
        m = MockGenome(0.5, 1, {})
        population.insert_genome(m, current_genome_number=1)

    # all three populations should each have a single genome after
    # three are inserted
    assert len(population.islands[0].population) == 1
    assert len(population.islands[1].population) == 1
    assert len(population.islands[2].population) == 1
    assert population.is_initializing()

    for i in range(3):
        m = MockGenome(0.5, 2, {})
        population.insert_genome(m, current_genome_number=1)

    # all three populations now have two genomes after
    # three are inserted
    assert len(population.islands[0].population) == 2
    assert len(population.islands[1].population) == 2
    assert len(population.islands[2].population) == 2
    assert population.is_initializing()

    for i in range(9):
        m = MockGenome(0.5, 5 + i, {})
        population.insert_genome(m, current_genome_number=1)

    # all three populations should now be at the max
    assert len(population.islands[0].population) == 5
    assert len(population.islands[1].population) == 5
    assert len(population.islands[2].population) == 5
    assert not population.is_initializing()

    for i in range(9):
        m = MockGenome(0.5, 15 + i, {})
        population.insert_genome(m, current_genome_number=1)

    # all three populations should now still be at the max
    # size (5), even if additional genomes added
    assert len(population.islands[0].population) == 5
    assert len(population.islands[1].population) == 5
    assert len(population.islands[2].population) == 5
    assert not population.is_initializing()

    # now test round robin getting of parents

    p, metadata = population.get_parent()
    assert metadata["target_island_id"] == 0

    p, metadata = population.get_parent()
    assert metadata["target_island_id"] == 1

    p, metadata = population.get_parent()
    assert metadata["target_island_id"] == 2

    p, metadata = population.get_parent()
    assert metadata["target_island_id"] == 0

    p, metadata = population.get_parent()
    assert metadata["target_island_id"] == 1

    p, metadata = population.get_parents(2)
    assert metadata["target_island_id"] == 2

    p, metadata = population.get_parents(3)
    assert metadata["target_island_id"] == 0
