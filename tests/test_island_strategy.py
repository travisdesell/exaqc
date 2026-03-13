from src.evolution.steady_state_islands import SteadyStateIslands


class MockGenome:
    def __init__(self, fitness: float, metadata: dict[str, any]):
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
        self.metadata = metadata


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
        m = MockGenome(0.5, {})
        population.insert_genome(m)

    # all three populations should each have a single genome after
    # three are inserted
    assert len(population.islands[0].population) == 1
    assert len(population.islands[1].population) == 1
    assert len(population.islands[2].population) == 1
    assert population.is_initializing()

    for i in range(3):
        m = MockGenome(0.5, {})
        population.insert_genome(m)

    # all three populations now have two genomes after
    # three are inserted
    assert len(population.islands[0].population) == 2
    assert len(population.islands[1].population) == 2
    assert len(population.islands[2].population) == 2
    assert population.is_initializing()

    for i in range(9):
        m = MockGenome(0.5, {})
        population.insert_genome(m)

    # all three populations should now be at the max
    assert len(population.islands[0].population) == 5
    assert len(population.islands[1].population) == 5
    assert len(population.islands[2].population) == 5
    assert not population.is_initializing()

    for i in range(9):
        m = MockGenome(0.5, {})
        population.insert_genome(m)

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
