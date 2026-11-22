"""Unit tests for :func:`src.evolution.topology.assign_topology`.

``assign_topology`` sets each island's ``neighbors`` list according to a named
connection topology. These tests validate the structure produced by each
topology (``fully_connected``, ``ring``, ``star``, ``2d_mesh``, ``tree`` and
``random``), the argument validation each performs, and the edge cases called
out in the task: no island should ever be its own neighbor, and the undirected
topologies should be symmetric and duplicate-free.

The tests are written against the behavior documented in ``topology``'s
docstrings (i.e. the intended contract), so a failure indicates the
implementation diverges from that contract rather than a problem with the test.
"""

from __future__ import annotations

import random

import pytest

from src.evolution.island import Island
from src.evolution.topology import assign_topology


def noop_compare(genome1: object, genome2: object) -> int:
    """A comparison that ranks everything equally.

    Topology assignment never consults the compare function, but ``Island``
    requires one, so this placeholder is used throughout.

    Args:
        genome1: Unused first genome.
        genome2: Unused second genome.

    Returns:
        ``0`` always.
    """

    return 0


def make_islands(n_islands: int) -> list[Island]:
    """Creates ``n_islands`` islands with ids ``0..n_islands - 1``.

    Args:
        n_islands: How many islands to create.

    Returns:
        A list of freshly constructed :class:`Island` objects with empty
        neighbor lists.
    """

    return [Island(id=i, max_size=10, compare=noop_compare) for i in range(n_islands)]


def neighbor_ids(island: Island) -> list[int]:
    """Returns the ids of an island's neighbors, in order (with duplicates).

    Args:
        island: The island whose neighbors to read.

    Returns:
        The ``id`` of each neighbor, preserving order and repeats so duplicate
        edges can be detected.
    """

    return [neighbor.id for neighbor in island.neighbors]


def assert_no_self_loops(islands: list[Island]) -> None:
    """Asserts no island lists itself among its neighbors.

    Args:
        islands: The islands to check.

    Returns:
        None. Raises ``AssertionError`` if any self-loop is found.
    """

    for island in islands:
        assert island not in island.neighbors, f"island {island.id} is its own neighbor"
        assert island.id not in neighbor_ids(island)


def assert_no_duplicate_neighbors(islands: list[Island]) -> None:
    """Asserts no island lists the same neighbor more than once.

    Args:
        islands: The islands to check.

    Returns:
        None. Raises ``AssertionError`` if a duplicate edge is found.
    """

    for island in islands:
        ids = neighbor_ids(island)
        assert len(ids) == len(
            set(ids)
        ), f"island {island.id} has duplicate neighbors: {ids}"


def assert_connected(islands: list[Island]) -> None:
    """Asserts every island is reachable from every other (one component).

    Treats the neighbor relation as **undirected** (an edge exists between ``a``
    and ``b`` if either lists the other), so this catches an island being left
    disconnected under both the symmetric topologies and the directed
    ``random`` one. A breadth-first walk from the first island must reach all of
    them; otherwise the graph has an isolated island or a split component.

    Args:
        islands: The islands to check. An empty list is trivially connected.

    Returns:
        None. Raises ``AssertionError`` if any island is unreachable.
    """

    if not islands:
        return

    id_to_island = {island.id: island for island in islands}

    # Build an undirected adjacency from the (possibly directed) edges.
    undirected: dict[int, set[int]] = {island.id: set() for island in islands}
    for island in islands:
        for neighbor in island.neighbors:
            undirected[island.id].add(neighbor.id)
            undirected[neighbor.id].add(island.id)

    reached: set[int] = {islands[0].id}
    frontier: list[int] = [islands[0].id]
    while frontier:
        current = frontier.pop()
        for neighbor_id in undirected[current]:
            if neighbor_id not in reached:
                reached.add(neighbor_id)
                frontier.append(neighbor_id)

    all_ids = set(id_to_island)
    assert reached == all_ids, f"disconnected islands: unreachable {all_ids - reached}"


def assert_symmetric(islands: list[Island]) -> None:
    """Asserts the neighbor relation is symmetric (an undirected graph).

    Args:
        islands: The islands to check.

    Returns:
        None. Raises ``AssertionError`` if ``a`` lists ``b`` but ``b`` does not
        list ``a``.
    """

    for island in islands:
        for neighbor in island.neighbors:
            assert (
                island in neighbor.neighbors
            ), f"edge {island.id}->{neighbor.id} is not symmetric"


# ---------------------------------------------------------------------------
# fully_connected
# ---------------------------------------------------------------------------


def test_fully_connected_connects_every_other_island() -> None:
    """Every island is connected to all others and nothing else."""

    islands = make_islands(4)
    assign_topology(islands, ["fully_connected"])
    for island in islands:
        assert set(neighbor_ids(island)) == {i for i in range(4) if i != island.id}


def test_fully_connected_is_the_default() -> None:
    """Omitting the topology argument uses ``fully_connected``."""

    islands = make_islands(3)
    assign_topology(islands)
    for island in islands:
        assert set(neighbor_ids(island)) == {i for i in range(3) if i != island.id}


def test_fully_connected_has_no_self_loops_or_duplicates() -> None:
    """A fully connected graph has no self-loops and no duplicate edges."""

    islands = make_islands(5)
    assign_topology(islands, ["fully_connected"])
    assert_no_self_loops(islands)
    assert_no_duplicate_neighbors(islands)
    assert_symmetric(islands)


def test_fully_connected_single_island_has_no_neighbors() -> None:
    """A lone island connects to nobody (edge case)."""

    islands = make_islands(1)
    assign_topology(islands, ["fully_connected"])
    assert islands[0].neighbors == []


# ---------------------------------------------------------------------------
# ring
# ---------------------------------------------------------------------------


def test_ring_connects_each_island_to_its_two_neighbors() -> None:
    """Each island is linked to the previous and next island, wrapping."""

    n = 5
    islands = make_islands(n)
    assign_topology(islands, ["ring"])
    for i, island in enumerate(islands):
        assert set(neighbor_ids(island)) == {(i - 1) % n, (i + 1) % n}


def test_ring_is_symmetric_without_self_loops_or_duplicates() -> None:
    """A ring is undirected, degree-2, with no self-loops or repeats."""

    islands = make_islands(6)
    assign_topology(islands, ["ring"])
    assert_no_self_loops(islands)
    assert_no_duplicate_neighbors(islands)
    assert_symmetric(islands)
    for island in islands:
        assert len(island.neighbors) == 2


def test_ring_two_islands_has_no_duplicate_edges() -> None:
    """A 2-island ring links the pair once, not twice (edge case)."""

    islands = make_islands(2)
    assign_topology(islands, ["ring"])
    assert_no_self_loops(islands)
    assert_no_duplicate_neighbors(islands)


# ---------------------------------------------------------------------------
# star
# ---------------------------------------------------------------------------


def test_star_center_connects_to_all_others() -> None:
    """Island 0 (the center) links to every other island."""

    islands = make_islands(5)
    assign_topology(islands, ["star"])
    assert set(neighbor_ids(islands[0])) == {1, 2, 3, 4}


def test_star_leaves_connect_only_to_center() -> None:
    """Every non-center island links to exactly the center."""

    islands = make_islands(5)
    assign_topology(islands, ["star"])
    for island in islands[1:]:
        assert neighbor_ids(island) == [0]


def test_star_is_symmetric_without_self_loops() -> None:
    """A star is undirected with no self-loops or duplicate edges."""

    islands = make_islands(5)
    assign_topology(islands, ["star"])
    assert_no_self_loops(islands)
    assert_no_duplicate_neighbors(islands)
    assert_symmetric(islands)


# ---------------------------------------------------------------------------
# 2d_mesh
# ---------------------------------------------------------------------------


def test_2d_mesh_grid_adjacency() -> None:
    """A 3x3 mesh connects each island to its up/down/left/right neighbors.

    Islands are laid out row-major: index ``x * y_dim + y``. The center has
    four neighbors, edges three, and corners two.
    """

    islands = make_islands(9)
    assign_topology(islands, ["2d_mesh", "3", "3"])
    # center (x=1, y=1) -> up/down/left/right
    assert set(neighbor_ids(islands[4])) == {1, 3, 5, 7}
    # corner (x=0, y=0) -> right and down
    assert set(neighbor_ids(islands[0])) == {1, 3}
    # corner (x=2, y=2) -> left and up
    assert set(neighbor_ids(islands[8])) == {5, 7}


def test_2d_mesh_is_symmetric_without_self_loops() -> None:
    """A 2x3 mesh is undirected with no self-loops or duplicate edges."""

    islands = make_islands(6)
    assign_topology(islands, ["2d_mesh", "2", "3"])
    assert_no_self_loops(islands)
    assert_no_duplicate_neighbors(islands)
    assert_symmetric(islands)


def test_2d_mesh_requires_two_arguments() -> None:
    """A mesh needs exactly ``<x_dim> <y_dim>``."""

    islands = make_islands(4)
    with pytest.raises(ValueError):
        assign_topology(islands, ["2d_mesh", "4"])


def test_2d_mesh_requires_integer_arguments() -> None:
    """Non-integer dimensions are rejected."""

    islands = make_islands(4)
    with pytest.raises(ValueError):
        assign_topology(islands, ["2d_mesh", "2", "x"])


def test_2d_mesh_dimensions_must_match_island_count() -> None:
    """``x_dim * y_dim`` must equal the number of islands."""

    islands = make_islands(5)
    with pytest.raises(ValueError):
        assign_topology(islands, ["2d_mesh", "2", "2"])


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------


def test_tree_binary_parent_child_links() -> None:
    """A binary tree links each parent ``i`` to children ``2i+1`` and ``2i+2``.

    With 7 islands and ``n_children = 2`` the tree is::

            0
          /   \\
         1     2
        / \\   / \\
       3   4 5   6
    """

    islands = make_islands(7)
    assign_topology(islands, ["tree", "2"])
    assert set(neighbor_ids(islands[0])) == {1, 2}
    assert set(neighbor_ids(islands[1])) == {0, 3, 4}
    assert set(neighbor_ids(islands[2])) == {0, 5, 6}
    assert set(neighbor_ids(islands[3])) == {1}
    assert set(neighbor_ids(islands[6])) == {2}


def test_tree_is_symmetric_without_self_loops() -> None:
    """A tree is undirected with no self-loops or duplicate edges."""

    islands = make_islands(7)
    assign_topology(islands, ["tree", "2"])
    assert_no_self_loops(islands)
    assert_no_duplicate_neighbors(islands)
    assert_symmetric(islands)


def test_tree_requires_one_integer_argument() -> None:
    """A tree needs exactly one integer ``<n_children>``."""

    islands = make_islands(7)
    with pytest.raises(ValueError):
        assign_topology(islands, ["tree"])
    islands = make_islands(7)
    with pytest.raises(ValueError):
        assign_topology(islands, ["tree", "x"])


# ---------------------------------------------------------------------------
# random (directed)
# ---------------------------------------------------------------------------


def test_random_out_degree_within_bounds() -> None:
    """Each island has between ``min_edges`` and ``max_edges`` outgoing edges."""

    random.seed(0)
    islands = make_islands(8)
    min_edges, max_edges = 2, 4
    assign_topology(islands, ["random", str(min_edges), str(max_edges)])
    for island in islands:
        assert min_edges <= len(island.neighbors) <= max_edges


def test_random_has_no_self_loops() -> None:
    """A random topology never connects an island to itself."""

    random.seed(1)
    islands = make_islands(8)
    assign_topology(islands, ["random", "2", "4"])
    assert_no_self_loops(islands)


def test_random_ring_guarantees_reachability() -> None:
    """Every island is reachable from island 0 along directed edges.

    ``random`` seeds the graph with a directed ring so the whole population
    stays connected; a breadth-first walk from island 0 should reach all.
    """

    random.seed(2)
    n = 8
    islands = make_islands(n)
    assign_topology(islands, ["random", "2", "4"])

    reached: set[int] = {0}
    frontier: list[Island] = [islands[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in current.neighbors:
            if neighbor.id not in reached:
                reached.add(neighbor.id)
                frontier.append(neighbor)
    assert reached == set(range(n))


def test_random_requires_two_integer_arguments() -> None:
    """A random topology needs exactly two integer arguments."""

    islands = make_islands(8)
    with pytest.raises(ValueError):
        assign_topology(islands, ["random", "2"])
    islands = make_islands(8)
    with pytest.raises(ValueError):
        assign_topology(islands, ["random", "2", "y"])


def test_random_rejects_max_less_than_min() -> None:
    """``max_edges`` must be at least ``min_edges``."""

    islands = make_islands(8)
    with pytest.raises(ValueError):
        assign_topology(islands, ["random", "4", "2"])


def test_random_rejects_max_edges_at_least_island_count() -> None:
    """``max_edges`` must be strictly fewer than the number of islands."""

    islands = make_islands(4)
    with pytest.raises(ValueError):
        assign_topology(islands, ["random", "2", "4"])


# ---------------------------------------------------------------------------
# cross-topology edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topology",
    [["fully_connected"], ["ring"], ["star"], ["2d_mesh", "2", "3"], ["tree", "2"]],
)
def test_no_topology_creates_a_self_loop(topology: list[str]) -> None:
    """No undirected topology ever makes an island its own neighbor."""

    islands = make_islands(6)
    assign_topology(islands, topology)
    assert_no_self_loops(islands)


@pytest.mark.parametrize(
    "topology",
    [["fully_connected"], ["ring"], ["star"], ["2d_mesh", "2", "3"], ["tree", "2"]],
)
def test_no_topology_leaves_disconnected_islands(topology: list[str]) -> None:
    """Every deterministic topology yields a single connected component.

    No island should end up isolated or split into a separate group: the whole
    population must be mutually reachable through neighbor links.
    """

    islands = make_islands(6)
    assign_topology(islands, topology)
    assert_connected(islands)
    for island in islands:
        assert island.neighbors, f"island {island.id} has no neighbors"


def test_random_leaves_no_disconnected_islands() -> None:
    """The random topology stays connected across many seeds.

    ``random`` seeds the graph with a directed ring before adding extra edges,
    so no island should ever be left disconnected regardless of the random
    draws; this checks connectivity over a range of seeds.
    """

    for seed in range(25):
        random.seed(seed)
        islands = make_islands(8)
        assign_topology(islands, ["random", "2", "4"])
        assert_connected(islands)
        for island in islands:
            assert island.neighbors, f"seed {seed}: island {island.id} has no neighbors"
