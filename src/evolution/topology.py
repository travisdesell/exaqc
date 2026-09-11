"""Assign inter-island connection topologies for the island-model search.

Each :class:`~src.evolution.island.Island` keeps a ``neighbors``
list naming the other islands it may draw genomes from for inter-island
crossover. :func:`assign_topology` populates those lists according to a named
topology -- fully connected, ring, star, 2-D mesh, n-ary tree, or a random
directed graph. Every builder mutates the islands in place and never returns a
value.
"""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evolution.island import Island


def assign_topology(
    islands: list[Island],
    topology: tuple[str, ...] = ("fully_connected",),
) -> None:
    """Sets each island's ``neighbors`` according to a connection topology.

    An island's neighbors are the other islands it may draw genomes from for
    inter-island crossover. Any neighbors already set are cleared first, so this
    is idempotent and independent of prior calls.

    Args:
        islands: The islands to arrange; their ``neighbors`` are set in place.
        topology: The topology name followed by any arguments it needs. One of:

            * ``["fully_connected"]`` (the default) -- every island connected to
              every other.
            * ``["ring"]`` -- each island connected to the previous and next,
              wrapping around.
            * ``["star"]`` -- the first island is the center, connected to every
              other; the others connect only to the center.
            * ``["2d_mesh", <x_dim>, <y_dim>]`` -- a grid, each island connected
              to its up/down/left/right neighbors (requires
              ``x_dim * y_dim == len(islands)``).
            * ``["tree", <n_children>]`` -- an n-ary tree where each node is
              connected to its parent and up to ``n_children`` children.
            * ``["random", <min_edges>, <max_edges>]`` -- a **directed** graph;
              the islands are first joined in a directed ring (guaranteeing
              every island is reachable from every other), then each island is
              given a total out-degree drawn uniformly from
              ``[min_edges, max_edges]`` by adding extra edges to randomly
              chosen islands.

    Returns:
        None. Mutates each island's ``neighbors`` list in place.

    Raises:
        ValueError: If ``topology`` names an unsupported topology, or its
            arguments are invalid (see the individual builders).
    """

    # Start from a clean slate so assignment is idempotent and independent of
    # any neighbors set by a previous call.
    for island in islands:
        island.neighbors = []

    topology_type = topology[0]
    other_args = list(topology[1:])

    if topology_type == "fully_connected":
        _connect_fully(islands)

    elif topology_type == "ring":
        _connect_ring(islands)

    elif topology_type == "star":
        _connect_star(islands)

    elif topology_type == "2d_mesh":
        _connect_2d_mesh(islands, other_args)

    elif topology_type == "tree":
        _connect_tree(islands, other_args)

    elif topology_type == "random":
        _connect_random(islands, other_args)

    else:
        raise ValueError(f"Unknown island topology specified: {topology_type}")


def _parse_int_args(
    other_args: list[str],
    names: list[str],
    topology_name: str,
) -> list[int]:
    """Validates and converts a topology's extra arguments to integers.

    Args:
        other_args: The raw string arguments that followed the topology name
            (as parsed from the command line).
        names: The expected argument names, in order; its length is the required
            number of arguments and the names are used in error messages.
        topology_name: The topology the arguments belong to, used in error
            messages.

    Returns:
        The parsed integer values, one per entry in ``names`` and in the same
        order.

    Raises:
        ValueError: If the number of arguments does not match ``names``, or any
            argument is not a non-negative integer.
    """

    if len(other_args) != len(names):
        expected = " ".join(f"<{name}>" for name in names)
        raise ValueError(
            f"{topology_name} topology requires {len(names)} additional "
            f"argument(s): {expected}"
        )

    for name, value in zip(names, other_args):
        if not value.isdigit():
            raise ValueError(
                f"{topology_name} topology requires {name} to be a non-negative "
                f"integer, but found {name}: {value}"
            )

    return [int(value) for value in other_args]


def _connect_fully(islands: list[Island]) -> None:
    """Connects every island to every other island.

    Args:
        islands: The islands to connect.

    Returns:
        None. Appends every other island to each island's ``neighbors``.
    """

    for island in islands:
        island.neighbors.extend(other for other in islands if other is not island)


def _connect_ring(islands: list[Island]) -> None:
    """Connects the islands in a ring.

    Each island is connected to the previous and next island in the list, and
    the first and last islands are connected to close the ring. For two islands
    the single connecting edge is added only once (there is no separate
    wrap-around), and for one island no edges are added.

    Args:
        islands: The islands to connect, in ring order.

    Returns:
        None. Appends the ring edges to each island's ``neighbors``.
    """

    n_islands = len(islands)

    for i in range(n_islands - 1):
        islands[i].neighbors.append(islands[i + 1])
        islands[i + 1].neighbors.append(islands[i])

    if n_islands > 2:
        # Close the ring by linking the first and last islands. With only two
        # islands they are already linked by the loop above.
        islands[0].neighbors.append(islands[-1])
        islands[-1].neighbors.append(islands[0])


def _connect_star(islands: list[Island]) -> None:
    """Connects the islands in a star around the first island.

    The first island is the center and is connected to every other island; each
    other island is connected only to the center.

    Args:
        islands: The islands to connect; ``islands[0]`` becomes the center.

    Returns:
        None. Appends the star edges to each island's ``neighbors``.
    """

    for island in islands[1:]:
        islands[0].neighbors.append(island)
        island.neighbors.append(islands[0])


def _connect_2d_mesh(islands: list[Island], other_args: list[str]) -> None:
    """Connects the islands in a 2-D mesh (grid).

    The islands are laid out row-major (island ``x * y_dim + y`` sits at grid
    position ``(x, y)``) and each island is connected to its up, down, left and
    right grid neighbors.

    Args:
        islands: The islands to connect.
        other_args: Exactly two strings, the x and y dimensions of the grid.

    Returns:
        None. Appends the mesh edges to each island's ``neighbors``.

    Raises:
        ValueError: If ``other_args`` is not two non-negative integers, or if
            ``x_dim * y_dim`` does not equal the number of islands.
    """

    x_dim, y_dim = _parse_int_args(other_args, ["x_dim", "y_dim"], "2d_mesh")

    if x_dim * y_dim != len(islands):
        raise ValueError(
            f"2d_mesh requires x_dim ({x_dim}) * y_dim ({y_dim}) == "
            f"n_islands ({len(islands)})"
        )

    # Row-major layout: island index == x * y_dim + y, so the up/down neighbors
    # are y_dim apart and the left/right neighbors are adjacent. Each edge is
    # appended from both endpoints, keeping the mesh symmetric.
    for idx, island in enumerate(islands):
        x, y = divmod(idx, y_dim)

        if x > 0:
            island.neighbors.append(islands[idx - y_dim])
        if x < x_dim - 1:
            island.neighbors.append(islands[idx + y_dim])
        if y > 0:
            island.neighbors.append(islands[idx - 1])
        if y < y_dim - 1:
            island.neighbors.append(islands[idx + 1])


def _connect_tree(islands: list[Island], other_args: list[str]) -> None:
    """Connects the islands in an n-ary tree.

    The islands form a tree stored in list order: island ``i`` is the parent of
    islands ``i * n_children + 1 .. i * n_children + n_children`` (whichever of
    those indices exist). Each parent/child pair is connected in both
    directions. The last islands in the list may have fewer than ``n_children``
    children if the tree does not divide evenly.

    Args:
        islands: The islands to connect.
        other_args: Exactly one string, the number of children per node.

    Returns:
        None. Appends the tree edges to each island's ``neighbors``.

    Raises:
        ValueError: If ``other_args`` is not a single non-negative integer.
    """

    n_children = _parse_int_args(other_args, ["n_children"], "tree")[0]
    n_islands = len(islands)

    for i in range(n_islands):
        first_child = (i * n_children) + 1

        for j in range(n_children):
            target = first_child + j

            if target < n_islands:
                islands[i].neighbors.append(islands[target])
                islands[target].neighbors.append(islands[i])


def _connect_random(islands: list[Island], other_args: list[str]) -> None:
    """Connects the islands as a random directed graph.

    The islands are first joined in a directed ring, which guarantees every
    island is reachable from every other. Each island is then given extra
    outgoing edges to randomly chosen islands so that its total out-degree is
    drawn uniformly from ``[min_edges, max_edges]``. Unlike the other
    topologies this graph is directed, and no duplicate edges are created.

    Args:
        islands: The islands to connect.
        other_args: Exactly two strings, the minimum and maximum out-degree
            (``min_edges`` and ``max_edges``) for each island.

    Returns:
        None. Appends the ring and random edges to each island's ``neighbors``.

    Raises:
        ValueError: If ``other_args`` is not two non-negative integers, if
            ``min_edges`` is less than 1, if ``max_edges`` is less than
            ``min_edges``, or if ``max_edges`` is not less than the number of
            islands.
    """

    min_edges, max_edges = _parse_int_args(
        other_args, ["min_edges", "max_edges"], "random"
    )
    n_islands = len(islands)

    if min_edges < 1:
        raise ValueError(f"random requires min_edges ({min_edges}) >= 1")
    if max_edges < min_edges:
        raise ValueError(
            f"random requires max_edges ({max_edges}) >= min_edges ({min_edges})"
        )
    if max_edges >= n_islands:
        raise ValueError(
            f"random requires max_edges ({max_edges}) < n_islands ({n_islands})"
        )

    # Seed a directed ring so there is a path from any island to any other.
    for i in range(n_islands - 1):
        islands[i].neighbors.append(islands[i + 1])
    islands[-1].neighbors.append(islands[0])

    # The ring already provides one outgoing edge per island, so draw the number
    # of *additional* edges from [min_edges - 1, max_edges - 1].
    for i, island in enumerate(islands):
        n_extra = random.randint(min_edges - 1, max_edges - 1)

        # Candidates are all other islands not already connected, so the ring
        # neighbor is never re-picked and no duplicate edges are created.
        already_connected = set(island.neighbors)
        candidates = [
            other
            for j, other in enumerate(islands)
            if j != i and other not in already_connected
        ]

        for target in random.sample(candidates, min(n_extra, len(candidates))):
            island.neighbors.append(target)
