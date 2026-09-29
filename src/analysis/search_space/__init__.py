"""Projections of a run's genomes into a low-dimensional search space.

The package has three layers, each extensible on its own:

* :mod:`~src.analysis.search_space.unitary` -- the unitary a genome's evolved
  gates implement.
* :mod:`~src.analysis.search_space.distances` -- :class:`DistanceMetric` and its
  registry: how far apart two genomes are (the unitary Fubini–Study angle, the
  structural Jaccard distance, ...).
* :mod:`~src.analysis.search_space.projections` -- :class:`Projection` and its
  registry: how a distance matrix becomes 2-D or 3-D coordinates (classical MDS,
  metric MDS, t-SNE, ...).

:func:`build_search_space` ties them together over a run's archive; the EXAQC
dashboard serves its result as the run's search-space view.
"""

from src.analysis.search_space.distances import (
    DISTANCE_METRICS,
    DistanceMetric,
    FubiniStudyDistance,
    JaccardDistance,
    UnsupportedGenome,
    get_distance,
    register_distance,
)
from src.analysis.search_space.projections import (
    PROJECTIONS,
    Projection,
    ProjectionResult,
    get_projection,
    register_projection,
)
from src.analysis.search_space.search_space import (
    DEFAULT_METRIC,
    DEFAULT_PROJECTION,
    SearchSpaceBuilder,
    best_path,
    build_search_space,
    options,
)

__all__ = [
    "DEFAULT_METRIC",
    "DEFAULT_PROJECTION",
    "DISTANCE_METRICS",
    "PROJECTIONS",
    "DistanceMetric",
    "FubiniStudyDistance",
    "JaccardDistance",
    "Projection",
    "ProjectionResult",
    "SearchSpaceBuilder",
    "UnsupportedGenome",
    "best_path",
    "build_search_space",
    "get_distance",
    "get_projection",
    "options",
    "register_distance",
    "register_projection",
]
