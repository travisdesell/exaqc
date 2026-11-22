"""Projections of a run's genomes into a low-dimensional search space.

The package has three layers, each extensible on its own:

* :mod:`~src.analysis.search_space.unitary` -- the unitary a genome's evolved
  gates implement.
* :mod:`~src.analysis.search_space.distances` -- :class:`DistanceMetric` and its
  registry: how far apart two genomes are (how differently they classify the
  run's data, how differently their circuits act on the measured observables,
  the unitary Fubini–Study angle, the structural Jaccard distance, ...).
* :mod:`~src.analysis.search_space.projections` -- :class:`Projection` and its
  registry: how a distance matrix becomes 2-D or 3-D coordinates (classical MDS,
  metric MDS, t-SNE, ...).

:func:`build_search_space` ties them together over a run's archive; the EXAQC
dashboard serves its result as the run's search-space view.
"""

from src.analysis.search_space.distances import (
    DISTANCE_METRICS,
    BehaviourDistance,
    DistanceMetric,
    FubiniStudyDistance,
    JaccardDistance,
    ReadoutOperatorDistance,
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
    CLASSIFICATION_METRIC,
    DEFAULT_METRIC,
    DEFAULT_PROJECTION,
    SearchSpaceBuilder,
    best_path,
    build_search_space,
    default_metric,
    island_separation,
    options,
    silhouette,
)

__all__ = [
    "CLASSIFICATION_METRIC",
    "DEFAULT_METRIC",
    "DEFAULT_PROJECTION",
    "DISTANCE_METRICS",
    "PROJECTIONS",
    "BehaviourDistance",
    "DistanceMetric",
    "FubiniStudyDistance",
    "JaccardDistance",
    "Projection",
    "ProjectionResult",
    "ReadoutOperatorDistance",
    "SearchSpaceBuilder",
    "UnsupportedGenome",
    "best_path",
    "build_search_space",
    "default_metric",
    "get_distance",
    "get_projection",
    "island_separation",
    "options",
    "register_distance",
    "register_projection",
    "silhouette",
]
