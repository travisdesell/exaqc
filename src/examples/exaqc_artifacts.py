"""Browse EXAQC runs -- their genomes, ancestry and progress -- in a web page.

Starts a local web server over one or more runs' ``genomes.sqlar`` archives and
serves a viewer for them. A run can be browsed while its search is still
running, and a genome's architecture diagram and training plot are drawn on
demand, so the search itself never has to draw them::

    python3 -m src.examples.exaqc_artifacts ./artifacts/iris
    python3 -m src.examples.exaqc_artifacts ./2026_ppsn_exaqc/classification --groups iris seeds wine

Then open the printed address (http://127.0.0.1:8000/ by default). To look at a
run on a remote machine, start the viewer there and forward its port, e.g.
``ssh -L 8000:localhost:8000 cluster``.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from src.utils.artifact_viewer.server import serve


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for the artifact viewer.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """

    parser = argparse.ArgumentParser(
        description="Browse EXAQC runs -- their genomes, ancestry and progress -- in a local web page."
    )

    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Run output directories or genomes.sqlar archives to view. Directories are searched "
            "recursively, so a directory holding many runs can be given."
        ),
    )

    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Substrings grouping runs for comparison: a run joins every group whose substring "
            "appears in its path (as with the analysis scripts' --groups)."
        ),
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Address to serve on. The default only accepts connections from this machine; view a "
            "remote run by forwarding the port over SSH."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (0 picks a free port).",
    )

    parser.add_argument(
        "--open_browser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the viewer in a web browser once it is running.",
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="""One of the 5 default logging levels for showing on terminal. Pick DEBUG to show everything.""",
    )

    return parser


def main() -> None:
    """Starts the artifact viewer and serves it until interrupted.

    Returns:
        None.
    """

    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.logging_level)

    try:
        serve(
            runs=args.runs,
            groups=args.groups,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
    except OSError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
