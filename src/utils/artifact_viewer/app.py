"""The web application serving the dashboard, its JSON API and its MCP endpoint.

:mod:`src.utils.artifact_viewer.server` holds the data layer: :class:`~src.utils
.artifact_viewer.server.ArtifactViewer` answers every question about the served
runs and returns plain JSON-safe dicts, knowing nothing about HTTP. This module
is the transport around it -- a Starlette application whose routes mirror those
payload methods one for one, plus the static files the browser loads and, when
one is given, an MCP server mounted at ``/mcp`` so agents reach the same runs
over the same port.

Endpoints are written as ordinary synchronous functions on purpose: Starlette
runs those in a worker thread, which is what the archive reads and the image
renderer need, and matches how the threaded server this replaced behaved.
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
import urllib.parse
import webbrowser
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import uvicorn
from loguru import logger
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

from src.utils.artifact_viewer.server import (
    ARCHIVE_FILENAME,
    DEFAULT_ANCESTRY_DEPTH,
    IMAGE_KINDS,
    MAX_ANCESTRY_DEPTH,
    RESCAN_INTERVAL_SECONDS,
    ArtifactViewer,
    RenderService,
    RunRegistry,
    json_safe,
    query_int,
)

#: Directory holding the dashboard's HTML, JavaScript, CSS and vendored uPlot.
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"

#: Content types for the static files served from :data:`STATIC_DIRECTORY`.
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}

#: Static file names that may be requested: one path segment, no directories, so
#: a request can never escape the static directory.
_STATIC_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def _query(request: Request) -> dict[str, str]:
    """Flattens a request's query parameters, keeping the last of any repeats.

    Args:
        request: The request whose query string is read.

    Returns:
        The parameters, keyed by name.
    """

    return {
        name: request.query_params.getlist(name)[-1] for name in request.query_params
    }


def _json(payload: Any, status: int = 200) -> Response:
    """Builds a JSON response the browser is not allowed to cache.

    Args:
        payload: The value to encode (see :func:`~src.utils.artifact_viewer
            .server.json_safe`).
        status: The HTTP status code.

    Returns:
        The response.
    """

    body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
    return Response(
        body,
        status_code=status,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _bytes(
    body: bytes,
    content_type: str,
    filename: str | None = None,
    cache: bool = False,
) -> Response:
    """Builds a binary response, optionally offered as a download.

    Args:
        body: The response bytes.
        content_type: The ``Content-Type`` header.
        filename: When given, the response is offered as a download under this
            name.
        cache: Whether the browser may cache the response.

    Returns:
        The response.
    """

    headers = {"Cache-Control": "max-age=3600" if cache else "no-store"}
    if filename is not None:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(body, media_type=content_type, headers=headers)


def create_app(viewer: ArtifactViewer, mcp_server: Any | None = None) -> Starlette:
    """Builds the application serving a set of runs.

    Args:
        viewer: The data layer answering every request.
        mcp_server: An ``MCPServer`` to mount at ``/mcp``, or ``None`` to serve
            only the dashboard. Its session manager is run for the application's
            lifetime, which the streamable-HTTP transport requires.

    Returns:
        The configured Starlette application.
    """

    def static_file(name: str) -> Response:
        """Serves one file from the static directory.

        Args:
            name: The file name (a single path segment).

        Returns:
            The file's response.

        Raises:
            KeyError: If there is no such static file.
        """

        path = STATIC_DIRECTORY / name
        if not _STATIC_NAME.fullmatch(name) or not path.is_file():
            raise KeyError(f"There is no static file {name!r}.")
        return _bytes(
            path.read_bytes(),
            _CONTENT_TYPES.get(path.suffix, "text/plain; charset=utf-8"),
        )

    def index(request: Request) -> Response:
        """Serves the dashboard page."""
        return static_file("index.html")

    def static(request: Request) -> Response:
        """Serves a file the dashboard page loads."""
        return static_file(request.path_params["name"])

    def api_runs(request: Request) -> Response:
        """Lists every served run."""
        return _json(viewer.runs_payload())

    def api_groups(request: Request) -> Response:
        """Compares groups of runs (``metric`` and ``conf`` parameters)."""
        query = _query(request)
        return _json(
            viewer.groups_payload(
                query.get("metric") or None, query.get("conf") or "std"
            )
        )

    def api_insertion_rates(request: Request) -> Response:
        """Tabulates insertion rates for a ``run``, a ``group`` or every group."""
        query = _query(request)
        run_index = query_int(query, "run", -1, minimum=0) if query.get("run") else None
        return _json(
            viewer.insertion_rates_payload(
                run_index=run_index, group=query.get("group") or None
            )
        )

    def api_run(request: Request) -> Response:
        """Describes one run."""
        return _json(viewer.run_payload(request.path_params["run"]))

    def api_genomes(request: Request) -> Response:
        """Lists a page of a run's genomes."""
        return _json(
            viewer.genomes_payload(request.path_params["run"], _query(request))
        )

    def api_points(request: Request) -> Response:
        """Returns a run's chart points (``y`` parameter)."""
        query = _query(request)
        return _json(
            viewer.points_payload(request.path_params["run"], query.get("y") or "loss")
        )

    def api_genealogy(request: Request) -> Response:
        """Returns a run's points and parent links (``y`` parameter)."""
        query = _query(request)
        return _json(
            viewer.genealogy_payload(
                request.path_params["run"], query.get("y") or "loss"
            )
        )

    def api_history(request: Request) -> Response:
        """Returns a run's search progress (optional ``metric`` parameter)."""
        return _json(
            viewer.history_payload(
                request.path_params["run"], _query(request).get("metric") or None
            )
        )

    def api_operators(request: Request) -> Response:
        """Returns a run's operator insert-type counts."""
        return _json(viewer.operators_payload(request.path_params["run"]))

    def api_compare(request: Request) -> Response:
        """Compares genomes ``a`` and ``b`` of a run.

        Raises:
            ValueError: If either genome is missing from the query.
        """

        query = _query(request)
        if not query.get("a") or not query.get("b"):
            raise ValueError(
                "Give the two genomes to compare as ?a=<number>&b=<number>."
            )
        return _json(
            viewer.compare_payload(
                request.path_params["run"],
                query_int(query, "a", 0, minimum=0),
                query_int(query, "b", 0, minimum=0),
            )
        )

    def api_genome(request: Request) -> Response:
        """Returns one genome's details."""
        return _json(
            viewer.genome_payload(
                request.path_params["run"], request.path_params["genome"]
            )
        )

    def api_genome_json(request: Request) -> Response:
        """Serves a genome's JSON as a download."""
        genome_number = request.path_params["genome"]
        body = viewer.genome_json(request.path_params["run"], genome_number)
        return _bytes(body, "application/json", filename=f"genome_{genome_number}.json")

    def api_genome_image(request: Request) -> Response:
        """Serves a genome's rendered diagram or training plot.

        Raises:
            KeyError: If the image kind is unknown or the image could not be
                drawn.
        """

        kind = request.path_params["kind"]
        if kind not in IMAGE_KINDS:
            raise KeyError(f"There is no {kind!r} image.")
        image = viewer.image(
            request.path_params["run"], request.path_params["genome"], kind
        )
        if image is None:
            if kind == "training":
                raise KeyError("This genome recorded no training metrics to plot.")
            raise KeyError("This genome's diagram could not be drawn.")
        return _bytes(image, "image/png", cache=True)

    def api_ancestry(request: Request) -> Response:
        """Returns a genome's ancestry graph (``depth`` parameter)."""
        depth = query_int(
            _query(request),
            "depth",
            DEFAULT_ANCESTRY_DEPTH,
            minimum=1,
            maximum=MAX_ANCESTRY_DEPTH,
        )
        return _json(
            viewer.ancestry_payload(
                request.path_params["run"], request.path_params["genome"], depth
            )
        )

    async def annotation_body(
        request: Request, required: bool = True
    ) -> dict[str, Any]:
        """Reads an annotation write's JSON body, refusing writes from other sites.

        A dashboard running on someone's machine can be sent a write by any web
        page they happen to have open, as a plain cross-site form post. Such a
        post cannot carry a JSON content type without a CORS preflight, which the
        dashboard never grants, and browsers name the page a request came from,
        so a write whose ``Origin`` is another site is refused outright.

        Args:
            request: The write request.
            required: Whether a body must be sent; removing a tag needs none.

        Returns:
            The decoded body, or an empty dict when none was needed or sent.

        Raises:
            PermissionError: If the request came from another site.
            ValueError: If a required body is missing, is not JSON, or is not a
                JSON object.
        """

        origin = request.headers.get("origin")
        if origin is not None and urllib.parse.urlsplit(
            origin
        ).netloc != request.headers.get("host"):
            raise PermissionError(
                "Notes and tags can only be written from the dashboard's own page."
            )
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        if content_type.lower() != "application/json":
            if required:
                raise ValueError(
                    "Send the annotation as a JSON body (Content-Type: application/json)."
                )
            return {}
        try:
            body = await request.json()
        except ValueError as error:
            raise ValueError(f"The request body is not valid JSON: {error}") from error
        if not isinstance(body, dict):
            raise ValueError("An annotation's JSON body must be an object.")
        return body

    def text_field(
        body: dict[str, Any], name: str, required: bool = True
    ) -> str | None:
        """Reads a string field from an annotation's body.

        Args:
            body: The decoded body.
            name: The field to read.
            required: Whether the field must be present.

        Returns:
            The string, or ``None`` when an optional field was left out.

        Raises:
            ValueError: If the field is missing when required, or is not a string.
        """

        value = body.get(name)
        if value is None and not required:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be given as a string.")
        return value

    def genome_field(body: dict[str, Any]) -> int | None:
        """Reads the optional genome an annotation's body is about.

        Args:
            body: The decoded body.

        Returns:
            The genome number, or ``None`` for an annotation about the run.

        Raises:
            ValueError: If a genome number is given but is not a whole number of
                zero or more.
        """

        value = body.get("genome_number")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("genome_number must be a whole number of zero or more.")
        return value

    def api_annotations(request: Request) -> Response:
        """Returns a run's notes and tags, or one genome's.

        The ``genome``, ``tag`` and ``include_removed`` parameters narrow what is
        returned.
        """
        query = _query(request)
        genome = (
            query_int(query, "genome", 0, minimum=0) if query.get("genome") else None
        )
        return _json(
            viewer.annotations_payload(
                request.path_params["run"],
                genome,
                query.get("tag") or None,
                query.get("include_removed") == "1",
            )
        )

    async def api_add_note(request: Request) -> Response:
        """Records a note about the run, or about the genome its body names."""
        viewer.require_annotations()
        body = await annotation_body(request)
        note = await run_in_threadpool(
            viewer.add_note,
            request.path_params["run"],
            text_field(body, "text"),
            "dashboard",
            text_field(body, "author", required=False),
            genome_field(body),
        )
        return _json(note, status=201)

    async def api_add_tag(request: Request) -> Response:
        """Tags a genome; a tag it already carries is left as it is."""
        viewer.require_annotations()
        body = await annotation_body(request)
        tag = await run_in_threadpool(
            viewer.add_tag,
            request.path_params["run"],
            request.path_params["genome"],
            text_field(body, "tag"),
            "dashboard",
            text_field(body, "author", required=False),
        )
        return _json(tag, status=201 if tag["created"] else 200)

    async def api_remove_tag(request: Request) -> Response:
        """Removes a tag from a genome, keeping the record that it applied."""
        viewer.require_annotations()
        body = await annotation_body(request, required=False)
        tag = await run_in_threadpool(
            viewer.remove_tag,
            request.path_params["run"],
            request.path_params["genome"],
            request.path_params["tag"],
            "dashboard",
            text_field(body, "author", required=False),
        )
        return _json(tag)

    def forbidden(request: Request, exception: Exception) -> Response:
        """Turns a write that is not allowed into a 403."""
        return _error(request, 403, str(exception))

    def not_found(request: Request, exception: Exception) -> Response:
        """Turns a missing run, genome or file into a 404."""
        message = str(exception.args[0]) if exception.args else "Not found."
        return _error(request, 404, message)

    def bad_request(request: Request, exception: Exception) -> Response:
        """Turns an invalid parameter into a 400."""
        return _error(request, 400, str(exception))

    def _error(request: Request, status: int, message: str) -> Response:
        """Renders an error as JSON for API requests and plain text otherwise."""
        if request.url.path.startswith("/api/"):
            return _json({"error": message}, status=status)
        return Response(
            message,
            status_code=status,
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    routes = [
        Route("/", index),
        Route("/index.html", index),
        Route("/static/{name}", static),
        Route("/api/runs", api_runs),
        Route("/api/groups", api_groups),
        Route("/api/insertion_rates", api_insertion_rates),
        Route("/api/runs/{run:int}", api_run),
        Route("/api/runs/{run:int}/genomes", api_genomes),
        Route("/api/runs/{run:int}/points", api_points),
        Route("/api/runs/{run:int}/genealogy", api_genealogy),
        Route("/api/runs/{run:int}/history", api_history),
        Route("/api/runs/{run:int}/operators", api_operators),
        Route("/api/runs/{run:int}/compare", api_compare),
        Route("/api/runs/{run:int}/genomes/{genome:int}.json", api_genome_json),
        Route("/api/runs/{run:int}/genomes/{genome:int}/{kind}.png", api_genome_image),
        Route("/api/runs/{run:int}/genomes/{genome:int}/ancestry", api_ancestry),
        Route("/api/runs/{run:int}/annotations", api_annotations),
        Route("/api/runs/{run:int}/notes", api_add_note, methods=["POST"]),
        Route(
            "/api/runs/{run:int}/genomes/{genome:int}/tags",
            api_add_tag,
            methods=["POST"],
        ),
        Route(
            "/api/runs/{run:int}/genomes/{genome:int}/tags/{tag}",
            api_remove_tag,
            methods=["DELETE"],
        ),
        Route("/api/runs/{run:int}/genomes/{genome:int}", api_genome),
    ]

    lifespan = None
    if mcp_server is not None:
        # Build the transport app first: session_manager is only available afterwards.
        # Mounted at the root with the transport owning "/mcp", so that path answers
        # directly instead of redirecting to "/mcp/" on every call.
        routes.append(
            Mount("", app=mcp_server.streamable_http_app(streamable_http_path="/mcp"))
        )

        @contextlib.asynccontextmanager
        async def mcp_lifespan(app: Starlette) -> AsyncIterator[None]:
            """Runs the MCP session manager for as long as the application serves."""
            async with mcp_server.session_manager.run():
                yield

        lifespan = mcp_lifespan

    return Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={
            KeyError: not_found,
            ValueError: bad_request,
            PermissionError: forbidden,
        },
    )


def serve(
    runs: list[str] | None = None,
    directory: str | None = None,
    groups: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
    render_processes: int = 1,
    rescan_interval: float = RESCAN_INTERVAL_SECONDS,
    mcp: bool = True,
    allow_annotations: bool = False,
) -> None:
    """Serves the dashboard until interrupted.

    Args:
        runs: Run output directories (or their archive files) to serve; a run
            whose archive has not been written yet (even one whose directory
            does not exist yet) appears once it is.
        directory: A directory to watch instead: every run below it is served,
            including runs started while the dashboard is running.
        groups: Substrings grouping runs for comparison.
        host: The address to listen on.
        port: The port to listen on (``0`` picks a free port).
        open_browser: Whether to open the dashboard in a web browser.
        render_processes: Worker processes rendering images.
        rescan_interval: The least time between scans for new runs, in seconds.
        mcp: Whether to serve the MCP interface at ``/mcp`` for agents.
        allow_annotations: Whether the dashboard and its MCP interface may write
            notes and tags, kept in each run's ``annotations.sqlite``. Archives are
            never written either way.

    Returns:
        None. Runs until interrupted with Ctrl+C.

    Raises:
        ValueError: If both or neither of ``runs`` and ``directory`` are given.
        FileNotFoundError: If the watched directory does not exist.
        NotADirectoryError: If the watched path is not a directory.
        OSError: If the server cannot listen on ``host:port``.
    """

    registry = RunRegistry(
        run_directories=runs,
        watch_directory=directory,
        groups=groups,
        rescan_interval=rescan_interval,
    )

    # Bind first, so the address is reported (and rejected) before anything starts,
    # and so port 0 can be logged as the port it actually picked.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
    except OSError:
        listener.close()
        raise
    bound_port = listener.getsockname()[1]

    renderer = RenderService(processes=render_processes)
    mcp_server = None
    if mcp:
        from src.utils.artifact_viewer.mcp_app import build_mcp_server

        mcp_server = build_mcp_server(
            registry,
            renderer,
            f"http://{host}:{bound_port}",
            allow_annotations=allow_annotations,
        )

    application = create_app(
        ArtifactViewer(registry, renderer, allow_annotations=allow_annotations),
        mcp_server,
    )
    url = f"http://{host}:{bound_port}/"

    if registry.watch_directory is not None:
        logger.info(
            "Watching {} for runs ({} found so far), serving at {} -- press Ctrl+C to stop.",
            registry.watch_directory,
            len(registry.runs),
            url,
        )
    else:
        logger.info(
            "Serving {} run(s) at {} -- press Ctrl+C to stop.", len(registry.runs), url
        )
        waiting = registry.source()["waiting"]
        if waiting:
            logger.info(
                "Waiting for {} to be written in: {}",
                ARCHIVE_FILENAME,
                ", ".join(waiting),
            )
    if mcp:
        logger.info("Serving the MCP interface at {}mcp for agents.", url)
    if allow_annotations:
        logger.info(
            "Notes and tags can be written, beside each run's archive in {}.",
            "annotations.sqlite",
        )
        if host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "Annotations are writable by anyone who can reach {} -- serve on "
                "127.0.0.1 and forward the port to limit who can write them.",
                url,
            )
    if open_browser:
        webbrowser.open(url)

    server = uvicorn.Server(uvicorn.Config(application, log_level="warning"))
    try:
        server.run(sockets=[listener])
    except KeyboardInterrupt:
        logger.info("Stopping the dashboard.")
    finally:
        listener.close()
        renderer.close()
