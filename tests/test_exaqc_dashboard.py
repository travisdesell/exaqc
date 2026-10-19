"""Tests for the EXAQC dashboard (``python3 -m src.examples.exaqc_dashboard``).

The dashboard serves a single-page app and a read-only JSON API over runs'
``genomes.sqlar`` archives, given as run directories or found by watching a
directory. These tests build small archives, start the real HTTP server on a
free port, and check the parser, finding and grouping runs (including runs
written after the dashboard started), every API route, the insertion-rate tables
and their LaTeX, image rendering and caching, and that static files cannot
escape their directory.
"""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from src.examples import exaqc_dashboard
from src.utils.artifact_viewer import server
from src.utils.artifact_viewer.app import create_app
from src.utils.artifact_viewer.server import (
    ArtifactViewer,
    RenderService,
    RunRegistry,
    assign_groups,
    compare_gates,
    find_archives,
    insertion_rates_latex,
)
from src.utils.genome_archive import ARCHIVE_FILENAME, GenomeArchive

#: The first eight bytes of any PNG file (the PNG signature).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class FakeGenome:
    """Stand-in genome that serializes to a fixed dict."""

    def __init__(
        self,
        genome_number: int,
        loss: float,
        target_metric: float,
        parents: list[int],
        generated_by: list[str],
        insert_type: str = "inserted",
        gates: list[dict[str, Any]] | None = None,
        task: str = "classification",
        task_target: str = "iris",
    ) -> None:
        """Creates the fake genome.

        Args:
            genome_number: The genome's number.
            loss: Its ``fitness["loss"]``.
            target_metric: Its ``fitness["target_metric"]``.
            parents: Its parent genome numbers.
            generated_by: The operators that generated it.
            insert_type: How it was inserted.
            gates: Its serialized gates.
            task: The task it was evolved for.
            task_target: The dataset or environment it was evolved on.
        """

        self.genome_number = genome_number
        self.serialized = {
            "genome_number": genome_number,
            "task": task,
            "task_target": task_target,
            "target": "pennylane",
            "fitness": {"loss": loss, "target_metric": target_metric},
            "hyperparameters": {"epochs": genome_number},
            "metadata": {
                "parent_genomes": parents,
                "generated_by": generated_by,
                "insert_type": insert_type,
            },
            "gates": gates or [],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializes the fake genome.

        Returns:
            The serialized genome.
        """

        return self.serialized


def gate(
    innovation: int, method: str = "rx", theta: float = 0.1, enabled: bool = True
) -> dict[str, Any]:
    """Builds a serialized gate.

    Args:
        innovation: The gate's innovation number.
        method: The gate method.
        theta: Its ``theta`` parameter.
        enabled: Whether it is enabled.

    Returns:
        The serialized gate.
    """

    return {
        "innovation_number": innovation,
        "method_name": method,
        "qubits": [["input", 0]],
        "parameters": {"theta": theta},
        "depth": innovation / 10,
        "enabled": enabled,
        "target": "pennylane",
    }


def build_run(
    directory,
    genomes: list[FakeGenome],
    history: bool = True,
    islands: list[int | None] | None = None,
    species: list[int | None] | None = None,
    run_info: dict[str, Any] | None = None,
) -> str:
    """Writes a run directory holding an archive, and its search progress.

    Args:
        directory: The run directory to create.
        genomes: The genomes to store, in insertion order.
        history: Whether to record the population changes a search would, which
            is what the progress charts are recomputed from.
        islands: The island each genome was inserted into, in the same order,
            for a run that used islands.
        species: The species each genome was assigned to, in the same order,
            for a run that used speciation.
        run_info: Further ``run_info`` values to record, such as an island
            topology or speciation config.

    Returns:
        The run directory, as a string.
    """

    with GenomeArchive.create(str(directory)) as archive:
        info = {
            "task": "classification",
            "task_target": "iris",
            "population_strategy": "SteadyStatePopulation",
            **(run_info or {}),
        }
        archive.set_run_info(**info)
        for insertion, genome in enumerate(genomes, start=1):
            island = islands[insertion - 1] if islands else None
            species_id = species[insertion - 1] if species else None
            archive.add_genome(
                genome, insertion=insertion, island=island, species=species_id
            )

        if history:
            # Three steps, growing the population one genome at a time, so a
            # run's series has the same shape a real search would produce.
            for step in range(1, 4):
                archive.record_population(step=step, population=genomes[:step])

    return str(directory)


def standard_genomes() -> list[FakeGenome]:
    """Builds a small lineage: two mutations from the seed, then crossovers.

    Returns:
        Genomes 1-4, where 3 is a crossover of 1 and 2, and 4 mutates 3.
    """

    return [
        FakeGenome(
            1,
            loss=0.5,
            target_metric=0.6,
            parents=[0],
            generated_by=["add_gate"],
            gates=[gate(1), gate(2)],
        ),
        FakeGenome(
            2,
            loss=0.4,
            target_metric=0.7,
            parents=[0],
            generated_by=["add_gate", "clone"],
            insert_type="global_best",
            gates=[gate(1, theta=0.2), gate(3)],
        ),
        FakeGenome(
            3,
            loss=0.3,
            target_metric=0.8,
            parents=[1, 2],
            generated_by=["n_ary_crossover"],
            insert_type="global_best",
            gates=[gate(1), gate(3)],
        ),
        FakeGenome(
            4,
            loss=0.9,
            target_metric=0.1,
            parents=[3],
            generated_by=["qubit_swap"],
            insert_type="discarded",
            gates=[gate(1)],
        ),
    ]


@contextmanager
def _running_server(
    registry: RunRegistry, allow_annotations: bool = False
) -> Iterator[str]:
    """Serves a registry's runs with uvicorn on a free port, for a block's duration.

    The socket is bound here rather than by uvicorn so the port is known before
    the server starts.

    Args:
        registry: The runs to serve.
        allow_annotations: Whether the dashboard may write notes and tags.

    Yields:
        The base URL, e.g. ``http://127.0.0.1:54321``.
    """

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    application = create_app(
        ArtifactViewer(
            registry, RenderService(processes=0), allow_annotations=allow_annotations
        )
    )
    running = uvicorn.Server(uvicorn.Config(application, log_level="error"))
    thread = threading.Thread(
        target=running.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not running.started and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        running.should_exit = True
        thread.join(timeout=10)
        listener.close()


@pytest.fixture
def viewer_url(tmp_path, monkeypatch) -> Iterator[str]:
    """Serves two runs (grouped as ``iris``) and yields the server's base URL.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to replace image rendering with a fake.

    Yields:
        The base URL, e.g. ``http://127.0.0.1:54321``.
    """

    rendered: list[tuple[str, int, str]] = []

    def fake_render(archive_path: str, genome_number: int, kind: str) -> bytes | None:
        """Pretends to render: diagrams are PNG bytes, training plots are missing."""
        rendered.append((archive_path, genome_number, kind))
        with GenomeArchive.open_readonly(archive_path) as reader:
            reader.get_genome_dict(genome_number)
        return _PNG_MAGIC + b"fake" if kind == "diagram" else None

    monkeypatch.setattr(server, "render_genome_image", fake_render)

    build_run(tmp_path / "runs" / "iris_1", standard_genomes())
    build_run(tmp_path / "runs" / "iris_2", standard_genomes()[:2])

    registry = RunRegistry(
        run_directories=[
            str(tmp_path / "runs" / "iris_1"),
            str(tmp_path / "runs" / "iris_2"),
        ],
        groups=["iris"],
    )
    with _running_server(registry) as url:
        yield url


@contextmanager
def serving(registry: RunRegistry, allow_annotations: bool = False) -> Iterator[str]:
    """Serves a registry's runs on a free port for the duration of a block.

    Args:
        registry: The runs to serve.
        allow_annotations: Whether the dashboard may write notes and tags.

    Yields:
        The base URL, e.g. ``http://127.0.0.1:54321``.
    """

    with _running_server(registry, allow_annotations) as url:
        yield url


def _headers(response: Any) -> dict[str, str]:
    """Collects a response's headers under lower-case names.

    HTTP header names are case-insensitive and servers choose their own casing,
    so tests compare against lower-case names rather than one server's spelling.

    Args:
        response: The response (or error) whose headers are read.

    Returns:
        The headers, keyed by lower-case name.
    """

    return {name.lower(): value for name, value in response.headers.items()}


def get(url: str) -> tuple[int, dict[str, str], bytes]:
    """Fetches a URL, returning error responses rather than raising.

    Args:
        url: The URL to fetch.

    Returns:
        The status code, the headers (keyed by lower-case name) and the body.
    """

    try:
        with urllib.request.urlopen(url) as response:
            return response.status, _headers(response), response.read()
    except urllib.error.HTTPError as error:
        return error.code, _headers(error), error.read()


def send(
    url: str,
    method: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    """Sends a write request with a JSON body, returning error responses too.

    Args:
        url: The URL to send to.
        method: The HTTP method, e.g. ``POST`` or ``DELETE``.
        body: The JSON body to send, if any.
        headers: Extra request headers, such as an ``Origin``.

    Returns:
        The status code and the decoded JSON response, or ``None`` when the
        response is not JSON.
    """

    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, None


def get_json(url: str) -> Any:
    """Fetches and decodes a successful JSON API response.

    Args:
        url: The URL to fetch.

    Returns:
        The decoded JSON.
    """

    status, _, body = get(url)
    assert status == 200, body
    return json.loads(body)


def test_parser_defaults() -> None:
    """The dashboard's arguments and defaults match the documentation."""

    args = exaqc_dashboard.build_parser().parse_args(
        ["--runs", "runs/iris_1", "runs/iris_2"]
    )

    assert args.runs == ["runs/iris_1", "runs/iris_2"]
    assert args.directory is None
    assert args.groups is None
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.open_browser is False
    assert args.allow_annotations is False
    assert args.logging_level == "INFO"

    watching = exaqc_dashboard.build_parser().parse_args(["--directory", "runs"])
    assert watching.directory == "runs"
    assert watching.runs is None

    # one of --runs and --directory is required, but not both (and no positional runs)
    for arguments in ([], ["--runs", "a", "--directory", "b"], ["runs/iris"]):
        with pytest.raises(SystemExit):
            exaqc_dashboard.build_parser().parse_args(arguments)


def test_registry_serves_given_run_directories_as_their_archives_appear(
    tmp_path: Path,
) -> None:
    """Given run directories are served, and one not written yet joins once it is.

    Runs are named from the directories' common parent, an archive file can be
    given for its directory, and a directory whose search has not written its
    archive yet (or not even created the directory) is reported as waiting,
    then appended after the runs already found once its archive appears.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    build_run(tmp_path / "exp" / "wine_1", standard_genomes()[:1], history=False)
    build_run(
        tmp_path / "exp" / "nested" / "wine_2", standard_genomes()[:1], history=False
    )
    (tmp_path / "exp" / "wine_3").mkdir()

    registry = RunRegistry(
        run_directories=[
            str(tmp_path / "exp" / "wine_3"),
            str(tmp_path / "exp" / "nested" / "wine_2" / ARCHIVE_FILENAME),
            str(tmp_path / "exp" / "wine_1"),
            str(tmp_path / "exp" / "wine_1"),
        ],
        rescan_interval=3600,
    )

    assert [(run.index, run.name) for run in registry.runs] == [
        (0, "nested/wine_2"),
        (1, "wine_1"),
    ]
    assert registry.source() == {
        "directory": None,
        "waiting": [str(tmp_path / "exp" / "wine_3")],
    }

    build_run(tmp_path / "exp" / "wine_3", standard_genomes()[:1], history=False)
    assert registry.refresh() is False  # the last scan was too recent
    assert registry.refresh(force=True) is True
    assert [(run.index, run.name) for run in registry.runs] == [
        (0, "nested/wine_2"),
        (1, "wine_1"),
        (2, "wine_3"),
    ]
    assert registry.source()["waiting"] == []

    # a run directory that does not exist yet (its search has not started) is waited for too
    later = RunRegistry(
        run_directories=[str(tmp_path / "later_run")], rescan_interval=3600
    )
    assert later.runs == []
    assert later.source()["waiting"] == [str(tmp_path / "later_run")]
    build_run(tmp_path / "later_run", standard_genomes()[:1], history=False)
    assert later.refresh(force=True) is True
    assert [run.name for run in later.runs] == ["later_run"]

    with pytest.raises(ValueError):
        RunRegistry(
            run_directories=[str(tmp_path / "exp" / "wine_1")],
            watch_directory=str(tmp_path / "exp"),
        )
    with pytest.raises(ValueError):
        RunRegistry()


def test_registry_watches_a_directory_for_new_runs(tmp_path: Path) -> None:
    """Every run below a watched directory is served, including runs added later.

    Runs are found at any depth and named relative to the watched directory,
    legacy ``all_genomes`` directories are not searched, and a run added later
    is appended (earlier runs keep their indexes) and joins its groups.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    watched = tmp_path / "experiments"
    build_run(watched / "wine_i30_2", standard_genomes()[:1], history=False)
    build_run(watched / "deep" / "wine_i10_1", standard_genomes()[:1], history=False)
    # an archive inside a legacy per-genome directory is not a run
    build_run(
        watched / "legacy" / "all_genomes" / "stray",
        standard_genomes()[:1],
        history=False,
    )

    registry = RunRegistry(
        watch_directory=str(watched), groups=["i30"], rescan_interval=3600
    )
    assert [run.name for run in registry.runs] == ["deep/wine_i10_1", "wine_i30_2"]
    assert registry.groups == {"i30": [1]}
    assert registry.source() == {"directory": str(watched), "waiting": []}

    build_run(watched / "wine_i30_1", standard_genomes()[:1], history=False)
    assert registry.refresh(force=True) is True
    assert [(run.index, run.name) for run in registry.runs] == [
        (0, "deep/wine_i10_1"),
        (1, "wine_i30_2"),
        (2, "wine_i30_1"),
    ]
    assert registry.groups == {"i30": [1, 2]}
    assert registry.run(2).groups == ["i30"]
    assert len(find_archives(str(watched))) == 3

    with pytest.raises(KeyError):
        registry.run(3)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert RunRegistry(watch_directory=str(empty)).runs == []
    with pytest.raises(FileNotFoundError):
        RunRegistry(watch_directory=str(tmp_path / "missing"))
    with pytest.raises(NotADirectoryError):
        RunRegistry(watch_directory=str(watched / "wine_i30_1" / ARCHIVE_FILENAME))


def test_the_run_list_picks_up_runs_started_later(tmp_path: Path) -> None:
    """A watched directory's new runs appear in the run list and can be opened.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    watched = tmp_path / "experiments"
    watched.mkdir()
    with serving(RunRegistry(watch_directory=str(watched), rescan_interval=0)) as url:
        payload = get_json(f"{url}/api/runs")
        assert payload["runs"] == []
        assert payload["source"] == {"directory": str(watched), "waiting": []}
        assert get(f"{url}/api/runs/0")[0] == 404

        build_run(watched / "first_run", standard_genomes()[:2])
        assert [run["name"] for run in get_json(f"{url}/api/runs")["runs"]] == [
            "first_run"
        ]
        assert get_json(f"{url}/api/runs/0")["genomes"] == 2

        # a run's page can be opened before the run list has been reloaded
        build_run(watched / "second_run", standard_genomes()[:1])
        assert get_json(f"{url}/api/runs/1")["name"] == "second_run"


def test_a_run_page_survives_an_archive_it_cannot_fully_read(tmp_path: Path) -> None:
    """An archive missing a column still opens, reporting what it could read.

    A run written by an older version has no ``final_metrics`` column, so asking
    what its genomes can be charted by fails. The page degrades to the rest of
    what the archive holds rather than failing outright, since a single such run
    under a watched directory would otherwise take its whole page down.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    watched = tmp_path / "experiments"
    watched.mkdir()
    build_run(watched / "older_run", standard_genomes()[:2])

    connection = sqlite3.connect(watched / "older_run" / ARCHIVE_FILENAME)
    connection.execute("ALTER TABLE genomes DROP COLUMN final_metrics")
    connection.commit()
    connection.close()

    with serving(RunRegistry(watch_directory=str(watched))) as url:
        assert get(f"{url}/api/runs/0")[0] == 200

        payload = get_json(f"{url}/api/runs/0")
        assert "final_metrics" in payload["error"]
        assert payload["metrics"] == []
        assert payload["primary_metrics"] == []
        # everything the archive could still answer comes through
        assert payload["genomes"] == 2
        assert payload["fitness_keys"] == ["loss", "target_metric"]

        # and the progress chart reports having nothing rather than erroring
        assert get(f"{url}/api/runs/0/history")[0] == 200


def test_annotations_are_read_only_unless_allowed(tmp_path: Path) -> None:
    """A dashboard started without --allow_annotations refuses every write.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    build_run(tmp_path / "iris_1", standard_genomes())
    with serving(RunRegistry(run_directories=[str(tmp_path / "iris_1")])) as url:
        assert get_json(f"{url}/api/runs/0")["annotations_enabled"] is False
        assert get_json(f"{url}/api/runs/0/annotations") == {
            "enabled": False,
            "genome_number": None,
            "notes": [],
            "tags": [],
        }

        status, body = send(f"{url}/api/runs/0/notes", "POST", {"text": "hello"})
        assert status == 403
        assert "--allow_annotations" in body["error"]
        tags = f"{url}/api/runs/0/genomes/3/tags"
        assert send(tags, "POST", {"tag": "candidate"})[0] == 403
        assert send(f"{tags}/candidate", "DELETE")[0] == 403

    # refusing, and reading, left the run directory as it was
    assert not (tmp_path / "iris_1" / "annotations.sqlite").exists()


def test_the_dashboard_writes_notes_and_tags_when_allowed(tmp_path: Path) -> None:
    """Notes and tags round-trip through the API, and the archive is untouched.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    build_run(tmp_path / "iris_1", standard_genomes())
    archive = tmp_path / "iris_1" / ARCHIVE_FILENAME
    before = archive.read_bytes()
    registry = RunRegistry(run_directories=[str(tmp_path / "iris_1")])

    with serving(registry, allow_annotations=True) as url:
        assert get_json(f"{url}/api/runs/0")["annotations_enabled"] is True

        status, note = send(
            f"{url}/api/runs/0/notes",
            "POST",
            {"text": "promising", "genome_number": 3, "author": "travis"},
        )
        assert status == 201
        assert (note["source"], note["author"], note["genome_number"]) == (
            "dashboard",
            "travis",
            3,
        )
        assert send(f"{url}/api/runs/0/notes", "POST", {"text": "run-level"})[0] == 201

        tags = f"{url}/api/runs/0/genomes/3/tags"
        status, tag = send(tags, "POST", {"tag": "candidate"})
        assert status == 201 and tag["created"] is True
        # tagging a genome that already carries the tag changes nothing
        status, again = send(tags, "POST", {"tag": "candidate"})
        assert status == 200 and again["created"] is False

        genome = get_json(f"{url}/api/runs/0/annotations?genome=3")
        assert [entry["text"] for entry in genome["notes"]] == ["promising"]
        assert [entry["tag"] for entry in genome["tags"]] == ["candidate"]
        assert len(get_json(f"{url}/api/runs/0/annotations")["notes"]) == 2

        status, removed = send(f"{tags}/candidate", "DELETE")
        assert status == 200 and removed["active"] is False
        assert get_json(f"{url}/api/runs/0/annotations?genome=3")["tags"] == []
        history = get_json(f"{url}/api/runs/0/annotations?genome=3&include_removed=1")
        assert len(history["tags"]) == 1

        # what cannot be written is refused, with a reason
        missing = f"{url}/api/runs/0/genomes/999/tags"
        assert send(missing, "POST", {"tag": "candidate"})[0] == 404
        assert send(tags, "POST", {"tag": "has space"})[0] == 400
        assert send(f"{url}/api/runs/0/notes", "POST", {"text": "   "})[0] == 400
        assert send(f"{tags}/candidate", "DELETE")[0] == 404

    assert archive.read_bytes() == before
    assert (tmp_path / "iris_1" / "annotations.sqlite").is_file()


def test_annotation_writes_from_other_sites_are_refused(tmp_path: Path) -> None:
    """A write the dashboard's own page did not send is not accepted.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    build_run(tmp_path / "iris_1", standard_genomes())
    registry = RunRegistry(run_directories=[str(tmp_path / "iris_1")])

    with serving(registry, allow_annotations=True) as url:
        notes = f"{url}/api/runs/0/notes"

        # a page on another site, which the browser names as the origin
        foreign = {"Origin": "https://elsewhere.example"}
        assert send(notes, "POST", {"text": "x"}, headers=foreign)[0] == 403

        # a plain form post needs no preflight, so it is refused for not being JSON
        form = urllib.request.Request(notes, data=b"text=x", method="POST")
        form.add_header("Content-Type", "application/x-www-form-urlencoded")
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(form)
        assert refused.value.code == 400

        # the dashboard's own page is accepted
        assert send(notes, "POST", {"text": "ok"}, headers={"Origin": url})[0] == 201


def test_assign_groups_uses_path_substrings(tmp_path) -> None:
    """A run joins every group whose substring is in its path.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    for name in ("iris_i30_1", "iris_i10_1", "wine_i30_1"):
        build_run(tmp_path / name, standard_genomes()[:1], history=False)
    runs = RunRegistry(watch_directory=str(tmp_path)).runs

    groups = assign_groups(runs, ["i30", "iris"])

    names = {run.index: run.name for run in runs}
    assert sorted(names[index] for index in groups["i30"]) == [
        "iris_i30_1",
        "wine_i30_1",
    ]
    assert sorted(names[index] for index in groups["iris"]) == [
        "iris_i10_1",
        "iris_i30_1",
    ]
    assert next(run for run in runs if run.name == "iris_i30_1").groups == [
        "i30",
        "iris",
    ]


def test_runs_and_run_payloads(viewer_url: str) -> None:
    """The run list and a run's description report its contents.

    Args:
        viewer_url: The test server's base URL.
    """

    payload = get_json(f"{viewer_url}/api/runs")
    assert payload["groups"] == ["iris"]
    first = payload["runs"][0]
    assert first["name"] == "iris_1"
    assert first["genomes"] == 4
    assert first["groups"] == ["iris"]
    assert first["best_loss"] == {"genome_number": 3, "value": 0.3}
    assert first["best_target_metric"] == {"genome_number": 3, "value": 0.8}
    assert first["task_target"] == "iris"

    run = get_json(f"{viewer_url}/api/runs/0")
    assert run["fitness_keys"] == ["loss", "target_metric"]
    assert run["filter_options"]["insert_type"] == [
        "discarded",
        "global_best",
        "inserted",
    ]
    assert run["filter_options"]["generated_by"] == [
        "add_gate",
        "clone",
        "n_ary_crossover",
        "qubit_swap",
    ]
    assert run["has_history"] is True
    # genomes 1 and 2 were mutated from genome 0, which (like a seed) is not stored
    assert run["unarchived_parents"] == [0]

    assert get(f"{viewer_url}/api/runs/7")[0] == 404


def test_genome_listing_sorts_filters_and_validates(viewer_url: str) -> None:
    """The genome table API sorts, filters, pages and rejects bad input.

    Args:
        viewer_url: The test server's base URL.
    """

    page = get_json(f"{viewer_url}/api/runs/0/genomes")
    assert page["total"] == 4
    assert [row["genome_number"] for row in page["rows"]] == [3, 2, 1, 4]
    assert page["rows"][0]["parents"] == [1, 2]

    by_target = get_json(
        f"{viewer_url}/api/runs/0/genomes?sort=target_metric&desc=1&limit=2&offset=1"
    )
    assert [row["genome_number"] for row in by_target["rows"]] == [2, 1]

    filtered = get_json(
        f"{viewer_url}/api/runs/0/genomes?generated_by=add_gate&insert_type=inserted"
    )
    assert filtered["total"] == 1
    assert filtered["rows"][0]["genome_number"] == 1

    assert get(f"{viewer_url}/api/runs/0/genomes?sort=loss;DROP")[0] == 400
    assert get(f"{viewer_url}/api/runs/0/genomes?limit=0")[0] == 400
    assert get(f"{viewer_url}/api/runs/0/genomes?max_genome=-1")[0] == 400


def test_genome_listing_pages_keep_to_their_snapshot(viewer_url: str) -> None:
    """Later pages keep to the genomes of the first page's snapshot.

    A live run can save genomes between page requests; passing the first page's
    ``max_genome_number`` back as ``max_genome`` leaves those genomes out, so
    they don't shift the rows of later pages.

    Args:
        viewer_url: The test server's base URL.
    """

    first = get_json(f"{viewer_url}/api/runs/0/genomes?limit=2")
    assert first["max_genome_number"] == 4

    snapshot = get_json(f"{viewer_url}/api/runs/0/genomes?max_genome=2")
    assert snapshot["max_genome_number"] == 2
    assert snapshot["total"] == 2
    assert sorted(row["genome_number"] for row in snapshot["rows"]) == [1, 2]


def test_points_and_genealogy(viewer_url: str) -> None:
    """Chart points and parent links cover every genome and link.

    Args:
        viewer_url: The test server's base URL.
    """

    points = get_json(f"{viewer_url}/api/runs/0/points?y=target_metric")
    assert points["genome_number"] == [1, 2, 3, 4]
    assert points["y"] == [0.6, 0.7, 0.8, 0.1]
    assert points["operator"] == [
        "add_gate",
        "add_gate",
        "n_ary_crossover",
        "qubit_swap",
    ]
    assert points["generated_by"][1] == ["add_gate", "clone"]

    genealogy = get_json(f"{viewer_url}/api/runs/0/genealogy?y=loss")
    assert genealogy["points"]["y"] == [0.5, 0.4, 0.3, 0.9]
    assert list(zip(genealogy["links"]["child"], genealogy["links"]["parent"])) == [
        (1, 0),
        (2, 0),
        (3, 1),
        (3, 2),
        (4, 3),
    ]


def test_genome_detail_json_and_commands(viewer_url: str) -> None:
    """A genome's detail, download and commands are served.

    Args:
        viewer_url: The test server's base URL.
    """

    detail = get_json(f"{viewer_url}/api/runs/0/genomes/3")
    assert detail["summary"]["parents"] == [1, 2]
    assert detail["children"] == [4]
    assert detail["genome"]["fitness"]["loss"] == 0.3
    assert "--archive" in detail["commands"]["refine_genome"]
    assert detail["commands"]["refine_genome"].endswith("--genome_number 3")
    assert "visualize_rl" not in detail["commands"]

    status, headers, body = get(f"{viewer_url}/api/runs/0/genomes/3.json")
    assert status == 200
    assert "genome_3.json" in headers["content-disposition"]
    assert json.loads(body) == detail["genome"]

    assert get(f"{viewer_url}/api/runs/0/genomes/99")[0] == 404


def test_images_are_rendered_once_and_missing_ones_are_404(viewer_url: str) -> None:
    """Diagrams are served as PNGs and cached; undrawable images are 404s.

    Args:
        viewer_url: The test server's base URL.
    """

    for _ in range(2):
        status, headers, body = get(f"{viewer_url}/api/runs/0/genomes/2/diagram.png")
        assert status == 200
        assert headers["content-type"] == "image/png"
        assert body.startswith(_PNG_MAGIC)

    status, _, body = get(f"{viewer_url}/api/runs/0/genomes/2/training.png")
    assert status == 404
    assert "no training metrics" in json.loads(body)["error"]

    assert get(f"{viewer_url}/api/runs/0/genomes/99/diagram.png")[0] == 404


def test_render_service_caches_and_shares_renders(monkeypatch) -> None:
    """Repeated requests for one image render it only once.

    Args:
        monkeypatch: Used to replace image rendering with a counter.
    """

    calls: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        server,
        "render_genome_image",
        lambda path, number, kind: calls.append((path, number, kind)) or b"png",
    )

    service = RenderService(processes=0, cache_size=1)
    assert service.render("a.sqlar", 1, "diagram") == b"png"
    assert service.render("a.sqlar", 1, "diagram") == b"png"
    assert len(calls) == 1

    service.render("a.sqlar", 2, "diagram")
    service.render("a.sqlar", 1, "diagram")  # evicted by the cache size of one
    assert len(calls) == 3
    service.close()


def test_ancestry_includes_the_seed(viewer_url: str) -> None:
    """A genome's ancestry reaches back to the (unarchived) seed genome.

    Args:
        viewer_url: The test server's base URL.
    """

    ancestry = get_json(f"{viewer_url}/api/runs/0/genomes/4/ancestry?depth=5")
    generations = {
        node["genome_number"]: node["generation"] for node in ancestry["nodes"]
    }
    assert generations == {4: 0, 3: 1, 1: 2, 2: 2, 0: 3}
    assert (
        next(node for node in ancestry["nodes"] if node["genome_number"] == 0)[
            "in_archive"
        ]
        is False
    )
    assert {(edge["child"], edge["parent"]) for edge in ancestry["edges"]} == {
        (4, 3),
        (3, 1),
        (3, 2),
        (1, 0),
        (2, 0),
    }

    shallow = get_json(f"{viewer_url}/api/runs/0/genomes/4/ancestry?depth=1")
    assert sorted(node["genome_number"] for node in shallow["nodes"]) == [3, 4]


def test_island_runs_show_their_topology_and_each_parents_island(tmp_path) -> None:
    """An island run's page learns its topology, and a genome its parents' islands.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    topology = {"topology": ["ring"], "neighbors": [[1, 2], [0, 2], [1, 0]]}
    build_run(
        tmp_path / "islands",
        standard_genomes(),
        islands=[0, 1, 1, 2],
        run_info={"island_topology": topology},
    )
    build_run(tmp_path / "steady", standard_genomes())
    registry = RunRegistry(
        run_directories=[str(tmp_path / "islands"), str(tmp_path / "steady")]
    )

    with serving(registry) as url:
        indexes = {
            run["name"]: run["index"] for run in get_json(f"{url}/api/runs")["runs"]
        }
        islands, steady = indexes["islands"], indexes["steady"]

        assert get_json(f"{url}/api/runs/{islands}")["island_topology"] == topology
        assert get_json(f"{url}/api/runs/{steady}")["island_topology"] is None

        # genome 3 is a crossover of genomes 1 (island 0) and 2 (island 1)
        crossover = get_json(f"{url}/api/runs/{islands}/genomes/3")
        assert crossover["summary"]["parents"] == [1, 2]
        assert crossover["parent_islands"] == [0, 1]
        # the seed genome is never stored, so its island is unknown
        assert get_json(f"{url}/api/runs/{islands}/genomes/1")["parent_islands"] == [
            None
        ]

        ancestry = get_json(f"{url}/api/runs/{islands}/genomes/4/ancestry?depth=5")
        assert {
            node["genome_number"]: node["island"] for node in ancestry["nodes"]
        } == {
            4: 2,
            3: 1,
            1: 0,
            2: 1,
            0: None,
        }


def test_speciation_runs_show_species_and_each_parents_species(tmp_path) -> None:
    """A speciation run's page exposes species filters and parent species.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    speciation = {
        "max_population_size": 30,
        "species_threshold": 0.6,
        "neat_c1": 1.0,
        "neat_c2": 1.0,
        "inter_species_parent_rate": 0.1,
    }
    build_run(
        tmp_path / "speciation",
        standard_genomes(),
        species=[0, 0, 1, 1],
        run_info={
            "population_strategy": "SteadyStateSpeciation",
            "speciation": speciation,
        },
    )
    build_run(tmp_path / "steady", standard_genomes())
    registry = RunRegistry(
        run_directories=[str(tmp_path / "speciation"), str(tmp_path / "steady")]
    )

    with serving(registry) as url:
        indexes = {
            run["name"]: run["index"] for run in get_json(f"{url}/api/runs")["runs"]
        }
        speciation_run, steady = indexes["speciation"], indexes["steady"]

        payload = get_json(f"{url}/api/runs/{speciation_run}")
        assert payload["speciation"] == speciation
        assert payload["filter_options"]["species"] == [0, 1]
        assert get_json(f"{url}/api/runs/{steady}")["speciation"] is None

        crossover = get_json(f"{url}/api/runs/{speciation_run}/genomes/3")
        assert crossover["summary"]["parents"] == [1, 2]
        assert crossover["summary"]["species"] == 1
        assert crossover["parent_species"] == [0, 0]

        points = get_json(f"{url}/api/runs/{speciation_run}/points?y=loss")
        assert points["species"] == [0, 0, 1, 1]

        ancestry = get_json(
            f"{url}/api/runs/{speciation_run}/genomes/4/ancestry?depth=5"
        )
        assert {
            node["genome_number"]: node["species"] for node in ancestry["nodes"]
        } == {
            4: 1,
            3: 1,
            1: 0,
            2: 0,
            0: None,
        }


def test_compare_two_genomes(viewer_url: str) -> None:
    """Two genomes are compared by fitness, hyperparameters and gates.

    Args:
        viewer_url: The test server's base URL.
    """

    comparison = get_json(f"{viewer_url}/api/runs/0/compare?a=1&b=2")

    assert [gate["innovation_number"] for gate in comparison["gates"]["only_a"]] == [2]
    assert [gate["innovation_number"] for gate in comparison["gates"]["only_b"]] == [3]
    assert comparison["gates"]["changed"][0]["differences"]["parameters"] == [
        {"theta": 0.1},
        {"theta": 0.2},
    ]
    assert comparison["gates"]["unchanged"] == 0
    assert {row["key"]: row["same"] for row in comparison["fitness"]} == {
        "loss": False,
        "target_metric": False,
    }
    assert comparison["hyperparameters"] == [
        {"key": "epochs", "a": 1, "b": 2, "same": False}
    ]

    assert get(f"{viewer_url}/api/runs/0/compare?a=1")[0] == 400


def test_operators_history_and_groups(viewer_url: str) -> None:
    """Operator counts, a run's history and the group comparison are served.

    Args:
        viewer_url: The test server's base URL.
    """

    operators = get_json(f"{viewer_url}/api/runs/0/operators")["operators"]
    assert operators["add_gate"] == {"global_best": 1, "inserted": 1}
    assert operators["qubit_swap"] == {"discarded": 1}

    progress = get_json(f"{viewer_url}/api/runs/0/history")
    assert progress["columns"]["step"] == [1, 2, 3]
    assert progress["columns"]["population_size"] == [1, 2, 3]
    # the series is recomputed from the archive, so it is charted per metric
    assert progress["metric"] == "loss"
    assert "n_gates" in progress["metrics"] and "loss" in progress["metrics"]

    groups = get_json(f"{viewer_url}/api/groups?metric=loss&conf=95ci")
    assert "loss" in groups["metrics"] and "n_enabled_gates" in groups["metrics"]
    (iris,) = groups["groups"]
    assert iris["name"] == "iris"
    assert [run["name"] for run in iris["runs"]] == ["iris_1", "iris_2"]
    # iris_2 holds two genomes, so it records no third step: it is carried
    # forward only across its own lifetime and drops out of the average past its
    # last recorded step, rather than appearing to level off there
    assert iris["history"]["step"] == [1, 2, 3]
    assert iris["history"]["runs_at_step"] == [2, 2, 1]
    assert iris["history"]["n_runs"] == 2
    assert iris["best_loss"]["n"] == 2
    assert iris["best_loss"]["min"] == pytest.approx(0.3)
    assert iris["kind"] == "group"
    # operator insertion counts moved to the insertion-rate tables
    assert "operators" not in iris

    # asked for no metric, the comparison picks one the runs recorded rather
    # than charting nothing: target_metric is what a search is judged on
    defaulted = get_json(f"{viewer_url}/api/groups")
    assert defaulted["metric"] == "target_metric"
    assert defaulted["groups"][0]["history"] is not None
    assert "target_metric" in defaulted["primary_metrics"]

    assert get(f"{viewer_url}/api/groups?conf=wide")[0] == 400


def test_insertion_rates_for_a_run_a_group_and_every_group(viewer_url: str) -> None:
    """Insertion counts are tabulated for one run, one group and every group.

    Args:
        viewer_url: The test server's base URL.
    """

    single = get_json(f"{viewer_url}/api/insertion_rates?run=0")
    assert single["scope"] == {"kind": "run", "index": 0, "name": "iris_1"}
    assert single["outcomes"] == ["global_best", "local_best", "inserted", "discarded"]
    assert single["operators"] == ["add_gate", "clone", "n_ary_crossover", "qubit_swap"]
    (column,) = single["columns"]
    assert (column["label"], column["kind"], column["genomes"]) == ("iris_1", "run", 4)
    # genome 2 was generated by both add_gate and clone, so it counts once for each
    assert column["counts"]["add_gate"] == {"inserted": 1, "global_best": 1, "total": 2}
    assert column["counts"]["clone"] == {"global_best": 1, "total": 1}
    assert column["counts"]["qubit_swap"] == {"discarded": 1, "total": 1}

    group = get_json(f"{viewer_url}/api/insertion_rates?group=iris")
    assert group["scope"] == {"kind": "group", "name": "iris"}
    assert [(column["label"], column["kind"]) for column in group["columns"]] == [
        ("iris", "group"),
        ("iris_1", "run"),
        ("iris_2", "run"),
    ]
    summed = group["columns"][0]
    assert summed["genomes"] == 6
    assert summed["counts"]["add_gate"] == {"inserted": 2, "global_best": 2, "total": 4}
    assert "qubit_swap" not in group["columns"][2]["counts"]

    every = get_json(f"{viewer_url}/api/insertion_rates")
    assert every["scope"] == {"kind": "all"}
    assert [column["label"] for column in every["columns"]] == ["iris"]
    assert every["latex"] == insertion_rates_latex([("iris", summed["counts"])])
    assert every["errors"] == []

    assert get(f"{viewer_url}/api/insertion_rates?group=wine")[0] == 404
    assert get(f"{viewer_url}/api/insertion_rates?run=9")[0] == 404
    assert get(f"{viewer_url}/api/insertion_rates?run=0&group=iris")[0] == 400


def test_insertion_rates_latex_matches_analyze_genome_generation() -> None:
    """The LaTeX table is laid out exactly as analyze_genome_generation prints it.

    The expected text follows that script's print statements: a column per
    group (underscores escaped), each operator's rows in global best, local
    best, inserted, discarded order (the inserted row with no space before its
    line break), shares to three decimals, and ``-`` for a group that has no
    genomes from the operator.
    """

    latex = insertion_rates_latex(
        [
            (
                "i30_p2",
                {
                    "add_gate": {"inserted": 3, "discarded": 1, "total": 4},
                    "n_ary_crossover": {"global_best": 1, "local_best": 1, "total": 2},
                },
            ),
            ("i10", {"add_gate": {"global_best": 1, "total": 1}}),
        ]
    )

    assert latex == (
        "\\begin{tabular}{lp{2cm}p{1.5cm}p{1.5cm}}\n"
        "\\toprule\n"
        " &\n"
        " & {\\bf i30\\_p2 } & {\\bf i10 }\\\\\n"
        "\\midrule\n"
        "\\multirowcell{4}{add\\\\gate} & global best & 0.000 & 1.000 \\\\\n"
        "& local best & 0.000 & 0.000 \\\\\n"
        "& inserted & 0.750 & 0.000\\\\\n"
        "& discarded & 0.250 & 0.000 \\\\\n"
        "\\hline\n"
        "\\multirowcell{4}{n-ary\\\\crossover} & global best & 0.500 & - \\\\\n"
        "& local best & 0.500 & - \\\\\n"
        "& inserted & 0.000 & -\\\\\n"
        "& discarded & 0.000 & - \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )


def test_static_files_are_served_safely(viewer_url: str) -> None:
    """The app is served, and nothing outside the static directory is.

    Args:
        viewer_url: The test server's base URL.
    """

    status, headers, body = get(f"{viewer_url}/")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert b"/static/app.js" in body

    for name in ("app.js", "app.css", "uPlot.iife.min.js", "uPlot.min.css"):
        assert get(f"{viewer_url}/static/{name}")[0] == 200

    assert get(f"{viewer_url}/static/../server.py")[0] == 404
    assert get(f"{viewer_url}/static/%2e%2e/server.py")[0] == 404
    assert get(f"{viewer_url}/nowhere")[0] == 404


def test_compare_gates_matches_by_innovation() -> None:
    """Gates are matched by innovation number, not position."""

    diff = compare_gates([gate(1), gate(2, enabled=False)], [gate(2), gate(1)])

    assert diff["only_a"] == [] and diff["only_b"] == []
    assert diff["unchanged"] == 1
    assert diff["changed"] == [
        {
            "innovation_number": 2,
            "method_name": "rx",
            "differences": {"enabled": [False, True]},
        }
    ]
