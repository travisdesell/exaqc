"""Tests for the artifact viewer (``python3 -m src.examples.exaqc_artifacts``).

The viewer serves a single-page app and a read-only JSON API over one or more
runs' ``genomes.sqlar`` archives. These tests build small archives, start the
real HTTP server on a free port, and check the parser, run discovery and
grouping, every API route, image rendering and caching, and that static files
cannot escape their directory.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from src.examples import exaqc_artifacts
from src.utils.artifact_viewer import server
from src.utils.artifact_viewer.server import (
    ArtifactViewer,
    ArtifactViewerServer,
    RenderService,
    assign_groups,
    compare_gates,
    discover_runs,
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


def build_run(directory, genomes: list[FakeGenome], history: bool = True) -> str:
    """Writes a run directory holding an archive (and optionally a history CSV).

    Args:
        directory: The run directory to create.
        genomes: The genomes to store, in insertion order.
        history: Whether to write an ``exaqc_history.csv``.

    Returns:
        The run directory, as a string.
    """

    with GenomeArchive.create(str(directory)) as archive:
        archive.set_run_info(
            task="classification",
            task_target="iris",
            population_strategy="SteadyStatePopulation",
        )
        for insertion, genome in enumerate(genomes, start=1):
            archive.add_genome(genome, insertion=insertion)

    history_path = directory / "exaqc_history.csv"
    if history:
        best = min(genome.serialized["fitness"]["loss"] for genome in genomes)
        history_path.write_text(
            "step,best,top5_mean\n"
            + "".join(f"{step},{best + 1 / step},{best * 2}\n" for step in range(1, 4))
        )
    elif history_path.exists():
        history_path.unlink()
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

    runs = discover_runs([str(tmp_path / "runs")])
    groups = assign_groups(runs, ["iris"])
    httpd = ArtifactViewerServer(
        ("127.0.0.1", 0), ArtifactViewer(runs, groups, RenderService(processes=0))
    )
    httpd.rendered = rendered
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(url: str) -> tuple[int, dict[str, str], bytes]:
    """Fetches a URL, returning error responses rather than raising.

    Args:
        url: The URL to fetch.

    Returns:
        The status code, headers and body.
    """

    try:
        with urllib.request.urlopen(url) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


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
    """The viewer's arguments and defaults match the documentation."""

    args = exaqc_artifacts.build_parser().parse_args(["runs/iris"])

    assert args.runs == ["runs/iris"]
    assert args.groups is None
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.open_browser is False
    assert args.logging_level == "INFO"

    with pytest.raises(SystemExit):
        exaqc_artifacts.build_parser().parse_args([])


def test_discover_runs_searches_directories_and_names_runs(tmp_path) -> None:
    """Run directories, directories of runs and archive files are all found once.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    build_run(tmp_path / "exp" / "wine_1", standard_genomes()[:1], history=False)
    build_run(
        tmp_path / "exp" / "nested" / "wine_2", standard_genomes()[:1], history=False
    )

    runs = discover_runs(
        [str(tmp_path / "exp"), str(tmp_path / "exp" / "wine_1" / ARCHIVE_FILENAME)]
    )

    assert [run.name for run in runs] == ["nested/wine_2", "wine_1"]
    assert [run.index for run in runs] == [0, 1]
    assert discover_runs([str(tmp_path / "exp" / "wine_1")])[0].name == "wine_1"

    with pytest.raises(FileNotFoundError):
        discover_runs([str(tmp_path / "missing")])


def test_assign_groups_uses_path_substrings(tmp_path) -> None:
    """A run joins every group whose substring is in its path.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    for name in ("iris_i30_1", "iris_i10_1", "wine_i30_1"):
        build_run(tmp_path / name, standard_genomes()[:1], history=False)
    runs = discover_runs([str(tmp_path)])

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
    assert "genome_3.json" in headers["Content-Disposition"]
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
        assert headers["Content-Type"] == "image/png"
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

    history = get_json(f"{viewer_url}/api/runs/0/history")["columns"]
    assert history["step"] == [1.0, 2.0, 3.0]

    groups = get_json(f"{viewer_url}/api/groups?metric=top5_mean&conf=95ci")
    assert groups["metrics"] == ["best", "top5_mean"]
    (iris,) = groups["groups"]
    assert iris["name"] == "iris"
    assert [run["name"] for run in iris["runs"]] == ["iris_1", "iris_2"]
    assert iris["history"]["step"] == [1, 2, 3]
    assert iris["history"]["n_runs"] == 2
    assert iris["best_loss"]["n"] == 2
    assert iris["best_loss"]["min"] == pytest.approx(0.3)
    assert iris["operators"]["add_gate"] == {"global_best": 2, "inserted": 2}

    assert get(f"{viewer_url}/api/groups?conf=wide")[0] == 400


def test_static_files_are_served_safely(viewer_url: str) -> None:
    """The app is served, and nothing outside the static directory is.

    Args:
        viewer_url: The test server's base URL.
    """

    status, headers, body = get(f"{viewer_url}/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
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
