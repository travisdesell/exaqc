"""Tests for :mod:`src.utils.genome_archive`.

A search writes every evaluated genome into a single SQLite archive rather than
several files per genome. These tests pin the parts other code and tools rely
on: the command-line arguments, the SQLite Archive layout that the stock
``sqlite3`` shell can extract, the summary and parent rows used for sorting and
ancestry, the SQLite settings for local and shared file systems, that a reader
never blocks the search's writes, the overwritten current-best files, and the
loaders the single-genome tools and analysis scripts use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from typing import Any

import pytest

from src.utils import genome_archive
from src.utils.genome_archive import (
    ARCHIVE_FILENAME,
    GenomeArchive,
    iter_run_genome_dicts,
    load_genome_dict,
    resolve_archive_path,
)


class FakeGenome:
    """Stand-in genome that serializes to a small fixed dict."""

    def __init__(
        self,
        genome_number: int,
        fitness: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        gates: list[dict[str, Any]] | None = None,
    ) -> None:
        """Creates the fake genome.

        Args:
            genome_number: The genome's number.
            fitness: Its fitness dict; defaults to a loss equal to the number.
            metadata: Its metadata dict.
            gates: Its serialized gates.
        """

        self.genome_number = genome_number
        self.fitness = (
            fitness if fitness is not None else {"loss": float(genome_number)}
        )
        self.metadata = metadata if metadata is not None else {}
        self.gates = gates if gates is not None else []

    def to_dict(self) -> dict[str, Any]:
        """Serializes the fake genome the way ``CircuitGenome.to_dict`` would.

        Returns:
            The serialized genome.
        """

        return {
            "fitness": self.fitness,
            "genome_number": self.genome_number,
            "metadata": self.metadata,
            "target": "pennylane",
            "gates": self.gates,
        }


def create_archive(tmp_path, shared_file_system: bool = False) -> GenomeArchive:
    """Creates a writable archive inside ``tmp_path``.

    Args:
        tmp_path: The directory to create the run output directory in.
        shared_file_system: Whether to use shared-file-system settings.

    Returns:
        The writable archive.
    """

    return GenomeArchive.create(
        str(tmp_path / "run"), shared_file_system=shared_file_system
    )


def journal_mode(archive: GenomeArchive) -> str:
    """Reads an archive connection's journal mode.

    Args:
        archive: The archive to inspect.

    Returns:
        The journal mode, lower case.
    """

    return archive.connection.execute("PRAGMA journal_mode").fetchone()[0].lower()


def test_parser_owns_the_output_arguments() -> None:
    """``--out_dir`` and ``--shared_file_system`` come from GenomeArchive."""

    parser = argparse.ArgumentParser()
    GenomeArchive.initialize_parser(parser)

    defaults = parser.parse_args([])
    assert defaults.out_dir == "artifacts"
    assert defaults.shared_file_system is False

    assert parser.parse_args(["--shared_file_system"]).shared_file_system is True
    assert parser.parse_args(["--no-shared_file_system"]).shared_file_system is False


def test_from_args_creates_the_output_directory(tmp_path) -> None:
    """The (nested) output directory and archive are created with WAL locally.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    out_dir = tmp_path / "fresh" / "nested" / "out"
    archive = GenomeArchive.from_args(
        argparse.Namespace(out_dir=str(out_dir), shared_file_system=False)
    )

    assert (out_dir / ARCHIVE_FILENAME).is_file()
    assert archive.out_dir == str(out_dir)
    assert journal_mode(archive) == "wal"
    assert archive.run_info()["format_version"] == genome_archive.ARCHIVE_FORMAT_VERSION

    archive.close()


def test_shared_file_system_uses_a_persistent_rollback_journal(tmp_path) -> None:
    """Shared file systems avoid WAL and per-commit journal files.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    archive = create_archive(tmp_path, shared_file_system=True)
    assert journal_mode(archive) == "persist"
    archive.add_genome(FakeGenome(1), insertion=1)
    archive.close()


@pytest.mark.parametrize("shared_file_system", [False, True])
def test_close_leaves_a_single_self_contained_file(
    tmp_path, shared_file_system: bool
) -> None:
    """After closing, no write-ahead log or journal file is left beside the archive.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        shared_file_system: Whether shared-file-system settings are used.
    """

    archive = create_archive(tmp_path, shared_file_system=shared_file_system)
    archive.add_genome(FakeGenome(1), insertion=1)
    archive.close()
    archive.close()  # closing twice is harmless

    leftovers = [
        name
        for name in os.listdir(tmp_path / "run")
        if name.startswith(ARCHIVE_FILENAME + "-")
    ]
    assert leftovers == []

    with GenomeArchive.open_readonly(str(tmp_path / "run")) as reader:
        assert reader.count() == 1


def test_add_genome_round_trips_and_summarizes(tmp_path) -> None:
    """A stored genome reads back unchanged, with its summary and parents.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome = FakeGenome(
        7,
        fitness={"loss": 0.25, "target_metric": 0.75, "env_id": "CartPole-v1"},
        metadata={
            "insert_type": "global_best",
            "generated_by": ["add_gate", "qubit_swap"],
            "crossover_type": "intra-island",
            "parent_genomes": [3, 5, 3],
        },
        gates=[
            {"method_name": "rx", "parameters": {"theta": 0.1}, "enabled": True},
            {
                "method_name": "rot",
                "parameters": {"phi": 0.2, "theta": 0.3, "omega": 0.4},
                "enabled": False,
            },
            {"method_name": "cx", "parameters": {}, "enabled": True},
        ],
    )

    with create_archive(tmp_path) as archive:
        assert archive.add_genome(genome, insertion=4, island=2) is True

        assert archive.get_genome_dict(7) == genome.to_dict()

        summary = archive.get_summary(7)
        assert summary["insertion"] == 4
        assert summary["island"] == 2
        assert summary["insert_type"] == "global_best"
        assert summary["generated_by"] == ["add_gate", "qubit_swap"]
        assert summary["crossover_type"] == "intra-island"
        assert summary["n_gates"] == 3
        assert summary["n_enabled_gates"] == 2
        assert summary["n_parameters"] == 4
        assert summary["fitness"] == genome.fitness
        assert summary["parents"] == [3, 5]

        assert archive.parents(7) == [3, 5]
        assert archive.children(3) == [7]
        assert archive.children(5) == [7]
        assert archive.fitness_keys() == ["env_id", "loss", "target_metric"]
        assert archive.max_genome_number() == 7

        # re-adding the same genome replaces it rather than duplicating it
        genome.metadata["parent_genomes"] = [5]
        archive.add_genome(genome, insertion=4)
        assert archive.count() == 1
        assert archive.children(3) == []


def test_members_match_the_legacy_genome_files(tmp_path) -> None:
    """Members are stored under the old path, byte-identical to the old files.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome = FakeGenome(12, metadata={"generated_by": ["clone"]})
    expected = json.dumps(genome.to_dict(), ensure_ascii=False, indent=4).encode(
        "utf-8"
    )

    with create_archive(tmp_path) as archive:
        archive.add_genome(genome, insertion=1)
        name, size, data = archive.connection.execute(
            "SELECT name, sz, data FROM sqlar"
        ).fetchone()

    assert name == "all_genomes/genome_12.json"
    assert size == len(expected)
    assert genome_archive._decode_member(size, data) == expected

    sqlite3_shell = shutil.which("sqlite3")
    if sqlite3_shell is None:
        pytest.skip("the sqlite3 shell is not installed")

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    subprocess.run(
        [sqlite3_shell, str(tmp_path / "run" / ARCHIVE_FILENAME), "-Ax"],
        cwd=extract_dir,
        check=True,
        capture_output=True,
    )
    assert (extract_dir / "all_genomes" / "genome_12.json").read_bytes() == expected


def test_list_genomes_sorts_filters_and_pages(tmp_path) -> None:
    """Listings sort by fitness keys or columns, filter, and put missing values last.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genomes = [
        FakeGenome(
            1,
            fitness={"loss": 0.5},
            metadata={"insert_type": "inserted", "generated_by": ["add_gate"]},
        ),
        FakeGenome(
            2,
            fitness={"loss": 0.1},
            metadata={
                "insert_type": "global_best",
                "generated_by": ["n_ary_crossover"],
            },
        ),
        FakeGenome(
            3,
            fitness={"loss": float("nan")},
            metadata={"insert_type": "discarded", "generated_by": ["add_gate"]},
        ),
        FakeGenome(
            4,
            fitness={"target_metric": 1.0},
            metadata={"insert_type": "inserted", "generated_by": ["clone", "add_gate"]},
        ),
        FakeGenome(
            5,
            fitness={"loss": 0.3},
            metadata={"insert_type": "inserted"},
            gates=[{"parameters": {}}] * 3,
        ),
    ]

    with create_archive(tmp_path) as archive:
        for insertion, genome in enumerate(genomes, start=1):
            archive.add_genome(genome, insertion=insertion)

        def numbers(**kwargs: Any) -> list[int]:
            """Lists genome numbers for the given listing arguments."""
            return [
                summary["genome_number"] for summary in archive.list_genomes(**kwargs)
            ]

        # NaN and missing losses sort last in both directions
        assert numbers() == [2, 5, 1, 3, 4]
        assert numbers(descending=True) == [1, 5, 2, 3, 4]
        assert numbers(sort_key="n_gates", descending=True)[0] == 5
        assert numbers(offset=1, limit=2) == [5, 1]

        assert numbers(filters={"insert_type": "inserted"}) == [5, 1, 4]
        assert numbers(filters={"generated_by": "add_gate"}) == [1, 3, 4]
        assert numbers(
            filters={
                "generated_by": "add_gate",
                "insert_type": "inserted",
                "island": None,
            }
        ) == [1, 4]
        assert archive.count(filters={"generated_by": "add_gate"}) == 3

        with pytest.raises(ValueError):
            archive.list_genomes(sort_key="loss; DROP TABLE genomes")
        with pytest.raises(ValueError):
            archive.list_genomes(filters={"fitness": 1})


def test_iter_genome_dicts_reads_every_genome_in_order(tmp_path) -> None:
    """Iteration crosses read batches and yields genomes in number order.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    n_genomes = genome_archive._ITERATION_BATCH_SIZE * 2 + 7

    with create_archive(tmp_path) as archive:
        for genome_number in reversed(range(1, n_genomes + 1)):
            archive.add_genome(FakeGenome(genome_number), insertion=genome_number)

    with GenomeArchive.open_readonly(str(tmp_path / "run")) as reader:
        read = list(reader.iter_genome_dicts())

    assert [genome_number for genome_number, _ in read] == list(range(1, n_genomes + 1))
    assert all(
        genome["genome_number"] == genome_number for genome_number, genome in read
    )


def test_a_reader_does_not_block_the_writer(tmp_path) -> None:
    """A search can keep writing while a viewer is part way through reading.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    writer = create_archive(tmp_path)
    for genome_number in range(1, 4):
        writer.add_genome(FakeGenome(genome_number), insertion=genome_number)

    reader = GenomeArchive.open_readonly(str(tmp_path / "run"))
    cursor = reader.connection.execute("SELECT data FROM sqlar")
    cursor.fetchone()  # the reader now holds an open read transaction

    started = time.perf_counter()
    assert writer.add_genome(FakeGenome(4), insertion=4) is True
    assert time.perf_counter() - started < 5.0

    cursor.fetchall()
    reader.close()
    writer.close()


def test_locked_writes_are_retried_then_given_up(tmp_path, monkeypatch) -> None:
    """A locked database delays a write, and never raises out of the search.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to skip the retry delay and shorten the retry count.
    """

    monkeypatch.setattr(genome_archive.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(genome_archive, "WRITE_RETRIES", 2)

    attempts: list[int] = []

    def locked_once(connection: Any) -> None:
        """Fails with a lock error on the first attempt only."""
        attempts.append(1)
        if len(attempts) == 1:
            raise genome_archive.sqlite3.OperationalError("database is locked")

    def always_locked(connection: Any) -> None:
        """Always fails with a lock error."""
        raise genome_archive.sqlite3.OperationalError("database is locked")

    with create_archive(tmp_path) as archive:
        assert archive._write("test", locked_once) is True
        assert len(attempts) == 2

        assert archive._write("test", always_locked) is False
        assert not archive.connection.in_transaction


def test_read_only_archives_refuse_writes(tmp_path) -> None:
    """Opening an archive read-only never lets it be modified.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    create_archive(tmp_path).close()

    with GenomeArchive.open_readonly(
        str(tmp_path / "run" / ARCHIVE_FILENAME)
    ) as reader:
        with pytest.raises(RuntimeError):
            reader.add_genome(FakeGenome(1), insertion=1)
        with pytest.raises(RuntimeError):
            reader.write_current_best(FakeGenome(1), "fitness")


def test_current_best_files_are_overwritten_in_place(tmp_path, monkeypatch) -> None:
    """Each new best replaces a fixed set of files instead of adding more.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to replace the (slow) image rendering.
    """

    import src.utils.genome_rendering as genome_rendering

    training_images: dict[int, bytes | None] = {1: b"training-1", 2: None}
    monkeypatch.setattr(
        genome_rendering,
        "render_diagram_png",
        lambda genome: f"diagram-{genome.genome_number}".encode(),
    )
    monkeypatch.setattr(
        genome_rendering,
        "render_training_png",
        lambda genome: training_images[genome.genome_number],
    )

    run_dir = tmp_path / "run"
    with create_archive(tmp_path) as archive:
        archive.write_current_best(FakeGenome(1), "fitness")
        first = set(os.listdir(run_dir))
        assert {
            "best_fitness.json",
            "best_fitness.png",
            "best_fitness_training.png",
        } <= first

        archive.write_current_best(FakeGenome(2), "fitness")

        with pytest.raises(ValueError):
            archive.write_current_best(FakeGenome(3), "accuracy")

    assert json.loads((run_dir / "best_fitness.json").read_text())["genome_number"] == 2
    assert (run_dir / "best_fitness.png").read_bytes() == b"diagram-2"
    # a training plot that could not be drawn does not leave the old best's plot
    assert not (run_dir / "best_fitness_training.png").exists()
    assert not [name for name in os.listdir(run_dir) if name.endswith(".tmp")]


def test_current_best_files_get_normal_file_permissions(tmp_path, monkeypatch) -> None:
    """Best-genome files are as readable as any other file the run writes.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
        monkeypatch: Used to replace the (slow) image rendering.
    """

    import src.utils.genome_rendering as genome_rendering

    monkeypatch.setattr(genome_rendering, "render_diagram_png", lambda genome: b"png")
    monkeypatch.setattr(genome_rendering, "render_training_png", lambda genome: None)

    run_dir = tmp_path / "run"
    with create_archive(tmp_path) as archive:
        archive.write_current_best(FakeGenome(1), "target_metric")

    reference = run_dir / "reference.txt"
    reference.write_text("written normally")

    for name in ("best_target_metric.json", "best_target_metric.png"):
        assert (
            run_dir / name
        ).stat().st_mode & 0o777 == reference.stat().st_mode & 0o777


def test_genome_source_arguments(tmp_path) -> None:
    """Single-genome tools take ``--genome_json`` or ``--archive`` with ``--genome_number``.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    parser = argparse.ArgumentParser()
    genome_archive.add_genome_source_arguments(parser, json_help="A genome.")

    def parse(*argv: str) -> argparse.Namespace:
        """Parses and checks the genome-source arguments."""
        args = parser.parse_args(argv)
        genome_archive.check_genome_source_arguments(parser, args)
        return args

    from_json = parse("--genome_json", "genome.json")
    assert (from_json.genome_json, from_json.archive, from_json.genome_number) == (
        "genome.json",
        None,
        None,
    )

    from_archive = parse("--archive", "run", "--genome_number", "7")
    assert (
        from_archive.genome_json,
        from_archive.archive,
        from_archive.genome_number,
    ) == (None, "run", 7)

    for argv in (
        [],
        ["--archive", "run"],
        ["--genome_json", "genome.json", "--genome_number", "7"],
        ["--genome_json", "genome.json", "--archive", "run"],
    ):
        with pytest.raises(SystemExit):
            parse(*argv)


def test_run_info_round_trips(tmp_path) -> None:
    """Run facts are stored as JSON values and replaced by key.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    with create_archive(tmp_path) as archive:
        archive.set_run_info(task="classification", task_target="iris", start_time=12.5)
        archive.set_run_info(task_target="wine")
        info = archive.run_info()

    assert info["task"] == "classification"
    assert info["task_target"] == "wine"
    assert info["start_time"] == 12.5


def test_resolve_archive_path(tmp_path) -> None:
    """Run directories and archive files both resolve; anything else is an error.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    create_archive(tmp_path).close()
    archive_path = str(tmp_path / "run" / ARCHIVE_FILENAME)

    assert resolve_archive_path(str(tmp_path / "run")) == archive_path
    assert resolve_archive_path(archive_path) == archive_path

    with pytest.raises(FileNotFoundError):
        resolve_archive_path(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        resolve_archive_path(str(tmp_path / "missing.sqlar"))


def test_load_genome_dict_reads_either_source(tmp_path) -> None:
    """Genomes load from a JSON file or an archive, with clear errors otherwise.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    genome = FakeGenome(3, metadata={"insert_type": "inserted"})
    with create_archive(tmp_path) as archive:
        archive.add_genome(genome, insertion=1)

    json_path = tmp_path / "genome_3.json"
    json_path.write_text(json.dumps(genome.to_dict()))

    assert load_genome_dict(genome_path=str(json_path)) == genome.to_dict()
    assert (
        load_genome_dict(archive=str(tmp_path / "run"), genome_number=3)
        == genome.to_dict()
    )

    with pytest.raises(ValueError):
        load_genome_dict()
    with pytest.raises(ValueError):
        load_genome_dict(
            genome_path=str(json_path), archive=str(tmp_path / "run"), genome_number=3
        )
    with pytest.raises(ValueError):
        load_genome_dict(archive=str(tmp_path / "run"))
    with pytest.raises(ValueError):
        load_genome_dict(genome_path=str(json_path), genome_number=3)
    with pytest.raises(ValueError, match="no genome 99"):
        load_genome_dict(archive=str(tmp_path / "run"), genome_number=99)


def test_iter_run_genome_dicts_reads_archives_and_legacy_directories(tmp_path) -> None:
    """Analysis reads new archive runs and old one-file-per-genome runs alike.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    with create_archive(tmp_path) as archive:
        archive.add_genome(FakeGenome(1), insertion=1)
        archive.add_genome(FakeGenome(2), insertion=2)

    archived = list(iter_run_genome_dicts(str(tmp_path / "run")))
    assert [genome["genome_number"] for _, genome in archived] == [1, 2]
    assert archived[0][0].endswith(f"{ARCHIVE_FILENAME}:all_genomes/genome_1.json")

    legacy_dir = tmp_path / "legacy" / "all_genomes"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "genome_5.json").write_text(json.dumps(FakeGenome(5).to_dict()))

    legacy = list(iter_run_genome_dicts(str(tmp_path / "legacy")))
    assert [genome["genome_number"] for _, genome in legacy] == [5]
    assert legacy[0][0] == str(legacy_dir / "genome_5.json")
