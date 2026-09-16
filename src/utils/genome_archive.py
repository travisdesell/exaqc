"""Single-file SQLite archive of the genomes an EXAQC search evaluates.

A search can evaluate tens of thousands of genomes, and writing several files
per genome overwhelms the metadata servers of HPC shared file systems. Every
evaluated genome is instead stored in one SQLite database, ``genomes.sqlar``,
inside the run's output directory.

The database uses the standard SQLite Archive (``sqlar``) layout, so the stock
``sqlite3`` shell can list and extract it (``sqlite3 genomes.sqlar -Atv`` /
``-Ax``), and the extracted ``all_genomes/genome_<n>.json`` files are identical
to the files earlier runs wrote. Alongside the archive table it keeps a summary
row and the parent links for every genome, so genomes can be sorted, filtered
and traced through their ancestry without decompressing them.

:class:`GenomeArchive` also owns the rest of a run's output directory: the
command-line arguments that locate and configure it (``--out_dir`` and
``--shared_file_system``), the current-best genome files that are overwritten
whenever the search improves, and the search-progress history written by
:class:`~src.utils.profiler.EXAQCProfiler`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.circuits.circuit import CircuitGenome
    from src.utils.profiler import EXAQCProfiler

#: File name of the archive inside a run's output directory.
ARCHIVE_FILENAME = "genomes.sqlar"

#: Version of the archive layout, recorded in ``run_info``.
ARCHIVE_FORMAT_VERSION = 1

#: The kinds of current-best genome files kept in the output directory: the best
#: genome by the search's own ranking and the best by ``fitness["target_metric"]``.
BEST_KINDS = ("fitness", "target_metric")

#: How long (in seconds) a connection waits on a lock before SQLite reports
#: ``database is locked``.
BUSY_TIMEOUT_SECONDS = 60.0

#: How many times a write is retried after ``database is locked``.
WRITE_RETRIES = 5

#: Delay (in seconds) before the first retry of a locked write; it doubles on
#: every further attempt.
WRITE_RETRY_DELAY_SECONDS = 1.0

#: Summary-table columns a genome listing can be sorted on directly. Any other
#: sort key is looked up in the genome's fitness.
SORTABLE_COLUMNS = frozenset(
    {
        "genome_number",
        "insertion",
        "saved_at",
        "insert_type",
        "crossover_type",
        "island",
        "n_gates",
        "n_enabled_gates",
        "n_parameters",
    }
)

#: Filters :meth:`GenomeArchive.list_genomes` understands.
FILTER_KEYS = frozenset(
    {"insert_type", "generated_by", "crossover_type", "island", "max_genome_number"}
)

#: File mode recorded for archive members (a regular, world-readable file).
_SQLAR_FILE_MODE = 0o100644

#: How many genomes :meth:`GenomeArchive.iter_genome_dicts` reads per query, so a
#: long scan never holds one read transaction open.
_ITERATION_BATCH_SIZE = 200

_SUMMARY_COLUMNS = (
    "genome_number, insertion, saved_at, insert_type, generated_by, "
    "crossover_type, island, n_gates, n_enabled_gates, n_parameters, fitness"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sqlar(
    name TEXT PRIMARY KEY,
    mode INT,
    mtime INT,
    sz INT,
    data BLOB
);
CREATE TABLE IF NOT EXISTS genomes(
    genome_number INTEGER PRIMARY KEY,
    insertion INTEGER,
    saved_at REAL,
    insert_type TEXT,
    generated_by TEXT,
    crossover_type TEXT,
    island INTEGER,
    n_gates INTEGER,
    n_enabled_gates INTEGER,
    n_parameters INTEGER,
    fitness TEXT
);
CREATE TABLE IF NOT EXISTS genome_parents(
    child INTEGER NOT NULL,
    parent INTEGER NOT NULL,
    PRIMARY KEY(child, parent)
);
CREATE INDEX IF NOT EXISTS genome_parents_parent ON genome_parents(parent);
CREATE TABLE IF NOT EXISTS run_info(
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def genome_member_name(genome_number: int) -> str:
    """Returns the archive member name a genome's JSON is stored under.

    Args:
        genome_number: The genome's number.

    Returns:
        The member path, e.g. ``all_genomes/genome_12.json`` -- the same
        relative path earlier runs wrote the file to.
    """

    return f"all_genomes/genome_{genome_number}.json"


def resolve_archive_path(path: str) -> str:
    """Resolves a run directory or an archive file to the archive's path.

    Args:
        path: Either a run output directory containing ``genomes.sqlar`` or the
            path of an archive file itself.

    Returns:
        The path of the archive file.

    Raises:
        FileNotFoundError: If ``path`` is a directory without an archive, or
            does not exist.
    """

    if os.path.isdir(path):
        candidate = os.path.join(path, ARCHIVE_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"{path!r} does not contain a {ARCHIVE_FILENAME} archive."
        )

    if os.path.isfile(path):
        return path

    raise FileNotFoundError(f"No genome archive found at {path!r}.")


def load_genome_dict(
    genome_path: str | None = None,
    archive: str | None = None,
    genome_number: int | None = None,
) -> dict[str, Any]:
    """Loads a serialized genome from a JSON file or from a run's archive.

    Exactly one source must be given: a ``genome_path`` to a JSON file, or an
    ``archive`` (a ``genomes.sqlar`` file or its run directory) together with
    the ``genome_number`` to read from it.

    Args:
        genome_path: Path to a genome JSON file.
        archive: Path to a genome archive or the run directory holding one.
        genome_number: The genome to read from ``archive``.

    Returns:
        The serialized genome dict, as written by ``CircuitGenome.to_dict``.

    Raises:
        ValueError: If both or neither source is given, if ``genome_number`` is
            missing for an archive (or given with a JSON file), or if the
            archive holds no such genome.
        FileNotFoundError: If the file or archive does not exist.
        json.JSONDecodeError: If a JSON file is not valid JSON.
    """

    if (genome_path is None) == (archive is None):
        raise ValueError(
            "Give either a genome JSON file or a genome archive, but not both."
        )

    if genome_path is not None:
        if genome_number is not None:
            raise ValueError(
                "A genome number is only used when loading from a genome archive."
            )
        with open(genome_path, "r", encoding="utf-8") as genome_file:
            return json.load(genome_file)

    if genome_number is None:
        raise ValueError(
            "A genome number is required to load a genome from a genome archive."
        )

    with GenomeArchive.open_readonly(archive) as reader:
        try:
            return reader.get_genome_dict(genome_number)
        except KeyError as error:
            raise ValueError(str(error.args[0])) from error


def add_genome_source_arguments(
    parser: argparse.ArgumentParser, json_help: str
) -> None:
    """Adds the arguments that choose which saved genome a tool loads.

    The single-genome entry points (``refine_genome``, ``evaluate`` and
    ``visualize_rl``) load a genome either from a JSON file (``--genome_json``)
    or out of a run's archive (``--archive`` together with ``--genome_number``),
    and all offer the same arguments for it. Exactly one of ``--genome_json``
    and ``--archive`` is required; :func:`check_genome_source_arguments` checks
    ``--genome_number`` once the arguments are parsed.

    Args:
        parser: The parser to add the arguments to.
        json_help: Help text for ``--genome_json``, describing the genome files
            the tool expects.

    Returns:
        None. Mutates ``parser`` by adding ``--genome_json`` and ``--archive``
        (as a required, mutually exclusive pair) and ``--genome_number``.
    """

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--genome_json", type=str, default=None, help=json_help)
    source.add_argument(
        "--archive",
        type=str,
        default=None,
        help=(
            f"A run's {ARCHIVE_FILENAME} archive, or the run directory holding it, to load "
            "the genome from; requires --genome_number."
        ),
    )
    parser.add_argument(
        "--genome_number",
        type=int,
        default=None,
        help="Number of the genome to load from --archive.",
    )


def check_genome_source_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Checks that ``--genome_number`` is given with ``--archive``, and only with it.

    Args:
        parser: The parser the arguments came from, used to report the error.
        args: The parsed arguments (see :func:`add_genome_source_arguments`).

    Returns:
        None.

    Raises:
        SystemExit: Via ``parser.error``, if ``--archive`` is given without
            ``--genome_number``, or ``--genome_number`` without ``--archive``.
    """

    if args.archive is not None and args.genome_number is None:
        parser.error("--genome_number is required with --archive.")
    if args.archive is None and args.genome_number is not None:
        parser.error("--genome_number can only be used with --archive.")


def iter_run_genome_dicts(run_dir: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterates over every genome a run saved, from its archive or legacy files.

    Runs written before the archive existed kept one JSON file per genome in an
    ``all_genomes/`` subdirectory; those are read when the run directory has no
    ``genomes.sqlar``, so existing results can still be analyzed.

    Args:
        run_dir: A run output directory.

    Yields:
        ``(source, genome)`` pairs: a label naming where the genome was read
        from (for logging) and the serialized genome dict.
    """

    archive_path = os.path.join(run_dir, ARCHIVE_FILENAME)
    if os.path.isfile(archive_path):
        with GenomeArchive.open_readonly(archive_path) as reader:
            for genome_number, genome in reader.iter_genome_dicts():
                yield f"{archive_path}:{genome_member_name(genome_number)}", genome
        return

    for genome_json in Path(run_dir, "all_genomes").glob("*.json"):
        with open(genome_json, "r", encoding="utf-8") as genome_file:
            yield str(genome_json), json.load(genome_file)


def _finite_or_none(value: Any) -> Any:
    """Replaces non-finite floats (NaN and infinities), recursively, with None.

    SQLite's JSON functions reject the ``NaN``/``Infinity`` tokens Python's
    ``json`` module writes, so summary values are stored without them.

    Args:
        value: A JSON-serializable value.

    Returns:
        The value with every non-finite float replaced by ``None``.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def _encode_member(data: bytes) -> bytes:
    """Compresses member bytes the way the SQLite Archive format expects.

    Args:
        data: The uncompressed member contents.

    Returns:
        The zlib-compressed bytes when that is smaller, otherwise ``data``
        unchanged (a member whose stored size equals its original size is
        uncompressed).
    """

    compressed = zlib.compress(data)
    return compressed if len(compressed) < len(data) else data


def _decode_member(size: int, data: bytes) -> bytes:
    """Restores member bytes read from the archive.

    Args:
        size: The member's original (uncompressed) size.
        data: The stored bytes.

    Returns:
        The uncompressed member contents.
    """

    return data if len(data) == size else zlib.decompress(data)


def _default_file_mode() -> int:
    """Returns the permissions a newly created file gets under the process umask.

    Returns:
        ``0o666`` with the current umask's bits cleared.
    """

    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _atomic_write(path: str, data: bytes) -> None:
    """Writes a file so readers never observe it partially written.

    The bytes go to a temporary file in the same directory, which then replaces
    ``path`` in one rename. The file gets the permissions a normally created file
    would, so a run's outputs stay readable by collaborators on a shared system.

    Args:
        path: Destination file path.
        data: Bytes to write.

    Returns:
        None. Creates or replaces ``path``.
    """

    descriptor, temporary_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", prefix=".", suffix=".tmp"
    )
    try:
        # mkstemp creates the file readable by its owner only
        os.chmod(temporary_path, _default_file_mode())
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(data)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _chunks(values: list[int], size: int = 500) -> Iterator[list[int]]:
    """Splits a list into consecutive chunks, to keep SQL ``IN`` lists short.

    Args:
        values: The values to split.
        size: The most values per chunk.

    Yields:
        Consecutive slices of ``values``.
    """

    for start in range(0, len(values), size):
        yield values[start : start + size]


def _remove_if_exists(path: str) -> None:
    """Deletes a file if it exists.

    Args:
        path: The file to delete.

    Returns:
        None. Removes ``path`` when present.
    """

    if os.path.exists(path):
        os.remove(path)


def _connect(path: str) -> sqlite3.Connection:
    """Opens a SQLite connection in autocommit mode with a long busy timeout.

    Args:
        path: Database file path.

    Returns:
        The open connection. Transactions are managed explicitly.
    """

    return sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)


def _sort_expression(sort_key: str) -> tuple[str, list[Any]]:
    """Builds the SQL expression a genome listing is ordered by.

    Args:
        sort_key: A summary column (see :data:`SORTABLE_COLUMNS`) or a key of the
            genomes' fitness dicts.

    Returns:
        The SQL expression and the parameters it binds.

    Raises:
        ValueError: If ``sort_key`` is neither a column nor a plain identifier.
    """

    if sort_key in SORTABLE_COLUMNS:
        return sort_key, []
    if re.fullmatch(r"[A-Za-z0-9_]+", sort_key):
        return "json_extract(fitness, ?)", [f"$.{sort_key}"]
    raise ValueError(f"Cannot sort genomes by {sort_key!r}.")


def _filter_clause(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Builds the WHERE clause for genome listing filters.

    Args:
        filters: Filter values keyed by :data:`FILTER_KEYS`; ``None`` values are
            ignored. ``generated_by`` matches genomes whose list of generating
            operators contains the value, and ``max_genome_number`` matches
            genomes numbered at most the value (so a listing paged while a run
            is still adding genomes keeps to the genomes of its first page).

    Returns:
        The WHERE clause (empty when nothing is filtered) and its parameters.

    Raises:
        ValueError: If a filter key is not supported.
    """

    conditions: list[str] = []
    parameters: list[Any] = []

    for key, value in (filters or {}).items():
        if value is None:
            continue
        if key not in FILTER_KEYS:
            raise ValueError(f"Cannot filter genomes by {key!r}.")
        if key == "generated_by":
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(genomes.generated_by) WHERE json_each.value = ?)"
            )
        elif key == "max_genome_number":
            conditions.append("genome_number <= ?")
        else:
            conditions.append(f"{key} = ?")
        parameters.append(value)

    if not conditions:
        return "", []
    return "WHERE " + " AND ".join(conditions), parameters


def _summary_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    """Converts a ``genomes`` table row into a summary dict.

    Args:
        row: The row's values, in :data:`_SUMMARY_COLUMNS` order.

    Returns:
        The summary, with the JSON columns decoded.
    """

    (
        genome_number,
        insertion,
        saved_at,
        insert_type,
        generated_by,
        crossover_type,
        island,
        n_gates,
        n_enabled_gates,
        n_parameters,
        fitness,
    ) = row

    return {
        "genome_number": genome_number,
        "insertion": insertion,
        "saved_at": saved_at,
        "insert_type": insert_type,
        "generated_by": json.loads(generated_by) if generated_by else [],
        "crossover_type": crossover_type,
        "island": island,
        "n_gates": n_gates,
        "n_enabled_gates": n_enabled_gates,
        "n_parameters": n_parameters,
        "fitness": json.loads(fitness) if fitness else None,
    }


class GenomeArchive:
    """A run's output directory: every evaluated genome, current bests and history.

    Open one for writing with :meth:`from_args` or :meth:`create` -- only on the
    serial run or the MPI master, so there is a single writer -- or for reading
    with :meth:`open_readonly`.

    Attributes:
        path: Path of the ``genomes.sqlar`` database.
        out_dir: The run's output directory (the directory holding ``path``).
        connection: The open SQLite connection.
        writable: Whether this archive was opened for writing.
        shared_file_system: Whether shared-file-system SQLite settings are used.
        profiler: The search-progress profiler (writers only, else ``None``).
    """

    @staticmethod
    def initialize_parser(parser: argparse.ArgumentParser) -> None:
        """Adds the run-output command-line arguments to a parser.

        Every search entry point calls this so they share the same output flags
        with the same defaults and help text.

        Args:
            parser: The parser to add the arguments to.

        Returns:
            None. Mutates ``parser`` by adding ``--out_dir`` and
            ``--shared_file_system``.
        """

        parser.add_argument(
            "--out_dir",
            type=str,
            default="artifacts",
            help=(
                "Directory the run's outputs are written into: the genomes.sqlar archive of "
                "every evaluated genome, the current best genome files, the search history "
                "and the run log."
            ),
        )

        parser.add_argument(
            "--shared_file_system",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Use SQLite settings that are safe on a shared network file system (NFS, "
                "Lustre, GPFS): a persistent rollback journal instead of write-ahead logging."
            ),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> GenomeArchive:
        """Creates a writable archive from parsed command-line arguments.

        Args:
            args: Parsed arguments carrying ``out_dir`` and
                ``shared_file_system`` (see :meth:`initialize_parser`).

        Returns:
            The archive, opened for writing.
        """

        return cls.create(
            out_dir=args.out_dir, shared_file_system=args.shared_file_system
        )

    @classmethod
    def create(cls, out_dir: str, shared_file_system: bool = False) -> GenomeArchive:
        """Creates (or reopens) a run's output directory and archive for writing.

        Local disks use write-ahead logging, so readers such as the artifact
        viewer never block the search's writes. Shared network file systems do
        not support write-ahead logging, so there a persistent rollback journal
        is used instead, which also avoids creating and deleting a journal file
        on every commit.

        Args:
            out_dir: The run's output directory; created if missing.
            shared_file_system: Whether to use shared-file-system settings.

        Returns:
            The archive, opened for writing.
        """

        # Imported here so that reading an archive does not load the profiler's
        # quantum-framework dependencies.
        from src.utils.profiler import EXAQCProfiler

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, ARCHIVE_FILENAME)

        connection = _connect(path)
        connection.execute(
            f"PRAGMA journal_mode={'PERSIST' if shared_file_system else 'WAL'}"
        )
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(_SCHEMA)

        archive = cls(
            path,
            connection,
            writable=True,
            shared_file_system=shared_file_system,
            profiler=EXAQCProfiler(out_dir=out_dir),
        )

        existing = archive.count()
        if existing:
            logger.warning(
                "{} already holds {} genomes; genomes with the same numbers will be replaced.",
                path,
                existing,
            )

        archive.set_run_info(format_version=ARCHIVE_FORMAT_VERSION)
        return archive

    @classmethod
    def open_readonly(cls, path: str) -> GenomeArchive:
        """Opens an existing archive for reading.

        Reads use short transactions, so they can run while a search is still
        writing to the archive.

        Args:
            path: A ``genomes.sqlar`` file or the run directory holding one.

        Returns:
            The archive, opened for reading only.

        Raises:
            FileNotFoundError: If no archive exists at ``path``.
        """

        archive_path = resolve_archive_path(path)
        connection = _connect(archive_path)
        connection.execute("PRAGMA query_only=ON")
        return cls(archive_path, connection, writable=False)

    def __init__(
        self,
        path: str,
        connection: sqlite3.Connection,
        *,
        writable: bool,
        shared_file_system: bool = False,
        profiler: EXAQCProfiler | None = None,
    ) -> None:
        """Wraps an open archive connection; use the factory methods instead.

        Args:
            path: Path of the archive database file.
            connection: The open SQLite connection to it.
            writable: Whether the archive was opened for writing.
            shared_file_system: Whether shared-file-system settings are in use.
            profiler: The search-progress profiler (writers only).
        """

        self.path = path
        self.out_dir = os.path.dirname(path) or "."
        self.connection = connection
        self.writable = writable
        self.shared_file_system = shared_file_system
        self.profiler = profiler
        self._closed = False

    def __enter__(self) -> GenomeArchive:
        """Enters a ``with`` block.

        Returns:
            This archive.
        """

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Closes the archive when a ``with`` block exits.

        Args:
            exception_type: The type of the exception that ended the block, if any.
            exception: The exception that ended the block, if any.
            traceback: The exception's traceback, if any.

        Returns:
            None. Closes the archive; exceptions are not suppressed.
        """

        self.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _require_writable(self) -> None:
        """Checks that this archive was opened for writing.

        Returns:
            None.

        Raises:
            RuntimeError: If the archive is read-only.
        """

        if not self.writable:
            raise RuntimeError(f"{self.path} was opened read-only.")

    def _write(
        self, description: str, operation: Callable[[sqlite3.Connection], None]
    ) -> bool:
        """Runs a write in one transaction, retrying while the database is locked.

        A long-running reader on a shared file system can hold a lock longer
        than the busy timeout. The search must never stop because of that, so
        the write is retried with a doubling delay and, if it still cannot be
        made, skipped with an error logged.

        Args:
            description: What is being written, for log messages.
            operation: Performs the writes on the connection, inside the
                transaction.

        Returns:
            True if the write was committed, False if it was given up on.
        """

        delay = WRITE_RETRY_DELAY_SECONDS
        for attempt in range(1, WRITE_RETRIES + 2):
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                try:
                    operation(self.connection)
                    self.connection.execute("COMMIT")
                except BaseException:
                    if self.connection.in_transaction:
                        self.connection.execute("ROLLBACK")
                    raise
                return True
            except sqlite3.OperationalError as error:
                message = str(error).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                if attempt > WRITE_RETRIES:
                    logger.error(
                        "Could not write {} to {} after {} attempts: {}",
                        description,
                        self.path,
                        attempt,
                        error,
                    )
                    return False
                logger.warning(
                    "{} was locked while writing {} (attempt {}); retrying in {:.0f}s.",
                    self.path,
                    description,
                    attempt,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
        return False

    def add_genome(
        self, genome: CircuitGenome, insertion: int, island: int | None = None
    ) -> bool:
        """Stores an evaluated genome, its summary row and its parent links.

        The genome's JSON is stored as the ``all_genomes/genome_<n>.json``
        archive member, formatted exactly as earlier runs wrote the file.

        Args:
            genome: The evaluated genome.
            insertion: How many genomes had been inserted into the search when
                this one was (its position in the search's insertion order).
            island: The island the genome was inserted into, for island
                strategies.

        Returns:
            True if the genome was stored, False if the write was given up on
            because the database stayed locked.

        Raises:
            RuntimeError: If the archive is read-only.
        """

        self._require_writable()

        serialized = genome.to_dict()
        genome_number = int(serialized["genome_number"])
        data = json.dumps(serialized, ensure_ascii=False, indent=4).encode("utf-8")

        metadata = serialized.get("metadata") or {}
        gates = serialized.get("gates") or []

        summary = (
            genome_number,
            int(insertion),
            time.time(),
            metadata.get("insert_type"),
            json.dumps(list(metadata.get("generated_by") or [])),
            metadata.get("crossover_type"),
            island,
            len(gates),
            sum(1 for gate in gates if gate.get("enabled", True)),
            sum(len(gate.get("parameters") or {}) for gate in gates),
            json.dumps(_finite_or_none(serialized.get("fitness"))),
        )
        parents = sorted(
            {int(parent) for parent in metadata.get("parent_genomes") or []}
        )

        def store(connection: sqlite3.Connection) -> None:
            """Writes the member, summary and parent rows for the genome."""

            connection.execute(
                "INSERT OR REPLACE INTO sqlar(name, mode, mtime, sz, data) VALUES (?, ?, ?, ?, ?)",
                (
                    genome_member_name(genome_number),
                    _SQLAR_FILE_MODE,
                    int(time.time()),
                    len(data),
                    _encode_member(data),
                ),
            )
            connection.execute(
                f"INSERT OR REPLACE INTO genomes({_SUMMARY_COLUMNS}) VALUES ({', '.join('?' * 11)})",
                summary,
            )
            connection.execute(
                "DELETE FROM genome_parents WHERE child = ?", (genome_number,)
            )
            connection.executemany(
                "INSERT OR IGNORE INTO genome_parents(child, parent) VALUES (?, ?)",
                [(genome_number, parent) for parent in parents],
            )

        return self._write(f"genome {genome_number}", store)

    def write_current_best(self, genome: CircuitGenome, kind: str) -> None:
        """Overwrites one set of current-best genome files in the output directory.

        Writes ``best_<kind>.json`` (the serialized genome),
        ``best_<kind>.png`` (its architecture diagram) and
        ``best_<kind>_training.png`` (its training history). Each file is
        replaced atomically, so it can be opened at any time during a run. An
        image that cannot be drawn is removed rather than left showing an
        earlier best.

        Args:
            genome: The new best genome.
            kind: Which best it is, one of :data:`BEST_KINDS`.

        Returns:
            None. Replaces the files in ``out_dir``.

        Raises:
            RuntimeError: If the archive is read-only.
            ValueError: If ``kind`` is not one of :data:`BEST_KINDS`.
        """

        self._require_writable()
        if kind not in BEST_KINDS:
            raise ValueError(
                f"Unknown best genome kind {kind!r}; expected one of {BEST_KINDS}."
            )

        # Imported here so that reading an archive does not load the plotting and
        # quantum-framework stacks.
        from src.utils.genome_rendering import render_diagram_png, render_training_png

        prefix = os.path.join(self.out_dir, f"best_{kind}")
        serialized = json.dumps(genome.to_dict(), ensure_ascii=False, indent=4).encode(
            "utf-8"
        )
        _atomic_write(f"{prefix}.json", serialized)

        for path, image in (
            (f"{prefix}.png", render_diagram_png(genome)),
            (f"{prefix}_training.png", render_training_png(genome)),
        ):
            if image is None:
                _remove_if_exists(path)
            else:
                _atomic_write(path, image)

        logger.info(
            "wrote current best ({}) genome {} to {}.*",
            kind,
            genome.genome_number,
            prefix,
        )

    def record_history(self, step: int, population: list[CircuitGenome]) -> None:
        """Appends a search-progress row for the current population.

        Args:
            step: The insertion the snapshot was taken at.
            population: The population, sorted best first.

        Returns:
            None. Appends a row to ``exaqc_history.csv``.

        Raises:
            RuntimeError: If the archive is read-only.
        """

        self._require_writable()
        if self.profiler is not None:
            self.profiler.record(step=step, population=population)

    def plot_history(self) -> None:
        """Redraws the search-progress curves (``exaqc_curves.png``).

        Returns:
            None. Replaces the plot; a drawing failure is logged, not raised.

        Raises:
            RuntimeError: If the archive is read-only.
        """

        self._require_writable()
        if self.profiler is None:
            return
        try:
            self.profiler.plot_single_run()
        except Exception as error:
            logger.warning("Could not plot the search history: {}", error)

    def set_run_info(self, **values: Any) -> None:
        """Records facts about the run, such as its task and command line.

        Args:
            **values: JSON-serializable values keyed by name; existing keys are
                replaced.

        Returns:
            None. Writes the ``run_info`` rows.

        Raises:
            RuntimeError: If the archive is read-only.
        """

        self._require_writable()

        def store(connection: sqlite3.Connection) -> None:
            """Writes one ``run_info`` row per value."""

            connection.executemany(
                "INSERT OR REPLACE INTO run_info(key, value) VALUES (?, ?)",
                [
                    (key, json.dumps(_finite_or_none(value)))
                    for key, value in values.items()
                ],
            )

        self._write("run info", store)

    def close(self) -> None:
        """Closes the archive.

        A writer first switches the database back to a plain rollback journal,
        which checkpoints and removes any write-ahead log or persistent journal,
        so a finished run's archive is a single self-contained file. If another
        connection is still reading, that step is skipped (SQLite checkpoints the
        log when the last connection closes).

        Returns:
            None. Closes the connection; calling it again does nothing.
        """

        if self._closed:
            return
        self._closed = True

        if self.writable:
            try:
                self.connection.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError as error:
                logger.debug(
                    "Left {} in its journal mode on close: {}", self.path, error
                )

        self.connection.close()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def run_info(self) -> dict[str, Any]:
        """Returns the facts recorded about the run.

        Returns:
            The ``run_info`` values keyed by name.
        """

        rows = self.connection.execute("SELECT key, value FROM run_info").fetchall()
        return {key: json.loads(value) for key, value in rows}

    def unarchived_parents(self) -> list[int]:
        """Returns the parents of stored genomes that are not stored themselves.

        In a run this is the seed genome (genome 1): the empty circuit the
        initial genomes are mutated from, which is never evaluated and so is
        never stored.

        Returns:
            Those parents' genome numbers, sorted.
        """

        rows = self.connection.execute(
            "SELECT DISTINCT parent FROM genome_parents "
            "WHERE parent NOT IN (SELECT genome_number FROM genomes) ORDER BY parent"
        ).fetchall()
        return [row[0] for row in rows]

    def last_saved_at(self) -> float | None:
        """Returns when the most recently stored genome was saved.

        Returns:
            The latest save time (Unix seconds), or ``None`` for an empty archive.
        """

        return self.connection.execute("SELECT MAX(saved_at) FROM genomes").fetchone()[
            0
        ]

    def best_value(self, key: str, higher_is_better: bool) -> dict[str, Any] | None:
        """Finds the genome with the best value of a fitness key or summary column.

        Args:
            key: A fitness key or summary column (see :data:`SORTABLE_COLUMNS`).
            higher_is_better: Whether larger values are better.

        Returns:
            ``genome_number`` and ``value`` of the best genome (ties go to the
            lowest genome number), or ``None`` if no genome has a value.

        Raises:
            ValueError: If ``key`` is not a valid key.
        """

        expression, parameters = _sort_expression(key)
        row = self.connection.execute(
            f"SELECT genome_number, {expression} AS value FROM genomes WHERE {expression} IS NOT NULL "
            f"ORDER BY value {'DESC' if higher_is_better else 'ASC'}, genome_number ASC LIMIT 1",
            [*parameters, *parameters],
        ).fetchone()
        return None if row is None else {"genome_number": row[0], "value": row[1]}

    def filter_options(self) -> dict[str, list[Any]]:
        """Returns the distinct values the stored genomes can be filtered by.

        Returns:
            The sorted distinct ``insert_type``, ``generated_by`` (operators),
            ``crossover_type`` and ``island`` values, keyed by filter.
        """

        def distinct(query: str) -> list[Any]:
            """Runs a single-column query and returns its values."""
            return [row[0] for row in self.connection.execute(query).fetchall()]

        return {
            "insert_type": distinct(
                "SELECT DISTINCT insert_type FROM genomes WHERE insert_type IS NOT NULL ORDER BY 1"
            ),
            "generated_by": distinct(
                "SELECT DISTINCT json_each.value FROM genomes, json_each(genomes.generated_by) ORDER BY 1"
            ),
            "crossover_type": distinct(
                "SELECT DISTINCT crossover_type FROM genomes WHERE crossover_type IS NOT NULL ORDER BY 1"
            ),
            "island": distinct(
                "SELECT DISTINCT island FROM genomes WHERE island IS NOT NULL ORDER BY 1"
            ),
        }

    def points(self, y_key: str) -> dict[str, list[Any]]:
        """Returns every stored genome as a point, in genome-number order.

        Args:
            y_key: The fitness key (or summary column) giving each point's value.

        Returns:
            Parallel lists: ``genome_number``, ``insertion``, ``y`` (``None`` when
            a genome has no value), ``insert_type``, ``generated_by`` (each
            genome's list of generating operators), ``operator`` (its first
            generating operator), ``crossover_type`` and ``island``.

        Raises:
            ValueError: If ``y_key`` is not a valid key.
        """

        expression, parameters = _sort_expression(y_key)
        rows = self.connection.execute(
            f"SELECT genome_number, insertion, {expression}, insert_type, generated_by, crossover_type, island "
            "FROM genomes ORDER BY genome_number",
            parameters,
        ).fetchall()
        generated_by = [json.loads(row[4]) if row[4] else [] for row in rows]
        return {
            "genome_number": [row[0] for row in rows],
            "insertion": [row[1] for row in rows],
            "y": [row[2] for row in rows],
            "insert_type": [row[3] for row in rows],
            "generated_by": generated_by,
            "operator": [
                operators[0] if operators else None for operators in generated_by
            ],
            "crossover_type": [row[5] for row in rows],
            "island": [row[6] for row in rows],
        }

    def parent_links(self) -> dict[str, list[int]]:
        """Returns every parent link in the archive.

        Returns:
            Parallel ``child`` and ``parent`` genome-number lists, ordered by child.
        """

        rows = self.connection.execute(
            "SELECT child, parent FROM genome_parents ORDER BY child, parent"
        ).fetchall()
        return {"child": [row[0] for row in rows], "parent": [row[1] for row in rows]}

    def ancestors(
        self, genome_number: int, depth: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Traces a genome's ancestry back a number of generations.

        Args:
            genome_number: The genome whose ancestors to trace.
            depth: How many generations back to go.

        Returns:
            ``nodes``: the genome and its ancestors, each with its
            ``genome_number``, ``generation`` (the fewest steps back it is
            reached in), whether it is ``in_archive`` (the seed genome never is),
            and its ``insert_type``, ``generated_by`` and ``fitness``; and
            ``edges``: the ``child``/``parent`` links among them.
        """

        rows = self.connection.execute(
            "WITH RECURSIVE lineage(genome, generation) AS ("
            "VALUES(?, 0) UNION "
            "SELECT genome_parents.parent, lineage.generation + 1 FROM genome_parents "
            "JOIN lineage ON genome_parents.child = lineage.genome WHERE lineage.generation < ?"
            ") SELECT genome, MIN(generation) FROM lineage GROUP BY genome",
            (int(genome_number), int(depth)),
        ).fetchall()
        generations = {genome: generation for genome, generation in rows}

        summaries: dict[int, dict[str, Any]] = {}
        for chunk in _chunks(sorted(generations)):
            for row in self.connection.execute(
                f"SELECT {_SUMMARY_COLUMNS} FROM genomes WHERE genome_number IN ({', '.join('?' * len(chunk))})",
                chunk,
            ):
                summary = _summary_from_row(row)
                summaries[summary["genome_number"]] = summary

        edges = []
        inner = sorted(
            genome for genome, generation in generations.items() if generation < depth
        )
        for chunk in _chunks(inner):
            for child, parent in self.connection.execute(
                f"SELECT child, parent FROM genome_parents WHERE child IN ({', '.join('?' * len(chunk))}) ORDER BY child, parent",
                chunk,
            ):
                if parent in generations:
                    edges.append({"child": child, "parent": parent})

        nodes = []
        for genome in sorted(generations):
            summary = summaries.get(genome, {})
            nodes.append(
                {
                    "genome_number": genome,
                    "generation": generations[genome],
                    "in_archive": genome in summaries,
                    "insert_type": summary.get("insert_type"),
                    "generated_by": summary.get("generated_by", []),
                    "fitness": summary.get("fitness"),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def operator_counts(self) -> dict[str, dict[str, int]]:
        """Counts how the genomes each generating operator produced were inserted.

        A genome generated by several operators counts once for each, as in
        ``src.analysis.analyze_genome_generation``.

        Returns:
            Insert-type counts keyed by operator, then by insert type.
        """

        counts: dict[str, dict[str, int]] = {}
        for operator, insert_type, count in self.connection.execute(
            "SELECT json_each.value, COALESCE(genomes.insert_type, 'unknown'), COUNT(*) "
            "FROM genomes, json_each(genomes.generated_by) GROUP BY 1, 2 ORDER BY 1, 2"
        ):
            counts.setdefault(operator, {})[insert_type] = count
        return counts

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Counts the genomes in the archive.

        Args:
            filters: Optional filters, as for :meth:`list_genomes`.

        Returns:
            How many genomes match.
        """

        where, parameters = _filter_clause(filters)
        return int(
            self.connection.execute(
                f"SELECT COUNT(*) FROM genomes {where}", parameters
            ).fetchone()[0]
        )

    def max_genome_number(self) -> int | None:
        """Returns the highest genome number stored.

        Returns:
            The highest genome number, or ``None`` for an empty archive.
        """

        return self.connection.execute(
            "SELECT MAX(genome_number) FROM genomes"
        ).fetchone()[0]

    def fitness_keys(self) -> list[str]:
        """Returns every key that appears in the stored genomes' fitness dicts.

        Returns:
            The fitness keys, sorted.
        """

        rows = self.connection.execute(
            "SELECT DISTINCT json_each.key FROM genomes, json_each(genomes.fitness) "
            "WHERE json_type(genomes.fitness) = 'object' ORDER BY json_each.key"
        ).fetchall()
        return [row[0] for row in rows]

    def list_genomes(
        self,
        sort_key: str = "loss",
        descending: bool = False,
        filters: dict[str, Any] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Lists genome summaries, sorted, filtered and paged.

        Genomes missing the sort value are listed last; ties are broken by
        genome number.

        Args:
            sort_key: A summary column (see :data:`SORTABLE_COLUMNS`) or a key of
                the genomes' fitness dicts.
            descending: Whether to sort largest first.
            filters: Filter values keyed by :data:`FILTER_KEYS`.
            offset: How many matching genomes to skip.
            limit: The most genomes to return.

        Returns:
            One summary dict per genome (see :meth:`get_summary`), each with its
            ``parents`` list.

        Raises:
            ValueError: If the sort key or a filter is not supported.
        """

        order_expression, order_parameters = _sort_expression(sort_key)
        where, where_parameters = _filter_clause(filters)
        direction = "DESC" if descending else "ASC"

        rows = self.connection.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM genomes {where} "
            f"ORDER BY ({order_expression}) IS NULL, {order_expression} {direction}, genome_number ASC "
            "LIMIT ? OFFSET ?",
            [
                *where_parameters,
                *order_parameters,
                *order_parameters,
                int(limit),
                int(offset),
            ],
        ).fetchall()

        summaries = [_summary_from_row(row) for row in rows]
        parents = self._parents_of([summary["genome_number"] for summary in summaries])
        for summary in summaries:
            summary["parents"] = parents.get(summary["genome_number"], [])
        return summaries

    def get_summary(self, genome_number: int) -> dict[str, Any]:
        """Returns one genome's summary row.

        Args:
            genome_number: The genome to look up.

        Returns:
            The summary: genome number, insertion, saved time, insert type,
            generating operators, crossover type, island, gate/parameter counts,
            fitness and ``parents``.

        Raises:
            KeyError: If the archive holds no such genome.
        """

        row = self.connection.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM genomes WHERE genome_number = ?",
            (int(genome_number),),
        ).fetchone()
        if row is None:
            raise KeyError(f"{self.path} holds no genome {genome_number}.")
        summary = _summary_from_row(row)
        summary["parents"] = self.parents(genome_number)
        return summary

    def get_genome_dict(self, genome_number: int) -> dict[str, Any]:
        """Reads one genome's serialized dict.

        Args:
            genome_number: The genome to read.

        Returns:
            The serialized genome, as written by ``CircuitGenome.to_dict``.

        Raises:
            KeyError: If the archive holds no such genome.
        """

        row = self.connection.execute(
            "SELECT sz, data FROM sqlar WHERE name = ?",
            (genome_member_name(int(genome_number)),),
        ).fetchone()
        if row is None:
            raise KeyError(f"{self.path} holds no genome {genome_number}.")
        return json.loads(_decode_member(row[0], row[1]))

    def iter_genome_dicts(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Iterates over every stored genome in genome-number order.

        Genomes are read in small batches, each in its own short read, so a scan
        of a large archive does not hold up a search still writing to it.

        Yields:
            ``(genome_number, genome)`` pairs.
        """

        last_genome_number = -1
        while True:
            rows = self.connection.execute(
                "SELECT genomes.genome_number, sqlar.sz, sqlar.data FROM genomes "
                "JOIN sqlar ON sqlar.name = 'all_genomes/genome_' || genomes.genome_number || '.json' "
                "WHERE genomes.genome_number > ? ORDER BY genomes.genome_number LIMIT ?",
                (last_genome_number, _ITERATION_BATCH_SIZE),
            ).fetchall()
            if not rows:
                return
            for genome_number, size, data in rows:
                yield genome_number, json.loads(_decode_member(size, data))
            last_genome_number = rows[-1][0]

    def parents(self, genome_number: int) -> list[int]:
        """Returns the genome numbers of a genome's parents.

        Args:
            genome_number: The child genome.

        Returns:
            Its parents' genome numbers, sorted. Parents need not be in the
            archive (the seed genome initial genomes come from never is).
        """

        rows = self.connection.execute(
            "SELECT parent FROM genome_parents WHERE child = ? ORDER BY parent",
            (int(genome_number),),
        ).fetchall()
        return [row[0] for row in rows]

    def children(self, genome_number: int) -> list[int]:
        """Returns the genome numbers of a genome's stored children.

        Args:
            genome_number: The parent genome.

        Returns:
            Its children's genome numbers, sorted.
        """

        rows = self.connection.execute(
            "SELECT child FROM genome_parents WHERE parent = ? ORDER BY child",
            (int(genome_number),),
        ).fetchall()
        return [row[0] for row in rows]

    def _parents_of(self, genome_numbers: list[int]) -> dict[int, list[int]]:
        """Looks up the parents of several genomes at once.

        Args:
            genome_numbers: The child genomes.

        Returns:
            Each child's sorted parent genome numbers, keyed by child.
        """

        if not genome_numbers:
            return {}
        placeholders = ", ".join("?" * len(genome_numbers))
        rows = self.connection.execute(
            f"SELECT child, parent FROM genome_parents WHERE child IN ({placeholders}) ORDER BY parent",
            genome_numbers,
        ).fetchall()
        parents: dict[int, list[int]] = {}
        for child, parent in rows:
            parents.setdefault(child, []).append(parent)
        return parents
