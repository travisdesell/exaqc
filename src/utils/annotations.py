"""Notes and tags that people and agents attach to a run after it has been searched.

A run's archive (``genomes.sqlar``) is the record of what its search did, and
nothing but the search writes to it. What someone concludes about the run
afterwards -- that a genome is a candidate worth refining, why another was set
aside -- is kept beside it instead, in ``annotations.sqlite`` in the same
directory. The archive therefore stays exactly as the search left it, however
the run is annotated, and deleting the sidecar removes every annotation without
touching the results.

Two kinds of annotation are kept:

* **Notes** are free text about a genome, or about the run as a whole. They are
  append-only -- never edited or deleted -- so they form a record of what was
  concluded and when.
* **Tags** are short labels on a genome that mark its current standing
  (``candidate``, ``rejected``). A tag can be removed, but the removal is
  stamped on the tag's row rather than deleting it, so a genome's tag history
  can still be read.

Neither the dashboard nor the MCP interface authenticates anyone, so an
annotation records only what can be known: which of them it came through, and
an optional name the writer gave.

Reading never creates the sidecar, so browsing a run leaves its directory
unchanged; the file appears on the first write. It uses SQLite's default
rollback journal rather than write-ahead logging, because a run directory may
sit on a shared file system that does not support it and annotation writes are
rare enough that the difference does not matter.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

from src.utils.genome_archive import (
    BUSY_TIMEOUT_SECONDS,
    WRITE_RETRIES,
    WRITE_RETRY_DELAY_SECONDS,
)

#: File name of a run's annotations, beside its ``genomes.sqlar``.
ANNOTATIONS_FILENAME = "annotations.sqlite"

#: Where an annotation can be written from.
SOURCES = ("mcp", "dashboard")

#: The longest note accepted, in characters.
MAX_NOTE_LENGTH = 10_000

#: The longest author name accepted, in characters.
MAX_AUTHOR_LENGTH = 100

#: What a tag may look like: a letter or digit, then up to 63 letters, digits or
#: ``_ . : -``. Short and space-free, so a tag reads as a label and can be used as
#: a SQL string without surprises.
TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")

#: The columns of the ``notes`` table, in order.
NOTE_COLUMNS = ("note_id", "genome_number", "text", "source", "author", "created_at")

#: The columns of the ``genome_tags`` table, in order.
TAG_COLUMNS = (
    "tag_id",
    "genome_number",
    "tag",
    "source",
    "author",
    "added_at",
    "removed_at",
    "removed_source",
    "removed_author",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes(
    note_id INTEGER PRIMARY KEY,
    genome_number INTEGER,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    author TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_genome ON notes(genome_number);
CREATE TABLE IF NOT EXISTS genome_tags(
    tag_id INTEGER PRIMARY KEY,
    genome_number INTEGER NOT NULL,
    tag TEXT NOT NULL,
    source TEXT NOT NULL,
    author TEXT,
    added_at REAL NOT NULL,
    removed_at REAL,
    removed_source TEXT,
    removed_author TEXT
);
CREATE INDEX IF NOT EXISTS genome_tags_genome ON genome_tags(genome_number);
CREATE INDEX IF NOT EXISTS genome_tags_tag ON genome_tags(tag);
CREATE UNIQUE INDEX IF NOT EXISTS genome_tags_active
    ON genome_tags(genome_number, tag) WHERE removed_at IS NULL;
"""

_Result = TypeVar("_Result")


def _clean_source(source: str) -> str:
    """Checks where an annotation is being written from.

    Args:
        source: The writer's source.

    Returns:
        The source, unchanged.

    Raises:
        ValueError: If ``source`` is not one of :data:`SOURCES`.
    """

    if source not in SOURCES:
        raise ValueError(
            f"An annotation's source must be one of {', '.join(SOURCES)}, not {source!r}."
        )
    return source


def _clean_author(author: str | None) -> str | None:
    """Normalizes an optional author name.

    Args:
        author: The name given, if any.

    Returns:
        The name with surrounding whitespace removed, or ``None`` when none was
        given or it was blank.

    Raises:
        ValueError: If the name is longer than :data:`MAX_AUTHOR_LENGTH`.
    """

    if author is None:
        return None
    author = author.strip()
    if not author:
        return None
    if len(author) > MAX_AUTHOR_LENGTH:
        raise ValueError(
            f"An author's name may be at most {MAX_AUTHOR_LENGTH} characters."
        )
    return author


def clean_tag(tag: str) -> str:
    """Checks that a tag is a short, space-free label.

    Args:
        tag: The tag given.

    Returns:
        The tag with surrounding whitespace removed.

    Raises:
        ValueError: If the tag does not match :data:`TAG_PATTERN`.
    """

    tag = (tag or "").strip()
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError(
            f"{tag!r} is not a tag: a tag starts with a letter or digit and holds at "
            "most 64 letters, digits, '_', '.', ':' or '-' (no spaces), such as "
            "'candidate' or 'needs:rerun'."
        )
    return tag


def _note(row: tuple[Any, ...]) -> dict[str, Any]:
    """Turns a ``notes`` row into a dict.

    Args:
        row: The row, in :data:`NOTE_COLUMNS` order.

    Returns:
        The note, keyed by column name.
    """

    return dict(zip(NOTE_COLUMNS, row))


def _tag(row: tuple[Any, ...]) -> dict[str, Any]:
    """Turns a ``genome_tags`` row into a dict.

    Args:
        row: The row, in :data:`TAG_COLUMNS` order.

    Returns:
        The tag, keyed by column name, with ``active`` saying whether it still
        applies.
    """

    tag = dict(zip(TAG_COLUMNS, row))
    tag["active"] = tag["removed_at"] is None
    return tag


class AnnotationStore:
    """The annotations kept beside one run's archive.

    Attributes:
        path: Path of the run's ``annotations.sqlite``, which need not exist.
    """

    def __init__(self, directory: str) -> None:
        """Locates the annotations of the run in a directory.

        Args:
            directory: The run's output directory, holding its ``genomes.sqlar``.
        """

        self.path = os.path.join(directory, ANNOTATIONS_FILENAME)

    @classmethod
    def beside(cls, archive_path: str) -> AnnotationStore:
        """Locates the annotations of the run an archive belongs to.

        Args:
            archive_path: The run's ``genomes.sqlar``.

        Returns:
            The store beside it.
        """

        return cls(os.path.dirname(os.path.abspath(archive_path)))

    def exists(self) -> bool:
        """Says whether anything has been annotated yet.

        Returns:
            Whether the sidecar file exists.
        """

        return os.path.isfile(self.path)

    @contextlib.contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection | None]:
        """Opens the sidecar for reading, without creating it.

        Yields:
            A read-only connection, or ``None`` when nothing has been annotated
            yet.
        """

        if not self.exists():
            yield None
            return
        connection = sqlite3.connect(
            f"{Path(self.path).resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=BUSY_TIMEOUT_SECONDS,
        )
        try:
            yield connection
        finally:
            connection.close()

    def _write(
        self, description: str, operation: Callable[[sqlite3.Connection], _Result]
    ) -> _Result:
        """Runs a write in one transaction, retrying while the file is locked.

        Unlike the archive, which gives up on a write rather than stop a search,
        an annotation is written because someone asked for it, so a write that
        still cannot be made after retrying is raised rather than dropped.

        Args:
            description: What is being written, for log messages.
            operation: Performs the writes on the connection, inside the
                transaction, and returns what the caller should receive.

        Returns:
            Whatever ``operation`` returned.

        Raises:
            sqlite3.OperationalError: If the file stays locked through every
                retry, or the write fails for another reason.
        """

        delay = WRITE_RETRY_DELAY_SECONDS
        for attempt in range(1, WRITE_RETRIES + 2):
            connection = sqlite3.connect(
                self.path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None
            )
            try:
                connection.executescript(_SCHEMA)
                connection.execute("BEGIN IMMEDIATE")
                try:
                    result = operation(connection)
                    connection.execute("COMMIT")
                except BaseException:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
                return result
            except sqlite3.OperationalError as error:
                message = str(error).lower()
                retryable = "locked" in message or "busy" in message
                if not retryable or attempt > WRITE_RETRIES:
                    raise
                logger.warning(
                    "{} was locked while writing {} (attempt {}); retrying in {:.0f}s.",
                    self.path,
                    description,
                    attempt,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            finally:
                connection.close()
        raise AssertionError("unreachable: the last attempt either returns or raises")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def notes(
        self, genome_number: int | None = None, run_only: bool = False
    ) -> list[dict[str, Any]]:
        """Lists notes, oldest first.

        Args:
            genome_number: Only list notes about this genome.
            run_only: Only list notes about the run as a whole; ignored when
                ``genome_number`` is given.

        Returns:
            The notes, each keyed by :data:`NOTE_COLUMNS`; empty when nothing has
            been annotated.
        """

        with self._reading() as connection:
            if connection is None:
                return []
            where, parameters = "", []
            if genome_number is not None:
                where, parameters = "WHERE genome_number = ?", [int(genome_number)]
            elif run_only:
                where = "WHERE genome_number IS NULL"
            rows = connection.execute(
                f"SELECT {', '.join(NOTE_COLUMNS)} FROM notes {where} "
                "ORDER BY created_at, note_id",
                parameters,
            ).fetchall()
        return [_note(row) for row in rows]

    def tags(
        self,
        genome_number: int | None = None,
        tag: str | None = None,
        include_removed: bool = False,
    ) -> list[dict[str, Any]]:
        """Lists genome tags, oldest first.

        Args:
            genome_number: Only list this genome's tags.
            tag: Only list this tag.
            include_removed: Also list tags that were removed, with when and by
                whom; by default only tags that still apply are listed.

        Returns:
            The tags, each keyed by :data:`TAG_COLUMNS` plus ``active``; empty
            when nothing has been annotated.
        """

        with self._reading() as connection:
            if connection is None:
                return []
            conditions: list[str] = []
            parameters: list[Any] = []
            if genome_number is not None:
                conditions.append("genome_number = ?")
                parameters.append(int(genome_number))
            if tag is not None:
                conditions.append("tag = ?")
                parameters.append(tag)
            if not include_removed:
                conditions.append("removed_at IS NULL")
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = connection.execute(
                f"SELECT {', '.join(TAG_COLUMNS)} FROM genome_tags {where} "
                "ORDER BY added_at, tag_id",
                parameters,
            ).fetchall()
        return [_tag(row) for row in rows]

    def table_rows(self, table: str) -> list[tuple[Any, ...]]:
        """Reads every row of one annotations table, for copying into a query.

        Args:
            table: ``notes`` or ``genome_tags``.

        Returns:
            The rows, in :data:`NOTE_COLUMNS` or :data:`TAG_COLUMNS` order; empty
            when nothing has been annotated.

        Raises:
            ValueError: If ``table`` is not an annotations table.
        """

        columns = {"notes": NOTE_COLUMNS, "genome_tags": TAG_COLUMNS}.get(table)
        if columns is None:
            raise ValueError(f"{table!r} is not an annotations table.")
        with self._reading() as connection:
            if connection is None:
                return []
            return connection.execute(
                f"SELECT {', '.join(columns)} FROM {table}"
            ).fetchall()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_note(
        self,
        text: str,
        source: str,
        author: str | None = None,
        genome_number: int | None = None,
    ) -> dict[str, Any]:
        """Records a note about a genome, or about the run when no genome is given.

        Notes are never edited or deleted, so each call adds one, even if the
        same text was noted before.

        Args:
            text: What to note.
            source: Where the note is written from, one of :data:`SOURCES`.
            author: The writer's name, if they gave one.
            genome_number: The genome the note is about, or ``None`` for the run.

        Returns:
            The note as recorded.

        Raises:
            ValueError: If the text is blank or longer than
                :data:`MAX_NOTE_LENGTH`, the source is unknown, or the author's
                name is too long.
        """

        text = (text or "").strip()
        if not text:
            raise ValueError("A note needs some text.")
        if len(text) > MAX_NOTE_LENGTH:
            raise ValueError(f"A note may be at most {MAX_NOTE_LENGTH} characters.")
        row = (
            None if genome_number is None else int(genome_number),
            text,
            _clean_source(source),
            _clean_author(author),
            time.time(),
        )

        def store(connection: sqlite3.Connection) -> dict[str, Any]:
            """Inserts the note and returns it."""

            cursor = connection.execute(
                "INSERT INTO notes(genome_number, text, source, author, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                row,
            )
            return _note((cursor.lastrowid, *row))

        return self._write("a note", store)

    def add_tag(
        self, genome_number: int, tag: str, source: str, author: str | None = None
    ) -> dict[str, Any]:
        """Tags a genome, doing nothing if it already carries the tag.

        Args:
            genome_number: The genome to tag.
            tag: The tag, as :data:`TAG_PATTERN` allows.
            source: Where the tag is written from, one of :data:`SOURCES`.
            author: The writer's name, if they gave one.

        Returns:
            The tag that now applies, with ``created`` saying whether this call
            added it or it was already there.

        Raises:
            ValueError: If the tag, source or author's name is invalid.
        """

        row = (
            int(genome_number),
            clean_tag(tag),
            _clean_source(source),
            _clean_author(author),
            time.time(),
        )

        def store(connection: sqlite3.Connection) -> dict[str, Any]:
            """Inserts the tag unless it already applies, and returns it."""

            existing = connection.execute(
                f"SELECT {', '.join(TAG_COLUMNS)} FROM genome_tags "
                "WHERE genome_number = ? AND tag = ? AND removed_at IS NULL",
                row[:2],
            ).fetchone()
            if existing is not None:
                return {**_tag(existing), "created": False}
            cursor = connection.execute(
                "INSERT INTO genome_tags(genome_number, tag, source, author, added_at) "
                "VALUES (?, ?, ?, ?, ?)",
                row,
            )
            return {
                **_tag((cursor.lastrowid, *row, None, None, None)),
                "created": True,
            }

        return self._write(f"tag {row[1]!r} on genome {row[0]}", store)

    def remove_tag(
        self, genome_number: int, tag: str, source: str, author: str | None = None
    ) -> dict[str, Any]:
        """Removes a tag from a genome, keeping a record that it was removed.

        Args:
            genome_number: The genome to untag.
            tag: The tag to remove.
            source: Where the removal is made from, one of :data:`SOURCES`.
            author: The remover's name, if they gave one.

        Returns:
            The tag's row, now stamped with when, where from and by whom it was
            removed.

        Raises:
            KeyError: If the genome does not currently carry the tag.
            ValueError: If the tag, source or author's name is invalid.
        """

        number = int(genome_number)
        cleaned = clean_tag(tag)
        removal = (time.time(), _clean_source(source), _clean_author(author))

        def store(connection: sqlite3.Connection) -> dict[str, Any]:
            """Stamps the active tag as removed and returns it."""

            existing = connection.execute(
                "SELECT tag_id FROM genome_tags "
                "WHERE genome_number = ? AND tag = ? AND removed_at IS NULL",
                (number, cleaned),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Genome {number} is not tagged {cleaned!r}.")
            connection.execute(
                "UPDATE genome_tags SET removed_at = ?, removed_source = ?, "
                "removed_author = ? WHERE tag_id = ?",
                (*removal, existing[0]),
            )
            return _tag(
                connection.execute(
                    f"SELECT {', '.join(TAG_COLUMNS)} FROM genome_tags WHERE tag_id = ?",
                    (existing[0],),
                ).fetchone()
            )

        if not self.exists():
            raise KeyError(f"Genome {number} is not tagged {cleaned!r}.")
        return self._write(f"removal of tag {cleaned!r} from genome {number}", store)
