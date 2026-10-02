"""Tests for the notes and tags kept beside a run's archive.

Annotations live in their own file so a run's archive is never written after its
search; these tests pin how that file behaves. Notes accumulate and are never
changed, tags can be removed but keep their history, reading never creates the
file, and what is written is checked before it is kept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.annotations import (
    ANNOTATIONS_FILENAME,
    MAX_NOTE_LENGTH,
    AnnotationStore,
    clean_tag,
)


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    """An empty store for a run directory.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).

    Returns:
        The store, with nothing annotated yet.
    """

    return AnnotationStore(str(tmp_path))


def test_reading_an_unannotated_run_creates_nothing(
    store: AnnotationStore, tmp_path: Path
) -> None:
    """Browsing a run leaves its directory exactly as it was.

    Args:
        store: The empty store.
        tmp_path: The run directory.
    """

    assert store.notes() == []
    assert store.tags() == []
    assert store.table_rows("notes") == []
    assert not store.exists()
    assert not (tmp_path / ANNOTATIONS_FILENAME).exists()


def test_the_store_sits_beside_its_archive(tmp_path: Path) -> None:
    """A run's annotations are found from its archive's path.

    Args:
        tmp_path: pytest per-test temporary directory (auto-removed).
    """

    store = AnnotationStore.beside(str(tmp_path / "genomes.sqlar"))
    assert Path(store.path) == tmp_path / ANNOTATIONS_FILENAME


def test_notes_accumulate_about_genomes_and_the_run(store: AnnotationStore) -> None:
    """Every note is kept, in order, whether it is about a genome or the run.

    Args:
        store: The empty store.
    """

    first = store.add_note("promising", "mcp", author="agent", genome_number=7)
    store.add_note("promising", "dashboard", genome_number=7)
    store.add_note("the ring topology converged early", "dashboard", author=" ")

    assert first["genome_number"] == 7
    assert first["source"] == "mcp"
    assert first["author"] == "agent"

    about_genome = store.notes(genome_number=7)
    # the same text noted twice is two notes: nothing is merged or replaced
    assert [note["text"] for note in about_genome] == ["promising", "promising"]
    assert [note["source"] for note in about_genome] == ["mcp", "dashboard"]

    (about_run,) = store.notes(run_only=True)
    assert about_run["genome_number"] is None
    # a blank author is recorded as no author rather than as whitespace
    assert about_run["author"] is None

    assert len(store.notes()) == 3


def test_a_removed_tag_keeps_its_history(store: AnnotationStore) -> None:
    """Removing a tag stamps it rather than deleting it, and it can be re-added.

    Args:
        store: The empty store.
    """

    added = store.add_tag(3, "candidate", "mcp", author="agent")
    assert added["created"] is True
    assert added["active"] is True
    assert [tag["tag"] for tag in store.tags(genome_number=3)] == ["candidate"]

    removed = store.remove_tag(3, "candidate", "dashboard", author="travis")
    assert removed["active"] is False
    assert removed["removed_source"] == "dashboard"
    assert removed["removed_author"] == "travis"
    assert removed["removed_at"] >= removed["added_at"]

    # no longer applies, but is still on record
    assert store.tags(genome_number=3) == []
    (history,) = store.tags(genome_number=3, include_removed=True)
    assert history["tag_id"] == added["tag_id"]

    readded = store.add_tag(3, "candidate", "dashboard")
    assert readded["created"] is True
    assert readded["tag_id"] != added["tag_id"]
    assert len(store.tags(genome_number=3, include_removed=True)) == 2
    assert len(store.tags(genome_number=3)) == 1


def test_tagging_twice_leaves_one_tag(store: AnnotationStore) -> None:
    """A tag that already applies is left as it is rather than duplicated.

    Args:
        store: The empty store.
    """

    first = store.add_tag(5, "rejected", "mcp")
    again = store.add_tag(5, "rejected", "dashboard", author="someone else")

    assert again["created"] is False
    assert again["tag_id"] == first["tag_id"]
    # the original writer is kept, not overwritten by the repeated call
    assert again["source"] == "mcp"
    assert len(store.tags(genome_number=5, include_removed=True)) == 1


def test_tags_can_be_listed_by_tag_across_genomes(store: AnnotationStore) -> None:
    """Asking for a tag finds every genome carrying it.

    Args:
        store: The empty store.
    """

    for number in (1, 4, 9):
        store.add_tag(number, "candidate", "mcp")
    store.add_tag(4, "rejected", "mcp")
    store.remove_tag(9, "candidate", "mcp")

    assert [tag["genome_number"] for tag in store.tags(tag="candidate")] == [1, 4]


def test_removing_a_tag_that_does_not_apply_is_refused(store: AnnotationStore) -> None:
    """A removal names the tag and genome that were not found.

    Args:
        store: The empty store.
    """

    with pytest.raises(KeyError, match="not tagged 'candidate'"):
        store.remove_tag(2, "candidate", "mcp")
    # refusing did not create the sidecar either
    assert not store.exists()

    store.add_tag(2, "candidate", "mcp")
    store.remove_tag(2, "candidate", "mcp")
    with pytest.raises(KeyError, match="not tagged 'candidate'"):
        store.remove_tag(2, "candidate", "mcp")


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda s: s.add_note("   ", "mcp"), "needs some text"),
        (lambda s: s.add_note("x" * (MAX_NOTE_LENGTH + 1), "mcp"), "at most"),
        (lambda s: s.add_note("fine", "email"), "source must be one of"),
        (lambda s: s.add_note("fine", "mcp", author="a" * 101), "author's name"),
        (lambda s: s.add_tag(1, "has spaces", "mcp"), "is not a tag"),
        (lambda s: s.add_tag(1, "-leading", "mcp"), "is not a tag"),
        (lambda s: s.add_tag(1, "", "mcp"), "is not a tag"),
    ],
)
def test_invalid_annotations_are_refused(
    store: AnnotationStore, call: object, message: str
) -> None:
    """What is written is checked, and a refusal says why.

    Args:
        store: The empty store.
        call: The invalid write to attempt.
        message: Part of the explanation expected.
    """

    with pytest.raises(ValueError, match=message):
        call(store)  # type: ignore[operator]
    assert not store.exists()


def test_tag_cleaning_keeps_useful_labels() -> None:
    """Short labels with separators are accepted, trimmed of surrounding spaces."""

    assert clean_tag("  candidate ") == "candidate"
    assert clean_tag("needs:rerun") == "needs:rerun"
    assert clean_tag("seed-3.v2") == "seed-3.v2"


def test_table_rows_are_copied_in_column_order(store: AnnotationStore) -> None:
    """Rows for queries come back complete, in the declared column order.

    Args:
        store: The empty store.
    """

    store.add_note("a", "mcp", genome_number=1)
    store.add_tag(1, "candidate", "mcp")

    (note,) = store.table_rows("notes")
    assert note[1:4] == (1, "a", "mcp")
    (tag,) = store.table_rows("genome_tags")
    assert tag[1:4] == (1, "candidate", "mcp")
    assert tag[6] is None  # removed_at

    with pytest.raises(ValueError, match="not an annotations table"):
        store.table_rows("genomes")
