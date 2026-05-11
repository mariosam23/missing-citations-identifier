"""Pydantic DTOs for the Phase 1 parsing pipeline.

These are the intermediate, in-memory shape of a parsed document — produced by
the TEI parser, consumed by the corpus ingestion script which writes rows into
``papers`` / ``source_documents`` / ``references`` / ``citation_contexts``.

The names mirror schema §6 columns where they overlap; everything that does not
end up in the database lives only on these objects.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedReference(BaseModel):
    """One ``<biblStruct>`` entry from a TEI bibliography."""

    ref_key: str = Field(..., description="TEI ``xml:id`` (e.g. ``b12``).")
    raw_reference_text: str | None = None
    parsed_title: str | None = None
    parsed_authors: list[str] = Field(default_factory=list)
    parsed_first_author: str | None = None
    parsed_year: int | None = None
    parsed_venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None


class ExtractedContext(BaseModel):
    """One citation occurrence — typically one ``<ref target="#bN">`` marker.

    Joint citations (e.g. ``[1, 2, 3]``) emit one ``ExtractedContext`` per
    marker; they all share the same ``citation_group_id`` and
    ``citation_group_size``.
    """

    section_name: str | None = None
    section_type: str | None = None
    paragraph_index: int
    sentence_index: int

    sentence_with_markers: str
    sentence_without_markers: str
    left_context: str | None = None
    right_context: str | None = None

    marker_text: str | None = None
    marker_start_char: int | None = None
    marker_end_char: int | None = None

    citation_group_id: str | None = None
    citation_group_size: int | None = None

    ref_key: str | None = Field(
        None,
        description=(
            "TEI bibkey this marker targets (joins to ExtractedReference.ref_key)."
            " ``None`` if the marker could not be linked."
        ),
    )

    local_window_text: str | None = None
    context_text_for_embedding: str | None = None


class ParsedDocument(BaseModel):
    """A single parsed PDF/TEI document, end-to-end."""

    openalex_id: str
    canonical_title: str | None = None
    authors: list[str] = Field(default_factory=list)
    first_author: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None

    tei_path: str
    pdf_path: str | None = None

    references: list[ExtractedReference] = Field(default_factory=list)
    contexts: list[ExtractedContext] = Field(default_factory=list)
