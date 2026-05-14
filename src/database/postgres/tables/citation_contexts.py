from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Computed,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class CitationContext(Base):
    """Citation occurrence. Schema §6.4 — the central table."""

    __tablename__ = "citation_contexts"

    context_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    citing_paper_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("papers.paper_id"), nullable=True
    )
    cited_paper_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("papers.paper_id"), nullable=True
    )
    reference_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("references.reference_id"), nullable=True
    )

    section_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sentence_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    sentence_with_markers: Mapped[str] = mapped_column(Text, nullable=False)
    sentence_without_markers: Mapped[str] = mapped_column(Text, nullable=False)
    left_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    right_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    marker_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    marker_start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    marker_end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_group_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_group_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    local_window_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_text_for_embedding: Mapped[str | None] = mapped_column(Text, nullable=True)

    citing_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cited_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age_at_citation: Mapped[int | None] = mapped_column(Integer, nullable=True)

    citation_role: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_role_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Sparse-retrieval index columns (Phase 6 — hybrid retrieval). Both are
    # ``GENERATED ALWAYS AS ... STORED`` from ``sentence_without_markers`` and
    # read-only on the Python side. ``english`` applies Snowball stemming
    # ("embeddings" → "embed"); ``simple`` preserves raw tokens so acronyms
    # ("LoRA") survive. The sparse query ORs across both. ``immutable_unaccent``
    # is the IMMUTABLE wrapper created in Alembic 0004 — a STORED generated
    # column cannot use the merely-STABLE ``unaccent(text)``.
    sentence_tsv_english: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', "
            "coalesce(immutable_unaccent(sentence_without_markers), ''))",
            persisted=True,
        ),
        nullable=True,
    )
    sentence_tsv_simple: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', "
            "coalesce(immutable_unaccent(sentence_without_markers), ''))",
            persisted=True,
        ),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("idx_contexts_cited_paper", "cited_paper_id"),
        Index("idx_contexts_citing_paper", "citing_paper_id"),
        Index("idx_contexts_years", "citing_year", "cited_year"),
        Index("idx_contexts_section_type", "section_type"),
        Index(
            "idx_contexts_sentence_tsv_english",
            "sentence_tsv_english",
            postgresql_using="gin",
        ),
        Index(
            "idx_contexts_sentence_tsv_simple",
            "sentence_tsv_simple",
            postgresql_using="gin",
        ),
    )
