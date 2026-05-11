from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Float, ForeignKey, Index, Integer, Text, func
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

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("idx_contexts_cited_paper", "cited_paper_id"),
        Index("idx_contexts_citing_paper", "citing_paper_id"),
        Index("idx_contexts_years", "citing_year", "cited_year"),
        Index("idx_contexts_section_type", "section_type"),
    )
