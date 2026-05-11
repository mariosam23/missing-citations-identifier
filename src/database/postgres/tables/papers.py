from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class Paper(Base):
    """Canonical paper identity. Schema §6.1."""

    __tablename__ = "papers"

    paper_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    canonical_title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    first_author: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(Text, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_survey: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("idx_papers_normalized_title", "normalized_title"),
        Index("idx_papers_year", "year"),
        Index("idx_papers_first_author_year", "first_author", "year"),
        Index("idx_papers_doi", "doi"),
        Index("idx_papers_arxiv_id", "arxiv_id"),
    )
