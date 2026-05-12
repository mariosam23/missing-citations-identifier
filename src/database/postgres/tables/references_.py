from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class Reference(Base):
    """Bibliography entry. Schema §6.3.

    Table name is the SQL-reserved-in-context word ``references``; SQLAlchemy
    quotes it correctly. Module is suffixed ``references_`` to avoid shadowing
    any future stdlib-style import.
    """

    __tablename__ = "references"

    reference_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    citing_paper_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("papers.paper_id"), nullable=True
    )
    raw_reference_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ref_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    cited_paper_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("papers.paper_id"), nullable=True
    )
    parsed_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_authors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    parsed_first_author: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parsed_venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(Text, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution_method: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
