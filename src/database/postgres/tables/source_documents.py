from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Float, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class SourceDocument(Base):
    """Parsed full-text document. Schema §6.2."""

    __tablename__ = "source_documents"

    doc_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    paper_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("papers.paper_id"), nullable=True
    )
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
