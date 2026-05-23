"""``recommendation_results`` — the candidate papers shown for one event.

Phase 9 feedback logging. One row per candidate returned by ``/recommend``,
carrying the rank, the raw blended score, and the citation key shown to the
user. ``feedback_events`` reference these rows so a thumbs-up/down or a
rejection reason can be tied to the exact candidate it was about.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Double,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class RecommendationResult(Base):
    """One candidate paper shown within a ``RecommendationEvent``."""

    __tablename__ = "recommendation_results"

    result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendation_events.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.paper_id"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Double, nullable=False)
    citation_key: Mapped[str] = mapped_column(Text, nullable=False)
    shown_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        Index("idx_rec_results_event", "event_id"),
        Index("idx_rec_results_paper", "paper_id"),
    )
