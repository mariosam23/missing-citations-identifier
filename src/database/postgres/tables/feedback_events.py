"""``feedback_events`` — user interactions with recommended candidates.

Phase 9 feedback logging. Each row records one interaction: the citation was
accepted/inserted, given a thumbs-up/down, copied as BibTeX, opened in the
browser, or rejected (with an optional free-text ``reason`` from the webview's
rejection-reason dropdown — e.g. "too generic", "wrong time period").

A ``UNIQUE NULLS NOT DISTINCT (event_id, result_id, feedback_type)`` constraint
(created in Alembic ``0012``) makes repeated feedback of the same type
idempotent: the ``/feedback`` route upserts, so clicking thumbs-up twice
updates the existing row instead of inserting a duplicate. ``NULLS NOT
DISTINCT`` (Postgres 15+) extends that to event-level feedback where
``result_id`` is NULL.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base

UQ_FEEDBACK_CONSTRAINT = "uq_feedback_event_result_type"


class FeedbackEvent(Base):
    """A single user interaction with a recommendation event or candidate."""

    __tablename__ = "feedback_events"

    feedback_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendation_events.event_id", ondelete="CASCADE"),
        nullable=False,
    )
    result_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendation_results.result_id", ondelete="CASCADE"),
        nullable=True,
    )
    feedback_type: Mapped[str] = mapped_column(Text, nullable=False)
    feedback_value: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "result_id",
            "feedback_type",
            name=UQ_FEEDBACK_CONSTRAINT,
            postgresql_nulls_not_distinct=True,
        ),
        Index("idx_feedback_event", "event_id"),
        Index("idx_feedback_result", "result_id"),
    )
