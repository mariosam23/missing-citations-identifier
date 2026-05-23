"""``recommendation_events`` — one row per ``/recommend`` invocation.

Phase 9 feedback logging. Stores the raw query, the (optional) document path
the request came from, and the target-year filter, in plaintext, so the
offline analysis pipeline can reconstruct exactly what was asked and when.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class RecommendationEvent(Base):
    """A single recommendation run. Parent of ``recommendation_results``."""

    __tablename__ = "recommendation_events"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    document_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
