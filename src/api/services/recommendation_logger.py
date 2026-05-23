"""Persist recommendation events and their result sets for offline analysis.

Phase 9 feedback logging. Every ``/recommend`` call records one
``recommendation_events`` row and one ``recommendation_results`` row per
candidate, so feedback (``/feedback``) can later be tied to the exact event and
candidate it was about.

Two deliberate design choices:

* **Best-effort.** A logging failure must never break the user-facing
  recommendation. DB errors are caught, logged, and turned into a ``None``
  return; the route then omits ``event_id``/``result_id`` from the response and
  feedback simply cannot be attached to that run.
* **Own session.** Logging uses a dedicated session (separate from the
  request's read-only retrieval session) so the write commits independently and
  the recommendation transaction stays read-only.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from database.postgres.tables.recommendation_events import RecommendationEvent
from database.postgres.tables.recommendation_results import RecommendationResult
from utils.logger import logger


@dataclass(slots=True, frozen=True)
class ResultToLog:
    """One candidate to persist, in the order it was shown to the user."""

    paper_id: int
    rank: int
    score: float
    citation_key: str


@dataclass(slots=True, frozen=True)
class LoggedRecommendation:
    """IDs assigned to a persisted event; ``result_ids`` align with the input."""

    event_id: uuid.UUID
    result_ids: list[uuid.UUID]


class RecommendationLogger:
    """Inserts recommendation events + results in a dedicated transaction."""

    def __init__(
        self, session_factory: Callable[[], Session] = get_session
    ) -> None:
        self._session_factory = session_factory

    def log(
        self,
        *,
        query_text: str,
        document_path: str | None,
        target_year: int | None,
        results: list[ResultToLog],
    ) -> LoggedRecommendation | None:
        """Persist one event and its results; return assigned IDs or ``None``.

        Returns ``None`` (without raising) if the write fails, so the caller can
        still return recommendations without feedback attribution.
        """
        session = self._session_factory()
        try:
            event = RecommendationEvent(
                event_id=uuid.uuid4(),
                query_text=query_text,
                document_path=document_path,
                target_year=target_year,
            )
            session.add(event)

            rows = [
                RecommendationResult(
                    result_id=uuid.uuid4(),
                    event_id=event.event_id,
                    paper_id=result.paper_id,
                    rank=result.rank,
                    score=result.score,
                    citation_key=result.citation_key,
                )
                for result in results
            ]
            session.add_all(rows)
            session.commit()

            return LoggedRecommendation(
                event_id=event.event_id,
                result_ids=[row.result_id for row in rows],
            )
        except SQLAlchemyError:
            session.rollback()
            logger.exception(
                "failed to log recommendation event (query=%r) — "
                "returning recommendations without feedback attribution",
                query_text,
            )
            return None
        finally:
            session.close()
