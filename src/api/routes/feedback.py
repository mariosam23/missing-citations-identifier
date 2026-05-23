"""POST /feedback — record a user interaction with a recommendation.

Phase 9 feedback logging. The webview (and the quickpick fallback) call this
after the user accepts, rejects, thumbs-up/downs, copies BibTeX, or opens a
candidate's URL. Payloads are stored in plaintext for offline analysis.

Idempotency: a ``UNIQUE NULLS NOT DISTINCT (event_id, result_id,
feedback_type)`` constraint (Alembic ``0012``) lets us upsert, so repeated
feedback of the same type updates the existing row instead of inserting a
duplicate. This keeps the dataset clean when the UI fires the same event twice
(double-click, retry).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from api.deps import db_session
from api.schemas import FeedbackRequest, FeedbackResponse
from database.postgres.tables.feedback_events import (
    UQ_FEEDBACK_CONSTRAINT,
    FeedbackEvent,
)
from database.postgres.tables.recommendation_events import RecommendationEvent
from database.postgres.tables.recommendation_results import RecommendationResult

router = APIRouter(tags=["feedback"])

DbSession = Annotated[Session, Depends(db_session)]


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    summary="Record a feedback interaction for a recommendation.",
)
def submit_feedback(
    request: FeedbackRequest,
    session: DbSession,
) -> FeedbackResponse:
    _verify_references(session, request)

    stmt = (
        pg_insert(FeedbackEvent)
        .values(
            event_id=request.event_id,
            result_id=request.result_id,
            feedback_type=request.feedback_type.value,
            feedback_value=request.feedback_value,
            reason=request.reason,
        )
        .on_conflict_do_update(
            constraint=UQ_FEEDBACK_CONSTRAINT,
            set_={
                "feedback_value": request.feedback_value,
                "reason": request.reason,
                "created_at": func.now(),
            },
        )
        .returning(FeedbackEvent.feedback_id)
    )

    feedback_id = session.execute(stmt).scalar_one()
    session.commit()
    return FeedbackResponse(feedback_id=feedback_id)


def _verify_references(session: Session, request: FeedbackRequest) -> None:
    """404 if the referenced event (or result) does not exist.

    Catches stale clients posting against a run that was never logged or has
    since been deleted, instead of failing later with an opaque FK error.
    """
    event_exists = session.execute(
        select(RecommendationEvent.event_id).where(
            RecommendationEvent.event_id == request.event_id
        )
    ).first()
    if event_exists is None:
        raise HTTPException(status_code=404, detail="event_id not found")

    if request.result_id is None:
        return

    result = session.execute(
        select(RecommendationResult.event_id).where(
            RecommendationResult.result_id == request.result_id
        )
    ).first()
    if result is None:
        raise HTTPException(status_code=404, detail="result_id not found")
    if result[0] != request.event_id:
        raise HTTPException(
            status_code=400,
            detail="result_id does not belong to event_id",
        )
