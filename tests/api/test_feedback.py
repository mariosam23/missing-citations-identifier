"""Integration tests for the Phase 9 ``POST /feedback`` route.

These are integration tests: they need a live Postgres with the Phase 9
feedback schema (Alembic ``0012``) applied, and skip — not fail — when it is
unavailable, so the pure-unit suite still passes anywhere.

Unlike the rolled-back ``pg_session`` tests, the ``/feedback`` route commits, so
this fixture commits its setup rows and deletes them on teardown (the
``ON DELETE CASCADE`` on ``recommendation_events`` removes results + feedback).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

_PAPER_ID = 92_000_001


@pytest.fixture
def feedback_env() -> Iterator[dict[str, object]]:
    """Commit one paper + event + result; yield their IDs; clean up after."""
    from database.postgres.engine import get_session
    from database.postgres.tables.recommendation_events import RecommendationEvent
    from database.postgres.tables.recommendation_results import RecommendationResult

    session = get_session()
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment-dependent
        session.close()
        pytest.skip(f"Postgres unavailable: {exc}")

    feedback_ready = session.execute(
        text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'feedback_events'"
        )
    ).first()
    if feedback_ready is None:
        session.close()
        pytest.skip("Phase 9 feedback schema not applied (run Alembic 0012)")

    event_id = uuid.uuid4()
    result_id = uuid.uuid4()
    try:
        session.execute(
            text(
                "INSERT INTO papers "
                "(paper_id, canonical_title, normalized_title, first_author, year) "
                "VALUES (:pid, :t, :nt, :a, :y)"
            ),
            {
                "pid": _PAPER_ID,
                "t": "Feedback Test Paper",
                "nt": "feedback test paper",
                "a": "Tester",
                "y": 2020,
            },
        )
        session.add(
            RecommendationEvent(
                event_id=event_id,
                query_text="a query about long-sequence modeling",
                document_path="paper.tex",
                target_year=2021,
            )
        )
        session.add(
            RecommendationResult(
                result_id=result_id,
                event_id=event_id,
                paper_id=_PAPER_ID,
                rank=1,
                score=0.5,
                citation_key="tester2020feedback",
            )
        )
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        session.close()
        raise

    try:
        yield {"event_id": event_id, "result_id": result_id, "session": session}
    finally:
        session.rollback()
        session.execute(
            text("DELETE FROM recommendation_events WHERE event_id = :e"),
            {"e": event_id},
        )
        session.execute(
            text("DELETE FROM papers WHERE paper_id = :p"), {"p": _PAPER_ID}
        )
        session.commit()
        session.close()


@pytest.fixture
def client() -> Iterator:
    """A TestClient over the real app — the route hits the real DB and commits."""
    from fastapi.testclient import TestClient

    from api.main import app

    yield TestClient(app)


def _count_feedback(
    session: Session, event_id: uuid.UUID, feedback_type: str
) -> int:
    session.rollback()  # end any open txn → fresh READ COMMITTED snapshot
    return session.execute(
        text(
            "SELECT COUNT(*) FROM feedback_events "
            "WHERE event_id = :e AND feedback_type = :t"
        ),
        {"e": event_id, "t": feedback_type},
    ).scalar_one()


def test_candidate_feedback_inserts_row(client, feedback_env) -> None:
    event_id = feedback_env["event_id"]
    result_id = feedback_env["result_id"]
    session: Session = feedback_env["session"]  # type: ignore[assignment]

    response = client.post(
        "/feedback",
        json={
            "event_id": str(event_id),
            "result_id": str(result_id),
            "feedback_type": "thumbs_up",
            "feedback_value": 1,
        },
    )

    assert response.status_code == 201, response.text
    assert "feedback_id" in response.json()
    assert _count_feedback(session, event_id, "thumbs_up") == 1


def test_duplicate_feedback_is_idempotent(client, feedback_env) -> None:
    event_id = feedback_env["event_id"]
    result_id = feedback_env["result_id"]
    session: Session = feedback_env["session"]  # type: ignore[assignment]

    payload = {
        "event_id": str(event_id),
        "result_id": str(result_id),
        "feedback_type": "thumbs_down",
        "feedback_value": -1,
    }
    first = client.post("/feedback", json=payload)
    second = client.post("/feedback", json=payload)

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    # Upsert on the unique constraint — the second call updates, not inserts.
    assert _count_feedback(session, event_id, "thumbs_down") == 1
    # Same row reused.
    assert first.json()["feedback_id"] == second.json()["feedback_id"]


def test_event_level_feedback_allows_null_result(client, feedback_env) -> None:
    event_id = feedback_env["event_id"]
    session: Session = feedback_env["session"]  # type: ignore[assignment]

    payload = {
        "event_id": str(event_id),
        "feedback_type": "rejected",
        "reason": "too generic",
    }
    first = client.post("/feedback", json=payload)
    second = client.post("/feedback", json=payload)

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    # NULLS NOT DISTINCT keeps event-level feedback idempotent too.
    assert _count_feedback(session, event_id, "rejected") == 1


def test_unknown_event_returns_404(client, feedback_env) -> None:
    response = client.post(
        "/feedback",
        json={"event_id": str(uuid.uuid4()), "feedback_type": "accepted"},
    )
    assert response.status_code == 404


def test_unknown_result_returns_404(client, feedback_env) -> None:
    event_id = feedback_env["event_id"]
    response = client.post(
        "/feedback",
        json={
            "event_id": str(event_id),
            "result_id": str(uuid.uuid4()),
            "feedback_type": "accepted",
        },
    )
    assert response.status_code == 404


def test_invalid_feedback_type_rejected(client, feedback_env) -> None:
    event_id = feedback_env["event_id"]
    response = client.post(
        "/feedback",
        json={"event_id": str(event_id), "feedback_type": "not_a_real_type"},
    )
    assert response.status_code == 422  # pydantic enum validation
