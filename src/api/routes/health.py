"""Liveness probe verifying both the DB and the pgvector extension."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import db_session

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz(session: Session = Depends(db_session)) -> dict[str, str]:
    try:
        session.execute(text("SELECT 1")).scalar_one()
    except Exception as exc:  # noqa: BLE001 — surface real error to caller
        raise HTTPException(status_code=503, detail=f"db unreachable: {exc}") from exc

    pgvector_version = session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one_or_none()
    if pgvector_version is None:
        raise HTTPException(status_code=503, detail="pgvector extension missing")

    return {"db": "ok", "pgvector": pgvector_version}
