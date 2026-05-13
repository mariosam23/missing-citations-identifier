"""FastAPI dependencies: DB session and lazy embedder singleton."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from pipeline.embedding.embedder import get_embedder as _get_embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def db_session() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def get_embedder() -> SentenceTransformer:
    """FastAPI-friendly accessor for the process-wide embedder singleton."""
    return _get_embedder()
