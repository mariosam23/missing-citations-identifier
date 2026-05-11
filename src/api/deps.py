"""FastAPI dependencies: DB session and lazy embedder singleton."""

from __future__ import annotations

from collections.abc import Iterator
from threading import Lock
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from database.postgres.engine import get_session

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def db_session() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


_embedder: SentenceTransformer | None = None
_embedder_lock = Lock()


def get_embedder() -> SentenceTransformer:
    """Return the process-wide SentenceTransformer, loading it on first call.

    Lazy so that import of this module (and therefore the FastAPI app) does not
    pull in torch/sentence-transformers at server start time.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            from utils.config import config

            _embedder = SentenceTransformer(config.EMBEDDER_MODEL_NAME)
    return _embedder
