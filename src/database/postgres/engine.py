from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from utils.config import config

from .base import Base

_engine: Engine | None = None
_sessionmaker: sessionmaker | None = None


def _on_connect(dbapi_connection, connection_record) -> None:
    """Register pgvector type adapter and set per-session GUCs."""
    from pgvector.psycopg import register_vector  # type: ignore[import-untyped]

    register_vector(dbapi_connection)
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SET hnsw.ef_search = 100")
    finally:
        cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(config.DB_URL, echo=False, future=True)
        event.listen(_engine, "connect", _on_connect)
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _sessionmaker


def get_session():
    return get_sessionmaker()()


def init_db() -> None:
    """Create all tables. Intended for test fixtures only.

    Production schema is managed by Alembic.
    """
    Base.metadata.create_all(bind=get_engine())
