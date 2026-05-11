from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine

from utils.config import config

from .base import Base

_engine: Engine | None = None
_sessionmaker: sessionmaker | None = None


def _register_pgvector_on_connect(dbapi_connection, connection_record) -> None:
    """Register pgvector psycopg adapters on every new DBAPI connection.

    Without this, ``psycopg.errors.UndefinedFunction`` is raised for ``<=>``.
    """
    from pgvector.psycopg import register_vector

    register_vector(dbapi_connection)


def _set_session_gucs(dbapi_connection, connection_record) -> None:
    """Set per-session GUCs that influence pgvector recall."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SET hnsw.ef_search = 100")
    finally:
        cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(config.DB_URL, echo=False, future=True)
        event.listen(_engine, "connect", _register_pgvector_on_connect)
        event.listen(_engine, "connect", _set_session_gucs)
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
