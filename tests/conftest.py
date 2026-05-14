"""Shared pytest fixtures.

The retrieval/API hybrid tests are *integration* tests: they need a live
Postgres with the Phase 6 sparse schema (Alembic 0004/0005) applied. They are
skipped — not failed — when the database is unreachable or the migration has
not run, so the pure-unit suite (e.g. ``test_fusion``) still passes anywhere.

Each ``pg_session`` test runs inside a transaction that is rolled back at
teardown, so test rows never persist.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


@pytest.fixture
def pg_session() -> Iterator[Session]:
    """Yield a Postgres session whose work is rolled back at teardown.

    Skips the test when Postgres is unreachable or the Phase 6 sparse columns
    are absent.
    """
    from database.postgres.engine import get_session

    session = get_session()
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment-dependent
        session.close()
        pytest.skip(f"Postgres unavailable: {exc}")

    sparse_ready = session.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'citation_contexts' "
            "AND column_name = 'sentence_tsv_english'"
        )
    ).first()
    if sparse_ready is None:
        session.close()
        pytest.skip("Phase 6 sparse schema not applied (run Alembic 0004/0005)")

    try:
        yield session
    finally:
        session.rollback()
        session.close()
