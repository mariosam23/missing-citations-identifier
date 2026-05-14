"""Build the GIN indexes on the sparse tsvector columns, CONCURRENTLY.

Phase 6 — hybrid retrieval. Split out from revision 0004 so the index build
can use ``CREATE INDEX CONCURRENTLY``, which does not take an
``ACCESS EXCLUSIVE`` lock on ``citation_contexts`` — important because the
table is read on every ``/recommend`` call.

``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction block, so the
statements run inside ``op.get_context().autocommit_block()``. A concurrent
build can leave an ``INVALID`` index behind if it is interrupted; if that
happens, ``DROP INDEX`` the invalid one and re-run this revision.

Two GIN indexes roughly double the sparse index footprint (~200MB at 80k
contexts); flag for review at ~1M contexts (Phase 14).

Revision ID: 0005_sparse_gin_indexes
Revises: 0004_add_sparse_index
Create Date: 2026-05-14

(The revision id is kept short — ``alembic_version.version_num`` is
``varchar(32)``; the descriptive name lives in the module filename.)
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005_sparse_gin_indexes"
down_revision: str | None = "0004_add_sparse_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEXES: tuple[tuple[str, str], ...] = (
    ("idx_contexts_sentence_tsv_english", "sentence_tsv_english"),
    ("idx_contexts_sentence_tsv_simple", "sentence_tsv_simple"),
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        for index_name, column in _INDEXES:
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} "
                f"ON citation_contexts USING gin ({column})"
            )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for index_name, _column in _INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}")
