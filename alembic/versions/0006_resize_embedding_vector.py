"""Resize ``citation_context_embeddings.embedding`` from vector(768) to vector(1024).

Phase 13a (embedder swap) — moving from ``BAAI/bge-base-en-v1.5`` (768) to
``dunzhang/stella_en_1.5B_v5`` (matryoshka, cut to 1024). The dim is part of
the column type in pgvector, so the swap requires an ``ALTER COLUMN`` and a
full re-embed pass.

Steps:

1. Drop the HNSW index (``citation_context_embedding_hnsw_idx``). It is
   bound to the column's current dim and would block the type change; it
   is also far cheaper to drop and rebuild in a follow-up revision once
   the new 1024-dim vectors are in place than to maintain it during the
   bulk re-embed insert path.

2. Snapshot the old vectors into ``citation_context_embeddings_backup_bge_base``.
   The snapshot table keeps the old ``vector(768)`` column type intact and
   makes rollback a single ``TRUNCATE`` + ``INSERT SELECT`` after a
   ``downgrade``. No re-embed needed if the new model loses the gate.

3. Truncate the live table — pgvector cannot ``ALTER TYPE`` a populated
   column to a different dim (vectors of one dim are not assignable to
   another). Truncating is equivalent to a drop-and-recreate of the rows
   and is what the bulk-embed script expects (resumable anti-join, empty
   destination ⇒ embed everything).

4. ``ALTER COLUMN embedding TYPE vector(1024)``. Now safe because the
   table is empty.

The HNSW rebuild is intentionally split into ``0007_rebuild_hnsw_1024``
so it runs **after** the re-embed pass, mirroring the original
``0001 → embed → 0003`` order.

Revision ID: 0006_resize_embedding_vector
Revises: 0005_sparse_gin_indexes
Create Date: 2026-05-16
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0006_resize_embedding_vector"
down_revision: str | None = "0005_sparse_gin_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HNSW_INDEX = "citation_context_embedding_hnsw_idx"
BACKUP_TABLE = "citation_context_embeddings_backup_bge_base"
OLD_DIM = 768
NEW_DIM = 1024


def upgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} AS
        SELECT * FROM citation_context_embeddings
        """
    )

    op.execute("TRUNCATE TABLE citation_context_embeddings")

    op.execute(
        f"ALTER TABLE citation_context_embeddings "
        f"ALTER COLUMN embedding TYPE vector({NEW_DIM})"
    )


def downgrade() -> None:
    op.execute("TRUNCATE TABLE citation_context_embeddings")

    op.execute(
        f"ALTER TABLE citation_context_embeddings "
        f"ALTER COLUMN embedding TYPE vector({OLD_DIM})"
    )

    op.execute(
        f"""
        INSERT INTO citation_context_embeddings (context_id, embedding, model_name, created_at)
        SELECT context_id, embedding, model_name, created_at FROM {BACKUP_TABLE}
        """
    )

    op.execute(f"DROP TABLE IF EXISTS {BACKUP_TABLE}")
