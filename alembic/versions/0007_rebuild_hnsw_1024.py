"""Rebuild the HNSW cosine index over the new 1024-dim embeddings.

Apply **after** the bulk re-embed pass (``scripts.embed_contexts``) that
follows revision ``0006_resize_embedding_vector``. Same operator class and
parameters as the original ``0003_create_hnsw_index`` — the only change is
that the underlying column is now ``vector(1024)``, so the index footprint
is ~1.3× larger and the build is correspondingly slower (still seconds at
current corpus size).

Same index name as 0003 so downstream session settings
(``hnsw.ef_search = 100`` in ``database.postgres.engine._on_connect``) and
query planner heuristics continue to apply without modification.

Revision ID: 0007_rebuild_hnsw_1024
Revises: 0006_resize_embedding_vector
Create Date: 2026-05-16
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0007_rebuild_hnsw_1024"
down_revision: str | None = "0006_resize_embedding_vector"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "citation_context_embedding_hnsw_idx"


def upgrade() -> None:
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {INDEX_NAME}
        ON citation_context_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
