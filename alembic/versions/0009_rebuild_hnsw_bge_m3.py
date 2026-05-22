"""Rebuild the HNSW cosine index over the new BGE-M3 1024-dim embeddings.

Apply **after** the bulk re-embed pass (``scripts.embed_contexts``) that
follows revision ``0008_prepare_bge_m3``.

Revision ID: 0009_rebuild_hnsw_bge_m3
Revises: 0008_prepare_bge_m3
Create Date: 2026-05-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009_rebuild_hnsw_bge_m3"
down_revision: str | None = "0008_prepare_bge_m3"
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
