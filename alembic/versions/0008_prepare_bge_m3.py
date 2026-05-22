"""Prepare database for BGE-M3 embedder swap.

Drops the HNSW index, snapshots the Stella embeddings into a backup table,
and truncates the live table to prepare for the re-embed pass.

Revision ID: 0008_prepare_bge_m3
Revises: 0007_rebuild_hnsw_1024
Create Date: 2026-05-22
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008_prepare_bge_m3"
down_revision: str | None = "0007_rebuild_hnsw_1024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HNSW_INDEX = "citation_context_embedding_hnsw_idx"
BACKUP_TABLE = "citation_context_embeddings_backup_stella"


def upgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX}")

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} AS
        SELECT * FROM citation_context_embeddings
        """
    )

    op.execute("TRUNCATE TABLE citation_context_embeddings")


def downgrade() -> None:
    op.execute("TRUNCATE TABLE citation_context_embeddings")

    op.execute(
        f"""
        INSERT INTO citation_context_embeddings (context_id, embedding, model_name, created_at)
        SELECT context_id, embedding, model_name, created_at FROM {BACKUP_TABLE}
        """
    )

    op.execute(f"DROP TABLE IF EXISTS {BACKUP_TABLE}")
