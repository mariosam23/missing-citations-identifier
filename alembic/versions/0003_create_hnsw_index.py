"""Build the HNSW cosine index on citation_context_embeddings.

Run **after** the bulk-embed script (``scripts.embed_contexts``) finishes —
HNSW builds are ~10× faster on a fully-populated table than they are when
the index has to be maintained during insert.

``vector_cosine_ops`` matches the ``<=>`` operator used by
``pipeline.retrieval.dense``. Embeddings are L2-normalized at write time,
so cosine distance equals 1 - dot product. Per-session ``hnsw.ef_search``
is set to 100 in ``database.postgres.engine._on_connect`` (the default 40
under-recalls at top-1000).

``m=16, ef_construction=64`` are the pgvector defaults — good enough for
~50k vectors. We will revisit if recall drops below acceptable on a
larger corpus.

Revision ID: 0003_create_hnsw_index
Revises: 0002_add_resolution_method
Create Date: 2026-05-12
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_create_hnsw_index"
down_revision: str | None = "0002_add_resolution_method"
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
