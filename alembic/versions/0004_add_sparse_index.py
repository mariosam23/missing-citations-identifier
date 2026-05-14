"""Add sparse-retrieval tsvector columns to citation_contexts.

Phase 6 — hybrid retrieval. Adds two ``GENERATED ALWAYS AS ... STORED``
``tsvector`` columns on ``citation_contexts``:

* ``sentence_tsv_english`` — ``'english'`` config (Snowball stemmer): folds
  "embedding" ↔ "embeddings", drops English stop-words.
* ``sentence_tsv_simple`` — ``'simple'`` config: no stemming, no stop-words, so
  acronyms ("LoRA", "RLHF") that the english stemmer mangles survive.

The sparse query (``pipeline.retrieval.sparse``) ORs ``ts_rank_cd`` across both.

**The ``unaccent`` gotcha.** A ``STORED`` generated column requires an
IMMUTABLE expression, but the single-argument ``unaccent(text)`` is only
STABLE (it resolves the default dictionary at call time). The two-argument
``unaccent(regdictionary, text)`` *is* immutable, so we wrap it in
``immutable_unaccent(text)`` pinned to the ``'unaccent'`` dictionary and use
that wrapper in both the generated columns and the sparse query.

The GIN indexes are intentionally **not** built here — see revision 0005,
which builds them with ``CREATE INDEX CONCURRENTLY`` so the index step does
not lock the table. This revision still rewrites the table once to populate
the generated columns on existing rows (~30s on 80k rows); plan for it.

Revision ID: 0004_add_sparse_index
Revises: 0003_create_hnsw_index
Create Date: 2026-05-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004_add_sparse_index"
down_revision: str | None = "0003_create_hnsw_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    # IMMUTABLE wrapper: the 2-arg unaccent() form is immutable; pin it to the
    # 'unaccent' dictionary so STORED generated columns can use it.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION immutable_unaccent(text)
          RETURNS text
          LANGUAGE sql
          IMMUTABLE PARALLEL SAFE STRICT
        AS $$ SELECT unaccent('unaccent', $1) $$
        """
    )

    op.execute(
        """
        ALTER TABLE citation_contexts
          ADD COLUMN sentence_tsv_english tsvector
            GENERATED ALWAYS AS (
              to_tsvector(
                'english',
                coalesce(immutable_unaccent(sentence_without_markers), '')
              )
            ) STORED,
          ADD COLUMN sentence_tsv_simple tsvector
            GENERATED ALWAYS AS (
              to_tsvector(
                'simple',
                coalesce(immutable_unaccent(sentence_without_markers), '')
              )
            ) STORED
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE citation_contexts "
        "DROP COLUMN IF EXISTS sentence_tsv_english, "
        "DROP COLUMN IF EXISTS sentence_tsv_simple"
    )
    op.execute("DROP FUNCTION IF EXISTS immutable_unaccent(text)")
