"""Sparse (tsvector) retrieval over ``citation_contexts``.

Mirrors the ``dense.py`` interface — same :class:`RetrievedContext` dataclass,
same ``target_year`` semantics — so the fusion layer is source-agnostic.

A single SQL runs ``plainto_tsquery`` against both the english-stemmed and the
simple (unstemmed) ``tsvector`` columns and ranks by ``ts_rank_cd``. The OR
across the two columns is the cheap way to recover acronyms ("LoRA", "RLHF")
that the english Snowball stemmer mangles or that land on stop-words, without
giving up real morphology ("embedding" ↔ "embeddings"). Both columns are
``GENERATED ALWAYS AS ... STORED`` with GIN indexes (Alembic 0004/0005).

``plainto_tsquery`` ANDs every lexeme together, which is wrong for this branch:
citation sentences are short (~25 tokens) and almost never contain *every*
query word, so an AND query matches nothing ("We used finetuned BERT model" →
``'use' & 'finetun' & 'bert' & 'model'`` → zero rows). Sparse here is a
*ranking* branch fused by RRF, not a strict filter, so we rewrite the query to
OR semantics (``&`` → ``|``); ``ts_rank_cd`` still floats sentences matching
more terms to the top. The rewrite is a textual ``&`` → ``|`` swap on the
``tsquery``; lexemes never contain a literal ``&``, so this is safe.

``similarity`` on the returned :class:`RetrievedContext` is the ``ts_rank_cd``
score — not comparable to dense cosine similarity, but that is fine: the fusion
layer ranks, it does not compare scores.

If the query reduces to zero lexemes after stop-word removal (e.g. "in the to
of"), ``plainto_tsquery`` yields an empty query, the ``@@`` match is false for
every row, and this function returns an empty list — the caller degrades to
dense-only.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from pipeline.retrieval.dense import DEFAULT_TOP_N, ContextSource, RetrievedContext

# ``immutable_unaccent`` is the IMMUTABLE wrapper around ``unaccent('unaccent',
# ...)`` created in Alembic 0004 — required because the stored generated
# columns are built with it, so the query must normalize identically.
_SPARSE_SQL = text(
    """
    WITH q AS (
      SELECT
        replace(
          plainto_tsquery('english', immutable_unaccent(:query))::text,
          ' & ', ' | '
        )::tsquery AS q_eng,
        replace(
          plainto_tsquery('simple', immutable_unaccent(:query))::text,
          ' & ', ' | '
        )::tsquery AS q_smp
    )
    SELECT cc.context_id,
           cc.cited_paper_id,
           cc.citing_paper_id,
           cc.citing_year,
           cc.sentence_without_markers,
           GREATEST(
             ts_rank_cd(cc.sentence_tsv_english, q.q_eng),
             ts_rank_cd(cc.sentence_tsv_simple,  q.q_smp)
           ) AS rank_score
    FROM citation_contexts cc, q
    WHERE (cc.sentence_tsv_english @@ q.q_eng OR cc.sentence_tsv_simple @@ q.q_smp)
      AND cc.cited_paper_id IS NOT NULL
      AND (CAST(:target_year AS INTEGER) IS NULL
           OR cc.citing_year IS NULL
           OR cc.citing_year <= CAST(:target_year AS INTEGER))
      AND (CAST(:exclude_citing_paper_id AS BIGINT) IS NULL
           OR cc.citing_paper_id IS DISTINCT FROM
              CAST(:exclude_citing_paper_id AS BIGINT))
    ORDER BY rank_score DESC
    LIMIT :top_n
    """
)


def retrieve_sparse(
    session: Session,
    query: str,
    *,
    top_n: int = DEFAULT_TOP_N,
    target_year: int | None = None,
    exclude_citing_paper_id: int | None = None,
) -> list[RetrievedContext]:
    """Return the top-N contexts ranked by ``ts_rank_cd`` over both tsvectors.

    ``query`` is the raw user query string; tokenisation, stemming and
    stop-word removal happen inside Postgres via ``plainto_tsquery``. Returns
    an empty list when the query produces zero lexemes.

    ``exclude_citing_paper_id``, when set, filters out all contexts whose
    ``citing_paper_id`` matches — used by the eval harness to prevent
    leakage from a test paper's own citations.
    """
    rows = session.execute(
        _SPARSE_SQL,
        {
            "query": query,
            "top_n": top_n,
            "target_year": target_year,
            "exclude_citing_paper_id": exclude_citing_paper_id,
        },
    ).all()

    results: list[RetrievedContext] = []
    for rank, row in enumerate(rows, start=1):
        results.append(
            RetrievedContext(
                context_id=row[0],
                cited_paper_id=row[1],
                citing_paper_id=row[2],
                citing_year=row[3],
                sentence=row[4] or "",
                similarity=float(row[5]),
                rank=rank,
                source=ContextSource.SPARSE,
            )
        )
    return results
