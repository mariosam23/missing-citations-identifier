"""Sparse (tsvector) retrieval over ``citation_contexts``.

Mirrors the ``dense.py`` interface — same :class:`RetrievedContext` dataclass,
same ``target_year`` semantics — so the fusion layer is source-agnostic.

A single SQL runs ``plainto_tsquery`` against both the english-stemmed and the
simple (unstemmed) ``tsvector`` columns and ranks by ``ts_rank_cd``. The OR
across the two columns is the cheap way to recover acronyms ("LoRA", "RLHF")
that the english Snowball stemmer mangles or that land on stop-words, without
giving up real morphology ("embedding" ↔ "embeddings"). Both columns are
``GENERATED ALWAYS AS ... STORED`` with GIN indexes (Alembic 0004/0005).

AND semantics are wrong for this branch: citation sentences are short (~25
tokens) and almost never contain *every* query word, so an AND query matches
nothing ("We used finetuned BERT model" → zero rows). Sparse here is a *ranking*
branch fused by RRF, not a strict filter, so we OR the lexemes; ``ts_rank_cd``
still floats sentences matching more terms to the top.

But OR-ing *every* lexeme is unselective — common lexemes ("use" ~21% of
contexts; the unstemmed ``simple`` column's bare stopwords like "the" ~67%)
match most of the corpus, so the planner abandons the GIN indexes and
seq-scans + ranks tens of thousands of rows (~2.6s/query). We therefore drop the
high-document-frequency lexemes listed in ``sparse_stoplexeme`` (built by
``scripts.build_sparse_stoplexeme``) and OR only the distinctive survivors. The
query then rides the GIN index (~7–100ms), and since common lexemes don't
discriminate citations, precision improves too. If every query lexeme is
stoplisted, that branch contributes nothing and retrieval degrades to
dense-only — the same graceful fallback as an empty query.

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
# Build each branch's tsquery from the query's own lexemes (``to_tsvector``
# stems exactly like the stored generated columns), drop the corpus-frequent
# lexemes listed in ``sparse_stoplexeme``, then OR the survivors. Each lexeme is
# re-quoted (``'lex'``) so it is matched literally — never re-stemmed — and
# embedded quotes are doubled. ``string_agg`` over an all-stoplisted query
# yields NULL, which the ``IS NOT NULL`` guards turn into "no sparse hits"
# (caller degrades to dense-only). An empty ``sparse_stoplexeme`` drops nothing,
# so the query stays correct (just unselective) until the cache is built.
_SPARSE_SQL = text(
    r"""
    WITH eng_q AS (
      SELECT string_agg('''' || replace(w, '''', '''''') || '''', ' | ') AS q
      FROM unnest(
             tsvector_to_array(to_tsvector('english', immutable_unaccent(:query)))
           ) AS w
      WHERE NOT EXISTS (
        SELECT 1 FROM sparse_stoplexeme s
        WHERE s.config = 'english' AND s.word = w
      )
    ),
    smp_q AS (
      SELECT string_agg('''' || replace(w, '''', '''''') || '''', ' | ') AS q
      FROM unnest(
             tsvector_to_array(to_tsvector('simple', immutable_unaccent(:query)))
           ) AS w
      WHERE NOT EXISTS (
        SELECT 1 FROM sparse_stoplexeme s
        WHERE s.config = 'simple' AND s.word = w
      )
    )
    SELECT cc.context_id,
           cc.cited_paper_id,
           cc.citing_paper_id,
           cc.citing_year,
           cc.sentence_without_markers,
           GREATEST(
             CASE WHEN eng_q.q IS NOT NULL
                  THEN ts_rank_cd(cc.sentence_tsv_english, eng_q.q::tsquery)
                  ELSE 0 END,
             CASE WHEN smp_q.q IS NOT NULL
                  THEN ts_rank_cd(cc.sentence_tsv_simple, smp_q.q::tsquery)
                  ELSE 0 END
           ) AS rank_score
    FROM citation_contexts cc, eng_q, smp_q
    WHERE cc.cited_paper_id IS NOT NULL
      AND (
        (eng_q.q IS NOT NULL AND cc.sentence_tsv_english @@ eng_q.q::tsquery)
        OR (smp_q.q IS NOT NULL AND cc.sentence_tsv_simple @@ smp_q.q::tsquery)
      )
      AND (CAST(:target_year AS INTEGER) IS NULL
           OR cc.citing_year IS NULL
           OR cc.citing_year <= CAST(:target_year AS INTEGER))
      AND (CAST(:exclude_citing_paper_id AS BIGINT) IS NULL
           OR cc.citing_paper_id IS DISTINCT FROM
              CAST(:exclude_citing_paper_id AS BIGINT))
      AND (CAST(:exclude_sentence AS TEXT) IS NULL
           OR lower(btrim(regexp_replace(
                cc.sentence_without_markers, '\s+', ' ', 'g')))
              <> lower(btrim(regexp_replace(
                CAST(:exclude_sentence AS TEXT), '\s+', ' ', 'g'))))
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
    exclude_sentence: str | None = None,
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
            "exclude_sentence": exclude_sentence,
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
