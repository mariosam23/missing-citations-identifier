"""Dense cosine-similarity retrieval over ``citation_context_embeddings``.

Single query, single SQL: returns the top-N contexts whose embedding is
closest (cosine) to a query vector, joined to ``citation_contexts`` to bring
back the metadata the aggregator needs without a second round-trip.

The ``<=>`` operator is pgvector's cosine *distance* (1 - cosine similarity);
because we always normalize embeddings (in both
``pipeline.embedding.embedder`` and the bulk-embed script), this is equivalent
to ``1 - dot(a, b)`` and the HNSW index built with ``vector_cosine_ops``
matches.

Per-session ``hnsw.ef_search`` is set to 100 in
``database.postgres.engine._on_connect``; with the default of 40 the top-1000
recall is noticeably worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

DEFAULT_TOP_N = 1000


@dataclass(slots=True, frozen=True)
class RetrievedContext:
    """One row from the dense-retrieval top-N."""

    context_id: int
    cited_paper_id: int
    citing_paper_id: int | None
    citing_year: int | None
    sentence: str
    similarity: float
    rank: int


_DENSE_SQL = text(
    """
    SELECT cce.context_id,
           cc.cited_paper_id,
           cc.citing_paper_id,
           cc.citing_year,
           cc.sentence_without_markers,
           1 - (cce.embedding <=> CAST(:query_embedding AS vector)) AS similarity
    FROM citation_context_embeddings cce
    JOIN citation_contexts cc
      ON cc.context_id = cce.context_id
    WHERE cc.cited_paper_id IS NOT NULL
      AND (CAST(:target_year AS INTEGER) IS NULL
           OR cc.citing_year IS NULL
           OR cc.citing_year <= CAST(:target_year AS INTEGER))
    ORDER BY cce.embedding <=> CAST(:query_embedding AS vector)
    LIMIT :top_n
    """
)


def retrieve_dense(
    session: Session,
    query_embedding: np.ndarray,
    *,
    top_n: int = DEFAULT_TOP_N,
    target_year: int | None = None,
) -> list[RetrievedContext]:
    """Return the top-N contexts ranked by cosine similarity.

    ``query_embedding`` must be a 1-D ``(EMBEDDER_DIM,)`` float32 array; the
    pgvector psycopg adapter is registered on the engine and accepts numpy
    arrays directly as ``vector`` bind values.
    """
    rows = session.execute(
        _DENSE_SQL,
        {
            "query_embedding": query_embedding,
            "top_n": top_n,
            "target_year": target_year,
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
            )
        )
    return results
