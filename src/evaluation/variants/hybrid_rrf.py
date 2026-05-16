"""Hybrid RRF retrieval variant — paper-level fusion.

Dense and sparse each produce their own *paper* ranking using their native
score scale, then RRF fuses the two paper rankings. This avoids the
context-level fusion pathology where sparse-only contexts entered the
aggregator with ``similarity = 0.0`` and dragged ``mean_top_3_similarity``
toward zero, making hybrid retrieval underperform dense-only.

Branch responsibilities:

* Dense branch: full aggregator (``mean_top_3_similarity`` + distinct-citers
  bonus + popularity penalty). Scoring stays in cosine space — the regime it
  was designed for.
* Sparse branch: rank papers by their best ``ts_rank_cd`` context. No
  popularity penalty here — applying it twice would punish frequently-cited
  papers asymmetrically across branches, and the scale of ``ts_rank_cd`` is
  too small for the existing log-additive penalty to behave sanely.

The production ``/recommend`` route still uses context-level fusion (for
evidence display); this paper-level path is currently scoped to the eval
harness only.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from pipeline.embedding.embedder import encode_query
from pipeline.retrieval.aggregate import (
    compute_features,
    group_by_paper,
    rank_papers,
    rank_papers_by_max_similarity,
)
from pipeline.retrieval.dense import DEFAULT_TOP_N, retrieve_dense
from pipeline.retrieval.fusion import fuse_paper_rankings
from pipeline.retrieval.sparse import retrieve_sparse


class HybridRRF:
    """Dense paper ranking + sparse paper ranking → RRF fusion."""

    name: str = "hybrid_rrf"

    def __init__(self, session: Session, *, top_n: int = DEFAULT_TOP_N) -> None:
        self._session = session
        self._top_n = top_n

    def candidates(
        self,
        query: str,
        *,
        target_year: int | None,
        exclude_citing_paper_id: int,
        top_k: int,
        query_embedding: np.ndarray | None = None,
    ) -> list[int]:
        """Return ranked ``cited_paper_id``s via paper-level hybrid RRF."""
        if query_embedding is None:
            query_embedding = encode_query(query)

        dense_ctxs = retrieve_dense(
            self._session,
            query_embedding,
            top_n=self._top_n,
            target_year=target_year,
            exclude_citing_paper_id=exclude_citing_paper_id,
        )
        sparse_ctxs = retrieve_sparse(
            self._session,
            query,
            top_n=self._top_n,
            target_year=target_year,
            exclude_citing_paper_id=exclude_citing_paper_id,
        )

        # Dense branch: full aggregator ranking. We over-fetch the paper
        # ranking (cap at ``top_n`` papers, not ``top_k``) so the fusion has a
        # deep ranking to draw from — a paper outside dense top-K but inside
        # sparse top-K can still surface after fusion.
        dense_paper_ranking: list[int] = []
        if dense_ctxs:
            dense_aggs = group_by_paper(dense_ctxs)
            compute_features(self._session, dense_aggs)
            dense_paper_ranking = [
                agg.cited_paper_id
                for agg in rank_papers(dense_aggs, top_k=self._top_n)
            ]

        # Sparse branch: rank papers by best ts_rank_cd. Empty when the query
        # reduces to stop-words; fusion then degrades to dense-only.
        sparse_paper_ranking = rank_papers_by_max_similarity(
            sparse_ctxs, top_k=self._top_n
        )

        return fuse_paper_rankings(
            [dense_paper_ranking, sparse_paper_ranking], top_k=top_k
        )
