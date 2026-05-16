"""Sparse-only retrieval variant.

Uses only the ``retrieve_sparse`` path (tsvector ``ts_rank_cd``) — no
dense branch, no fusion. Useful for measuring the raw keyword-matching
quality of the BM25-like path.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from pipeline.retrieval.aggregate import (
    compute_features,
    group_by_paper,
    rank_papers,
)
from pipeline.retrieval.dense import DEFAULT_TOP_N
from pipeline.retrieval.sparse import retrieve_sparse


class SparseOnly:
    """Sparse tsvector retrieval + aggregation, no dense branch."""

    name: str = "sparse_only"

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
        """Return ranked cited_paper_ids via sparse retrieval only.

        ``query_embedding`` is accepted for protocol conformance but
        unused — sparse retrieval operates on the raw query text.
        """
        contexts = retrieve_sparse(
            self._session,
            query,
            top_n=self._top_n,
            target_year=target_year,
            exclude_citing_paper_id=exclude_citing_paper_id,
        )
        if not contexts:
            return []

        aggregates = group_by_paper(contexts)
        compute_features(self._session, aggregates)
        ranked = rank_papers(aggregates, top_k=top_k)
        return [agg.cited_paper_id for agg in ranked]
