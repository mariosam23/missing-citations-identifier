"""Dense-only retrieval variant.

Uses only the ``retrieve_dense`` path (cosine similarity over
``citation_context_embeddings``) — no sparse branch, no fusion.
Serves as the Phase 3 baseline.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from pipeline.embedding.embedder import encode_query
from pipeline.retrieval.aggregate import (
    compute_features,
    group_by_paper,
    rank_papers,
)
from pipeline.retrieval.dense import DEFAULT_TOP_N, retrieve_dense


class DenseOnly:
    """Dense cosine retrieval + aggregation, no sparse branch."""

    name: str = "dense_only"

    def __init__(
        self,
        session: Session,
        *,
        top_n: int = DEFAULT_TOP_N,
        leak_free: bool = False,
    ) -> None:
        self._session = session
        self._top_n = top_n
        self._leak_free = leak_free

    def candidates(
        self,
        query: str,
        *,
        target_year: int | None,
        exclude_citing_paper_id: int,
        top_k: int,
        query_embedding: np.ndarray | None = None,
    ) -> list[int]:
        """Return ranked cited_paper_ids via dense retrieval only."""
        if query_embedding is None:
            query_embedding = encode_query(query)

        contexts = retrieve_dense(
            self._session,
            query_embedding,
            top_n=self._top_n,
            target_year=target_year,
            exclude_citing_paper_id=exclude_citing_paper_id,
            exclude_sentence=query if self._leak_free else None,
        )
        if not contexts:
            return []

        aggregates = group_by_paper(contexts)
        compute_features(self._session, aggregates)
        ranked = rank_papers(aggregates, top_k=top_k)
        return [agg.cited_paper_id for agg in ranked]
