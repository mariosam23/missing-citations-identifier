"""Hybrid retrieval variant — context-level fusion (the production ranker).

Delegates to :func:`pipeline.retrieval.hybrid.hybrid_rank`, the same canonical
context-level path the ``/recommend`` route uses — so this variant measures
exactly what the extension serves. RRF fuses the dense + sparse *context*
rankings, the fused contexts are grouped by paper, and the cosine-space
aggregator (``mean_top_3_similarity`` + distinct-citers bonus) picks the top-K.

On full val this beat both dense-only and the paper-level ``hybrid_rrf``
ablation on every metric, which is why it is the production ranker. A
sparse-only context enters the aggregator at ``similarity = 0.0`` (see
``pipeline.retrieval.fusion``); the sparse branch therefore mainly adds recall
of lexically-exact tokens (acronyms, dataset names) without letting bare keyword
matches outrank semantic ones.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from pipeline.embedding.embedder import encode_query
from pipeline.retrieval.dense import DEFAULT_TOP_N
from pipeline.retrieval.hybrid import hybrid_rank


class HybridContext:
    """Context-level RRF + cosine-space aggregation (production ranker)."""

    name: str = "hybrid_context"

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
        """Return ranked ``cited_paper_id``s via context-level fusion."""
        if query_embedding is None:
            query_embedding = encode_query(query)

        ranked = hybrid_rank(
            self._session,
            query,
            query_embedding,
            top_n=self._top_n,
            top_k=top_k,
            target_year=target_year,
            exclude_citing_paper_id=exclude_citing_paper_id,
            exclude_sentence=query if self._leak_free else None,
        )
        return [agg.cited_paper_id for agg in ranked]
