"""Hybrid RRF retrieval variant — paper-level fusion (ablation).

Dense and sparse each produce their own *paper* ranking using their native
score scale, then RRF fuses the two paper rankings (see
:func:`pipeline.retrieval.hybrid.hybrid_rank_paper_level`).

This is **not** the production ranker. On full val, paper-level fusion gives the
lexical branch equal weight to the semantic one and promotes lexical-only papers
that are usually not the right citation, so it underperformed both dense-only and
context-level fusion (hit@1 / mrr worst). Production and the ``hybrid_context``
variant use the context-level :func:`pipeline.retrieval.hybrid.hybrid_rank`
instead; this variant is retained for that before/after ablation.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from pipeline.embedding.embedder import encode_query
from pipeline.retrieval.dense import DEFAULT_TOP_N
from pipeline.retrieval.hybrid import hybrid_rank_paper_level


class HybridRRF:
    """Dense paper ranking + sparse paper ranking → RRF fusion."""

    name: str = "hybrid_rrf"

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
        """Return ranked ``cited_paper_id``s via paper-level hybrid RRF."""
        if query_embedding is None:
            query_embedding = encode_query(query)

        ranked = hybrid_rank_paper_level(
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
