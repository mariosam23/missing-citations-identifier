"""Shared hybrid ranking — the single source of truth for prod + eval.

Production ``POST /recommend`` and the evaluation variants call into this module
so the served system and the reported metrics rank identically (they used to
fork, describing a different ranker than the extension served).

Two fusion strategies live here; they differ only in *where* the dense and
sparse branches are combined:

* **Context-level** (:func:`hybrid_rank`, the **canonical** path) — RRF fuses
  the dense + sparse *context* rankings, the fused contexts are grouped by
  paper, and the cosine-space aggregator (``mean_top_3_similarity`` +
  distinct-citers bonus) picks the top-K. This is what ``/recommend`` and the
  ``hybrid_context`` eval variant use. On full val it beat the alternative on
  every metric, so it is the production ranker.
* **Paper-level** (:func:`hybrid_rank_paper_level`, an **ablation**) — each
  branch first reduces to its own *paper* ranking (dense by the aggregator,
  sparse by best ``ts_rank_cd``), then RRF fuses the two paper rankings. Surfaces
  lexical-only papers more aggressively, but on full val that *hurt* precision
  (it promotes lexical matches that are usually not the right citation), so it
  is kept only for the ``hybrid_rrf`` comparison.

Both strategies return ranked :class:`PaperAggregate`\\ s carrying the contexts
used for evidence display; the context-level ``score`` is the cosine-space
aggregator score, the paper-level ``score`` is the RRF fused score.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sqlalchemy.orm import Session

from pipeline.retrieval.aggregate import (
    PaperAggregate,
    compute_features,
    group_by_paper,
    rank_papers,
    rank_papers_by_max_similarity,
)
from pipeline.retrieval.dense import (
    DEFAULT_TOP_N,
    RetrievedContext,
    retrieve_dense,
)
from pipeline.retrieval.fusion import (
    fuse_paper_rankings_scored,
    reciprocal_rank_fusion,
)
from pipeline.retrieval.sparse import retrieve_sparse
from utils.logger import logger


def _retrieve_dense_sparse(
    session: Session,
    query: str,
    query_embedding: np.ndarray,
    *,
    top_n: int,
    target_year: int | None,
    exclude_citing_paper_id: int | None,
    exclude_sentence: str | None,
) -> tuple[list[RetrievedContext], list[RetrievedContext]]:
    """Run both retrieval branches with shared filters; return ``(dense, sparse)``."""
    dense_ctxs = retrieve_dense(
        session,
        query_embedding,
        top_n=top_n,
        target_year=target_year,
        exclude_citing_paper_id=exclude_citing_paper_id,
        exclude_sentence=exclude_sentence,
    )
    sparse_ctxs = retrieve_sparse(
        session,
        query,
        top_n=top_n,
        target_year=target_year,
        exclude_citing_paper_id=exclude_citing_paper_id,
        exclude_sentence=exclude_sentence,
    )
    logger.debug(
        "hybrid retrieval — dense=%d sparse=%d (top_n=%d)",
        len(dense_ctxs),
        len(sparse_ctxs),
        top_n,
    )
    return dense_ctxs, sparse_ctxs


# ---------------------------------------------------------------------------
# Context-level fusion (canonical)
# ---------------------------------------------------------------------------

def rank_paper_aggregates_context_level(
    session: Session | None,
    dense_ctxs: list[RetrievedContext],
    sparse_ctxs: list[RetrievedContext],
    *,
    top_n: int,
    top_k: int,
) -> list[PaperAggregate]:
    """Context-level RRF → group by paper → cosine-space score → top-K.

    Sparse-only contexts enter the aggregator at ``similarity = 0.0`` (see
    ``pipeline.retrieval.fusion``), so a paper is ranked by its strongest
    *dense* evidence plus the distinct-citers bonus. ``score`` is the aggregator
    score. Unit-testable with ``session=None`` (penalty disabled).
    """
    fused = reciprocal_rank_fusion([dense_ctxs, sparse_ctxs], top_n=top_n)
    if not fused:
        return []
    aggregates = group_by_paper(fused)
    compute_features(session, aggregates)
    return rank_papers(aggregates, top_k=top_k)


def hybrid_rank(
    session: Session,
    query: str,
    query_embedding: np.ndarray,
    *,
    top_n: int = DEFAULT_TOP_N,
    top_k: int,
    target_year: int | None = None,
    exclude_citing_paper_id: int | None = None,
    exclude_sentence: str | None = None,
) -> list[PaperAggregate]:
    """Canonical hybrid ranking (context-level fusion).

    The entry point ``/recommend`` and the ``hybrid_context`` eval variant share.
    ``exclude_citing_paper_id`` is the eval leakage guard (omit in production);
    ``exclude_sentence`` is the leak-free guard that drops verbatim duplicates.
    """
    dense_ctxs, sparse_ctxs = _retrieve_dense_sparse(
        session,
        query,
        query_embedding,
        top_n=top_n,
        target_year=target_year,
        exclude_citing_paper_id=exclude_citing_paper_id,
        exclude_sentence=exclude_sentence,
    )
    return rank_paper_aggregates_context_level(
        session, dense_ctxs, sparse_ctxs, top_n=top_n, top_k=top_k
    )


# ---------------------------------------------------------------------------
# Paper-level fusion (ablation — hybrid_rrf)
# ---------------------------------------------------------------------------

def rank_paper_aggregates(
    session: Session | None,
    dense_ctxs: list[RetrievedContext],
    sparse_ctxs: list[RetrievedContext],
    *,
    top_n: int,
    top_k: int,
) -> list[PaperAggregate]:
    """Paper-level RRF: fuse the two branches' *paper* rankings.

    ``top_n`` caps each branch's paper ranking before fusion (deeper than
    ``top_k`` so a paper outside one branch's top-K can still surface);
    ``top_k`` caps the fused result. Each aggregate's ``score`` is the RRF fused
    score. Unit-testable with ``session=None`` (penalty disabled).
    """
    dense_aggs = group_by_paper(dense_ctxs)
    compute_features(session, dense_aggs)
    dense_ranking = [
        agg.cited_paper_id for agg in rank_papers(dense_aggs, top_k=top_n)
    ]

    sparse_aggs = group_by_paper(sparse_ctxs)
    sparse_ranking = rank_papers_by_max_similarity(sparse_ctxs, top_k=top_n)

    fused = fuse_paper_rankings_scored(
        [dense_ranking, sparse_ranking], top_k=top_k
    )

    ranked: list[PaperAggregate] = []
    for paper_id, rrf_score in fused:
        agg = dense_aggs.get(paper_id)
        if agg is None:
            # Sparse-only paper: no dense (cosine) context survived the top-N
            # cut. Carry its sparse contexts for evidence but zero their
            # similarity so the cosine-space ``mean_top_3_similarity`` and the
            # client confidence meter stay well-defined in [0, 1].
            sparse_agg = sparse_aggs[paper_id]
            agg = PaperAggregate(
                cited_paper_id=paper_id,
                contexts=[replace(c, similarity=0.0) for c in sparse_agg.contexts],
            )
        agg.score = rrf_score
        ranked.append(agg)
    return ranked


def hybrid_rank_paper_level(
    session: Session,
    query: str,
    query_embedding: np.ndarray,
    *,
    top_n: int = DEFAULT_TOP_N,
    top_k: int,
    target_year: int | None = None,
    exclude_citing_paper_id: int | None = None,
    exclude_sentence: str | None = None,
) -> list[PaperAggregate]:
    """Paper-level fusion ablation — used only by the ``hybrid_rrf`` variant."""
    dense_ctxs, sparse_ctxs = _retrieve_dense_sparse(
        session,
        query,
        query_embedding,
        top_n=top_n,
        target_year=target_year,
        exclude_citing_paper_id=exclude_citing_paper_id,
        exclude_sentence=exclude_sentence,
    )
    return rank_paper_aggregates(
        session, dense_ctxs, sparse_ctxs, top_n=top_n, top_k=top_k
    )
