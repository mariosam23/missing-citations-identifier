"""Aggregate dense-retrieval contexts into ranked paper candidates.

Pipeline:

    retrieve_dense(top-1000 contexts)
        → group by cited_paper_id
        → compute features (mean_top_3_similarity, distinct_citing_papers, ...)
        → score = mean_top_3_similarity
                + 0.3 * log1p(distinct_citing_papers)
                - 0.15 * log1p(global_context_count / max(distinct, 1))
        → sort desc, keep top-K
        → hydrate paper metadata + top-3 evidence contexts

Why this score? At a 1k-paper corpus, summing raw similarity surfaces
"Attention Is All You Need" and BERT for every query (the §31.2 famous-paper
trap). ``mean_top_3_similarity`` measures *how well* the strongest evidence
matches; ``distinct_citing_papers`` measures *how broadly* it is cited (an
encyclopedia citation count is not the same signal as four different teams
independently citing for the same reason); the popularity penalty knocks
down universally-cited papers from #1 on every query without removing them
from the ranking.

The penalty acts on ``global_count / distinct_in_top_n`` rather than raw
``global_count``. The ratio is "for every paper that cites X in a context
similar to the query, how many cite X globally?" — high when a paper is
cited *everywhere but for this reason* (the Transformer trap), low when a
paper is cited a lot *for this reason* (BERT on a BERT query). Using the
raw global count over-fired: BERT lost a "we use BERT to encode sentences"
query by 0.002 because the penalty ate its entire distinct-citers bonus.

The constants 0.3 and 0.15 are placeholders for the eventually-fitted
weights in §14.1; they were picked to be small enough that
``mean_top_3_similarity`` still dominates ordering.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

from pipeline.retrieval.dense import RetrievedContext

DISTINCT_CITERS_WEIGHT = 0.3
POPULARITY_PENALTY_WEIGHT = 0.15
TOP_M_FOR_MEAN = 3
DEFAULT_EVIDENCE_COUNT = 3


@dataclass(slots=True)
class PaperAggregate:
    """All retrieved contexts pointing at one ``cited_paper_id``, plus features."""

    cited_paper_id: int
    contexts: list[RetrievedContext] = field(default_factory=list)

    # Features (populated by ``compute_features``).
    mean_top_3_similarity: float = 0.0
    distinct_citing_papers: int = 0
    global_context_count: int = 0
    score: float = 0.0

    def top_evidence(self, n: int = DEFAULT_EVIDENCE_COUNT) -> list[RetrievedContext]:
        return sorted(self.contexts, key=lambda c: c.similarity, reverse=True)[:n]


def group_by_paper(
    contexts: Iterable[RetrievedContext],
) -> dict[int, PaperAggregate]:
    """Bucket retrieved contexts by ``cited_paper_id``."""
    buckets: dict[int, PaperAggregate] = {}
    for ctx in contexts:
        agg = buckets.get(ctx.cited_paper_id)
        if agg is None:
            agg = PaperAggregate(cited_paper_id=ctx.cited_paper_id)
            buckets[ctx.cited_paper_id] = agg
        agg.contexts.append(ctx)
    return buckets


def _fetch_global_counts(
    session: Session, paper_ids: list[int]
) -> dict[int, int]:
    """Return a mapping ``paper_id → total citation_contexts pointing at it``.

    Used as the popularity normalizer; bigger means "more famous", which we
    *penalize* lightly so non-Transformer-non-BERT papers can ever win.
    """
    if not paper_ids:
        return {}
    rows = session.execute(
        text(
            """
            SELECT cited_paper_id, COUNT(*)
            FROM citation_contexts
            WHERE cited_paper_id = ANY(:paper_ids)
            GROUP BY cited_paper_id
            """
        ),
        {"paper_ids": paper_ids},
    ).all()
    return {row[0]: int(row[1]) for row in rows}


def compute_features(
    session: Session, aggregates: dict[int, PaperAggregate]
) -> None:
    """Populate ``mean_top_3_similarity``, ``distinct_citing_papers``,
    ``global_context_count`` and ``score`` on each aggregate in-place.
    """
    paper_ids = list(aggregates.keys())
    global_counts = _fetch_global_counts(session, paper_ids)

    for paper_id, agg in aggregates.items():
        sims = sorted((c.similarity for c in agg.contexts), reverse=True)
        top = sims[:TOP_M_FOR_MEAN]
        agg.mean_top_3_similarity = sum(top) / len(top) if top else 0.0

        citing_papers = {
            c.citing_paper_id for c in agg.contexts if c.citing_paper_id is not None
        }
        agg.distinct_citing_papers = len(citing_papers)

        agg.global_context_count = global_counts.get(paper_id, len(agg.contexts))

        popularity_ratio = agg.global_context_count / max(agg.distinct_citing_papers, 1)
        agg.score = (
            agg.mean_top_3_similarity
            + DISTINCT_CITERS_WEIGHT * math.log1p(agg.distinct_citing_papers)
            - POPULARITY_PENALTY_WEIGHT * math.log1p(popularity_ratio)
        )


def rank_papers(
    aggregates: dict[int, PaperAggregate], top_k: int
) -> list[PaperAggregate]:
    """Return aggregates sorted by ``score`` desc, capped at ``top_k``."""
    return sorted(aggregates.values(), key=lambda a: a.score, reverse=True)[:top_k]
