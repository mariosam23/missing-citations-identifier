"""Aggregate dense-retrieval contexts into ranked paper candidates.

Pipeline:

    retrieve_dense(top-1000 contexts)
        → group by cited_paper_id
        → compute features (mean_top_m_similarity, distinct_citing_papers, ...)
        → score = mean_top_m_similarity
                + 0.1 * log1p(distinct_citing_papers)
        → sort desc, keep top-K
        → hydrate paper metadata + top-3 evidence contexts

Why this score? ``mean_top_3_similarity`` measures *how well* the strongest
evidence matches; ``distinct_citing_papers`` measures *how broadly* it is
cited — an answer corroborated by several independent citers, each phrasing
the citation similarly to the query, is more trustworthy than a single match.

History — the popularity penalty (removed): the score used to subtract
``0.15 * log1p(global_context_count / max(distinct, 1))`` to fight the
"famous-paper trap" (BERT/Transformer ranking #1 on every query). On the
full val split (1,202 queries, bge-large) that penalty *halved* recall —
recall@20 0.397 → 0.188 — because it demoted the genuinely-correct papers far
more than it suppressed noise. Removing it ~doubled recall@10/@20 with no
re-embedding, so it is gone. If the famous-paper trap resurfaces, prefer a
much smaller weight (the sweep showed even 0.05 cost ~9 recall@20 points) or a
different mechanism entirely. ``global_context_count`` is still computed as a
diagnostic feature (surfaced by ``scripts.debug_recommend``).

The weights ``DISTINCT_CITERS_WEIGHT = 0.1`` and ``TOP_M_FOR_MEAN = 1`` were
fitted by ``scripts.sweep_scoring`` on the seed-42 split (see the constant
comments below); the similarity term still dominates ordering.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

from pipeline.retrieval.dense import RetrievedContext

# Tuned by ``scripts.sweep_scoring`` on the seed-42 split (2026-05-29). A grid
# over distinct_weight × top_m, confirmed on both val (1,711 q) and test
# (2,282 q), beat the previous 0.3 / top_m=3 placeholder on every metric:
# val recall@5 0.253→0.297, mrr@20 0.182→0.228; test recall@5 0.367→0.416,
# mrr@20 0.249→0.308. Two findings: the single strongest context (top_m=1)
# scores better than averaging the top 3 (averaging dilutes the best
# evidence), and the distinct-citers bonus helps but was over-weighted
# (0.1 > 0.3 > 0.0).
DISTINCT_CITERS_WEIGHT = 0.1
# Popularity penalty disabled: on the val split it ~halved recall by demoting
# correct papers. Kept at 0.0 (rather than deleting the term) so the lever is
# discoverable and re-tunable. See module docstring.
POPULARITY_PENALTY_WEIGHT = 0.0
# Number of strongest contexts averaged into ``mean_top_3_similarity`` (the
# field name predates this becoming tunable). 1 wins the sweep.
TOP_M_FOR_MEAN = 1
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
    session: Session,
    aggregates: dict[int, PaperAggregate],
    *,
    distinct_weight: float = DISTINCT_CITERS_WEIGHT,
    top_m: int = TOP_M_FOR_MEAN,
    penalty_weight: float = POPULARITY_PENALTY_WEIGHT,
) -> None:
    """Populate ``mean_top_3_similarity``, ``distinct_citing_papers``,
    ``global_context_count`` and ``score`` on each aggregate in-place.

    The three weights default to the module constants so existing callers keep
    the production scoring; ``scripts.sweep_scoring`` overrides them to grid-search
    the val split. ``mean_top_3_similarity`` is the mean of the best ``top_m``
    contexts (the field name predates the tunable cutoff).

    The popularity normalizer requires a DB round-trip for global citation
    counts; it is skipped entirely when ``penalty_weight == 0`` (the production
    default), so scoring stays fully in-memory and the sweep needs no DB calls.
    """
    paper_ids = list(aggregates.keys())
    global_counts = (
        _fetch_global_counts(session, paper_ids) if penalty_weight else {}
    )

    for paper_id, agg in aggregates.items():
        sims = sorted((c.similarity for c in agg.contexts), reverse=True)
        top = sims[:top_m]
        agg.mean_top_3_similarity = sum(top) / len(top) if top else 0.0

        citing_papers = {
            c.citing_paper_id for c in agg.contexts if c.citing_paper_id is not None
        }
        agg.distinct_citing_papers = len(citing_papers)

        agg.global_context_count = global_counts.get(paper_id, len(agg.contexts))

        score = agg.mean_top_3_similarity + distinct_weight * math.log1p(
            agg.distinct_citing_papers
        )
        if penalty_weight:
            popularity_ratio = agg.global_context_count / max(
                agg.distinct_citing_papers, 1
            )
            score -= penalty_weight * math.log1p(popularity_ratio)
        agg.score = score


def rank_papers(
    aggregates: dict[int, PaperAggregate], top_k: int
) -> list[PaperAggregate]:
    """Return aggregates sorted by ``score`` desc, capped at ``top_k``."""
    return sorted(aggregates.values(), key=lambda a: a.score, reverse=True)[:top_k]


def rank_papers_by_max_similarity(
    contexts: Iterable[RetrievedContext], top_k: int
) -> list[int]:
    """Rank ``cited_paper_id``s by their best context's ``similarity``, capped at ``top_k``.

    Branch-agnostic: works for dense (cosine) or sparse (ts_rank_cd) contexts
    because the ranking is intra-branch — the absolute scale never crosses
    branches. Used by the paper-level RRF fusion path where each branch
    produces its own paper ranking before fusion.
    """
    best: dict[int, float] = {}
    for ctx in contexts:
        prev = best.get(ctx.cited_paper_id)
        if prev is None or ctx.similarity > prev:
            best[ctx.cited_paper_id] = ctx.similarity
    return [pid for pid, _ in sorted(best.items(), key=lambda kv: -kv[1])[:top_k]]
