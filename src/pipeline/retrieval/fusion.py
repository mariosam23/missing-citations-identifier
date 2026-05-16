"""Reciprocal Rank Fusion over multiple :class:`RetrievedContext` rankings.

Merge by ``context_id``. The RRF score of a context is the sum, over every
ranking it appears in, of ``1 / (k + rank)``. Sort by that score, re-rank from
1, and return the top-N. ``k = 60`` is the §10.5 default — it has no data to
tune and is robust to the wildly different score scales of dense cosine
similarity and sparse ``ts_rank_cd``.

The fused list keeps the :class:`RetrievedContext` shape so the aggregator is
unchanged, with two deliberate conventions:

* ``rank`` is the *post-fusion* rank (1-based).
* ``similarity`` is the *dense* cosine similarity if the context was in the
  dense ranking, else ``0.0``. The aggregator's ``mean_top_3_similarity`` is
  defined in cosine space; a sparse ``ts_rank_cd`` value there would be
  nonsense. A sparse-only context scoring ``0.0`` is the desired behaviour — we
  trust a dense semantic match more than a bare keyword match.
* ``source`` records whether the context came from the dense branch only, the
  sparse branch only, or both — surfaced by ``scripts.debug_recommend``.

Ties on RRF score are broken by dense similarity (desc), which keeps a strong
dense match ahead of a sparse-only context at the same fused score.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.retrieval.dense import DEFAULT_TOP_N, ContextSource, RetrievedContext

DEFAULT_K = 60


def _resolve_source(seen: set[ContextSource]) -> ContextSource:
    """Collapse the set of branches a context appeared in into one tag."""
    if ContextSource.DENSE in seen and ContextSource.SPARSE in seen:
        return ContextSource.BOTH
    # Exactly one branch (or already-fused inputs); return it as-is.
    return next(iter(seen))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedContext]],
    *,
    k: int = DEFAULT_K,
    top_n: int = DEFAULT_TOP_N,
) -> list[RetrievedContext]:
    """Fuse multiple rankings into a single re-ranked top-N list.

    ``rankings`` is typically ``[dense_ctxs, sparse_ctxs]`` but any number of
    rankings works. An empty ranking (e.g. a stop-word-only sparse query)
    contributes nothing and the fusion degenerates gracefully to the remaining
    branches.

    Raises ``ValueError`` if ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    rrf_score: dict[int, float] = {}
    dense_similarity: dict[int, float] = {}
    seen_sources: dict[int, set[ContextSource]] = {}
    # Metadata representative per context; a dense row is preferred because it
    # carries the cosine similarity, but any branch has the same columns.
    representative: dict[int, RetrievedContext] = {}

    for ranking in rankings:
        for ctx in ranking:
            cid = ctx.context_id
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (k + ctx.rank)
            seen_sources.setdefault(cid, set()).add(ctx.source)

            if ctx.source is ContextSource.DENSE:
                dense_similarity[cid] = ctx.similarity

            current = representative.get(cid)
            if current is None or (
                current.source is not ContextSource.DENSE
                and ctx.source is ContextSource.DENSE
            ):
                representative[cid] = ctx

    ordered = sorted(
        rrf_score.items(),
        key=lambda item: (item[1], dense_similarity.get(item[0], 0.0)),
        reverse=True,
    )

    fused: list[RetrievedContext] = []
    for new_rank, (cid, _score) in enumerate(ordered[:top_n], start=1):
        base = representative[cid]
        fused.append(
            RetrievedContext(
                context_id=base.context_id,
                cited_paper_id=base.cited_paper_id,
                citing_paper_id=base.citing_paper_id,
                citing_year=base.citing_year,
                sentence=base.sentence,
                similarity=dense_similarity.get(cid, 0.0),
                rank=new_rank,
                source=_resolve_source(seen_sources[cid]),
            )
        )
    return fused


def fuse_paper_rankings(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = DEFAULT_K,
    top_k: int,
) -> list[int]:
    """Fuse multiple paper-id rankings via RRF; return the top-K paper ids.

    Operates one level up from :func:`reciprocal_rank_fusion`: instead of
    fusing context rankings (where sparse-only contexts contaminate the
    aggregator's cosine-space score), each branch first reduces its own
    contexts to a *paper* ranking using its native score scale, and we fuse
    those rankings. Tie-break is by paper-id ascending — stable and
    deterministic.

    Raises ``ValueError`` if ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    rrf_score: dict[int, float] = {}
    for ranking in rankings:
        for rank, paper_id in enumerate(ranking, start=1):
            rrf_score[paper_id] = rrf_score.get(paper_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(rrf_score.items(), key=lambda kv: (-kv[1], kv[0]))
    return [pid for pid, _ in ordered[:top_k]]
