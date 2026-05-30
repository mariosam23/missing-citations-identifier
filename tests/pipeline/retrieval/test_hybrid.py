"""Unit tests for ``pipeline.retrieval.hybrid`` — shared hybrid ranking.

Pure-Python, no database: the ranking cores accept ``session=None`` because the
popularity penalty is disabled, so ``compute_features`` never makes its DB
round-trip. The tests pin:

* both fusion cores' ordering (context-level = canonical, paper-level =
  ablation) and the score each stamps onto its aggregates;
* a sparse-only candidate surfaces under paper-level fusion, with its evidence
  similarity zeroed so the client's cosine-percentage meter stays in ``[0, 1]``;
* the behavioural divergence between the strategies (paper-level lifts a strong
  lexical-only paper that context-level buries — which is why paper-level lost
  on precision and is kept only as an ablation);
* prod/eval parity — each variant delegates to its shared core, so the thesis
  numbers describe the served ranker.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.retrieval.aggregate import (
    compute_features,
    group_by_paper,
    rank_papers,
)
from pipeline.retrieval.dense import ContextSource, RetrievedContext
from pipeline.retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion
from pipeline.retrieval.hybrid import (
    rank_paper_aggregates,
    rank_paper_aggregates_context_level,
)


def _ctx(
    context_id: int,
    cited_paper_id: int,
    rank: int,
    similarity: float,
    source: ContextSource,
) -> RetrievedContext:
    return RetrievedContext(
        context_id=context_id,
        cited_paper_id=cited_paper_id,
        citing_paper_id=cited_paper_id * 10,
        citing_year=2020,
        sentence=f"ctx{context_id} for P{cited_paper_id}",
        similarity=similarity,
        rank=rank,
        source=source,
    )


def _context_level_order(
    dense: list[RetrievedContext],
    sparse: list[RetrievedContext],
    *,
    top_n: int,
    top_k: int,
) -> list[int]:
    """Reproduce the old context-level path (what /recommend used to serve)."""
    fused = reciprocal_rank_fusion([dense, sparse], top_n=top_n)
    aggs = group_by_paper(fused)
    compute_features(None, aggs)
    return [agg.cited_paper_id for agg in rank_papers(aggs, top_k=top_k)]


class TestRankPaperAggregates:
    def test_paper_level_order_and_rrf_score(self) -> None:
        """Order and stamped score match hand-computed paper-level RRF."""
        # Dense paper ranking: [100, 200]; sparse paper ranking: [200, 300].
        dense = [
            _ctx(1, 100, 1, 0.9, ContextSource.DENSE),
            _ctx(2, 200, 2, 0.4, ContextSource.DENSE),
        ]
        sparse = [
            _ctx(3, 200, 1, 5.0, ContextSource.SPARSE),
            _ctx(4, 300, 2, 3.0, ContextSource.SPARSE),
        ]

        ranked = rank_paper_aggregates(None, dense, sparse, top_n=50, top_k=10)
        order = [agg.cited_paper_id for agg in ranked]
        scores = {agg.cited_paper_id: agg.score for agg in ranked}

        k = DEFAULT_K
        # p200: dense r2 + sparse r1; p100: dense r1; p300: sparse r2.
        assert order == [200, 100, 300]
        assert scores[200] == pytest.approx(1 / (k + 2) + 1 / (k + 1))
        assert scores[100] == pytest.approx(1 / (k + 1))
        assert scores[300] == pytest.approx(1 / (k + 2))

    def test_sparse_only_candidate_evidence_is_zeroed(self) -> None:
        """A sparse-only paper surfaces; its evidence similarity is 0.0."""
        dense = [_ctx(1, 100, 1, 0.5, ContextSource.DENSE)]
        sparse = [_ctx(2, 900, 1, 7.0, ContextSource.SPARSE)]  # no dense ctx

        ranked = rank_paper_aggregates(None, dense, sparse, top_n=50, top_k=10)
        by_id = {agg.cited_paper_id: agg for agg in ranked}

        assert 900 in by_id, "sparse-only paper should still surface"
        sparse_only = by_id[900]
        assert sparse_only.contexts, "evidence contexts should be carried"
        assert all(c.similarity == 0.0 for c in sparse_only.contexts), (
            "ts_rank_cd must not leak into cosine-space evidence similarity"
        )
        # The dense paper keeps its real cosine evidence.
        assert by_id[100].contexts[0].similarity == pytest.approx(0.5)

    def test_top_k_caps_output(self) -> None:
        dense = [_ctx(i, i, i, 1.0 / i, ContextSource.DENSE) for i in range(1, 6)]
        ranked = rank_paper_aggregates(None, dense, [], top_n=50, top_k=2)
        assert len(ranked) == 2

    def test_empty_sparse_degrades_to_dense(self) -> None:
        dense = [
            _ctx(1, 10, 1, 0.9, ContextSource.DENSE),
            _ctx(2, 20, 2, 0.8, ContextSource.DENSE),
        ]
        ranked = rank_paper_aggregates(None, dense, [], top_n=50, top_k=10)
        assert [agg.cited_paper_id for agg in ranked] == [10, 20]


class TestContextLevelCore:
    """The canonical context-level core (``/recommend`` + ``hybrid_context``)."""

    def test_matches_reference_context_level_order(self) -> None:
        """``rank_paper_aggregates_context_level`` == the manual reference path."""
        dense = [
            _ctx(1, 100, 1, 0.9, ContextSource.DENSE),
            _ctx(2, 200, 2, 0.4, ContextSource.DENSE),
        ]
        sparse = [_ctx(3, 300, 1, 5.0, ContextSource.SPARSE)]

        got = [
            agg.cited_paper_id
            for agg in rank_paper_aggregates_context_level(
                None, dense, sparse, top_n=50, top_k=10
            )
        ]
        assert got == _context_level_order(dense, sparse, top_n=50, top_k=10)

    def test_score_is_cosine_aggregator_not_rrf(self) -> None:
        """Context-level ``score`` is the cosine-space aggregator value."""
        dense = [_ctx(1, 100, 1, 0.8, ContextSource.DENSE)]
        ranked = rank_paper_aggregates_context_level(
            None, dense, [], top_n=50, top_k=10
        )
        # 0.8 (mean_top_3) + 0.1*log1p(1 distinct citer) ≈ 0.8693, not a ~0.016 RRF.
        assert ranked[0].score == pytest.approx(0.8 + 0.1 * np.log1p(1))


class TestProdEvalParity:
    """Each variant delegates to its shared core, so prod == eval by construction."""

    # Five weak/moderate dense papers + one strong sparse-only paper.
    _DENSE = [
        _ctx(1, 101, 1, 0.30, ContextSource.DENSE),
        _ctx(2, 102, 2, 0.25, ContextSource.DENSE),
        _ctx(3, 103, 3, 0.20, ContextSource.DENSE),
        _ctx(4, 104, 4, 0.15, ContextSource.DENSE),
        _ctx(5, 105, 5, 0.10, ContextSource.DENSE),
    ]
    _SPARSE = [_ctx(6, 200, 1, 9.0, ContextSource.SPARSE)]

    def test_paper_level_lifts_sparse_only_vs_context_level(self) -> None:
        """Characterises why paper-level was rejected as the production ranker.

        The #1 lexical-only hit (P200) scores 0.0 cosine and ranks *last* under
        context-level fusion; paper-level keeps its rank-1 weight and lifts it
        near the top. That aggressive promotion of lexical-only papers is
        exactly what cost paper-level precision (hit@1/mrr) on full val, so the
        canonical ranker is context-level and this is the documented trade-off.
        """
        paper_level = [
            agg.cited_paper_id
            for agg in rank_paper_aggregates(
                None, self._DENSE, self._SPARSE, top_n=200, top_k=20
            )
        ]
        context_level = [
            agg.cited_paper_id
            for agg in rank_paper_aggregates_context_level(
                None, self._DENSE, self._SPARSE, top_n=200, top_k=20
            )
        ]

        assert paper_level.index(200) < context_level.index(200)
        assert context_level[-1] == 200, "context-level buries the sparse-only paper"
        assert paper_level.index(200) <= 1, "paper-level surfaces it near the top"

    def test_variants_delegate_to_their_shared_cores(self, monkeypatch) -> None:
        """hybrid_context → context-level core; hybrid_rrf → paper-level core."""
        from evaluation.variants.hybrid_context import HybridContext
        from evaluation.variants.hybrid_rrf import HybridRRF

        dense, sparse = self._DENSE, self._SPARSE
        monkeypatch.setattr(
            "pipeline.retrieval.hybrid.retrieve_dense", lambda *a, **k: dense
        )
        monkeypatch.setattr(
            "pipeline.retrieval.hybrid.retrieve_sparse", lambda *a, **k: sparse
        )
        emb = np.zeros(4, dtype=np.float32)
        ctx_variant = HybridContext(session=None, top_n=200)  # type: ignore[arg-type]
        rrf_variant = HybridRRF(session=None, top_n=200)  # type: ignore[arg-type]

        ctx_ids = ctx_variant.candidates(
            "q", target_year=None, exclude_citing_paper_id=0, top_k=20,
            query_embedding=emb,
        )
        rrf_ids = rrf_variant.candidates(
            "q", target_year=None, exclude_citing_paper_id=0, top_k=20,
            query_embedding=emb,
        )

        assert ctx_ids == [
            a.cited_paper_id
            for a in rank_paper_aggregates_context_level(
                None, dense, sparse, top_n=200, top_k=20
            )
        ]
        assert rrf_ids == [
            a.cited_paper_id
            for a in rank_paper_aggregates(None, dense, sparse, top_n=200, top_k=20)
        ]
        # The two strategies genuinely differ on this input (sanity).
        assert ctx_ids != rrf_ids
