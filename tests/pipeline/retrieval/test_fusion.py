"""Unit tests for ``pipeline.retrieval.fusion`` — reciprocal rank fusion.

Pure-Python; no database. The RRF arithmetic is checked by hand against
``1 / (k + rank)`` so a regression in the scoring is caught exactly.
"""

from __future__ import annotations

import pytest

from pipeline.retrieval.dense import ContextSource, RetrievedContext
from pipeline.retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion


def _ctx(
    context_id: int,
    rank: int,
    *,
    source: ContextSource,
    cited_paper_id: int = 1,
    similarity: float = 0.0,
) -> RetrievedContext:
    """Build a minimal RetrievedContext for fusion tests."""
    return RetrievedContext(
        context_id=context_id,
        cited_paper_id=cited_paper_id,
        citing_paper_id=None,
        citing_year=None,
        sentence=f"sentence-{context_id}",
        similarity=similarity,
        rank=rank,
        source=source,
    )


class TestReciprocalRankFusion:
    def test_rrf_math_and_ordering(self) -> None:
        """Fused order and ranks match the hand-computed 1/(k+rank) sums."""
        k = DEFAULT_K  # 60
        dense = [
            _ctx(10, 1, source=ContextSource.DENSE, similarity=0.9),
            _ctx(20, 2, source=ContextSource.DENSE, similarity=0.8),
            _ctx(30, 3, source=ContextSource.DENSE, similarity=0.7),
        ]
        sparse = [
            _ctx(20, 1, source=ContextSource.SPARSE, similarity=0.5),
            _ctx(40, 2, source=ContextSource.SPARSE, similarity=0.4),
            _ctx(10, 3, source=ContextSource.SPARSE, similarity=0.3),
        ]

        fused = reciprocal_rank_fusion([dense, sparse], k=k, top_n=10)

        # c20: 1/62 + 1/61  | c10: 1/61 + 1/63 | c40: 1/62 | c30: 1/63
        expected_order = [20, 10, 40, 30]
        assert [c.context_id for c in fused] == expected_order
        assert [c.rank for c in fused] == [1, 2, 3, 4]

    def test_dense_similarity_is_preserved(self) -> None:
        """Fused similarity carries the dense cosine value, or 0.0 if sparse-only."""
        dense = [_ctx(10, 1, source=ContextSource.DENSE, similarity=0.77)]
        sparse = [
            _ctx(10, 5, source=ContextSource.SPARSE, similarity=0.2),
            _ctx(99, 1, source=ContextSource.SPARSE, similarity=0.9),
        ]

        fused = {c.context_id: c for c in reciprocal_rank_fusion([dense, sparse])}

        assert fused[10].similarity == pytest.approx(0.77)  # dense value kept
        assert fused[99].similarity == 0.0  # sparse-only → 0.0, not ts_rank_cd

    def test_source_resolution(self) -> None:
        """Each fused context is tagged dense / sparse / both correctly."""
        dense = [
            _ctx(10, 1, source=ContextSource.DENSE),
            _ctx(20, 2, source=ContextSource.DENSE),
        ]
        sparse = [
            _ctx(10, 1, source=ContextSource.SPARSE),
            _ctx(30, 2, source=ContextSource.SPARSE),
        ]

        fused = {c.context_id: c.source for c in reciprocal_rank_fusion([dense, sparse])}

        assert fused[10] is ContextSource.BOTH
        assert fused[20] is ContextSource.DENSE
        assert fused[30] is ContextSource.SPARSE

    def test_tie_broken_by_dense_similarity(self) -> None:
        """Equal RRF scores: the context with higher dense similarity wins."""
        dense = [_ctx(10, 1, source=ContextSource.DENSE, similarity=0.5)]
        sparse = [_ctx(99, 1, source=ContextSource.SPARSE, similarity=0.99)]

        fused = reciprocal_rank_fusion([dense, sparse])

        # Both have RRF score 1/61; dense-backed context 10 ranks ahead of the
        # sparse-only context 99 (which contributes 0.0 dense similarity).
        assert [c.context_id for c in fused] == [10, 99]

    def test_top_n_caps_output(self) -> None:
        dense = [_ctx(i, i, source=ContextSource.DENSE) for i in range(1, 11)]

        fused = reciprocal_rank_fusion([dense], top_n=3)

        assert len(fused) == 3
        assert [c.context_id for c in fused] == [1, 2, 3]

    def test_empty_sparse_degrades_to_dense(self) -> None:
        """A stop-word-only sparse query yields []; fusion is dense-only."""
        dense = [
            _ctx(10, 1, source=ContextSource.DENSE, similarity=0.9),
            _ctx(20, 2, source=ContextSource.DENSE, similarity=0.8),
        ]

        fused = reciprocal_rank_fusion([dense, []])

        assert [c.context_id for c in fused] == [10, 20]
        assert all(c.source is ContextSource.DENSE for c in fused)

    def test_all_empty_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []

    def test_non_positive_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            reciprocal_rank_fusion([[]], k=0)
