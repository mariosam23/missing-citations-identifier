"""Unit tests for ``pipeline.retrieval.aggregate`` — pure-Python helpers.

Only the DB-free functions are covered here; ``compute_features`` and
``rank_papers`` are exercised in higher-level eval tests against a real
session.
"""

from __future__ import annotations

from pipeline.retrieval.aggregate import rank_papers_by_max_similarity
from pipeline.retrieval.dense import ContextSource, RetrievedContext


def _ctx(
    context_id: int,
    cited_paper_id: int,
    similarity: float,
    *,
    rank: int = 1,
    source: ContextSource = ContextSource.DENSE,
) -> RetrievedContext:
    return RetrievedContext(
        context_id=context_id,
        cited_paper_id=cited_paper_id,
        citing_paper_id=None,
        citing_year=None,
        sentence=f"s-{context_id}",
        similarity=similarity,
        rank=rank,
        source=source,
    )


class TestRankPapersByMaxSimilarity:
    def test_ranks_papers_by_their_best_context(self) -> None:
        """Each paper is represented by its single highest-similarity context."""
        contexts = [
            _ctx(1, cited_paper_id=10, similarity=0.5),
            _ctx(2, cited_paper_id=20, similarity=0.9),
            _ctx(3, cited_paper_id=10, similarity=0.8),  # paper 10 max = 0.8
            _ctx(4, cited_paper_id=30, similarity=0.7),
        ]

        ranked = rank_papers_by_max_similarity(contexts, top_k=10)

        assert ranked == [20, 10, 30]

    def test_top_k_caps_output(self) -> None:
        contexts = [_ctx(i, cited_paper_id=i, similarity=1.0 / i) for i in range(1, 6)]
        ranked = rank_papers_by_max_similarity(contexts, top_k=2)
        assert ranked == [1, 2]

    def test_empty_input(self) -> None:
        assert rank_papers_by_max_similarity([], top_k=5) == []

    def test_works_for_ts_rank_cd_scale(self) -> None:
        """Sparse contexts (ts_rank_cd ~ 0.0–0.1) rank correctly — the function
        is scale-agnostic because it ranks within a single branch."""
        contexts = [
            _ctx(1, cited_paper_id=10, similarity=0.001, source=ContextSource.SPARSE),
            _ctx(2, cited_paper_id=20, similarity=0.08, source=ContextSource.SPARSE),
            _ctx(3, cited_paper_id=30, similarity=0.04, source=ContextSource.SPARSE),
        ]
        assert rank_papers_by_max_similarity(contexts, top_k=10) == [20, 30, 10]
