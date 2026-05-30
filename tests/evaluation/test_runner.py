"""Tests for ``evaluation.runner``.

Uses a stub ``Variant`` returning fixed lists on a tiny in-memory split.
Asserts per-query and aggregate metrics match hand-computed values.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.dataset import EvalQuery


class _StubVariant:
    """A variant that returns a fixed ranking regardless of query."""

    name: str = "stub"

    def __init__(self, fixed_ranking: list[int]) -> None:
        self._ranking = fixed_ranking

    def candidates(
        self,
        query: str,
        *,
        target_year: int | None,
        exclude_citing_paper_id: int,
        top_k: int,
        query_embedding: np.ndarray | None = None,
    ) -> list[int]:
        return self._ranking[:top_k]


class _PositionAwareStubVariant:
    """A variant that always returns the gold at a specific position."""

    name: str = "position_stub"

    def __init__(self, position: int) -> None:
        """``position`` is 1-indexed: gold goes at ``ranked[position-1]``."""
        self._position = position

    def candidates(
        self,
        query: str,
        *,
        target_year: int | None,
        exclude_citing_paper_id: int,
        top_k: int,
        query_embedding: np.ndarray | None = None,
    ) -> list[int]:
        # Build a ranking where a "unique" gold per query is at position.
        # We use exclude_citing_paper_id as a stand-in for unique numbering.
        gold = exclude_citing_paper_id  # stub convention
        fillers = [-(i + 1) for i in range(top_k)]
        if self._position <= top_k:
            fillers[self._position - 1] = gold
        return fillers[:top_k]


class TestEvalRunner:
    """Runner integration with stub variants."""

    def test_perfect_ranking(self) -> None:
        """Variant always returns gold at rank 1 → all metrics = 1.0."""
        queries = [
            EvalQuery(
                sentence="We use BERT",
                gold_paper_id=100,
                citing_paper_id=1,
                citing_year=2020,
            ),
            EvalQuery(
                sentence="Transformer architecture",
                gold_paper_id=200,
                citing_paper_id=2,
                citing_year=2021,
            ),
        ]

        # Variant returns [gold, ...others] for each query.
        class _PerfectVariant:
            name = "perfect"

            def candidates(
                self,
                query: str,
                *,
                target_year: int | None,
                exclude_citing_paper_id: int,
                top_k: int,
                query_embedding: np.ndarray | None = None,
            ) -> list[int]:
                # Map query back to gold.
                gold_map = {"We use BERT": 100, "Transformer architecture": 200}
                gold = gold_map.get(query, -1)
                return [gold] + [-i for i in range(1, top_k)]

        # Run manually since we can't use session.
        from evaluation.metrics import compute_all_metrics
        from evaluation.report import Report

        per_query: dict[str, list[float]] = {}
        for q in queries:
            metrics = compute_all_metrics(
                _PerfectVariant().candidates(
                    q.sentence,
                    target_year=None,
                    exclude_citing_paper_id=q.citing_paper_id,
                    top_k=20,
                ),
                q.gold_paper_id,
            )
            for key, val in metrics.items():
                per_query.setdefault(key, []).append(val)

        report = Report(
            variant_name="perfect",
            split_name="test",
            num_citing_papers=2,
            num_queries=2,
            target_year=None,
            top_k=20,
            per_query=per_query,
        )
        report.compute_aggregates()

        assert report.aggregates["hit@1"]["mean"] == 1.0
        assert report.aggregates["mrr@20"]["mean"] == 1.0
        assert report.aggregates["ndcg@20"]["mean"] == pytest.approx(1.0)

    def test_total_miss(self) -> None:
        """Gold never in candidates → all metrics = 0.0."""
        from evaluation.metrics import compute_all_metrics
        from evaluation.report import Report

        queries = [
            EvalQuery(
                sentence="Missing paper",
                gold_paper_id=999,
                citing_paper_id=1,
                citing_year=2020,
            ),
        ]

        # Variant never returns gold.
        ranking = [10, 20, 30, 40, 50]

        per_query: dict[str, list[float]] = {}
        for q in queries:
            metrics = compute_all_metrics(ranking, q.gold_paper_id)
            for key, val in metrics.items():
                per_query.setdefault(key, []).append(val)

        report = Report(
            variant_name="miss",
            split_name="test",
            num_citing_papers=1,
            num_queries=1,
            target_year=None,
            top_k=20,
            per_query=per_query,
        )
        report.compute_aggregates()

        for metric_stats in report.aggregates.values():
            assert metric_stats["mean"] == 0.0

    def test_mixed_queries(self) -> None:
        """One hit, one miss → mean = 0.5 for binary metrics."""
        from evaluation.metrics import compute_all_metrics
        from evaluation.report import Report

        q_hit = EvalQuery(
            sentence="Hit query",
            gold_paper_id=42,
            citing_paper_id=1,
            citing_year=2020,
        )
        q_miss = EvalQuery(
            sentence="Miss query",
            gold_paper_id=99,
            citing_paper_id=2,
            citing_year=2021,
        )

        ranking_hit = [42, 10, 20]  # Gold at position 1.
        ranking_miss = [10, 20, 30]  # Gold not present.

        per_query: dict[str, list[float]] = {}
        for ranking, q in [
            (ranking_hit, q_hit),
            (ranking_miss, q_miss),
        ]:
            metrics = compute_all_metrics(ranking, q.gold_paper_id)
            for key, val in metrics.items():
                per_query.setdefault(key, []).append(val)

        report = Report(
            variant_name="mixed",
            split_name="test",
            num_citing_papers=2,
            num_queries=2,
            target_year=None,
            top_k=20,
            per_query=per_query,
        )
        report.compute_aggregates()

        assert report.aggregates["hit@1"]["mean"] == pytest.approx(0.5)
        assert report.aggregates["mrr@20"]["mean"] == pytest.approx(0.5)
        assert report.aggregates["ndcg@20"]["mean"] == pytest.approx(0.5)

    def test_sequential_excludes_failed_queries(self) -> None:
        """A query whose variant raises is counted and excluded, not fatal.

        Guards the regression where a failed query silently shrank the metric
        vectors while ``num_queries`` still advertised the attempted count.
        """
        from evaluation.runner import EvalRunner

        queries = [
            EvalQuery(
                sentence="ok",
                gold_paper_id=1,
                citing_paper_id=10,
                citing_year=2020,
            ),
            EvalQuery(
                sentence="boom",
                gold_paper_id=2,
                citing_paper_id=11,
                citing_year=2021,
            ),
            EvalQuery(
                sentence="ok again",
                gold_paper_id=3,
                citing_paper_id=12,
                citing_year=2022,
            ),
        ]

        class _FlakyVariant:
            name = "flaky"

            def candidates(
                self,
                query: str,
                *,
                target_year: int | None,
                exclude_citing_paper_id: int,
                top_k: int,
                query_embedding: np.ndarray | None = None,
            ) -> list[int]:
                if query == "boom":
                    raise RuntimeError("simulated retrieval failure")
                return [exclude_citing_paper_id]

        # split and session are unused by _eval_sequential with a stub variant.
        runner = EvalRunner(
            variant=_FlakyVariant(),
            split=None,  # type: ignore[arg-type]
            session=None,  # type: ignore[arg-type]
        )
        embeddings = [np.zeros(4, dtype=np.float32) for _ in queries]

        metrics, num_failed = runner._eval_sequential(queries, embeddings)

        assert num_failed == 1
        assert len(metrics) == 2
