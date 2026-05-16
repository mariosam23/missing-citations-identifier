"""Tests for ``evaluation.metrics``.

Known ranks and gold sets, hand-verified. Edge cases: gold not in
candidates (all zero), gold at position 1 (Hit@1 = MRR = NDCG = 1.0),
empty candidates.
"""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import (
    compute_all_metrics,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestHitAtK:
    """Hit@K — binary presence check in top-k."""

    def test_gold_at_position_1(self) -> None:
        assert hit_at_k([10, 20, 30], gold=10, k=1) == 1.0

    def test_gold_at_position_3_with_k5(self) -> None:
        assert hit_at_k([10, 20, 30, 40, 50], gold=30, k=5) == 1.0

    def test_gold_at_position_3_with_k2(self) -> None:
        assert hit_at_k([10, 20, 30], gold=30, k=2) == 0.0

    def test_gold_not_in_candidates(self) -> None:
        assert hit_at_k([10, 20, 30], gold=99, k=3) == 0.0

    def test_empty_candidates(self) -> None:
        assert hit_at_k([], gold=10, k=5) == 0.0

    def test_k_larger_than_list(self) -> None:
        assert hit_at_k([10, 20], gold=20, k=100) == 1.0


class TestRecallAtK:
    """Recall@K — identical to Hit@K for single-relevant."""

    def test_identical_to_hit(self) -> None:
        ranked = [5, 10, 15, 20]
        assert recall_at_k(ranked, gold=15, k=3) == hit_at_k(
            ranked, gold=15, k=3
        )


class TestReciprocalRank:
    """MRR@K — 1/position if found."""

    def test_gold_at_position_1(self) -> None:
        assert reciprocal_rank([10, 20, 30], gold=10, k=10) == 1.0

    def test_gold_at_position_3(self) -> None:
        assert reciprocal_rank([10, 20, 30], gold=30, k=10) == pytest.approx(
            1 / 3
        )

    def test_gold_beyond_k(self) -> None:
        assert reciprocal_rank([10, 20, 30], gold=30, k=2) == 0.0

    def test_gold_not_present(self) -> None:
        assert reciprocal_rank([10, 20, 30], gold=99, k=10) == 0.0

    def test_empty_candidates(self) -> None:
        assert reciprocal_rank([], gold=10, k=10) == 0.0


class TestNDCGAtK:
    """NDCG@K — binary relevance, single gold, IDCG=1.0."""

    def test_gold_at_position_1(self) -> None:
        # 1 / log2(2) = 1.0
        assert ndcg_at_k([10, 20, 30], gold=10, k=10) == pytest.approx(1.0)

    def test_gold_at_position_2(self) -> None:
        # 1 / log2(3)
        expected = 1.0 / math.log2(3)
        assert ndcg_at_k([10, 20, 30], gold=20, k=10) == pytest.approx(
            expected
        )

    def test_gold_at_position_5(self) -> None:
        # 1 / log2(6)
        ranked = [1, 2, 3, 4, 5]
        expected = 1.0 / math.log2(6)
        assert ndcg_at_k(ranked, gold=5, k=10) == pytest.approx(expected)

    def test_gold_beyond_k(self) -> None:
        assert ndcg_at_k([10, 20, 30], gold=30, k=2) == 0.0

    def test_gold_not_present(self) -> None:
        assert ndcg_at_k([10, 20, 30], gold=99, k=10) == 0.0

    def test_empty_candidates(self) -> None:
        assert ndcg_at_k([], gold=10, k=10) == 0.0


class TestComputeAllMetrics:
    """Integration of all metrics into a single dict."""

    def test_gold_at_position_1(self) -> None:
        result = compute_all_metrics([42, 10, 20], gold=42, ks=(1, 5, 10, 20))
        assert result["hit@1"] == 1.0
        assert result["hit@5"] == 1.0
        assert result["recall@1"] == 1.0
        assert result["mrr@20"] == 1.0
        assert result["ndcg@20"] == pytest.approx(1.0)

    def test_gold_at_position_3(self) -> None:
        result = compute_all_metrics(
            [10, 20, 42, 30, 40], gold=42, ks=(1, 5, 10, 20)
        )
        assert result["hit@1"] == 0.0
        assert result["hit@5"] == 1.0
        assert result["recall@1"] == 0.0
        assert result["recall@5"] == 1.0
        assert result["mrr@20"] == pytest.approx(1 / 3)
        assert result["ndcg@20"] == pytest.approx(1.0 / math.log2(4))

    def test_gold_not_present(self) -> None:
        result = compute_all_metrics([10, 20, 30], gold=99, ks=(1, 5, 10))
        assert all(v == 0.0 for v in result.values())
