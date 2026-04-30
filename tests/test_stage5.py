from __future__ import annotations

# pyright: reportMissingImports=false

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from entities import Decomposition, RetrievalResult, Subclaim  # noqa: E402
from pipeline.aggregator import DecomposedRetriever, weighted_rrf_aggregate  # noqa: E402
from pipeline.claim_decomposer import ClaimDecomposer  # noqa: E402


def result(paper_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(paper_id=paper_id, title=f"Paper {paper_id}", score=score)


class FakeClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def complete(
        self,
        system: str,
        user: str,
        response_mime_type: str | None = None,
    ) -> str:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.results_by_query.get(query, [])[:top_k]


class Stage5AggregationTests(unittest.TestCase):
    def test_single_subclaim_preserves_ranking_order(self) -> None:
        papers = [result("p1", 0.9), result("p2", 0.8), result("p3", 0.7)]
        ranked = weighted_rrf_aggregate(
            [(Subclaim("atomic claim", importance=1.0), papers)],
            top_k=3,
        )

        self.assertEqual([paper.paper_id for paper in ranked], ["p1", "p2", "p3"])

    def test_weighted_aggregation_deduplicates_and_combines_scores(self) -> None:
        ranked = weighted_rrf_aggregate(
            [
                (Subclaim("claim one", importance=0.7), [result("p1", 0.9), result("p2", 0.8)]),
                (Subclaim("claim two", importance=0.3), [result("p2", 0.95), result("p3", 0.7)]),
            ],
            top_k=3,
        )

        self.assertEqual(ranked[0].paper_id, "p2")
        self.assertEqual(ranked[0].contributions, {0: 2, 1: 1})
        self.assertAlmostEqual(ranked[0].result.score, 0.95)

    def test_top_k_zero_returns_empty_without_retrieval(self) -> None:
        decomposition = Decomposition(
            original_text="claim",
            subclaims=(Subclaim("claim"),),
        )
        retriever = FakeRetriever({"claim": [result("p1", 1.0)]})
        decomposed = DecomposedRetriever(retriever)

        self.assertEqual(decomposed.retrieve_and_aggregate(decomposition, top_k=0), [])


class Stage5DecomposerTests(unittest.TestCase):
    def test_decomposer_normalizes_valid_json_response(self) -> None:
        client = FakeClient(
            """
            {
              "subclaims": [
                {"text": "Transformers improve biomedical NER", "importance": 2},
                {"text": "Transformers replaced BiLSTM models", "importance": 1}
              ],
              "aggregation": "WEIGHTED"
            }
            """
        )
        decomposition = ClaimDecomposer(client=client).decompose("compound claim")

        self.assertEqual(len(decomposition.subclaims), 2)
        self.assertAlmostEqual(decomposition.subclaims[0].importance, 2 / 3)
        self.assertAlmostEqual(decomposition.subclaims[1].importance, 1 / 3)

    def test_decomposer_falls_back_to_single_subclaim_on_invalid_response(self) -> None:
        decomposition = ClaimDecomposer(client=FakeClient("not json")).decompose("original claim")

        self.assertEqual(len(decomposition.subclaims), 1)
        self.assertEqual(decomposition.subclaims[0].text, "original claim")
        self.assertEqual(decomposition.subclaims[0].importance, 1.0)


if __name__ == "__main__":
    unittest.main()
