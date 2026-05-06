# pyright: reportMissingImports=false

import sys
import unittest
from entities import Decomposition, RetrievalResult, Subclaim
from evaluation.runner import EvaluationResult
from evaluation.benchmarks.common import BenchmarkExample
from experiments.stage4_experiments import (
    VariantOutput,
    build_stage5_stratified_metrics,
    build_stage5_subclaim_histogram,
    select_stage5_qualitative_examples,
    stage5_facet_label,
)
from pipeline.aggregator import DecomposedRetriever, weighted_rrf_aggregate
from pipeline.claim_decomposer import ClaimDecomposer


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


class Stage5ArtifactHelperTests(unittest.TestCase):
    def test_facet_label_uses_benchmark_value_before_decomposition_fallback(self) -> None:
        decompositions = {
            "e1": Decomposition(
                original_text="compound",
                subclaims=(Subclaim("a"), Subclaim("b")),
            )
        }

        self.assertEqual(stage5_facet_label("e1", False, decompositions), "single")
        self.assertEqual(stage5_facet_label("e1", None, decompositions), "multi")
        self.assertEqual(stage5_facet_label("missing", None, decompositions), "unknown")

    def test_stratified_metrics_group_v3_v4_by_facet(self) -> None:
        decompositions = {
            "single": Decomposition(original_text="one", subclaims=(Subclaim("one"),)),
            "multi": Decomposition(
                original_text="two",
                subclaims=(Subclaim("a"), Subclaim("b")),
            ),
        }
        result = EvaluationResult(
            n_examples=2,
            n_evaluated=2,
            overall={},
            per_section={},
            per_intent={},
            per_multi_facet={},
            per_example=[
                {
                    "example_id": "single",
                    "is_multi_facet": None,
                    "hidden_count": 1,
                    "recall@10": 0.0,
                    "ndcg@10": 0.25,
                    "skipped": False,
                },
                {
                    "example_id": "multi",
                    "is_multi_facet": None,
                    "hidden_count": 1,
                    "recall@10": 1.0,
                    "ndcg@10": 0.75,
                    "skipped": False,
                },
            ],
        )

        rows = build_stage5_stratified_metrics(
            {
                "V3": VariantOutput("V3", result),
                "V4": VariantOutput("V4", result),
            },
            decompositions,
        )

        by_key = {(row["variant"], row["facet"]): row for row in rows}
        self.assertEqual(by_key[("V4", "single")]["n"], 1)
        self.assertEqual(by_key[("V4", "multi")]["n"], 1)
        self.assertAlmostEqual(by_key[("V3", "multi")]["recall@10"], 1.0)
        self.assertAlmostEqual(by_key[("V4", "single")]["ndcg@10"], 0.25)

    def test_subclaim_histogram_counts_decompositions(self) -> None:
        histogram = build_stage5_subclaim_histogram(
            {
                "e1": Decomposition(original_text="one", subclaims=(Subclaim("one"),)),
                "e2": Decomposition(
                    original_text="two",
                    subclaims=(Subclaim("a"), Subclaim("b")),
                ),
                "e3": Decomposition(
                    original_text="three",
                    subclaims=(Subclaim("a"), Subclaim("b")),
                ),
            }
        )

        self.assertEqual(histogram, {"1": 1, "2": 2})

    def test_qualitative_selection_returns_multi_facet_examples_with_results(self) -> None:
        examples = [
            BenchmarkExample(
                example_id="single",
                query_text="single query",
                hidden_paper_ids=frozenset({"p1"}),
            ),
            BenchmarkExample(
                example_id="multi",
                query_text="multi query",
                hidden_paper_ids=frozenset({"p2"}),
            ),
        ]
        decompositions = {
            "single": Decomposition(original_text="one", subclaims=(Subclaim("one"),)),
            "multi": Decomposition(
                original_text="two",
                subclaims=(Subclaim("a"), Subclaim("b")),
            ),
        }
        selected = select_stage5_qualitative_examples(
            examples,
            decompositions,
            {"single": [result("p1", 0.9)], "multi": [result("p2", 0.8)]},
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["example_id"], "multi")
        self.assertEqual(len(selected[0]["subclaims"]), 2)


if __name__ == "__main__":
    unittest.main()
