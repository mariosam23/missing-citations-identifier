"""Tests for the urgency scorer's batched probe path."""

import unittest
from collections.abc import Sequence
from dataclasses import dataclass

from entities import CitationIntent, SentenceRecord
from entities.retrieval_result import RetrievalResult
from pipeline.urgency_scorer import UrgencyScorer


@dataclass
class _BatchRetriever:
    """Fake retriever that records how often retrieve / retrieve_batch were called."""

    payload: dict[str, list[RetrievalResult]]
    batch_calls: int = 0
    single_calls: int = 0

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.single_calls += 1
        return self.payload.get(query, [])

    def retrieve_batch(
        self, queries: Sequence[str], top_k: int = 10
    ) -> list[list[RetrievalResult]]:
        self.batch_calls += 1
        return [self.payload.get(q, []) for q in queries]


@dataclass
class _SingleRetriever:
    """Fake retriever exposing only retrieve(); urgency_scorer should fall back."""

    payload: dict[str, list[RetrievalResult]]
    calls: int = 0

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.calls += 1
        return self.payload.get(query, [])


def _candidate(text: str) -> SentenceRecord:
    return SentenceRecord(
        text=text,
        section="introduction",
        position_in_section=0.1,
        has_citation=False,
        citation_intent=CitationIntent.METHOD,
        retrieval_text=text,
        citation_worthy=True,
        worthiness_score=0.9,
    )


class TestUrgencyScorerBatching(unittest.TestCase):
    def test_uses_retrieve_batch_when_available(self):
        sentences = [_candidate(f"claim about topic {i}") for i in range(4)]
        payload = {
            s.get_retrieval_text(): [RetrievalResult(f"p{i}", "T", 0.7)]
            for i, s in enumerate(sentences)
        }
        retriever = _BatchRetriever(payload=payload)

        scorer = UrgencyScorer(retriever)
        features = scorer.score_sentences(sentences)

        # All four candidates scored in a single batch round trip.
        self.assertEqual(len(features), 4)
        self.assertEqual(retriever.batch_calls, 1)
        self.assertEqual(retriever.single_calls, 0)

    def test_falls_back_to_per_query_retrieve(self):
        sentences = [_candidate(f"claim {i}") for i in range(3)]
        payload = {
            s.get_retrieval_text(): [RetrievalResult(f"p{i}", "T", 0.5)]
            for i, s in enumerate(sentences)
        }
        retriever = _SingleRetriever(payload=payload)

        scorer = UrgencyScorer(retriever)
        features = scorer.score_sentences(sentences)

        self.assertEqual(len(features), 3)
        # No retrieve_batch attribute -> one call per candidate.
        self.assertEqual(retriever.calls, 3)


if __name__ == "__main__":
    unittest.main()
