"""Tests for the urgency scorer's batched probe and small-N normalization."""

from entities import CitationWorthiness
import unittest
from collections.abc import Sequence
from dataclasses import dataclass

from entities import CitationIntent, SentenceRecord
from entities.retrieval_result import RetrievalResult
from entities.sentence_record import CitationState
from pipeline.urgency_scorer import UrgencyScorer


@dataclass
class _DenseProbeRetriever:
    """Fake retriever exposing the dense-cosine probe used in production."""

    payload: dict[str, list[RetrievalResult]]
    probe_calls: int = 0
    last_max_year: int | None = None

    def probe_dense_cosine_batch(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        max_year: int | None = None,
    ) -> list[list[RetrievalResult]]:
        self.probe_calls += 1
        self.last_max_year = max_year
        return [self.payload.get(q, []) for q in queries]


def _candidate(text: str, intent: CitationIntent = CitationIntent.METHOD) -> SentenceRecord:
    return SentenceRecord(
        text=text,
        section="introduction",
        position_in_section=0.1,
        has_citation=False,
        citation_intent=intent,
        retrieval_text=text,
        citation_state=CitationState.MISSING_CITATION,
        worthiness_score=CitationWorthiness.HIGH,
    )


class TestUrgencyScorerProbe(unittest.TestCase):
    def test_calls_dense_cosine_probe_in_a_single_batch(self):
        sentences = [_candidate(f"claim about topic {i}") for i in range(4)]
        payload = {
            s.get_retrieval_text(): [RetrievalResult(f"p{i}", "T", 0.7)]
            for i, s in enumerate(sentences)
        }
        retriever = _DenseProbeRetriever(payload=payload)

        scorer = UrgencyScorer(retriever)
        _, features = scorer.score_sentences(sentences, max_year=2024)

        self.assertEqual(len(features), 4)
        self.assertEqual(retriever.probe_calls, 1)
        self.assertEqual(retriever.last_max_year, 2024)


class TestUrgencyScorerNormalization(unittest.TestCase):
    def test_single_candidate_does_not_force_normalized_similarity_to_one(self):
        """Regression: with one candidate, _minmax used to return 1.0 always.

        After the fix, normalized_similarity falls back to the raw cosine
        clipped to [0, 1] when fewer than 3 candidates exist.
        """
        sentences = [_candidate("only candidate")]
        payload = {"only candidate": [RetrievalResult("p", "T", 0.4)]}
        retriever = _DenseProbeRetriever(payload=payload)

        scorer = UrgencyScorer(retriever)
        _, features = scorer.score_sentences(sentences)
        feature = next(iter(features.values()))

        self.assertAlmostEqual(feature.similarity, 0.4)
        self.assertAlmostEqual(feature.normalized_similarity, 0.4)
        self.assertNotEqual(feature.normalized_similarity, 1.0)

    def test_two_candidates_keep_absolute_signal(self):
        """With 2 candidates, fallback should still preserve absolute scale."""
        s1 = _candidate("first claim")
        s2 = _candidate("second claim")
        payload = {
            "first claim": [RetrievalResult("p1", "T", 0.2)],
            "second claim": [RetrievalResult("p2", "T", 0.8)],
        }
        retriever = _DenseProbeRetriever(payload=payload)

        scorer = UrgencyScorer(retriever)
        _, features = scorer.score_sentences([s1, s2])

        normalized = sorted(f.normalized_similarity for f in features.values())
        self.assertNotIn(0.0, normalized)  # min isn't forced to 0
        self.assertNotIn(1.0, normalized)  # max isn't forced to 1
        self.assertAlmostEqual(normalized[0], 0.2, places=5)
        self.assertAlmostEqual(normalized[1], 0.8, places=5)

    def test_three_or_more_candidates_use_minmax_normalization(self):
        sents = [_candidate(f"claim {i}") for i in range(3)]
        payload = {
            "claim 0": [RetrievalResult("p0", "T", 0.2)],
            "claim 1": [RetrievalResult("p1", "T", 0.5)],
            "claim 2": [RetrievalResult("p2", "T", 0.9)],
        }
        retriever = _DenseProbeRetriever(payload=payload)

        scorer = UrgencyScorer(retriever)
        _, features = scorer.score_sentences(sents)

        normalized = sorted(f.normalized_similarity for f in features.values())
        self.assertAlmostEqual(normalized[0], 0.0, places=5)
        self.assertAlmostEqual(normalized[-1], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
