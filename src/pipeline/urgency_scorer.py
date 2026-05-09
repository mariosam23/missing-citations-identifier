"""Stage 4B - Urgency scoring.

Prioritizes uncited, citation-worthy sentences before full retrieval. The
scorer runs a lightweight hybrid probe, derives a retrieval-support signal,
mixes it with intent and section priors, and stores the result on each
``SentenceRecord`` as ``urgency_score``.
"""


from typing import TYPE_CHECKING

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

if TYPE_CHECKING:
    from pipeline.retriever import HybridRetriever

from entities import CitationIntent, SentenceRecord
from entities.sentence_record import CitationState

from utils import logger


_MIN_CANDIDATES_FOR_NORMALIZATION = 3


@dataclass(frozen=True)
class UrgencyFeatures:
    """Scoring features extracted for one candidate sentence."""

    top1_score: float
    mean_top5: float
    similarity: float
    normalized_similarity: float
    intent_weight: float
    section_weight: float
    support_factor: float
    urgency: float


class UrgencyScorer:
    """Assign urgency scores to worthy, uncited sentences and rank them.

    1. quick hybrid probe against Qdrant
    2. compute ``top1_score`` and ``mean_top5``
    3. normalize similarity within the current document
    4. combine normalized similarity, intent prior, and section prior
    5. dampen when retrieval support is close to zero
    """

    DEFAULT_INTENT_WEIGHTS: Mapping[CitationIntent | None, float] = {
        CitationIntent.METHOD: 1.0,
        CitationIntent.RESULT: 0.85,
        CitationIntent.BACKGROUND: 0.65,
        CitationIntent.OTHER: 0.5,
        None: 0.5,
    }

    DEFAULT_SECTION_WEIGHTS: Mapping[str, float] = {
        "abstract": 0.55,
        "introduction": 0.9,
        "background": 0.85,
        "related work": 0.85,
        "literature review": 0.85,
        "method": 1.0,
        "methods": 1.0,
        "methodology": 1.0,
        "approach": 0.95,
        "experimental setup": 0.95,
        "experiments": 0.8,
        "results": 0.8,
        "discussion": 0.7,
        "analysis": 0.7,
        "conclusion": 0.45,
        "future work": 0.35,
    }

    def __init__(
        self,
        retriever: "HybridRetriever",
        *,
        probe_k: int = 10,
        mean_k: int = 5,
        similarity_threshold: float = 0.15,
        similarity_mix_top1: float = 0.7,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        intent_weights: Mapping[CitationIntent | None, float] | None = None,
        section_weights: Mapping[str, float] | None = None,
    ) -> None:
        sim_w, intent_w, section_w = weights

        self.retriever = retriever
        self.probe_k = probe_k
        self.mean_k = mean_k
        self.similarity_threshold = similarity_threshold
        self.similarity_mix_top1 = similarity_mix_top1
        self.similarity_mix_mean = 1.0 - similarity_mix_top1
        self.similarity_weight = sim_w
        self.intent_weight = intent_w
        self.section_weight = section_w
        self.intent_weights = dict(self.DEFAULT_INTENT_WEIGHTS)
      
        if intent_weights:
            self.intent_weights.update(intent_weights)
       
        self.section_weights = {
            key.casefold(): value for key, value in self.DEFAULT_SECTION_WEIGHTS.items()
        }
       
        if section_weights:
            self.section_weights.update(
                {key.casefold(): value for key, value in section_weights.items()}
            )

    def score_sentences(
        self,
        sentences: Sequence[SentenceRecord],
        max_year: int | None = None,
    ) -> tuple[list[SentenceRecord], dict[int, UrgencyFeatures]]:
        """Annotate candidate sentences and return a new list with updated scores, along with their features.

        Only worthy, uncited sentences are scored. All other sentences keep
        ``urgency_score=None``. The original list is not modified.
        """
        updated_sentences = [replace(sentence, urgency_score=None) for sentence in sentences]
        candidate_indices = [
            idx for idx, sentence in enumerate(sentences) if self._is_candidate(sentence)
        ]

        if not candidate_indices:
            logger.info("Out of %d sentences, no uncited sentences were found.", len(sentences))
            return updated_sentences, {}

        queries = [updated_sentences[idx].get_retrieval_text() for idx in candidate_indices]
        probe_results = self._probe_similarity_batch(queries, max_year=max_year)
        raw_scores: dict[int, tuple[float, float, float]] = {}

        for index, (top1_score, mean_top5) in zip(candidate_indices, probe_results):
            similarity = self._combine_similarity(top1_score, mean_top5)
            raw_scores[index] = (top1_score, mean_top5, similarity)

        sim_values = [similarity for _, _, similarity in raw_scores.values()]
        normalize_within_doc = len(sim_values) >= _MIN_CANDIDATES_FOR_NORMALIZATION
        sim_min = min(sim_values) if normalize_within_doc else 0.0
        sim_max = max(sim_values) if normalize_within_doc else 0.0

        features_by_index: dict[int, UrgencyFeatures] = {}
        for index in candidate_indices:
            sentence = updated_sentences[index]
            top1_score, mean_top5, similarity = raw_scores[index]
            if normalize_within_doc:
                normalized_similarity = self._minmax(similarity, sim_min, sim_max)
            else:
                normalized_similarity = max(0.0, min(similarity, 1.0))
            intent_weight = self._intent_prior(sentence.citation_intent)
            section_weight = self._section_prior(sentence.section)
            base_urgency = (
                self.similarity_weight * normalized_similarity
                + self.intent_weight * intent_weight
                + self.section_weight * section_weight
            )
          
            support_factor = self._support_factor(similarity)
            urgency = base_urgency * support_factor

            updated_sentences[index] = replace(updated_sentences[index], urgency_score=urgency)

            features_by_index[index] = UrgencyFeatures(
                top1_score=top1_score,
                mean_top5=mean_top5,
                similarity=similarity,
                normalized_similarity=normalized_similarity,
                intent_weight=intent_weight,
                section_weight=section_weight,
                support_factor=support_factor,
                urgency=urgency,
            )

        logger.debug("Scored %d candidate sentences for urgency", len(features_by_index))
        return updated_sentences, features_by_index

    def rank_sentences(
        self,
        sentences: Sequence[SentenceRecord],
        max_year: int | None = None,
    ) -> list[SentenceRecord]:
        """Return candidate sentences sorted by descending urgency."""
        updated_sentences, _ = self.score_sentences(sentences, max_year=max_year)
        ranked = [sentence for sentence in updated_sentences if sentence.urgency_score is not None]
        ranked.sort(
            key=lambda sentence: (
                sentence.urgency_score,
                sentence.worthiness_score if sentence.worthiness_score is not None else -1.0,
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _is_candidate(sentence: SentenceRecord) -> bool:
        return sentence.citation_state == CitationState.MISSING_CITATION

    def _probe_similarity_batch(
        self, queries: Sequence[str], max_year: int | None = None
    ) -> list[tuple[float, float]]:
        if not queries:
            return []

        results_per_query = self.retriever.probe_dense_cosine_batch(
            list(queries), top_k=self.probe_k, max_year=max_year
        )

        out: list[tuple[float, float]] = []
        for query, results in zip(queries, results_per_query):
            if not results:
                logger.warning("Probe retrieved no results for query=%r", query[:80])
                out.append((0.0, 0.0))
                continue
          
            scores = [max(float(result.score), 0.0) for result in results]
            top1_score = scores[0]
            mean_top5 = sum(scores[: self.mean_k]) / min(len(scores), self.mean_k)
            out.append((top1_score, mean_top5))
        return out

    def _combine_similarity(self, top1_score: float, mean_top5: float) -> float:
        return (
            self.similarity_mix_top1 * top1_score
            + self.similarity_mix_mean * mean_top5
        )

    def _support_factor(self, similarity: float) -> float:
        if similarity <= 0.0:
            return 0.0
       
        if self.similarity_threshold <= 0.0:
            return 1.0
        return min(similarity / self.similarity_threshold, 1.0)

    def _intent_prior(self, intent: CitationIntent | None) -> float:
        return float(self.intent_weights.get(intent, self.intent_weights[None]))

    def _section_prior(self, section: str) -> float:
        section_key = (section or "").strip().casefold()
        if not section_key:
            return 0.5

        if section_key in self.section_weights:
            return float(self.section_weights[section_key])

        for known_section, weight in self.section_weights.items():
            if known_section in section_key:
                return float(weight)
        return 0.5

    @staticmethod
    def _minmax(value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0 if value > 0.0 else 0.0
        return (value - lower) / (upper - lower)
