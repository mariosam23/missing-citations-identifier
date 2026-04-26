"""Stage 4B — Cross-encoder reranking (V3).

Reranks the candidate papers returned by the hybrid retriever using a
cross-encoder that scores each (query, title) pair jointly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from entities import RetrievalResult

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class _CrossEncoderLike(Protocol):
    def predict(self, pairs: list[tuple[str, str]]):
        """Return one score per (query, candidate text) pair."""


class CrossEncoderReranker:
    """Rerank retrieval candidates with a sentence-transformers CrossEncoder.

    The model is loaded lazily on the first call to ``rerank`` so importing this
    module does not download or initialize the cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model: _CrossEncoderLike | None = None,
    ) -> None:
        self._model_name = model_name
        self._model = model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Return candidates sorted by cross-encoder relevance score.

        Scores from the retriever are replaced with cross-encoder scores in the
        returned ``RetrievalResult`` objects.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not candidates:
            return []

        model = self._get_model()
        pairs = [(query, self._candidate_text(candidate)) for candidate in candidates]
        scores = self._as_scores(model.predict(pairs))

        if len(scores) != len(candidates):
            raise ValueError(
                f"Cross-encoder returned {len(scores)} scores for "
                f"{len(candidates)} candidates."
            )

        rescored = [
            (candidate.with_score(float(score)), index)
            for index, (candidate, score) in enumerate(zip(candidates, scores))
        ]
        rescored.sort(key=lambda item: (item[0].score, -item[1]), reverse=True)

        logger.debug(
            "Reranked %d candidates for query=%r; returning top %d",
            len(candidates),
            query[:80],
            top_k,
        )
        return [candidate for candidate, _ in rescored[:top_k]]

    def _get_model(self) -> _CrossEncoderLike:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    @staticmethod
    def _candidate_text(candidate: RetrievalResult) -> str:
        return candidate.title or candidate.paper_id

    @staticmethod
    def _as_scores(scores) -> "Sequence[float]":
        if hasattr(scores, "tolist"):
            return scores.tolist()
        return list(scores)
