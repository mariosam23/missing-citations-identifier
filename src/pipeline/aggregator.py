"""Stage 5 — aggregate per-subclaim retrieval results."""


from collections.abc import Sequence
from typing import Protocol

from entities import AggregationStrategy, Decomposition, RankedPaper, RetrievalResult, Subclaim

from utils import logger

PerSubclaimResults = Sequence[tuple[Subclaim, Sequence[RetrievalResult]]]


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 10, max_year: int | None = None) -> list[RetrievalResult]:
        ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        ...


def weighted_rrf_aggregate(
    per_subclaim_results: PerSubclaimResults,
    *,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[RankedPaper]:
    """Merge subclaim rankings with weighted reciprocal rank fusion."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if top_k == 0:
        return []

    scores: dict[str, float] = {}
    representatives: dict[str, RetrievalResult] = {}
    contributions: dict[str, dict[int, int]] = {}

    for subclaim_index, (subclaim, results) in enumerate(per_subclaim_results):
        for rank, result in enumerate(results, start=1):
            paper_id = str(result.paper_id)
            scores[paper_id] = scores.get(paper_id, 0.0) + (
                subclaim.importance / (rrf_k + rank)
            )
            contributions.setdefault(paper_id, {})[subclaim_index] = rank

            previous = representatives.get(paper_id)
            if previous is None or result.score > previous.score:
                representatives[paper_id] = result

    ranked = [
        RankedPaper(
            result=representatives[paper_id],
            aggregate_score=score,
            contributions=contributions.get(paper_id, {}),
        )
        for paper_id, score in scores.items()
    ]
    ranked.sort(
        key=lambda paper: (
            paper.aggregate_score,
            paper.result.score,
            paper.result.paper_id,
        ),
        reverse=True,
    )
    return ranked[:top_k]


class DecomposedRetriever:
    """Retrieve evidence per subclaim and aggregate the rankings."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        candidates_per_subclaim: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if candidates_per_subclaim < 1:
            raise ValueError("candidates_per_subclaim must be positive")
        self.retriever = retriever
        self.reranker = reranker
        self.candidates_per_subclaim = candidates_per_subclaim
        self.rrf_k = rrf_k

    def retrieve_ranked(
        self,
        decomposition: Decomposition,
        top_k: int = 10,
        max_year: int | None = None,
    ) -> list[RankedPaper]:
        if decomposition.aggregation != AggregationStrategy.WEIGHTED:
            raise ValueError(f"Unsupported aggregation strategy: {decomposition.aggregation}")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        per_subclaim: list[tuple[Subclaim, list[RetrievalResult]]] = []
        for subclaim in decomposition.subclaims:
            hits = self.retriever.retrieve(
                subclaim.text,
                top_k=self.candidates_per_subclaim,
                max_year=max_year,
            )
            if self.reranker is not None:
                hits = self.reranker.rerank(
                    subclaim.text,
                    hits,
                    top_k=self.candidates_per_subclaim,
                )
            per_subclaim.append((subclaim, hits))

        logger.debug(
            "Aggregating %d subclaim rankings for query=%r",
            len(per_subclaim),
            decomposition.original_text[:80],
        )
        return weighted_rrf_aggregate(
            per_subclaim,
            top_k=top_k,
            rrf_k=self.rrf_k,
        )

    def retrieve_and_aggregate(
        self,
        decomposition: Decomposition,
        top_k: int = 10,
        max_year: int | None = None,
    ) -> list[RetrievalResult]:
        """Return aggregated results using the existing RetrievalResult contract."""
        return [
            paper.to_retrieval_result()
            for paper in self.retrieve_ranked(
                decomposition,
                top_k=top_k,
                max_year=max_year,
            )
        ]
