"""Aggregated paper result produced by Stage 5 subclaim retrieval."""


from typing import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from entities.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class RankedPaper:
    """A paper after merging evidence from one or more subclaim rankings."""

    result: RetrievalResult
    aggregate_score: float
    contributions: Mapping[int, int] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def paper_id(self) -> str:
        return self.result.paper_id

    def to_retrieval_result(self) -> RetrievalResult:
        """Return the representative result with the aggregate score."""
        return self.result.with_score(self.aggregate_score)

    def to_dict(self) -> dict:
        return {
            **self.to_retrieval_result().to_dict(),
            "aggregate_score": self.aggregate_score,
            "contributions": dict(self.contributions),
        }
