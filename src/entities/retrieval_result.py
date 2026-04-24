"""Data contract for a single result returned by the retriever."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RetrievalResult:
    """A single paper returned by the hybrid retriever.

    ``paper_id`` is the Semantic Scholar / corpus identifier used as the
    primary key in both Postgres and Qdrant.
    ``score`` is the post-fusion relevance score (higher is better).
    Optional fields are populated from Qdrant payload when available.
    """

    paper_id: str
    title: str
    score: float
    year: int | None = None
    venue: str | None = None
    cited_by_count: int | None = None

    def with_score(self, new_score: float) -> "RetrievalResult":
        """Return a copy with an updated score (used by rerankers)."""
        return replace(self, score=new_score)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "score": self.score,
            "year": self.year,
            "venue": self.venue,
            "cited_by_count": self.cited_by_count,
        }

    def __str__(self) -> str:
        return (
            f"RetrievalResult(id={self.paper_id!r}, score={self.score:.4f}, "
            f"title={self.title[:60]!r})"
        )
