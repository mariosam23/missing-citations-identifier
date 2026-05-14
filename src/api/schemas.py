"""Public request/response schemas for the recommender API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Query sentence/snippet.")
    top_k: int = Field(10, ge=1, le=100)
    target_year: int | None = Field(
        None, description="Filter to papers published on/before this year."
    )
    language: str | None = Field(
        None,
        description=(
            "VS Code document languageId (e.g. 'latex', 'markdown'); used by the "
            "client to pick a citation format, not by retrieval."
        ),
    )


class Evidence(BaseModel):
    sentence: str
    citing_year: int | None
    similarity: float


class Candidate(BaseModel):
    paper_id: int
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    citation_key: str
    score: float
    evidence: list[Evidence]
    bibtex: str


class RecommendResponse(BaseModel):
    candidates: list[Candidate]
