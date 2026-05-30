"""Public request/response schemas for the recommender API."""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field

from pipeline.missing_citations.detector import CitationNeedLabel


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
    document_path: str | None = Field(
        default=None,
        description=(
            "Workspace-relative path of the document the query came from. "
            "Logged in plaintext for offline analysis; not used by retrieval."
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
    result_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "ID of the persisted recommendation_results row for this candidate. "
            "Echo it back in POST /feedback to attribute feedback to this "
            "candidate. Null when logging failed."
        ),
    )


class RecommendResponse(BaseModel):
    candidates: list[Candidate]
    event_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "ID of the persisted recommendation_events row for this run. Echo it "
            "back in POST /feedback. Null when logging failed."
        ),
    )


class ScanRequest(BaseModel):
    """Request for document-wide missing-citation scanning."""

    text: str = Field(..., min_length=1, description="Full document text.")
    top_k: int = Field(5, ge=1, le=50)
    target_year: int | None = Field(
        None, description="Filter to papers published on/before this year."
    )
    language: str | None = Field(
        None,
        description="VS Code document languageId, e.g. 'latex' or 'markdown'.",
    )
    document_path: str | None = Field(
        default=None,
        description="Workspace-relative path of the scanned document.",
    )
    max_sentences: int = Field(250, ge=1, le=1000)
    min_confidence: float = Field(0.55, ge=0.0, le=1.0)


class ScanItem(BaseModel):
    """One actionable missing-citation finding."""

    sentence_id: str
    text: str
    start_offset: int
    end_offset: int
    label: CitationNeedLabel
    confidence: float
    reasons: list[str]
    candidates: list[Candidate]
    recommendation_event_id: uuid.UUID | None = None


class ScanResponse(BaseModel):
    items: list[ScanItem]


class FeedbackType(StrEnum):
    """The interactions the webview/quickpick can report for a candidate."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    COPIED_BIBTEX = "copied_bibtex"
    OPENED_URL = "opened_url"


class FeedbackRequest(BaseModel):
    event_id: uuid.UUID = Field(
        ..., description="Event the feedback belongs to (from RecommendResponse)."
    )
    result_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Candidate the feedback is about (from Candidate.result_id). Null for "
            "event-level feedback that is not tied to a single candidate."
        ),
    )
    feedback_type: FeedbackType
    feedback_value: int | None = Field(
        default=None,
        description="Optional numeric payload, e.g. +1/-1 for thumbs up/down.",
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Optional free-text reason, e.g. the rejection-reason dropdown value "
            "('too generic', 'wrong time period', 'irrelevant')."
        ),
    )


class FeedbackResponse(BaseModel):
    feedback_id: uuid.UUID
