"""Resolved reference entity mapping raw references to database papers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedReference:
    """A reference parsed from text and mapped to a database paper."""

    raw_reference: str
    resolved_paper_id: str | None = None
    openalex_id: str | None = None
    title: str | None = None
    doi: str | None = None
    method: str | None = None  # "exact_doi", "fuzzy_title", "openalex", "openalex_external", "unresolved"
    confidence: float = 0.0
    unresolved_reason: str | None = None

    @property
    def is_resolved(self) -> bool:
        """True if the reference was successfully linked to a paper."""
        return self.resolved_paper_id is not None
