from dataclasses import dataclass, field
from enum import Enum


class CitationIntent(Enum):
    BACKGROUND = "BACKGROUND"
    METHOD = "METHOD"
    RESULT = "RESULT"
    OTHER = "OTHER"


class CitationWorthiness(Enum):
    HIGH = "HIGH"      # core claim — essential to support with a citation
    MEDIUM = "MEDIUM"  # notable claim — should be cited
    LOW = "LOW"        # minor or borderline claim


class CitationState(Enum):
    MISSING_CITATION = "MISSING_CITATION"
    COVERED_BY_BLOCK = "COVERED_BY_BLOCK"
    HAS_CITATION = "HAS_CITATION"
    NOT_CITATION_WORTHY = "NOT_CITATION_WORTHY"


@dataclass
class SentenceRecord:
    text: str                              # with citation markers intact
    section: str
    position_in_section: float             # 0.0=start, 1.0=end
    has_citation: bool
    citation_intent: CitationIntent | None = None

    retrieval_text: str | None = None      # markers stripped (for embedding)
    previous_sentence: str | None = None
    next_sentence: str | None = None

    # GROBID bibliography keys (e.g. "b3") of every reference cited in this
    # sentence. Empty when ``has_citation`` is False or when GROBID failed to
    # link a marker to a bib entry.
    cited_bibkeys: list[str] = field(default_factory=list)

    # --- Filled by Phase 2 (classification) ---
    citation_state: CitationState | None = None
    worthiness_score: CitationWorthiness | None = None

    # --- Filled by Phase 3 (urgency scoring) ---
    urgency_score: float | None = None

    def get_retrieval_text(self) -> str:
        """Return the citation-stripped view, falling back to `text` if not set."""
        return self.retrieval_text if self.retrieval_text is not None else self.text
    
    def get_retrieval_text_with_context(self) -> str:
        return " ".join(
            filter(None, [self.previous_sentence, self.get_retrieval_text(), self.next_sentence])
        )

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "retrieval_text": self.retrieval_text,
            "section": self.section,
            "position_in_section": self.position_in_section,
            "has_citation": self.has_citation,
            "previous_sentence": self.previous_sentence,
            "next_sentence": self.next_sentence,
            "cited_bibkeys": list(self.cited_bibkeys),
            "citation_intent": self.citation_intent.name if self.citation_intent else None,
            "citation_state": self.citation_state.name if self.citation_state else None,
            "worthiness_score": self.worthiness_score.value if self.worthiness_score else None,
            "urgency_score": self.urgency_score,
        }

    def __str__(self) -> str:
        preview = self.text[:60] + ("..." if len(self.text) > 60 else "")
        return (
            f"SentenceRecord(text={preview!r}, section={self.section!r}, "
            f"pos={self.position_in_section:.2f}, has_cite={self.has_citation}, "
            f"citation_intent={self.citation_intent.name if self.citation_intent else None}, "
            f"citation_state={self.citation_state.name if self.citation_state else None}, "
            f"worthiness_score={self.worthiness_score.name if self.worthiness_score else None}, "
            f"urgency_score={self.urgency_score})"
        )
