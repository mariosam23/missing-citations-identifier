from dataclasses import dataclass
from enum import Enum


class CitationIntent(Enum):
    BACKGROUND = "BACKGROUND"
    METHOD = "METHOD"
    RESULT = "RESULT"


@dataclass
class SentenceRecord:
    text: str
    section: str
    position_in_section: float
    has_citation: bool
    citation_intent: CitationIntent
    
    retrieval_text: str | None = None
    previous_sentence: str | None = None
    next_sentence: str | None = None

    def get_retrieval_text(self) -> str:
        """Return the citation-stripped view, falling back to `text` if not set."""
        return self.retrieval_text if self.retrieval_text is not None else self.text

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "retrieval_text": self.retrieval_text,
            "section": self.section,
            "position_in_section": self.position_in_section,
            "has_citation": self.has_citation,
            "previous_sentence": self.previous_sentence,
            "next_sentence": self.next_sentence,
            "citation_intent": self.citation_intent.name,
        }

    def __str__(self) -> str:
        preview = self.text[:60] + ("..." if len(self.text) > 60 else "")
        return (
            f"AnnotatedSentence(text={preview!r}, section={self.section!r}, "
            f"pos={self.position_in_section:.2f}, has_cite={self.has_citation}, "
            f"citation_intent={self.citation_intent.name})"
        )
