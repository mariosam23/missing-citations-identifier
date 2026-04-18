from dataclasses import dataclass, field


@dataclass
class ParsedPaper:
    """
    Data class representing a parsed academic paper.
    """

    title: str
    abstract: str
    references: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all sections into a single string."""
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}")

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")

        for name, text in self.sections.items():
            parts.append(f"{name}:\n{text}")

        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "references": self.references,
            "sections": self.sections,
        }

    def __str__(self) -> str:
        refs_preview = f"{len(self.references)} refs"
        secs_preview = f"{len(self.sections)} sections"

        abstract_preview = (
            self.abstract[:80] + "..." if len(self.abstract) > 80 else self.abstract
        )

        return (
            f"ParsedPaper(title={self.title!r}, "
            f"abstract={abstract_preview!r}, {refs_preview}, {secs_preview})"
        )
