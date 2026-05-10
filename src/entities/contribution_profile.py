from dataclasses import dataclass


@dataclass(frozen=True)
class ContributionProfile:
    """Structured summary of what a paper proposes (novel) versus what it uses (prior work).

    Used by the classifier to avoid flagging sentences that describe the authors'
    own novel contributions as MISSING_CITATION.

    - ``proposes`` lists components that originate in this paper. Sentences that
      describe items in this list should default to NOT_CITATION_WORTHY.
    - ``uses`` lists external building blocks the paper relies on but did not
      invent. Mentions of these still need a citation.
    """

    system_names: tuple[str, ...]
    novel_contributions: tuple[str, ...]
    uses: tuple[str, ...]
    proposes: tuple[str, ...]
    source_sections: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "system_names": list(self.system_names),
            "novel_contributions": list(self.novel_contributions),
            "uses": list(self.uses),
            "proposes": list(self.proposes),
            "source_sections": list(self.source_sections),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContributionProfile":
        return cls(
            system_names=tuple(data.get("system_names", [])),
            novel_contributions=tuple(data.get("novel_contributions", [])),
            uses=tuple(data.get("uses", [])),
            proposes=tuple(data.get("proposes", [])),
            source_sections=tuple(data.get("source_sections", [])),
        )

    def __str__(self) -> str:
        return (
            f"ContributionProfile("
            f"system_names={list(self.system_names)}, "
            f"proposes={list(self.proposes)}, "
            f"uses={list(self.uses)}, "
            f"novel_contributions={len(self.novel_contributions)} bullets, "
            f"source_sections={list(self.source_sections)})"
        )
