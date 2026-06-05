"""LLM-based binary citation-need identification.

The *identifying* stage that runs before recommendation: for each sentence in a
draft, an LLM answers a single yes/no question — "should this sentence cite a
source?". Sentences answered YES that do not already carry a citation are the
candidate *missing citations* the user can then send to ``/recommend``.

This is deliberately separate from ``pipeline.missing_citations`` (the older
rule-based, multi-label detector). Only the offset-preserving sentence
segmentation and the explicit-citation guard are reused from there.
"""

from pipeline.citation_need.identifier import (
    CitationNeedIdentifier,
    CitationNeedJudgement,
    CitationNeedQuery,
    CitationNeedResult,
    get_identifier,
)
from pipeline.citation_need.sanitize import clean_for_llm, is_classifiable

__all__ = [
    "CitationNeedIdentifier",
    "CitationNeedJudgement",
    "CitationNeedQuery",
    "CitationNeedResult",
    "clean_for_llm",
    "get_identifier",
    "is_classifiable",
]
