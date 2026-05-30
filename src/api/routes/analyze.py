"""POST /analyze — binary citation-need identification (the stage before /recommend).

The LLM identifier decides, per sentence, whether it should cite a source. This
route returns only the sentences it flagged (``needs_citation`` true and above
``min_confidence``); the client highlights their spans and the user can then send
any flagged sentence to ``POST /recommend`` to retrieve candidate papers. No
retrieval happens here — identification and recommendation are kept separate.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_citation_need_identifier
from api.schemas import AnalyzeItem, AnalyzeRequest, AnalyzeResponse
from pipeline.citation_need.identifier import CitationNeedIdentifier

router = APIRouter(tags=["analyze"])

Identifier = Annotated[
    CitationNeedIdentifier, Depends(get_citation_need_identifier)
]


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, identifier: Identifier) -> AnalyzeResponse:
    """Flag sentences that should cite a source but currently do not."""
    if not request.text.strip():
        return AnalyzeResponse(items=[])

    results = identifier.analyze(
        request.text, max_sentences=request.max_sentences
    )
    items = [
        AnalyzeItem(
            sentence_id=result.sentence_id,
            text=result.text,
            start_offset=result.start_offset,
            end_offset=result.end_offset,
            needs_citation=result.needs_citation,
            confidence=round(result.confidence, 4),
            section_type=result.section_type,
        )
        for result in results
        if result.needs_citation and result.confidence >= request.min_confidence
    ]
    items.sort(key=lambda item: item.confidence, reverse=True)
    return AnalyzeResponse(items=items)
