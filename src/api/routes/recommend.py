"""POST /recommend — hybrid (dense + sparse) retrieval over the citation DB.

Ranking is delegated to ``pipeline.retrieval.hybrid.hybrid_rank`` — the same
canonical context-level path the ``hybrid_context`` evaluation variant uses, so
the extension serves exactly what the thesis numbers measure. This route owns
only the user-facing concerns around that shared core.

Flow:

1. Embed the query sentence with the singleton encoder.
2. ``hybrid_rank``: top-1000 dense (cosine over ``citation_context_embeddings``)
   and top-1000 sparse (``ts_rank_cd`` over the tsvector columns) retrieval,
   fused by reciprocal rank fusion (``k=60``) at the *context* level, grouped by
   ``cited_paper_id``, scored by the cosine-space aggregator, and capped to
   top-K.
3. Hydrate paper metadata; build BibTeX-style citation keys; attach top-3
   evidence contexts; apply the optional ``target_year`` candidate filter.

``Candidate.score`` is the aggregator score
(``mean_top_3_similarity + 0.1*log1p(distinct_citing_papers)``); evidence
``similarity`` is dense cosine, with a sparse-only context carrying ``0.0`` (it
never outranks a semantic match, and the client's confidence meter stays in
``[0, 1]``). Hybrid retrieval (§10.5) adds the sparse branch so lexically-exact
tokens — acronyms ("LoRA"), named datasets ("GLUE") — are not lost to sub-word
tokenisation. (Paper-level RRF fusion was tried and lost on full val; see
``pipeline.retrieval.hybrid``.)
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import db_session
from api.schemas import (
    Candidate,
    RecommendRequest,
    RecommendResponse,
    ScanItem,
    ScanRequest,
    ScanResponse,
)
from api.services.candidate_builder import CandidateBuildResult, build_candidates
from api.services.recommendation_logger import (
    RecommendationLogger,
    ResultToLog,
)
from pipeline.embedding.embedder import encode_query
from pipeline.missing_citations.detector import (
    CitationNeedLabel,
    Detection,
    scan_document,
)
from pipeline.retrieval.aggregate import PaperAggregate
from pipeline.retrieval.dense import DEFAULT_TOP_N
from pipeline.retrieval.hybrid import hybrid_rank

router = APIRouter(tags=["recommend"])

DbSession = Annotated[Session, Depends(db_session)]

# Best-effort, dedicated-session logger (see recommendation_logger docstring).
_recommendation_logger = RecommendationLogger()


@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest,
    session: DbSession,
) -> RecommendResponse:
    query = request.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="text must not be empty")

    ranked = _rank_candidates(
        session,
        query,
        top_k=request.top_k,
        target_year=request.target_year,
    )
    built = build_candidates(
        session,
        ranked,
        target_year=request.target_year,
    )
    event_id = _log_recommendation(
        query_text=query,
        document_path=request.document_path,
        target_year=request.target_year,
        candidates=built.candidates,
        results_to_log=built.results_to_log,
    )
    return RecommendResponse(candidates=built.candidates, event_id=event_id)


@router.post("/scan", response_model=ScanResponse)
def scan(
    request: ScanRequest,
    session: DbSession,
) -> ScanResponse:
    """Scan a document for citation-worthy uncited sentences."""
    if not request.text.strip():
        return ScanResponse(items=[])

    detections = scan_document(request.text, max_sentences=request.max_sentences)
    items: list[ScanItem] = []

    for detection in detections:
        if detection.label != CitationNeedLabel.MISSING_CITATION:
            continue

        query = detection.sentence.text.strip()
        if not query:
            continue

        ranked = _rank_candidates(
            session,
            query,
            top_k=request.top_k,
            target_year=request.target_year,
        )
        confidence = _combine_scan_confidence(detection, ranked)
        if confidence < request.min_confidence:
            continue

        built = build_candidates(
            session,
            ranked,
            target_year=request.target_year,
        )
        if not built.candidates and detection.confidence < 0.5:
            continue

        event_id = None
        if built.candidates:
            event_id = _log_recommendation(
                query_text=query,
                document_path=request.document_path,
                target_year=request.target_year,
                candidates=built.candidates,
                results_to_log=built.results_to_log,
            )
        items.append(
            _build_scan_item(
                detection=detection,
                confidence=confidence,
                ranked=ranked,
                built=built,
                event_id=event_id,
            )
        )

    return ScanResponse(
        items=sorted(items, key=lambda item: item.confidence, reverse=True)
    )


def _rank_candidates(
    session: Session,
    query: str,
    *,
    top_k: int,
    target_year: int | None,
) -> list[PaperAggregate]:
    query_embedding = encode_query(query)
    return hybrid_rank(
        session,
        query,
        query_embedding,
        top_n=DEFAULT_TOP_N,
        top_k=top_k,
        target_year=target_year,
    )


def _log_recommendation(
    *,
    query_text: str,
    document_path: str | None,
    target_year: int | None,
    candidates: list[Candidate],
    results_to_log: list[ResultToLog],
) -> uuid.UUID | None:
    """Persist the run and stamp each candidate with its ``result_id``.

    Best-effort: returns ``None`` and leaves ``result_id`` unset on every
    candidate if logging failed, so a logging outage never affects the response.
    """
    logged = _recommendation_logger.log(
        query_text=query_text,
        document_path=document_path,
        target_year=target_year,
        results=results_to_log,
    )
    if logged is None:
        return None
    for candidate, result_id in zip(candidates, logged.result_ids, strict=True):
        candidate.result_id = result_id
    return logged.event_id


def _build_scan_item(
    *,
    detection: Detection,
    confidence: float,
    ranked: list[PaperAggregate],
    built: CandidateBuildResult,
    event_id: uuid.UUID | None,
) -> ScanItem:
    return ScanItem(
        sentence_id=detection.sentence.sentence_id,
        text=detection.sentence.text,
        start_offset=detection.sentence.start_offset,
        end_offset=detection.sentence.end_offset,
        label=CitationNeedLabel.MISSING_CITATION,
        confidence=round(confidence, 4),
        reasons=_scan_reasons(detection, ranked),
        candidates=built.candidates,
        recommendation_event_id=event_id,
    )


def _combine_scan_confidence(
    detection: Detection,
    ranked: list[PaperAggregate],
) -> float:
    """Blend rule and retrieval confidence.

    The rule score decides *whether* a sentence is citation-worthy; retrieval
    refines *how confident* we are in surfacing it. Strong rule hits must not
    disappear when hybrid scores are only moderate.
    """
    rule_confidence = detection.confidence
    if not ranked:
        return rule_confidence if rule_confidence >= 0.5 else 0.0

    top_score = _bounded_score(ranked[0].score)
    second_score = _bounded_score(ranked[1].score) if len(ranked) > 1 else 0.0
    margin_confidence = min(1.0, max(0.0, (top_score - second_score) / 0.25))
    retrieval_confidence = (0.7 * top_score) + (0.3 * margin_confidence)

    combined = (0.6 * rule_confidence) + (0.4 * retrieval_confidence)
    if rule_confidence >= 0.5:
        combined = max(combined, rule_confidence)

    return min(1.0, combined)


def _bounded_score(score: float) -> float:
    return min(1.0, max(0.0, score))


def _scan_reasons(
    detection: Detection,
    ranked: list[PaperAggregate],
) -> list[str]:
    reasons = list(detection.reasons)
    if ranked:
        reasons.append(f"hybrid top score {ranked[0].score:.3f}")
        if len(ranked) > 1:
            margin = ranked[0].score - ranked[1].score
            reasons.append(f"hybrid rank margin {margin:.3f}")
    return reasons
