"""POST /recommend — dense retrieval over the citation-context DB.

Flow:

1. Embed the query sentence with the singleton encoder.
2. Top-1000 dense retrieval over ``citation_context_embeddings``.
3. Group by ``cited_paper_id``; compute features.
4. Score and pick top-K.
5. Hydrate paper metadata; build BibTeX-style citation keys; attach top-3
   evidence contexts.

The score blends three signals (see ``pipeline.retrieval.aggregate``):
``mean_top_3_similarity + 0.3*log1p(distinct_citing_papers) -
0.15*log1p(global_context_count)``. This avoids the §31.2 failure mode
where summing similarity surfaces Transformer/BERT for every query.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import db_session
from api.schemas import Candidate, Evidence, RecommendRequest, RecommendResponse
from database.postgres.tables.papers import Paper
from pipeline.bibtex.formatter import paper_to_bibtex
from pipeline.embedding.embedder import encode_query
from pipeline.retrieval.aggregate import (
    DEFAULT_EVIDENCE_COUNT,
    compute_features,
    group_by_paper,
    rank_papers,
)
from pipeline.retrieval.dense import DEFAULT_TOP_N, retrieve_dense
from utils.logger import logger

router = APIRouter(tags=["recommend"])

DbSession = Annotated[Session, Depends(db_session)]


# Words too generic to anchor a citation key. Lowercase, ASCII-only.
_TITLE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the",
        "on", "in", "of", "for", "to", "at", "by", "with", "from", "as",
        "and", "or", "but", "nor", "so", "yet",
        "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those",
        "we", "i", "our", "their",
        "via", "using", "towards", "toward",
    }
)
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest,
    session: DbSession,
) -> RecommendResponse:
    query = request.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="text must not be empty")

    query_embedding = encode_query(query)

    retrieved = retrieve_dense(
        session,
        query_embedding,
        top_n=DEFAULT_TOP_N,
        target_year=request.target_year,
    )
    logger.debug("dense retrieval — contexts=%d", len(retrieved))

    if not retrieved:
        return RecommendResponse(candidates=[])

    aggregates = group_by_paper(retrieved)
    compute_features(session, aggregates)
    ranked = rank_papers(aggregates, top_k=request.top_k)

    if not ranked:
        return RecommendResponse(candidates=[])

    paper_rows = session.execute(
        select(Paper).where(Paper.paper_id.in_([a.cited_paper_id for a in ranked]))
    ).scalars().all()
    paper_by_id = {p.paper_id: p for p in paper_rows}

    # Two-pass citation-key generation so we can suffix duplicates.
    raw_keys: list[str] = []
    for agg in ranked:
        paper = paper_by_id.get(agg.cited_paper_id)
        raw_keys.append(_build_citation_key(paper))
    final_keys = _disambiguate_keys(raw_keys)

    candidates: list[Candidate] = []
    for agg, key in zip(ranked, final_keys, strict=True):
        paper = paper_by_id.get(agg.cited_paper_id)
        if paper is None:
            # Hydration miss — should be rare; skip rather than break the response.
            logger.warning("paper_id=%s missing from papers table", agg.cited_paper_id)
            continue
        if request.target_year is not None and paper.year is not None and paper.year > request.target_year:
            continue

        evidence = [
            Evidence(
                sentence=e.sentence,
                citing_year=e.citing_year,
                similarity=round(e.similarity, 4),
            )
            for e in agg.top_evidence(DEFAULT_EVIDENCE_COUNT)
        ]
        bibtex = paper_to_bibtex(paper, key)
        candidates.append(
            Candidate(
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                authors=_extract_author_list(paper),
                year=paper.year,
                venue=paper.venue,
                citation_key=key,
                score=round(agg.score, 6),
                evidence=evidence,
                bibtex=bibtex,
            )
        )

    return RecommendResponse(candidates=candidates)


# ---------------------------------------------------------------------------
# Citation-key generation
# ---------------------------------------------------------------------------

def _ascii_slug(value: str) -> str:
    """NFKD-decompose, drop combining marks, lowercase, strip non-alphanum."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_only if ch.isalnum()).lower()


def _first_significant_title_word(title: str | None) -> str:
    if not title:
        return "untitled"
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
    for word in _WORD_PATTERN.findall(ascii_title):
        if word not in _TITLE_STOPWORDS and not word.isdigit():
            return word
    # Fallback: first token of any kind.
    tokens = _WORD_PATTERN.findall(ascii_title)
    return tokens[0] if tokens else "untitled"


def _build_citation_key(paper: Paper | None) -> str:
    """``{surname}{year}{firstword}`` — matches the §18 BibTeX convention."""
    if paper is None:
        return "unknown"
    surname = _ascii_slug(paper.first_author or "") or "anon"
    year = str(paper.year) if paper.year else "nodate"
    word = _first_significant_title_word(paper.canonical_title)
    return f"{surname}{year}{word}"


def _disambiguate_keys(keys: list[str]) -> list[str]:
    """Append ``a``, ``b``, ``c``... to duplicates in the order they appear."""
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1

    seen: dict[str, int] = {}
    out: list[str] = []
    for k in keys:
        if counts[k] == 1:
            out.append(k)
            continue
        idx = seen.get(k, 0)
        suffix = chr(ord("a") + idx) if idx < 26 else f"_{idx}"
        seen[k] = idx + 1
        out.append(f"{k}{suffix}")
    return out


def _extract_author_list(paper: Paper) -> list[str]:
    """Best-effort flatten of the JSONB authors blob into ``["Last, First", ...]``."""
    blob = paper.authors
    if not isinstance(blob, dict):
        return [paper.first_author] if paper.first_author else []
    raw = blob.get("list")
    if not isinstance(raw, list):
        return [paper.first_author] if paper.first_author else []

    out: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("display_name")
            if isinstance(name, str) and name:
                out.append(name)
    if not out and paper.first_author:
        out.append(paper.first_author)
    return out
