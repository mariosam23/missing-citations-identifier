"""Build API candidate DTOs from ranked paper aggregates."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import Candidate, Evidence
from api.services.recommendation_logger import ResultToLog
from database.postgres.tables.papers import Paper
from pipeline.bibtex.formatter import paper_to_bibtex
from pipeline.retrieval.aggregate import DEFAULT_EVIDENCE_COUNT, PaperAggregate
from utils.logger import logger


@dataclass(frozen=True, slots=True)
class CandidateBuildResult:
    """Candidates plus matching rows ready for recommendation logging."""

    candidates: list[Candidate]
    results_to_log: list[ResultToLog]


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


def build_candidates(
    session: Session,
    ranked: list[PaperAggregate],
    *,
    target_year: int | None = None,
) -> CandidateBuildResult:
    """Hydrate paper aggregates into public API candidate objects."""
    if not ranked:
        return CandidateBuildResult(candidates=[], results_to_log=[])

    paper_rows = session.execute(
        select(Paper).where(Paper.paper_id.in_([a.cited_paper_id for a in ranked]))
    ).scalars().all()
    paper_by_id = {p.paper_id: p for p in paper_rows}

    raw_keys: list[str] = []
    for agg in ranked:
        paper = paper_by_id.get(agg.cited_paper_id)
        raw_keys.append(_build_citation_key(paper))
    final_keys = _disambiguate_keys(raw_keys)

    candidates: list[Candidate] = []
    results_to_log: list[ResultToLog] = []
    for agg, key in zip(ranked, final_keys, strict=True):
        paper = paper_by_id.get(agg.cited_paper_id)
        if paper is None:
            logger.warning("paper_id=%s missing from papers table", agg.cited_paper_id)
            continue
        if (
            target_year is not None
            and paper.year is not None
            and paper.year > target_year
        ):
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
        results_to_log.append(
            ResultToLog(
                paper_id=paper.paper_id,
                rank=len(candidates),
                score=agg.score,
                citation_key=key,
            )
        )

    return CandidateBuildResult(
        candidates=candidates,
        results_to_log=results_to_log,
    )


def _ascii_slug(value: str) -> str:
    """NFKD-decompose, drop combining marks, lowercase, strip non-alphanum."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_only if ch.isalnum()).lower()


def _first_significant_title_word(title: str | None) -> str:
    if not title:
        return "untitled"
    ascii_title = (
        unicodedata.normalize("NFKD", title)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    for word in _WORD_PATTERN.findall(ascii_title):
        if word not in _TITLE_STOPWORDS and not word.isdigit():
            return word
    tokens = _WORD_PATTERN.findall(ascii_title)
    return tokens[0] if tokens else "untitled"


def _build_citation_key(paper: Paper | None) -> str:
    """``{surname}{year}{firstword}`` citation-key convention."""
    if paper is None:
        return "unknown"
    surname = _ascii_slug(paper.first_author or "") or "anon"
    year = str(paper.year) if paper.year else "nodate"
    word = _first_significant_title_word(paper.canonical_title)
    return f"{surname}{year}{word}"


def _disambiguate_keys(keys: list[str]) -> list[str]:
    """Append ``a``, ``b``, ``c``... to duplicate keys in rank order."""
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1

    seen: dict[str, int] = {}
    out: list[str] = []
    for key in keys:
        if counts[key] == 1:
            out.append(key)
            continue
        idx = seen.get(key, 0)
        suffix = chr(ord("a") + idx) if idx < 26 else f"_{idx}"
        seen[key] = idx + 1
        out.append(f"{key}{suffix}")
    return out


def _extract_author_list(paper: Paper) -> list[str]:
    """Best-effort flatten of the JSONB authors blob."""
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

