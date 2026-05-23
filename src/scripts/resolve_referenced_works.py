"""P1: resolve references using each citing paper's OpenAlex ``referenced_works``.

The discovery JSONL records, per citing Work, the exact list of OpenAlex IDs it
cites (``referenced_works``, avg ~67/work). Matching a GROBID bibliography entry
against that *bounded* candidate set is far more accurate than fuzzing it against
all of OpenAlex, and resolving to the canonical OpenAlex ID means two papers that
cite the same work land on the **same** ``Paper`` row — which is exactly the
co-citation density the retrieval task needs (a target is only reachable when
≥2 distinct papers cite it).

Pipeline:

1. Map citing-paper ``openalex_id → referenced_works[]`` from the discovery JSONL.
2. For every citing paper with ≥1 unresolved reference, build a candidate set
   from its ``referenced_works`` metadata (batch-fetched from OpenAlex, cached
   across papers since foundational works recur), reusing already-ingested
   ``papers`` rows where possible.
3. Match each unresolved reference to the best candidate: DOI/arXiv exact first,
   else bounded fuzzy title (``token_set_ratio``) with surname/year sanity.
4. ``get_or_create`` the matched ``Paper`` (only on a hit — no unmatched stubs),
   set ``references.cited_paper_id`` + method/confidence, and finally propagate
   into ``citation_contexts``.

Runs *before* the generic ``resolve_references`` cascade, or after it as a
gap-filler — either way it only touches references with ``cited_paper_id IS NULL``
so it is idempotent and resumable. Supports ``--dry-run`` and ``--limit``.

Usage::

    python -m scripts.resolve_referenced_works --dry-run --limit 20
    python -m scripts.resolve_referenced_works
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import typer
from rapidfuzz import fuzz
from sqlalchemy import select, text
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from database.postgres.tables.papers import Paper
from database.postgres.tables.references_ import Reference
from pipeline.resolution.normalize import (
    normalize_arxiv_id,
    normalize_author,
    normalize_doi,
    normalize_title,
)
from pipeline.resolution.openalex_client import OpenAlexClient
from pipeline.resolution.resolver import ReferenceResolver
from utils.logger import logger

DEFAULT_INPUT = Path("data/corpus/openalex_works.jsonl")
_OPENALEX_ID_RE = re.compile(r"(W\d+)")

# Bounded-set fuzzy thresholds. The candidate set is the citing work's actual
# citation list, so a moderate title match is very likely correct; the real risk
# is picking the wrong neighbour, which the surname/year guards mitigate.
_FUZZY_ACCEPT = 0.88
_FUZZY_ACCEPT_NO_AUTHOR = 0.93
_YEAR_TOLERANCE = 2

app = typer.Typer(add_completion=False)


@dataclass(slots=True)
class _Candidate:
    """One referenced_work as a match target."""

    openalex_id: str
    title_norm: str
    surname: str
    year: int | None
    doi: str | None
    arxiv_id: str | None
    paper_id: int | None  # set if already an ingested papers row
    fields: dict[str, Any] | None  # OpenAlex fields, for lazy get_or_create


def _bare_id(value: str | None) -> str | None:
    if not value:
        return None
    m = _OPENALEX_ID_RE.search(value)
    return m.group(1) if m else None


def _load_referenced_works(input_path: Path) -> dict[str, list[str]]:
    """Map citing-paper bare OpenAlex id → list of bare referenced-work ids."""
    out: dict[str, list[str]] = {}
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            work = json.loads(line)
            oid = _bare_id(work.get("id"))
            if not oid:
                continue
            refs = [
                rid
                for rid in (
                    _bare_id(r) for r in (work.get("referenced_works") or [])
                )
                if rid
            ]
            if refs:
                out[oid] = refs
    return out


def _citing_papers_with_unresolved(session: Session) -> dict[int, str]:
    """Return ``{citing_paper_id: openalex_id}`` for papers with unresolved refs."""
    rows = session.execute(
        text(
            """
            SELECT DISTINCT p.paper_id, p.url
            FROM papers p
            JOIN "references" r ON r.citing_paper_id = p.paper_id
            WHERE r.cited_paper_id IS NULL
              AND (r.parsed_title IS NOT NULL
                   OR r.doi IS NOT NULL
                   OR r.arxiv_id IS NOT NULL)
            """
        )
    ).all()
    out: dict[int, str] = {}
    for paper_id, url in rows:
        oid = _bare_id(url)
        if oid:
            out[int(paper_id)] = oid
    return out


def _candidate_from_paper_row(row: Any) -> _Candidate:
    paper_id, url, title, _norm, first_author, year, doi, arxiv_id = row
    return _Candidate(
        openalex_id=_bare_id(url) or "",
        title_norm=normalize_title(title or ""),
        surname=normalize_author(first_author or "") or "",
        year=int(year) if year is not None else None,
        doi=doi,
        arxiv_id=arxiv_id,
        paper_id=int(paper_id),
        fields=None,
    )


def _candidate_from_fields(oid: str, fields: dict[str, Any]) -> _Candidate:
    return _Candidate(
        openalex_id=oid,
        title_norm=normalize_title(str(fields.get("canonical_title") or "")),
        surname=normalize_author(str(fields.get("first_author") or "")) or "",
        year=fields.get("year") if isinstance(fields.get("year"), int) else None,
        doi=fields.get("doi"),
        arxiv_id=fields.get("arxiv_id"),
        paper_id=None,
        fields=fields,
    )


def _build_candidate_index(
    session: Session,
    oa: OpenAlexClient,
    needed_ids: set[str],
) -> dict[str, _Candidate]:
    """Build ``{openalex_id: _Candidate}`` for every needed referenced-work id.

    Existing ``papers`` rows are reused (no API call); the rest are batch-fetched
    from OpenAlex. Candidates are *not* turned into ``papers`` rows here — that
    happens lazily, only when a reference actually matches one.
    """
    index: dict[str, _Candidate] = {}

    # 1. Reuse already-ingested papers (matched by their OpenAlex URL).
    urls = [f"https://openalex.org/{oid}" for oid in needed_ids]
    for start in range(0, len(urls), 1000):
        chunk = urls[start : start + 1000]
        rows = session.execute(
            select(
                Paper.paper_id,
                Paper.url,
                Paper.canonical_title,
                Paper.normalized_title,
                Paper.first_author,
                Paper.year,
                Paper.doi,
                Paper.arxiv_id,
            ).where(Paper.url.in_(chunk))
        ).all()
        for row in rows:
            cand = _candidate_from_paper_row(row)
            if cand.openalex_id:
                index[cand.openalex_id] = cand

    # 2. Batch-fetch the rest from OpenAlex.
    missing = sorted(needed_ids - set(index))
    logger.info(
        "candidate index: %d reused from papers, %d to fetch from OpenAlex",
        len(index),
        len(missing),
    )
    if missing:
        works = oa.fetch_many_by_openalex_ids(missing)
        for oid, work in works.items():
            fields = oa.extract_paper_fields(work)
            if not str(fields.get("canonical_title") or "").strip():
                continue
            index[oid] = _candidate_from_fields(oid, fields)
    return index


def _match_reference(
    ref: Reference, candidates: list[_Candidate]
) -> tuple[_Candidate | None, str | None, float | None]:
    """Pick the best candidate for ``ref``; return (candidate, method, conf)."""
    # Exact identifier matches win outright.
    ref_doi = normalize_doi(ref.doi) if ref.doi else None
    if ref_doi:
        for cand in candidates:
            if cand.doi and cand.doi == ref_doi:
                return cand, "refworks_doi", 1.0
    ref_arxiv = normalize_arxiv_id(ref.arxiv_id) if ref.arxiv_id else None
    if ref_arxiv:
        for cand in candidates:
            if cand.arxiv_id and cand.arxiv_id == ref_arxiv:
                return cand, "refworks_arxiv", 1.0

    if not ref.parsed_title:
        return None, None, None
    ref_title = normalize_title(ref.parsed_title)
    if not ref_title:
        return None, None, None
    ref_surname = normalize_author(ref.parsed_first_author or "") or ""
    ref_len = len(ref_title.split())

    best: _Candidate | None = None
    best_score = 0.0
    for cand in candidates:
        if not cand.title_norm:
            continue
        # Length guard: token_set_ratio scores a pure subset (e.g. ref "to what
        # extent is moral behavior guided by social heuristics" vs candidate
        # "moral heuristics") at 1.0. Reject candidates whose token count is
        # wildly out of proportion so a short namesake can't shadow the real one.
        cand_len = len(cand.title_norm.split())
        if min(ref_len, cand_len) / max(ref_len, cand_len, 1) < 0.5:
            continue
        score = fuzz.token_set_ratio(ref_title, cand.title_norm) / 100.0
        if score > best_score:
            best_score = score
            best = cand

    if best is None:
        return None, None, None

    # Surname guard: when both sides have a surname they must agree; the title
    # acceptance bar is raised when we cannot check the author.
    surnames_known = bool(ref_surname and best.surname)
    if surnames_known and ref_surname != best.surname:
        return None, None, None
    threshold = _FUZZY_ACCEPT if surnames_known else _FUZZY_ACCEPT_NO_AUTHOR
    if best_score < threshold:
        return None, None, None

    # Year sanity (soft — only rejects when both years exist and diverge).
    if (
        ref.parsed_year is not None
        and best.year is not None
        and abs(ref.parsed_year - best.year) > _YEAR_TOLERANCE
    ):
        return None, None, None

    return best, "refworks_fuzzy", best_score


def _resolve_candidate_paper_id(
    session: Session, resolver: ReferenceResolver, cand: _Candidate
) -> int | None:
    """Return the candidate's paper_id, creating the row on first need."""
    if cand.paper_id is not None:
        return cand.paper_id
    if cand.fields is None:
        return None
    paper_id = resolver._get_or_create_from_fields(session, cand.fields)
    cand.paper_id = paper_id  # cache so repeat matches reuse the row
    return paper_id


def _propagate_cited_paper_id(session: Session) -> int:
    cursor = cast(
        CursorResult[Any],
        session.execute(
            text(
                """
                UPDATE citation_contexts cc
                SET    cited_paper_id = r.cited_paper_id
                FROM   "references" r
                WHERE  cc.reference_id    = r.reference_id
                  AND  r.cited_paper_id  IS NOT NULL
                  AND  cc.cited_paper_id IS NULL
                """
            )
        ),
    )
    return cursor.rowcount


@app.command()
def main(
    input_path: Path = typer.Option(DEFAULT_INPUT, "--input"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Match and log but do not commit."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Process at most N citing papers (smoke test)."
    ),
) -> None:
    """Resolve unresolved references via the citing work's referenced_works."""
    if not input_path.exists():
        raise typer.BadParameter(f"input not found: {input_path}")

    t0 = time.monotonic()
    oid_to_refworks = _load_referenced_works(input_path)
    logger.info("loaded referenced_works for %d citing works", len(oid_to_refworks))

    stats = {"papers": 0, "refs_seen": 0, "refworks_doi": 0,
             "refworks_arxiv": 0, "refworks_fuzzy": 0, "unmatched": 0}

    session = get_session()
    with OpenAlexClient() as oa:
        resolver = ReferenceResolver(oa)
        try:
            citing = _citing_papers_with_unresolved(session)
            citing_items = list(citing.items())
            if limit is not None:
                citing_items = citing_items[:limit]
            logger.info(
                "%d citing papers with unresolved references (processing %d)",
                len(citing),
                len(citing_items),
            )

            # Gather every referenced-work id we will need, then build the index.
            needed: set[str] = set()
            for _pid, oid in citing_items:
                needed.update(oid_to_refworks.get(oid, []))
            if not needed:
                typer.echo("No referenced_works available for these papers.")
                return
            index = _build_candidate_index(session, oa, needed)

            for paper_id, oid in citing_items:
                ref_oids = oid_to_refworks.get(oid, [])
                candidates = [
                    index[r] for r in ref_oids if r in index and index[r] is not None
                ]
                if not candidates:
                    continue

                unresolved = (
                    session.query(Reference)
                    .filter(
                        Reference.citing_paper_id == paper_id,
                        Reference.cited_paper_id.is_(None),
                    )
                    .all()
                )
                for ref in unresolved:
                    stats["refs_seen"] += 1
                    cand, method, conf = _match_reference(ref, candidates)
                    if cand is None or method is None:
                        stats["unmatched"] += 1
                        continue
                    cited_id = _resolve_candidate_paper_id(session, resolver, cand)
                    if cited_id is None or cited_id == paper_id:
                        stats["unmatched"] += 1
                        continue
                    ref.cited_paper_id = cited_id
                    ref.resolution_method = method
                    ref.resolution_confidence = conf
                    stats[method] += 1

                stats["papers"] += 1
                if not dry_run:
                    session.commit()
                else:
                    session.flush()

            if not dry_run:
                ctx = _propagate_cited_paper_id(session)
                session.commit()
                logger.info("propagated cited_paper_id to %d contexts", ctx)
            else:
                session.rollback()
        finally:
            session.close()

    elapsed = time.monotonic() - t0
    resolved = stats["refworks_doi"] + stats["refworks_arxiv"] + stats["refworks_fuzzy"]
    seen = stats["refs_seen"] or 1
    typer.echo(
        f"\nreferenced_works resolution {'(DRY RUN) ' if dry_run else ''}summary\n"
        f"  citing papers   : {stats['papers']}\n"
        f"  references seen  : {stats['refs_seen']}\n"
        f"  resolved        : {resolved} ({resolved * 100.0 / seen:.1f}%)\n"
        f"    doi exact     : {stats['refworks_doi']}\n"
        f"    arxiv exact   : {stats['refworks_arxiv']}\n"
        f"    fuzzy title   : {stats['refworks_fuzzy']}\n"
        f"  unmatched       : {stats['unmatched']}\n"
        f"  elapsed         : {elapsed:.1f}s"
    )


if __name__ == "__main__":
    app()
