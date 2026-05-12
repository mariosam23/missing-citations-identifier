"""Reference resolver — Phase 2 full cascade.

Resolution strategies applied in order (first hit wins):

1. DOI exact match                 → method='doi',               confidence=1.0
2. Normalized title exact match    → method='title_exact',       confidence=0.97
3. Title + first author + year ±1  → method='title_author_year', confidence=0.92
4. Fuzzy title (token_set_ratio)
     ≥ 0.95  → method='fuzzy',           confidence=score
   0.85–0.95 → method='fuzzy_tentative', confidence=score
     < 0.85  → unresolved

Strategy 1 additionally calls OpenAlex to create the canonical ``Paper`` row
when the DOI is valid but no local row exists yet.
Strategies 2–4 only match against the existing ``papers`` table.

arXiv ID matching was measured at <0.1% hit rate on this corpus and removed
to avoid the OpenAlex round-trip cost.
"""

from __future__ import annotations

from rapidfuzz import fuzz
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.postgres.tables.papers import Paper
from database.postgres.tables.references_ import Reference
from pipeline.resolution.normalize import (
    db_normalize_title,
    normalize_author,
    normalize_doi,
    normalize_title,
)
from pipeline.resolution.openalex_client import OpenAlexClient
from utils.logger import logger

# Resolution result: (cited_paper_id, method, confidence) or (None, None, None)
_Result = tuple[int | None, str | None, float | None]

_FUZZY_ACCEPT = 0.95
_FUZZY_TENTATIVE_MIN = 0.85


class ReferenceResolver:
    """Resolves a single ``Reference`` row to a canonical ``Paper``.

    Instantiate once per batch run and share across all calls::

        with OpenAlexClient() as oa:
            resolver = ReferenceResolver(oa)
            for ref in unresolved:
                paper_id, method, conf = resolver.resolve(session, ref)
    """

    def __init__(self, openalex_client: OpenAlexClient) -> None:
        self._oa = openalex_client

    def resolve(self, session: Session, ref: Reference) -> _Result:
        """Try each strategy in order; return on first hit."""
        if ref.doi:
            result = self._resolve_doi(session, ref.doi)
            if result[0] is not None:
                return result

        if ref.parsed_title:
            result = self._resolve_title_exact(session, ref.parsed_title)
            if result[0] is not None:
                return result

            result = self._resolve_title_author_year(session, ref)
            if result[0] is not None:
                return result

            result = self._resolve_fuzzy(session, ref)
            if result[0] is not None:
                return result

        return None, None, None

    # ------------------------------------------------------------------
    # Strategy 1: DOI exact match
    # ------------------------------------------------------------------

    def _resolve_doi(self, session: Session, doi_raw: str) -> _Result:
        normalized = normalize_doi(doi_raw)
        if not normalized:
            return None, None, None

        existing = session.execute(
            select(Paper).where(Paper.doi == normalized)
        ).scalar_one_or_none()
        if existing is not None:
            return existing.paper_id, "doi", 1.0

        try:
            work = self._oa.fetch_by_doi(normalized)
        except Exception:
            logger.warning("openalex doi lookup failed for %s", normalized, exc_info=True)
            return None, None, None

        if work is None:
            return None, None, None

        paper_id = self._get_or_create_from_work(session, work)
        return paper_id, "doi", 1.0

    # ------------------------------------------------------------------
    # Strategy 2: normalized title exact match
    # ------------------------------------------------------------------

    def _resolve_title_exact(self, session: Session, title_raw: str) -> _Result:
        db_norm = db_normalize_title(title_raw)
        if not db_norm:
            return None, None, None

        matches = (
            session.execute(
                select(Paper).where(Paper.normalized_title == db_norm)
            )
            .scalars()
            .all()
        )

        if len(matches) == 1:
            return matches[0].paper_id, "title_exact", 0.97

        # Multiple papers share the same normalized title — ambiguous.
        # Fall through to step 4 which disambiguates with author + year.
        return None, None, None

    # ------------------------------------------------------------------
    # Strategy 3: title + first author + year ±1
    # ------------------------------------------------------------------

    def _resolve_title_author_year(self, session: Session, ref: Reference) -> _Result:
        if not ref.parsed_title:
            return None, None, None

        db_norm = db_normalize_title(ref.parsed_title)
        stmt = select(Paper).where(Paper.normalized_title == db_norm)

        if ref.parsed_year:
            stmt = stmt.where(
                Paper.year.between(ref.parsed_year - 1, ref.parsed_year + 1)
            )

        candidates = session.execute(stmt).scalars().all()
        if not candidates:
            return None, None, None

        if ref.parsed_first_author:
            ref_surname = normalize_author(ref.parsed_first_author)
            if ref_surname:
                candidates = [
                    p
                    for p in candidates
                    if p.first_author
                    and normalize_author(p.first_author) == ref_surname
                ]

        if len(candidates) == 1:
            return candidates[0].paper_id, "title_author_year", 0.92

        return None, None, None

    # ------------------------------------------------------------------
    # Strategy 4: fuzzy title + first author + year ±1
    # ------------------------------------------------------------------

    def _resolve_fuzzy(self, session: Session, ref: Reference) -> _Result:
        if not ref.parsed_title or not ref.parsed_year:
            return None, None, None

        stmt = select(Paper).where(
            Paper.year.between(ref.parsed_year - 1, ref.parsed_year + 1)
        )

        # Loose SQL prefilter to shrink the candidate set; strict surname
        # equality is applied in Python below to avoid e.g. "lee" matching
        # "mcleelan".
        ref_surname: str | None = None
        if ref.parsed_first_author:
            ref_surname = normalize_author(ref.parsed_first_author) or None
            if ref_surname:
                stmt = stmt.where(
                    func.lower(Paper.first_author).contains(ref_surname)
                )

        candidates = session.execute(stmt).scalars().all()
        if not candidates:
            return None, None, None

        if ref_surname:
            candidates = [
                p
                for p in candidates
                if p.first_author
                and normalize_author(p.first_author) == ref_surname
            ]
            if not candidates:
                return None, None, None

        ref_norm = normalize_title(ref.parsed_title)
        best_score = 0.0
        best_paper: Paper | None = None

        for paper in candidates:
            paper_norm = normalize_title(paper.canonical_title)
            score = fuzz.token_set_ratio(ref_norm, paper_norm) / 100.0
            if score > best_score:
                best_score = score
                best_paper = paper

        if best_paper is None or best_score < _FUZZY_TENTATIVE_MIN:
            return None, None, None

        method = "fuzzy" if best_score >= _FUZZY_ACCEPT else "fuzzy_tentative"
        return best_paper.paper_id, method, best_score

    # ------------------------------------------------------------------
    # Shared helper: get or create Paper from OpenAlex Work
    # ------------------------------------------------------------------

    def _get_or_create_from_work(
        self, session: Session, work: dict[str, object]
    ) -> int:
        """Return the ``paper_id`` of an existing or freshly inserted ``Paper``.

        Checks in order: OpenAlex URL → DOI → arXiv ID → (normalized title +
        first-author surname). The last fallback catches the common case where
        ``parse_corpus`` already inserted this paper from the base corpus but
        without a DOI, so a fresh OpenAlex-by-DOI lookup returns a different
        OpenAlex Work ID and would otherwise create a duplicate row.

        When a fallback hit fills in missing identifiers on the existing row,
        we backfill them so the next lookup matches earlier.
        """
        fields = self._oa.extract_paper_fields(work)  # type: ignore[arg-type]

        if fields.get("url"):
            existing = session.execute(
                select(Paper).where(Paper.url == fields["url"])
            ).scalar_one_or_none()
            if existing is not None:
                self._backfill_identifiers(existing, fields)
                return existing.paper_id

        if fields.get("doi"):
            existing = session.execute(
                select(Paper).where(Paper.doi == fields["doi"])
            ).scalar_one_or_none()
            if existing is not None:
                self._backfill_identifiers(existing, fields)
                return existing.paper_id

        if fields.get("arxiv_id"):
            existing = session.execute(
                select(Paper).where(Paper.arxiv_id == fields["arxiv_id"])
            ).scalar_one_or_none()
            if existing is not None:
                self._backfill_identifiers(existing, fields)
                return existing.paper_id

        existing = self._find_by_title_author(session, fields)
        if existing is not None:
            self._backfill_identifiers(existing, fields)
            return existing.paper_id

        paper = Paper(**fields)
        session.add(paper)
        session.flush()
        logger.debug(
            "created paper from openalex: paper_id=%s title=%r",
            paper.paper_id,
            fields.get("canonical_title", "")[:60],
        )
        return paper.paper_id

    @staticmethod
    def _find_by_title_author(
        session: Session, fields: dict[str, object]
    ) -> Paper | None:
        """Look up an existing Paper by normalized title + surname.

        Matches the storage normalization used by ``parse_corpus`` (lowercase
        + whitespace collapse, punctuation preserved) so the indexed column
        can be used directly.
        """
        normalized = fields.get("normalized_title")
        if not isinstance(normalized, str) or not normalized:
            return None

        candidates = (
            session.execute(
                select(Paper).where(Paper.normalized_title == normalized)
            )
            .scalars()
            .all()
        )
        if not candidates:
            return None

        first_author = fields.get("first_author")
        if not isinstance(first_author, str) or not first_author:
            # Title-only match is too risky for the dedup fallback; let the
            # caller insert a new row rather than collapse unrelated papers.
            return None

        ref_surname = normalize_author(first_author)
        if not ref_surname:
            return None

        for paper in candidates:
            if paper.first_author and normalize_author(paper.first_author) == ref_surname:
                return paper
        return None

    @staticmethod
    def _backfill_identifiers(existing: Paper, fields: dict[str, object]) -> None:
        """Copy missing identifiers from a freshly-fetched Work onto the row.

        Only fills fields that are currently NULL on ``existing`` — never
        overwrites an existing value, since the local one is authoritative
        for its origin (e.g., parse_corpus already validated the URL).
        """
        for key in ("doi", "arxiv_id", "url", "venue", "abstract"):
            new_value = fields.get(key)
            if new_value and getattr(existing, key, None) in (None, ""):
                setattr(existing, key, new_value)
