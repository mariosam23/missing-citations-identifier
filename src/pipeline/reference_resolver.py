"""Reference resolution pipeline mapping raw citations to database papers.

Resolution strategy (in order):

1. Exact DOI match against the local Postgres ``papers`` table.
2. Fuzzy title match: pull a small candidate set from Postgres using a
   token-based ``ILIKE`` filter on the longest words in the raw reference,
   then rank with ``difflib`` and accept the best match above
   ``fuzzy_threshold``.
3. OpenAlex fallback. If OpenAlex returns a paper that *is* in the local
   corpus (via DOI lookup), resolve to that local id. Otherwise, return a
   "openalex_external" reference that still carries the OpenAlex DOI / title
   so callers can decide what to do with out-of-corpus matches instead of
   throwing the API result away.
"""


import difflib
import logging
import re
from collections.abc import Sequence

import requests
from sqlalchemy import or_

from entities.resolved_reference import ResolvedReference
from database.postgres.engine import get_session
from database.postgres.tables.paper import Paper
from utils.config import config

logger = logging.getLogger(__name__)

# Basic DOI regex pattern. Look for 10.NNNN/....
DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")

# Match OpenAlex Work IDs in either bare (W123…) or URL form. The local
# ``papers.paperId`` column stores them in bare form so we strip the prefix.
OPENALEX_ID_PATTERN = re.compile(r"(?:openalex\.org/)?(W\d{6,})", re.IGNORECASE)


def extract_openalex_id(text: str | None) -> str | None:
    """Pull an OpenAlex Work ID from a raw string or URL, normalizing case."""
    if not text:
        return None
    match = OPENALEX_ID_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).upper()


# Stop-word-ish tokens to ignore when picking ILIKE candidate filters.
_FUZZY_STOP_TOKENS = {
    "the", "and", "for", "with", "from", "into", "this", "that", "their",
    "via", "using", "based", "toward", "towards", "against", "between",
    "of", "in", "on", "to", "an", "a", "is", "are", "be", "by", "as",
    "we", "our", "its", "it", "or", "not", "but", "et", "al", "eds",
    "vol", "pp", "no", "ed", "proc", "proceedings", "journal", "conference",
}


def extract_doi(text: str) -> str | None:
    """Extract and normalize a DOI from raw reference text."""
    match = DOI_PATTERN.search(text)
    if not match:
        return None
    doi = match.group(1).strip().lower()
    # Remove any trailing punctuation that might have been caught
    doi = re.sub(r"[,.;]+$", "", doi)
    return doi


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching."""
    if not title:
        return ""
    # Lowercase, remove punctuation, normalize spaces
    normalized = re.sub(r"[^\w\s]", "", title.lower())
    return " ".join(normalized.split())


def _fuzzy_candidate_tokens(raw_reference: str, *, top_n: int = 3) -> list[str]:
    """Pick a few content tokens from a raw reference for ILIKE filtering."""
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", raw_reference)
    cleaned = [t.lower() for t in tokens if t.lower() not in _FUZZY_STOP_TOKENS]
    # Longer tokens are more selective and less likely to be common words.
    cleaned.sort(key=len, reverse=True)
    seen: set[str] = set()
    out: list[str] = []
    for token in cleaned:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= top_n:
            break
    return out


class ReferenceResolver:
    """Resolves raw citation strings to database papers."""

    def __init__(
        self,
        fuzzy_threshold: float = 0.90,
        fuzzy_candidate_limit: int = 50,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_candidate_limit = fuzzy_candidate_limit
        # Cache resolution results in memory
        self._cache: dict[str, ResolvedReference] = {}
        # Keep track of methods used for metrics
        self.stats: dict[str, int] = {
            "exact_doi": 0,
            "fuzzy_title": 0,
            "openalex": 0,
            "openalex_external": 0,
            "unresolved": 0,
        }

    def resolve(self, raw_reference: str) -> ResolvedReference:
        """Resolve a single raw reference string."""
        if raw_reference in self._cache:
            return self._cache[raw_reference]

        # 1. Exact DOI match
        doi = extract_doi(raw_reference)
        if doi:
            result = self._resolve_by_doi(raw_reference, doi)
            if result:
                self.stats["exact_doi"] += 1
                self._cache[raw_reference] = result
                return result

        # 2. Fuzzy title match against a small Postgres candidate set
        result = self._resolve_by_fuzzy_title(raw_reference)
        if result:
            self.stats["fuzzy_title"] += 1
            self._cache[raw_reference] = result
            return result

        # 3. OpenAlex fallback. If OpenAlex finds a paper in our corpus we
        # link to the local id; if not, we still return what OpenAlex gave us
        # rather than discarding it.
        result = self._resolve_by_openalex(raw_reference)
        if result:
            self.stats[result.method or "openalex"] += 1
            self._cache[raw_reference] = result
            return result

        unresolved = ResolvedReference(
            raw_reference=raw_reference,
            method="unresolved",
            unresolved_reason="No match found via DOI, fuzzy title, or OpenAlex.",
        )
        self.stats["unresolved"] += 1
        self._cache[raw_reference] = unresolved
        return unresolved

    def resolve_batch(self, raw_references: Sequence[str]) -> list[ResolvedReference]:
        """Resolve a batch of raw references."""
        return [self.resolve(ref) for ref in raw_references]

    def _resolve_by_doi(self, raw_reference: str, doi: str) -> ResolvedReference | None:
        """Attempt to resolve using DOI against local Postgres."""
        try:
            with get_session() as session:
                paper = session.query(Paper).filter(Paper.doi == doi).first()
                if paper:
                    return ResolvedReference(
                        raw_reference=raw_reference,
                        resolved_paper_id=str(paper.paperId),
                        title=paper.title,
                        doi=paper.doi,
                        method="exact_doi",
                        confidence=1.0,
                    )
        except Exception as e:
            logger.warning("DB lookup failed for DOI %s: %s", doi, e)
        return None

    def _resolve_by_fuzzy_title(self, raw_reference: str) -> ResolvedReference | None:
        """Fuzzy-match the reference against a small set of Postgres candidates.

        We avoid scanning the whole DB by first filtering with a few content
        tokens from the reference (case-insensitive ``ILIKE``). The resulting
        candidate set is small, so per-pair ``difflib`` matching is cheap.
        """
        tokens = _fuzzy_candidate_tokens(raw_reference)
        if not tokens:
            return None

        try:
            with get_session() as session:
                conditions = [Paper.title.ilike(f"%{token}%") for token in tokens]
                rows = (
                    session.query(Paper.paperId, Paper.title, Paper.doi)
                    .filter(Paper.title.is_not(None))
                    .filter(or_(*conditions))
                    .limit(self.fuzzy_candidate_limit)
                    .all()
                )
        except Exception as e:
            logger.warning("Fuzzy title candidate query failed: %s", e)
            return None

        if not rows:
            return None

        candidates = {str(row.paperId): (row.title or "", row.doi) for row in rows}
        match = self._fuzzy_match_title(
            raw_reference, {pid: title for pid, (title, _) in candidates.items()}
        )
        if match is None:
            return None

        best_id, score = match
        title, doi = candidates[best_id]
        return ResolvedReference(
            raw_reference=raw_reference,
            resolved_paper_id=best_id,
            title=title,
            doi=doi,
            method="fuzzy_title",
            confidence=score,
        )

    def _resolve_by_openalex(self, raw_reference: str) -> ResolvedReference | None:
        """Fallback to the OpenAlex Works search API.

        The local ``papers.paperId`` column stores OpenAlex Work IDs (``W…``)
        so once OpenAlex returns a candidate work we read its ID directly and
        check it against the local corpus — no DOI round-trip required. If
        the work isn't in our corpus we still surface its metadata under
        ``openalex_external`` instead of discarding the API result.
        """
        if not getattr(config, "OPEN_ALEX_EMAIL", "") or not config.OPEN_ALEX_EMAIL:
            return None

        email = config.OPEN_ALEX_EMAIL
        api_key = getattr(config, "OPEN_ALEX_API_KEY", None) or None
        base_url = getattr(config, "OPENALEX_BASE_URL", "https://api.openalex.org")

        url = f"{base_url.rstrip('/')}/works"
        params: dict[str, str | int] = {
            "search": raw_reference,
            "mailto": email,
            "per-page": 1,
        }
        if api_key:
            params["api_key"] = api_key

        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.debug("OpenAlex returned status %d", resp.status_code)
                return None

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None

            best = results[0]
            best_title = best.get("title", "") or ""
            doi_url = best.get("doi") or ""
            doi = doi_url.replace("https://doi.org/", "").lower() if doi_url else None
            openalex_id = extract_openalex_id(best.get("id"))

            # Confidence proxy: similarity between the candidate title and the
            # raw reference text. Anchors the threshold so callers can compare
            # confidence across resolution methods on the same scale.
            confidence = (
                difflib.SequenceMatcher(
                    None,
                    normalize_title(best_title),
                    normalize_title(raw_reference),
                ).ratio()
                if best_title
                else 0.0
            )

            local_id = self._lookup_local_paper(openalex_id, doi)
            if local_id:
                return ResolvedReference(
                    raw_reference=raw_reference,
                    resolved_paper_id=local_id,
                    title=best_title,
                    doi=doi,
                    method="openalex",
                    confidence=confidence,
                )

            # External hit: paper exists in OpenAlex but not in our corpus.
            # Return the metadata so callers can still cite it instead of
            # throwing this resolution away.
            return ResolvedReference(
                raw_reference=raw_reference,
                resolved_paper_id=None,
                title=best_title or None,
                doi=doi,
                method="openalex_external",
                confidence=confidence,
                unresolved_reason="Found in OpenAlex but not in local corpus.",
            )

        except requests.RequestException as e:
            logger.warning("OpenAlex request failed: %s", e)
        except ValueError as e:  # JSON decode
            logger.warning("OpenAlex returned non-JSON response: %s", e)
        except Exception as e:
            logger.warning("OpenAlex fallback failed: %s", e)

        return None

    @staticmethod
    def _lookup_local_paper(openalex_id: str | None, doi: str | None) -> str | None:
        """Try the OpenAlex Work ID first (the local primary key), DOI as a fallback."""
        if not openalex_id and not doi:
            return None
        try:
            with get_session() as session:
                if openalex_id:
                    paper = session.get(Paper, openalex_id)
                    if paper:
                        return str(paper.paperId)
                if doi:
                    paper = session.query(Paper).filter(Paper.doi == doi.lower()).first()
                    if paper:
                        return str(paper.paperId)
        except Exception as e:
            logger.warning("Local lookup failed for openalex_id=%s doi=%s: %s", openalex_id, doi, e)
        return None

    def _fuzzy_match_title(
        self, query: str, candidates: dict[str, str]
    ) -> tuple[str, float] | None:
        """Fuzzy match a query against a set of candidate titles."""
        norm_query = normalize_title(query)
        if not norm_query:
            return None

        best_score = 0.0
        best_id: str | None = None

        for cand_id, title in candidates.items():
            norm_cand = normalize_title(title)
            if not norm_cand:
                continue

            # Use partial-ratio-style scoring: the candidate title is usually a
            # substring of the (much longer) raw reference, so SequenceMatcher
            # over the normalized query and the candidate window gives a more
            # forgiving score than full-string ratio.
            score = difflib.SequenceMatcher(None, norm_cand, norm_query).ratio()
            # Also consider whether the candidate title appears verbatim
            # inside the normalized reference — common, and a strong signal.
            if norm_cand in norm_query:
                score = max(score, 0.95)
            if score > best_score:
                best_score = score
                best_id = cand_id

        if best_id and best_score >= self.fuzzy_threshold:
            return best_id, best_score

        return None
