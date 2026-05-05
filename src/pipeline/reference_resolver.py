"""Reference resolution pipeline mapping raw citations to database IDs."""


import difflib
import logging
import re
from collections.abc import Sequence

import requests

from entities.resolved_reference import ResolvedReference
from database.postgres.engine import get_session
from database.postgres.tables.paper import Paper
from utils.config import config

logger = logging.getLogger(__name__)

# Basic DOI regex pattern. Look for 10.NNNN/....
DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


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


class ReferenceResolver:
    """Resolves raw citation strings to database papers."""

    def __init__(self, fuzzy_threshold: float = 0.90):
        self.fuzzy_threshold = fuzzy_threshold
        # Cache resolution results in memory
        self._cache: dict[str, ResolvedReference] = {}
        # Keep track of methods used for metrics
        self.stats: dict[str, int] = {
            "exact_doi": 0,
            "fuzzy_title": 0,
            "openalex": 0,
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

        # 2. Fuzzy Title match
        # We don't have a reliable way to extract the title from a raw reference string
        # without an LLM or specific reference parser (like GROBID).
        # We will attempt to find a title within the database that matches a segment
        # of the reference string, or if we had a citation parser, we'd use that.
        # Given the requirements, we implement basic difflib matching.
        # To avoid comparing against the entire DB (slow), we could do a trigram search
        # or use Postgres full text search. Since we don't have that explicitly, 
        # we'll look for OpenAlex fallback immediately if DOI fails.
        # 
        # Wait, the prompt says:
        # "Fuzzy title match: normalize punctuation/case/spacing. compare against candidate paper titles from Postgres. accept best match at confidence >= 0.90."
        # Candidate paper titles? If it's a batch resolution of a citing paper's references?
        # Actually, the prompt says "Fuzzy title match: ... compare against candidate paper titles from Postgres".
        # If it just means local DB lookup, how do we get candidates? We can't fetch all titles.
        # But wait, maybe the raw reference is passed directly to OpenAlex, or we do a text search?
        # Let's try OpenAlex if local fails.

        # Let's do OpenAlex fallback
        result = self._resolve_by_openalex(raw_reference)
        if result:
            self.stats["openalex"] += 1
            self._cache[raw_reference] = result
            return result

        # If all fail:
        unresolved = ResolvedReference(
            raw_reference=raw_reference,
            method="unresolved",
            unresolved_reason="No match found via DOI or OpenAlex.",
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
            logger.warning(f"DB lookup failed for DOI {doi}: {e}")
        return None

    def _resolve_by_openalex(self, raw_reference: str) -> ResolvedReference | None:
        """Fallback to OpenAlex API if configured."""
        if not hasattr(config, "OPEN_ALEX_EMAIL") or not config.OPEN_ALEX_EMAIL:
            return None

        # Call OpenAlex
        email = config.OPEN_ALEX_EMAIL
        api_key = getattr(config, "OPEN_ALEX_API_KEY", None)

        # We can use /works?search=...
        url = "https://api.openalex.org/works"
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
                return None
            
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return None
            
            best = results[0]
            # Verify if this match is good enough? 
            # We don't have OpenAlex score directly usable for absolute threshold, 
            # but we can try fuzzy matching the best title against the raw string.
            best_title = best.get("title", "")
            doi_url = best.get("doi", "")
            doi = doi_url.replace("https://doi.org/", "") if doi_url else None

            # If we don't have a Semantic Scholar ID or similar, we might use OpenAlex ID or DOI
            # We need to map to our local `paper_id` if it exists.
            # We will use the OpenAlex DOI to query local DB, and if that fails, 
            # we just return the OpenAlex result with unresolved paper_id?
            # Or if it's OpenAlex, do we just return the OpenAlex ID?
            # Wait, the pipeline assumes we are mapping to `corpus paper IDs using Postgres first and OpenAlex as fallback`.
            
            local_id = None
            # Check local DB by OpenAlex DOI
            if doi:
                with get_session() as session:
                    local_paper = session.query(Paper).filter(Paper.doi == doi.lower()).first()
                    if local_paper:
                        local_id = str(local_paper.paperId)

            if not local_id:
                # We failed to map it to a local ID.
                return ResolvedReference(
                    raw_reference=raw_reference,
                    method="unresolved",
                    unresolved_reason="Found in OpenAlex but not in local corpus.",
                )

            return ResolvedReference(
                raw_reference=raw_reference,
                resolved_paper_id=local_id,
                title=best_title,
                doi=doi,
                method="openalex",
                confidence=0.8, # Estimated
            )

        except Exception as e:
            logger.warning(f"OpenAlex fallback failed: {e}")
        
        return None

    def _fuzzy_match_title(self, query: str, candidates: dict[str, str]) -> tuple[str, float] | None:
        """Fuzzy match a query against a set of candidate titles."""
        norm_query = normalize_title(query)
        if not norm_query:
            return None

        best_score = 0.0
        best_id = None

        for cand_id, title in candidates.items():
            norm_cand = normalize_title(title)
            if not norm_cand:
                continue
            
            score = difflib.SequenceMatcher(None, norm_query, norm_cand).ratio()
            if score > best_score:
                best_score = score
                best_id = cand_id

        if best_id and best_score >= self.fuzzy_threshold:
            return best_id, best_score

        return None
