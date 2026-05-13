"""Synchronous OpenAlex client for reference resolution.

Used exclusively by the reference resolver to enrich references that have a
DOI or arXiv ID but no matching ``papers`` row yet. One instance is created
per script run and shared across all resolver calls.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pipeline.resolution.normalize import normalize_arxiv_id, normalize_doi
from utils.config import config
from utils.logger import logger

_POLITE_DELAY_S = 0.1  # 10 req/s polite-pool ceiling
_WHITESPACE_RE = re.compile(r"\s+")


class OpenAlexClient:
    """Thin httpx wrapper for looking up OpenAlex Works by DOI or arXiv ID."""

    def __init__(self) -> None:
        self._client = httpx.Client(
            base_url=config.OPENALEX_BASE_URL,
            timeout=30.0,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    def _get(self, path: str, **params: Any) -> dict[str, Any] | None:
        all_params: dict[str, Any] = {"mailto": config.OPEN_ALEX_EMAIL}
        if config.OPEN_ALEX_API_KEY:
            all_params["api_key"] = config.OPEN_ALEX_API_KEY
        all_params.update(params)
        try:
            r = self._client.get(path, params=all_params)
            if r.status_code in (400, 404):
                return None
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]
        finally:
            time.sleep(_POLITE_DELAY_S)

    def fetch_by_doi(self, normalized_doi: str) -> dict[str, Any] | None:
        """Return a raw OpenAlex Work dict, or ``None`` if not found.

        ``normalized_doi`` must already be stripped of any URL prefix
        (e.g. ``10.18653/v1/P18-1168``, not the full ``https://doi.org/…``
        form). OpenAlex accepts both, but a bare DOI avoids double-encoding.
        """
        result = self._get(f"/works/doi:{normalized_doi}")
        if result and "id" in result:
            logger.debug("openalex doi hit: %s", normalized_doi)
            return result
        return None

    def fetch_by_openalex_id(self, work_id: str) -> dict[str, Any] | None:
        """Return a raw OpenAlex Work dict by its Work ID (e.g. ``W2963341956``).

        Accepts either a bare ID or a full ``https://openalex.org/W…`` URL.
        """
        match = re.search(r"(W\d+)", work_id)
        if not match:
            return None
        result = self._get(f"/works/{match.group(1)}")
        if result and "id" in result:
            return result
        return None

    def fetch_by_arxiv_id(self, normalized_arxiv_id: str) -> dict[str, Any] | None:
        """Return a raw OpenAlex Work dict, or ``None`` if not found.

        OpenAlex does not expose arXiv IDs under the ``ids`` namespace.
        The arXiv landing page URL is stored in ``locations[].landing_page_url``,
        so we filter on that field with the full canonical URL.
        """
        url = f"https://arxiv.org/abs/{normalized_arxiv_id}"
        result = self._get("/works", filter=f"locations.landing_page_url:{url}")
        if not result:
            return None
        results: list[dict[str, Any]] = result.get("results") or []
        if results:
            logger.debug("openalex arxiv hit: %s", normalized_arxiv_id)
            return results[0]
        return None

    @staticmethod
    def extract_paper_fields(work: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw OpenAlex Work to ``Paper`` constructor kwargs.

        All identifier fields are normalized before being returned so the
        caller can store them consistently and perform exact-match lookups.
        """
        # OpenAlex's canonical title field is ``display_name``. The ``title``
        # alias is sometimes absent on Works returned from ``/works/doi:...``,
        # which previously caused us to fall through to the URL fallback below.
        title: str = (work.get("display_name") or work.get("title") or "").strip()

        authorships: list[dict[str, Any]] = work.get("authorships") or []
        authors: list[str] = []
        for authorship in authorships:
            author = authorship.get("author") or {}
            name = author.get("display_name") or ""
            if name:
                authors.append(name)
        first_author = authors[0] if authors else None

        year: int | None = work.get("publication_year")

        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        venue: str | None = source.get("display_name") or None

        ids: dict[str, Any] = work.get("ids") or {}

        doi: str | None = None
        doi_raw = ids.get("doi") or work.get("doi")
        if doi_raw:
            doi = normalize_doi(str(doi_raw))

        arxiv_id: str | None = None
        arxiv_raw = ids.get("arxiv")
        if arxiv_raw:
            arxiv_id = normalize_arxiv_id(str(arxiv_raw))

        openalex_url: str | None = work.get("id") or None

        abstract: str | None = work.get("abstract")
        if not abstract:
            inverted = work.get("abstract_inverted_index")
            if inverted:
                abstract = _reconstruct_abstract(inverted)

        return {
            # Do NOT fall back to the URL here — historically this wrote rows
            # like ``canonical_title='https://openalex.org/W…'`` which the
            # citation-key builder then mangled into ``…https`` keys.
            "canonical_title": title,
            "normalized_title": _quick_normalize(title),
            "authors": {"list": authors} if authors else None,
            "first_author": first_author,
            "year": year,
            "venue": venue,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": openalex_url,
            "source": "openalex",
            "abstract": abstract,
        }


def _quick_normalize(title: str) -> str:
    return _WHITESPACE_RE.sub(" ", title.lower()).strip()


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """Reconstruct plain text from OpenAlex's inverted index format."""
    max_pos = max(
        (pos for positions in inverted_index.values() for pos in positions),
        default=-1,
    )
    if max_pos < 0:
        return ""
    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word
    return " ".join(words)