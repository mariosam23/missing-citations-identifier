"""Semantic Scholar client for title-based reference resolution.

S2's ``/paper/search/match`` endpoint is purpose-built for citation-to-paper
matching: given a bibliography entry's title, it returns the single best
match in the S2 corpus. We use it as the last step in the resolver cascade
for references that:

* have no DOI / arXiv ID (so OpenAlex DOI/arXiv enrichment can't help), or
* whose DOI is bogus (e.g. CrossRef-consolidation hallucinations) and whose
  title doesn't match any locally-ingested paper.

S2 has excellent coverage of pre-2018 classics that our 2018-2024 OpenAlex
corpus filter excludes (BERT, Transformer, word2vec, Adam, etc.) — exactly
the gap we observed in resolver coverage.

Rate limit: 1 RPS for all endpoints. ``_POLITE_DELAY_S = 1.05`` is enforced
inside ``_get`` via a finally-block ``time.sleep`` so the limit is honored
even on retry paths.

Auth: passes ``x-api-key`` when ``config.S2_API_Key`` is set; falls back to
the (slower) unauthenticated pool otherwise.
"""

from __future__ import annotations

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

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,authors,year,venue,externalIds,abstract"
_POLITE_DELAY_S = 0.8  # 1 RPS ceiling, with a touch of headroom


class SemanticScholarClient:
    """Thin httpx wrapper around the Semantic Scholar Graph API.

    Currently exposes only the title-match endpoint that the resolver
    needs; expand as needed in later phases.
    """

    def __init__(self) -> None:
        headers: dict[str, str] = {}
        if config.S2_API_Key:
            headers["x-api-key"] = config.S2_API_Key
        else:
            logger.warning(
                "no S2 API key set; falling back to the unauthenticated pool"
            )
        self._client = httpx.Client(
            base_url=_BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SemanticScholarClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    def _get(self, path: str, **params: Any) -> dict[str, Any] | None:
        try:
            r = self._client.get(path, params=params)
            if r.status_code in (400, 404):
                return None
            if r.status_code == 429:
                logger.warning("s2 rate-limit (429) — tenacity will retry")
            # S2 returns 429 on rate-limit breaches; let tenacity retry.
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]
        finally:
            time.sleep(_POLITE_DELAY_S)

    def get_open_access_pdf(
        self,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
    ) -> str | None:
        """Return an open-access PDF URL for a paper identified by DOI/arXiv.

        Uses the single-paper lookup ``GET /paper/{id}`` with an ``externalId``
        prefix (``DOI:`` or ``ARXIV:``). Returns ``None`` when S2 has no record
        or no open-access PDF on file. The first available identifier wins;
        DOI is tried before arXiv.
        """
        paper_key: str | None = None
        if doi:
            paper_key = f"DOI:{doi.strip()}"
        elif arxiv_id:
            paper_key = f"ARXIV:{arxiv_id.strip()}"
        if not paper_key:
            return None

        result = self._get(f"/paper/{paper_key}", fields="openAccessPdf")
        if not result:
            return None
        oa = result.get("openAccessPdf") or {}
        url = oa.get("url")
        return url.strip() if isinstance(url, str) and url.strip() else None

    def match_by_title(self, title: str) -> dict[str, Any] | None:
        """Return the best-matching S2 paper for the given title, or ``None``.

        Wraps ``GET /paper/search/match`` which returns at most one result
        (the model's own best guess); we don't second-guess its ranking but
        the caller is expected to verify the match (author surname, year)
        before accepting.
        """
        cleaned = title.strip()
        if not cleaned:
            return None
        result = self._get(
            "/paper/search/match",
            query=cleaned,
            fields=_FIELDS,
        )
        if not result:
            return None
        data = result.get("data") or []
        if not data:
            return None
        match: dict[str, Any] = data[0]
        logger.debug(
            "s2 title-match hit: query=%r → %r (paperId=%s)",
            cleaned[:60],
            (match.get("title") or "")[:60],
            match.get("paperId"),
        )
        return match

    @staticmethod
    def extract_paper_fields(paper: dict[str, Any]) -> dict[str, Any]:
        """Convert an S2 paper dict into ``Paper`` constructor kwargs.

        Shape matches ``OpenAlexClient.extract_paper_fields`` so the resolver's
        ``_get_or_create_from_fields`` helper is source-agnostic.
        """
        title = (paper.get("title") or "").strip()

        authors_raw: list[dict[str, Any]] = paper.get("authors") or []
        authors: list[str] = []
        for entry in authors_raw:
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())
        first_author = authors[0] if authors else None

        year_raw = paper.get("year")
        year = int(year_raw) if isinstance(year_raw, int) else None

        venue_raw = paper.get("venue")
        venue = venue_raw.strip() if isinstance(venue_raw, str) and venue_raw.strip() else None

        ext: dict[str, Any] = paper.get("externalIds") or {}
        doi: str | None = None
        if ext.get("DOI"):
            doi = normalize_doi(str(ext["DOI"]))
        arxiv_id: str | None = None
        if ext.get("ArXiv"):
            arxiv_id = normalize_arxiv_id(str(ext["ArXiv"]))

        abstract_raw = paper.get("abstract")
        abstract = abstract_raw.strip() if isinstance(abstract_raw, str) and abstract_raw.strip() else None

        paper_id_s2 = paper.get("paperId")
        url = (
            f"https://www.semanticscholar.org/paper/{paper_id_s2}"
            if isinstance(paper_id_s2, str) and paper_id_s2
            else None
        )

        return {
            "canonical_title": title,
            "normalized_title": _quick_normalize(title),
            "authors": {"list": authors} if authors else None,
            "first_author": first_author,
            "year": year,
            "venue": venue,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": url,
            "source": "semanticscholar",
            "abstract": abstract,
        }


def _quick_normalize(title: str) -> str:
    import re

    return re.sub(r"\s+", " ", title.lower()).strip()
