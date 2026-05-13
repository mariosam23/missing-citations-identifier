"""Thin HTTP wrapper around GROBID's ``processFulltextDocument`` endpoint.

We call GROBID directly with ``httpx`` rather than via the third-party
``grobid-client-python`` package because:

* The official client's defaults (e.g. ``segmentSentences=1``) interact poorly
  with our pysbd-based sentence segmentation.
* We need precise control over ``consolidateCitations`` and timeouts.
* One HTTP POST is trivial to write and removes a dependency surface.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.config import config
from utils.logger import logger

DEFAULT_TIMEOUT_S = 300.0
ENDPOINT = "/api/processFulltextDocument"


class GrobidError(RuntimeError):
    """Raised when GROBID returns a non-2xx response."""


class GrobidTimeoutError(GrobidError):
    """Raised on read/connect timeout — distinguishable for parse_status."""


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.ConnectError,)),
    reraise=True,
)
def process_fulltext(
    pdf_path: Path,
    *,
    consolidate_citations: int = 1,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Send ``pdf_path`` to GROBID and return the TEI XML as text.

    Raises ``GrobidTimeoutError`` on timeout, ``GrobidError`` on any other
    non-2xx response.
    """
    url = config.GROBID_URL.rstrip("/") + ENDPOINT
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    files = {"input": (pdf_path.name, pdf_path.open("rb"), "application/pdf")}
    data = {
        "consolidateCitations": str(consolidate_citations),
        # `teiCoordinates` deliberately omitted: 3–5× TEI bloat for no MVP value.
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, files=files, data=data)
    except httpx.ReadTimeout as exc:
        raise GrobidTimeoutError(f"GROBID read timeout on {pdf_path.name}") from exc
    except httpx.ConnectTimeout as exc:
        raise GrobidTimeoutError(f"GROBID connect timeout on {pdf_path.name}") from exc
    finally:
        files["input"][1].close()

    if response.status_code >= 400:
        logger.error(
            "GROBID %s -> %s: %s",
            pdf_path.name,
            response.status_code,
            response.text[:200],
        )
        raise GrobidError(
            f"GROBID {response.status_code} for {pdf_path.name}"
        )
    return response.text
