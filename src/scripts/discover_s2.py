"""Discover ML/AI papers via the Semantic Scholar Graph API.

Strategy: use the `/paper/search` endpoint with queries like "machine learning",
"artificial intelligence", and "natural language processing". 

Writes one JSON object per line to ``data/corpus/s2_works.jsonl``. Each
record is the raw S2 Paper; downstream scripts consume only the fields
they need.

Usage:
    python -m scripts.discover_s2 --target 1000
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import typer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.config import config
from utils.logger import logger

DEFAULT_OUTPUT = Path("data/corpus/s2_works.jsonl")
PAGE_SIZE = 100
POLITE_DELAY_S = 1.0  # 1 req/s API limit

app = typer.Typer(add_completion=False)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError,)),
    reraise=True,
)
def _fetch_page(
    client: httpx.Client,
    query: str,
    offset: int,
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "query": query,
        "offset": offset,
        "limit": PAGE_SIZE,
        "fields": "title,authors,year,venue,externalIds,abstract,citationCount,openAccessPdf",
        "fieldsOfStudy": "Computer Science",
    }
    response = client.get("/paper/search", params=params, timeout=60.0)
    
    if response.status_code == 429:
        logger.warning("s2 rate-limit (429) — tenacity will retry")
    response.raise_for_status()
    
    return response.json()


def _has_pdf_url(work: dict[str, Any]) -> bool:
    """Cheap pre-filter to drop works that obviously won't yield a PDF."""
    oa = work.get("openAccessPdf")
    if oa and oa.get("url"):
        return True
    return False


@app.command()
def main(
    target: int = typer.Option(
        1000,
        "--target",
        help="How many Works to collect before stopping.",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        help="Destination JSONL file.",
    ),
    queries: list[str] = typer.Option(
        ["machine learning", "deep learning", "natural language processing", "artificial intelligence"],
        "--query",
        help="Search queries to run.",
    ),
    min_citations: int = typer.Option(
        100,
        "--min-citations",
        help="Floor on citationCount.",
    ),
) -> None:
    """Paginate the S2 search endpoint until ``target`` is reached."""
    if not config.S2_API_Key:
        logger.warning("No S2_API_Key found in .env. Falling back to unauthenticated pool (may rate limit).")

    output.parent.mkdir(parents=True, exist_ok=True)
    
    headers: dict[str, str] = {}
    if config.S2_API_Key:
        headers["x-api-key"] = config.S2_API_Key

    written = 0
    skipped_no_pdf = 0

    with (
        httpx.Client(base_url="https://api.semanticscholar.org/graph/v1", headers=headers) as client,
        output.open("w", encoding="utf-8") as fh,
    ):
        for query in queries:
            if written >= target:
                break
                
            logger.info("Starting Semantic Scholar discovery for query: %s", query)
            offset = 0
            
            while written < target and offset <= 900: # S2 restricts offset beyond 999
                try:
                    payload = _fetch_page(client, query, offset)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 400:
                        logger.warning("Hit max offset for query %s, moving to next query.", query)
                        break
                    raise

                results = payload.get("data") or []
                if not results:
                    logger.info("Empty page; ending pagination for query: %s", query)
                    break

                for work in results:
                    citations = work.get("citationCount") or 0
                    if citations < min_citations:
                        continue
                        
                    if not _has_pdf_url(work):
                        skipped_no_pdf += 1
                        continue
                        
                    fh.write(json.dumps(work, ensure_ascii=False))
                    fh.write("\n")
                    written += 1
                    if written >= target:
                        break

                meta = payload.get("next")
                if not meta:
                    logger.info("No more pages for query: %s", query)
                    break
                    
                offset += PAGE_SIZE
                
                logger.info(
                    "page done — query=%s written=%d skipped_no_pdf=%d next_offset=%d",
                    query,
                    written,
                    skipped_no_pdf,
                    offset,
                )
                time.sleep(POLITE_DELAY_S)

    logger.info(
        "discover finished — written=%d skipped_no_pdf=%d output=%s",
        written,
        skipped_no_pdf,
        output,
    )


if __name__ == "__main__":
    app()
