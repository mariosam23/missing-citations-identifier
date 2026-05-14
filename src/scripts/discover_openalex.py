"""Discover ML/AI papers via the OpenAlex Works API.

Strategy: walk the AI/ML/NLP concept index sorted by ``cited_by_count`` (most
cited first). No publication-date lower bound — foundational papers like
Word2Vec (2013), AlexNet (2012), and Attention Is All You Need (2017) are
exactly what a citation recommender needs to surface.

Writes one JSON object per line to ``data/corpus/openalex_works.jsonl``. Each
record is the raw OpenAlex Work; downstream scripts consume only the fields
they need.

Usage:
    python -m scripts.discover_openalex --target 1500
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

# OpenAlex concept IDs (https://api.openalex.org/concepts).
CONCEPT_AI = "C154945302"  # Artificial Intelligence
CONCEPT_NLP = "C204321447"  # Natural Language Processing
CONCEPT_ML = "C119857082"  # Machine Learning

DEFAULT_OUTPUT = Path("data/corpus/openalex_works.jsonl")
DEFAULT_MIN_CITATIONS = 100
PAGE_SIZE = 200
POLITE_DELAY_S = 0.1  # 10 req/s polite-pool ceiling

app = typer.Typer(add_completion=False)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError,)),
    reraise=True,
)
def _fetch_page(
    client: httpx.Client,
    filter_str: str,
    cursor: str,
    mailto: str,
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "filter": filter_str,
        "sort": "cited_by_count:desc",
        "per_page": PAGE_SIZE,
        "cursor": cursor,
        "mailto": mailto,
    }
    if config.OPEN_ALEX_API_KEY:
        params["api_key"] = config.OPEN_ALEX_API_KEY
    response = client.get("/works", params=params, timeout=60.0)
    response.raise_for_status()
    return response.json()


def _build_filter(
    concept_ids: list[str],
    min_citations: int,
    from_date: str | None,
    to_date: str | None,
) -> str:
    concept_clause = "|".join(concept_ids)
    parts = [
        f"concepts.id:{concept_clause}",
        "is_oa:true",
        "has_fulltext:true",
        f"cited_by_count:>{min_citations}",
    ]
    if from_date:
        parts.append(f"from_publication_date:{from_date}")
    if to_date:
        parts.append(f"to_publication_date:{to_date}")
    return ",".join(parts)


def _has_pdf_url(work: dict[str, Any]) -> bool:
    """Cheap pre-filter to drop works that obviously won't yield a PDF."""
    locations = (
        work.get("best_oa_location"),
        work.get("primary_location"),
    )
    for loc in locations:
        if loc and loc.get("pdf_url"):
            return True
    oa = work.get("open_access") or {}
    return bool(oa.get("oa_url"))


@app.command()
def main(
    target: int = typer.Option(
        1500,
        "--target",
        help="How many Works to collect before stopping.",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        help="Destination JSONL file.",
    ),
    concepts: list[str] = typer.Option(
        [CONCEPT_AI, CONCEPT_ML, CONCEPT_NLP],
        "--concept",
        help="OpenAlex concept ID(s). Pass multiple times to OR-join.",
    ),
    min_citations: int = typer.Option(
        DEFAULT_MIN_CITATIONS,
        "--min-citations",
        help="Floor on cited_by_count. Keep high for 'popular papers only'.",
    ),
    from_date: str | None = typer.Option(
        None,
        "--from-date",
        help="Optional lower bound on publication date (YYYY-MM-DD).",
    ),
    to_date: str | None = typer.Option(
        None,
        "--to-date",
        help="Optional upper bound on publication date (YYYY-MM-DD).",
    ),
) -> None:
    """Cursor-paginate the Works endpoint until ``target`` is reached."""
    if not config.OPEN_ALEX_EMAIL:
        raise typer.BadParameter(
            "OPEN_ALEX_EMAIL is required (polite-pool mailto). Set it in .env."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    filter_str = _build_filter(concepts, min_citations, from_date, to_date)
    logger.info("OpenAlex filter: %s", filter_str)

    written = 0
    skipped_no_pdf = 0
    cursor = "*"

    with (
        httpx.Client(base_url=config.OPENALEX_BASE_URL) as client,
        output.open("w", encoding="utf-8") as fh,
    ):
        while written < target and cursor:
            payload = _fetch_page(
                client, filter_str, cursor, config.OPEN_ALEX_EMAIL
            )
            results = payload.get("results") or []
            if not results:
                logger.info("Empty page; ending pagination.")
                break

            for work in results:
                if not _has_pdf_url(work):
                    skipped_no_pdf += 1
                    continue
                fh.write(json.dumps(work, ensure_ascii=False))
                fh.write("\n")
                written += 1
                if written >= target:
                    break

            meta = payload.get("meta") or {}
            cursor = meta.get("next_cursor") or ""
            logger.info(
                "page done — written=%d skipped_no_pdf=%d next_cursor=%s",
                written,
                skipped_no_pdf,
                "<end>" if not cursor else cursor[:24],
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
