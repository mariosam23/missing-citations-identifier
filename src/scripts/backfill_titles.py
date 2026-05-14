"""Backfill canonical_title (and friends) for papers with URL-shaped titles.

Some ``papers`` rows were inserted with ``canonical_title`` set to the
OpenAlex Work URL — the result of a now-fixed fallback in
``pipeline.resolution.openalex_client.extract_paper_fields`` that wrote
the URL whenever ``display_name``/``title`` came back empty. This script
finds those rows, re-fetches the Work via OpenAlex, and fills in the
real title plus any other identifier fields that are still missing.

Strategy: never overwrite a non-empty existing value (the local row may
have been hand-corrected or merged from another source). Only:

* replace ``canonical_title`` if it currently looks like a URL or is empty
* recompute ``normalized_title`` from the new title
* set ``authors``, ``first_author``, ``venue``, ``doi``, ``arxiv_id``,
  ``abstract``, ``year`` only when the existing column is NULL/empty

Fully resumable: re-running after a crash only touches rows whose title
still looks broken.

Usage::

    python -m scripts.backfill_titles
    python -m scripts.backfill_titles --limit 50 --dry-run
"""

from __future__ import annotations

import re
import time
from typing import Any

import typer
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from database.postgres.tables.papers import Paper
from pipeline.resolution.openalex_client import OpenAlexClient
from utils.logger import logger

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Match ``OpenAlexClient._quick_normalize`` for ``normalized_title``."""
    return _WHITESPACE_RE.sub(" ", title.lower()).strip()

app = typer.Typer(add_completion=False)


def _looks_broken(title: str | None) -> bool:
    if not title:
        return True
    return title.startswith("http://") or title.startswith("https://")


def _fetch_broken(session: Session, limit: int | None) -> list[Paper]:
    """Return all papers whose canonical_title is URL-shaped or empty."""
    stmt = (
        select(Paper)
        .where(
            (Paper.canonical_title.is_(None))
            | (Paper.canonical_title.startswith("http://"))
            | (Paper.canonical_title.startswith("https://"))
            | (Paper.canonical_title == "")
        )
        .order_by(Paper.paper_id)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars().all())


_FALLBACK_TITLE_SQL = text(
    """
    SELECT parsed_title, COUNT(*) AS n
    FROM "references"
    WHERE cited_paper_id = :paper_id
      AND parsed_title IS NOT NULL
      AND length(trim(parsed_title)) > 0
    GROUP BY parsed_title
    ORDER BY n DESC, length(parsed_title) DESC
    LIMIT 1
    """
)


def _fallback_title_from_references(session: Session, paper_id: int) -> str | None:
    """Last-resort title: the most common GROBID-parsed reference text for
    citations resolved to this paper. Works for canonical papers that aren't
    in our ingested corpus (e.g. BERT, pre-2018 classics) but ARE referenced
    by it — the citing papers' bibliography entries contain the real title.
    """
    row = session.execute(_FALLBACK_TITLE_SQL, {"paper_id": paper_id}).first()
    if row is None:
        return None
    candidate = (row[0] or "").strip()
    if not candidate or _looks_broken(candidate):
        return None
    return candidate


def _work_id_from_paper(paper: Paper) -> str | None:
    """Recover the OpenAlex Work ID from the paper's ``url`` or broken title."""
    for candidate in (paper.url, paper.canonical_title):
        if candidate and "openalex.org/" in candidate:
            return candidate.rsplit("/", 1)[-1]
    return None


def _apply_update(
    paper: Paper, fields: dict[str, Any], *, force_title: bool
) -> list[str]:
    """Mutate ``paper`` in place; return the list of column names changed."""
    changed: list[str] = []

    new_title = fields.get("canonical_title")
    if force_title and isinstance(new_title, str) and new_title.strip():
        paper.canonical_title = new_title.strip()
        changed.append("canonical_title")
        new_norm = fields.get("normalized_title")
        if isinstance(new_norm, str):
            paper.normalized_title = new_norm
            changed.append("normalized_title")

    fill_if_empty: tuple[str, ...] = (
        "first_author",
        "year",
        "venue",
        "doi",
        "arxiv_id",
        "abstract",
    )
    for col in fill_if_empty:
        current = getattr(paper, col, None)
        if current in (None, ""):
            new_value = fields.get(col)
            if new_value not in (None, ""):
                setattr(paper, col, new_value)
                changed.append(col)

    if paper.authors in (None, {}):
        new_authors = fields.get("authors")
        if isinstance(new_authors, dict) and new_authors.get("list"):
            paper.authors = new_authors
            changed.append("authors")

    return changed


@app.command()
def main(
    limit: int | None = typer.Option(
        None, "--limit", help="Stop after this many papers (smoke test)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Fetch and log but do not commit."
    ),
) -> None:
    """Re-query OpenAlex for any paper row whose title is a URL or empty."""
    session = get_session()
    stats = {
        "scanned": 0,
        "fetched": 0,
        "updated_openalex": 0,
        "updated_references": 0,
        "no_work_id": 0,
        "no_data": 0,
        "still_broken": 0,
        "errors": 0,
    }
    started = time.monotonic()

    try:
        broken = _fetch_broken(session, limit)
        logger.info("found %d papers with broken/empty titles", len(broken))
        if not broken:
            typer.echo("No broken titles found.")
            return

        with OpenAlexClient() as oa:
            for paper in broken:
                stats["scanned"] += 1
                work_id = _work_id_from_paper(paper)
                if not work_id:
                    stats["no_work_id"] += 1
                    logger.warning(
                        "paper_id=%s has no recoverable OpenAlex Work ID",
                        paper.paper_id,
                    )
                    continue

                try:
                    work = oa.fetch_by_openalex_id(work_id)
                except Exception:
                    stats["errors"] += 1
                    logger.error(
                        "openalex lookup failed for paper_id=%s work_id=%s",
                        paper.paper_id,
                        work_id,
                        exc_info=True,
                    )
                    continue

                if work is None:
                    stats["no_data"] += 1
                    logger.warning(
                        "openalex returned no Work for paper_id=%s work_id=%s",
                        paper.paper_id,
                        work_id,
                    )
                    continue

                stats["fetched"] += 1
                fields = OpenAlexClient.extract_paper_fields(work)
                changed = _apply_update(
                    paper, fields, force_title=_looks_broken(paper.canonical_title)
                )

                if changed:
                    stats["updated_openalex"] += 1
                    logger.info(
                        "paper_id=%s updated from openalex: %s — new_title=%r",
                        paper.paper_id,
                        ",".join(changed),
                        paper.canonical_title[:80],
                    )

                # If OpenAlex couldn't supply a usable title, fall back to the
                # citing-side bibliography text. We see this on canonical papers
                # outside the ingest corpus (BERT, pre-2018 classics) whose
                # OpenAlex Work record has a NULL display_name.
                if _looks_broken(paper.canonical_title):
                    fallback = _fallback_title_from_references(
                        session, paper.paper_id
                    )
                    if fallback:
                        paper.canonical_title = fallback
                        paper.normalized_title = _normalize_title(fallback)
                        stats["updated_references"] += 1
                        logger.info(
                            "paper_id=%s updated from references — new_title=%r",
                            paper.paper_id,
                            fallback[:80],
                        )
                    else:
                        stats["still_broken"] += 1
                        logger.warning(
                            "paper_id=%s still has broken title after backfill",
                            paper.paper_id,
                        )

                if not dry_run and stats["scanned"] % 20 == 0:
                    session.commit()

            if not dry_run:
                session.commit()
            else:
                session.rollback()
    finally:
        session.close()

    elapsed = time.monotonic() - started
    typer.echo(
        f"\nBackfill summary\n"
        f"  scanned             : {stats['scanned']}\n"
        f"  fetched             : {stats['fetched']}\n"
        f"  updated (openalex)  : {stats['updated_openalex']}\n"
        f"  updated (references): {stats['updated_references']}\n"
        f"  no work id          : {stats['no_work_id']}\n"
        f"  no openalex data    : {stats['no_data']}\n"
        f"  still broken        : {stats['still_broken']}\n"
        f"  errors              : {stats['errors']}\n"
        f"  elapsed             : {elapsed:.1f}s\n"
        f"  mode                : {'dry-run' if dry_run else 'committed'}"
    )


if __name__ == "__main__":
    app()
