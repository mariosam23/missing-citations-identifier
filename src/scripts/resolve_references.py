"""Resolve bibliography references to canonical paper rows.

Phase 2 — full cascade: DOI, arXiv, title exact, title+author+year, fuzzy.

For every ``references`` row where ``cited_paper_id IS NULL`` and at least
one resolvable field is present, the resolver tries five strategies in order:

1. DOI exact match against ``papers.doi``
2. arXiv ID exact match against ``papers.arxiv_id``
3. Normalized title exact match against ``papers.normalized_title``
4. Normalized title + first author + year ±1
5. Fuzzy title (rapidfuzz token_set_ratio) + first author + year ±1

For strategies 1–2, if no local match exists, the resolver calls OpenAlex to
enrich and creates a new ``papers`` row before linking.

After all references are resolved, a single bulk UPDATE propagates
``cited_paper_id`` from ``references`` into ``citation_contexts`` so that
Phase 3 retrieval can join directly on the contexts table.

Fully resumable: already-resolved rows (``cited_paper_id IS NOT NULL``) are
skipped.

Usage::

    python -m scripts.resolve_references
    python -m scripts.resolve_references --batch-size 200 --dry-run
"""

from __future__ import annotations

import time
from typing import Any, cast

import typer
from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from database.postgres.tables.references_ import Reference
from pipeline.resolution.openalex_client import OpenAlexClient
from pipeline.resolution.resolver import ReferenceResolver
from pipeline.resolution.semantic_scholar_client import SemanticScholarClient
from utils.logger import logger

DEFAULT_BATCH_SIZE = 100

app = typer.Typer(add_completion=False)


def _fetch_batch(
    session: Session,
    batch_size: int,
    after_id: int,
) -> list[Reference]:
    """Cursor-paginate unresolved references by ``reference_id``.

    Using ``reference_id > after_id`` instead of SQL OFFSET guarantees
    forward progress regardless of whether rows were resolved in prior
    batches — resolved rows are skipped by the ``cited_paper_id IS NULL``
    filter, but unresolvable rows are advanced past by the cursor.
    """
    rows = (
        session.query(Reference)
        .filter(
            Reference.reference_id > after_id,
            Reference.cited_paper_id.is_(None),
            (Reference.doi.isnot(None))
            | (Reference.arxiv_id.isnot(None))
            | (Reference.parsed_title.isnot(None)),
        )
        .order_by(Reference.reference_id)
        .limit(batch_size)
        .all()
    )
    return rows


def _propagate_cited_paper_id(session: Session) -> int:
    """Copy ``references.cited_paper_id`` into matching ``citation_contexts`` rows.

    Returns the number of contexts updated.
    """
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
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="References to process per DB commit.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve and log but do not commit changes.",
    ),
) -> None:
    """Resolve unresolved references via DOI and arXiv exact match."""
    logger.info(
        "resolve_references started — batch_size=%d dry_run=%s",
        batch_size,
        dry_run,
    )
    t0 = time.monotonic()

    stats: dict[str, int] = {
        "processed": 0,
        "resolved_doi": 0,
        "resolved_arxiv": 0,
        "resolved_title_exact": 0,
        "resolved_title_author_year": 0,
        "resolved_fuzzy": 0,
        "resolved_fuzzy_tentative": 0,
        "resolved_title_search_s2": 0,
        "unresolved": 0,
        "errors": 0,
    }

    with OpenAlexClient() as oa_client, SemanticScholarClient() as s2_client:
        resolver = ReferenceResolver(oa_client, s2_client)
        session = get_session()
        try:
            last_id = 0
            while True:
                batch = _fetch_batch(session, batch_size, after_id=last_id)
                if not batch:
                    break

                for ref in batch:
                    try:
                        paper_id, method, confidence = resolver.resolve(session, ref)
                    except Exception:
                        logger.error(
                            "resolver error for reference_id=%s",
                            ref.reference_id,
                            exc_info=True,
                        )
                        stats["errors"] += 1
                        stats["processed"] += 1
                        continue

                    if paper_id is not None:
                        ref.cited_paper_id = paper_id
                        ref.resolution_method = method
                        ref.resolution_confidence = confidence
                        stats[f"resolved_{method}"] = stats.get(f"resolved_{method}", 0) + 1
                    else:
                        stats["unresolved"] += 1

                    stats["processed"] += 1

                # Advance cursor past this batch regardless of resolution outcome.
                last_id = batch[-1].reference_id

                elapsed = time.monotonic() - t0
                if not dry_run:
                    session.commit()
                else:
                    session.rollback()

                logger.info(
                    "batch done — last_id=%d processed=%d doi=%d arxiv=%d s2=%d "
                    "unresolved=%d errors=%d elapsed=%.1fs",
                    last_id,
                    stats["processed"],
                    stats.get("resolved_doi", 0),
                    stats.get("resolved_arxiv", 0),
                    stats.get("resolved_title_search_s2", 0),
                    stats["unresolved"],
                    stats["errors"],
                    elapsed,
                )

            if not dry_run:
                ctx_updated = _propagate_cited_paper_id(session)
                session.commit()
                logger.info(
                    "propagated cited_paper_id to %d citation_contexts rows",
                    ctx_updated,
                )
        finally:
            session.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "resolve_references finished — processed=%d doi=%d arxiv=%d s2=%d "
        "unresolved=%d errors=%d elapsed=%.1fs",
        stats["processed"],
        stats.get("resolved_doi", 0),
        stats.get("resolved_arxiv", 0),
        stats.get("resolved_title_search_s2", 0),
        stats["unresolved"],
        stats["errors"],
        elapsed,
    )

    _print_summary(stats, elapsed)


def _print_summary(stats: dict[str, Any], elapsed: float) -> None:
    total = stats["processed"]
    resolved = sum(
        v for k, v in stats.items() if k.startswith("resolved_")
    )
    pct = resolved * 100.0 / total if total else 0.0
    typer.echo(
        f"\nResolution summary\n"
        f"  processed        : {total}\n"
        f"  resolved         : {resolved} ({pct:.1f}%)\n"
        f"    doi            : {stats.get('resolved_doi', 0)}\n"
        f"    arxiv          : {stats.get('resolved_arxiv', 0)}\n"
        f"    title_exact    : {stats.get('resolved_title_exact', 0)}\n"
        f"    title+author+yr: {stats.get('resolved_title_author_year', 0)}\n"
        f"    fuzzy          : {stats.get('resolved_fuzzy', 0)}\n"
        f"    fuzzy_tentative: {stats.get('resolved_fuzzy_tentative', 0)}\n"
        f"    s2 title-match : {stats.get('resolved_title_search_s2', 0)}\n"
        f"  unresolved       : {stats['unresolved']}\n"
        f"  errors           : {stats['errors']}\n"
        f"  elapsed          : {elapsed:.1f}s"
    )


if __name__ == "__main__":
    app()