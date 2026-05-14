"""Bulk-embed citation contexts with BAAI/bge-base-en-v1.5.

Phase 3 — produces the ``citation_context_embeddings`` rows that back the
``POST /recommend`` dense retrieval path. Fully resumable: the inner query
left-joins ``citation_context_embeddings`` and only pulls rows that have not
been embedded yet, so re-running after a crash, a DB wipe, or after
``parse_corpus`` adds more documents Just Works.

Order of operations matters: bulk-insert all embeddings FIRST, then build
the HNSW index in the follow-up Alembic revision (``0003_create_hnsw_index``).
Inserting into an existing HNSW index is ~10× slower.

Usage::

    python -m scripts.embed_contexts
    python -m scripts.embed_contexts --page-size 5000 --batch-size 64
    python -m scripts.embed_contexts --limit 1000     # smoke test
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import typer
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from database.postgres.engine import get_session
from pipeline.embedding.embedder import encode_texts, get_embedder
from utils.config import config
from utils.logger import logger

DEFAULT_PAGE_SIZE = 5000

app = typer.Typer(add_completion=False)


_FETCH_PAGE_SQL = text(
    """
    SELECT cc.context_id, cc.context_text_for_embedding
    FROM citation_contexts cc
    LEFT JOIN citation_context_embeddings cce
           ON cce.context_id = cc.context_id
    WHERE cce.context_id IS NULL
      AND cc.context_text_for_embedding IS NOT NULL
      AND length(trim(cc.context_text_for_embedding)) > 0
    ORDER BY cc.context_id
    LIMIT :page_size
    """
)

_INSERT_SQL = text(
    """
    INSERT INTO citation_context_embeddings (context_id, embedding, model_name)
    VALUES (:context_id, :embedding, :model_name)
    ON CONFLICT (context_id) DO NOTHING
    """
)


def _fetch_page(session: Session, page_size: int) -> list[tuple[int, str]]:
    rows = session.execute(_FETCH_PAGE_SQL, {"page_size": page_size}).all()
    return [(r[0], r[1]) for r in rows]


def _insert_page(
    session: Session,
    ids: list[int],
    vectors: np.ndarray,
    model_name: str,
) -> None:
    """Bulk-insert one page of embeddings.

    pgvector's psycopg adapter (registered on connect in ``engine.py``)
    accepts numpy arrays directly as ``vector`` values.
    """
    params: list[dict[str, Any]] = [
        {
            "context_id": cid,
            "embedding": vectors[i],
            "model_name": model_name,
        }
        for i, cid in enumerate(ids)
    ]
    session.execute(_INSERT_SQL, params)


def _total_remaining(session: Session) -> int:
    return session.execute(
        text(
            """
            SELECT COUNT(*)
            FROM citation_contexts cc
            LEFT JOIN citation_context_embeddings cce
                   ON cce.context_id = cc.context_id
            WHERE cce.context_id IS NULL
              AND cc.context_text_for_embedding IS NOT NULL
              AND length(trim(cc.context_text_for_embedding)) > 0
            """
        )
    ).scalar_one()


@app.command()
def main(
    page_size: int = typer.Option(
        DEFAULT_PAGE_SIZE, "--page-size", help="Rows fetched per DB round-trip."
    ),
    batch_size: int = typer.Option(
        config.EMBEDDER_BATCH_SIZE,
        "--batch-size",
        help="Sentences per encoder forward pass.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Stop after this many contexts (for smoke tests)."
    ),
) -> None:
    """Embed every un-embedded citation context with the configured model."""
    model_name = config.EMBEDDER_MODEL_NAME
    logger.info(
        "embed_contexts started — model=%s page_size=%d batch_size=%d limit=%s",
        model_name,
        page_size,
        batch_size,
        limit,
    )

    # Force model load so the first page does not include the cold-start cost
    # in its throughput numbers.
    get_embedder()

    session = get_session()
    total = 0
    started = time.monotonic()
    try:
        remaining = _total_remaining(session)
        if limit is not None:
            remaining = min(remaining, limit)
        logger.info("contexts to embed: %d", remaining)
        if remaining == 0:
            typer.echo("Nothing to embed.")
            return

        with tqdm(total=remaining, unit="ctx") as bar:
            while True:
                fetch_n = page_size
                if limit is not None:
                    fetch_n = min(page_size, limit - total)
                    if fetch_n <= 0:
                        break

                page = _fetch_page(session, fetch_n)
                if not page:
                    break

                ids = [cid for cid, _ in page]
                texts = [t for _, t in page]
                vectors = encode_texts(
                    texts, batch_size=batch_size, show_progress_bar=False
                )
                if vectors.shape != (len(texts), config.EMBEDDER_DIM):
                    raise RuntimeError(
                        f"embedder returned shape={vectors.shape}, "
                        f"expected ({len(texts)}, {config.EMBEDDER_DIM})"
                    )

                _insert_page(session, ids, vectors, model_name)
                session.commit()

                total += len(page)
                bar.update(len(page))

        elapsed = time.monotonic() - started
        rate = total / elapsed if elapsed > 0 else 0.0
        logger.info(
            "embed_contexts finished — embedded=%d elapsed=%.1fs rate=%.1f/s",
            total,
            elapsed,
            rate,
        )
        typer.echo(
            f"\nEmbedding summary\n"
            f"  embedded : {total}\n"
            f"  elapsed  : {elapsed:.1f}s\n"
            f"  rate     : {rate:.1f}/s"
        )
    finally:
        session.close()


if __name__ == "__main__":
    app()
