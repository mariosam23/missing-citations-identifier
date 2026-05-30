"""Build the sparse-retrieval stoplexeme cache.

``retrieve_sparse`` is fast only when its ``tsquery`` is *selective*. The
``&``→``|`` OR-rewrite (needed so short citation sentences match at all) makes
the query match a huge fraction of the corpus when it contains common lexemes —
"use" alone is in ~21% of contexts, and the unstemmed ``simple`` column carries
bare stopwords ("the" ~67%). The planner then abandons the GIN indexes and
sequentially scans + ``ts_rank_cd``-ranks ~60k+ rows (~2.6s/query).

This script materialises the high-document-frequency lexemes per text-search
config into ``sparse_stoplexeme``; ``retrieve_sparse`` drops them from the query
so only distinctive (low-DF, high-IDF) lexemes remain. That keeps the query on
the GIN index (~7–100ms) and, because common lexemes don't discriminate
citations anyway, also sharpens precision.

Derived/regenerable cache (not an Alembic-managed schema object): re-run after
the corpus grows. ``CREATE TABLE IF NOT EXISTS`` + ``TRUNCATE`` make it
idempotent. If the table is empty, ``retrieve_sparse`` simply drops nothing and
behaves as before (correct, just slow) — so this is safe to run anytime.

Usage::

    python -m scripts.build_sparse_stoplexeme                 # threshold 2%
    python -m scripts.build_sparse_stoplexeme --min-df-frac 0.01
"""

from __future__ import annotations

import typer
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from utils.logger import logger

app = typer.Typer(add_completion=False)

# (config name, the GENERATED tsvector column it indexes).
_CONFIGS: tuple[tuple[str, str], ...] = (
    ("english", "sentence_tsv_english"),
    ("simple", "sentence_tsv_simple"),
)

_CREATE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS sparse_stoplexeme (
        config text NOT NULL,
        word   text NOT NULL,
        ndoc   integer NOT NULL,
        PRIMARY KEY (config, word)
    )
    """
)


def _total_contexts(session: Session) -> int:
    return session.execute(
        text("SELECT count(*) FROM citation_contexts")
    ).scalar_one()


def _populate_config(
    session: Session, config: str, column: str, min_ndoc: int
) -> int:
    """Insert lexemes for one config whose document frequency exceeds the floor."""
    result = session.execute(
        text(
            f"""
            INSERT INTO sparse_stoplexeme (config, word, ndoc)
            SELECT :config, word, ndoc
            FROM ts_stat('SELECT {column} FROM citation_contexts')
            WHERE ndoc > :min_ndoc
            ON CONFLICT (config, word) DO UPDATE SET ndoc = EXCLUDED.ndoc
            """
        ),
        {"config": config, "min_ndoc": min_ndoc},
    )
    return result.rowcount


@app.command()
def main(
    min_df_frac: float = typer.Option(
        0.02,
        "--min-df-frac",
        min=0.0,
        max=1.0,
        help="Exclude lexemes appearing in more than this fraction of contexts.",
    ),
) -> None:
    """Rebuild ``sparse_stoplexeme`` from the live corpus."""
    session = get_session()
    try:
        session.execute(_CREATE_SQL)
        total = _total_contexts(session)
        if total == 0:
            typer.echo("No contexts; nothing to do.")
            return

        min_ndoc = int(total * min_df_frac)
        logger.info(
            "building sparse stoplexeme — total_contexts=%d min_df_frac=%.3f "
            "min_ndoc=%d",
            total,
            min_df_frac,
            min_ndoc,
        )

        session.execute(text("TRUNCATE sparse_stoplexeme"))
        counts = {
            config: _populate_config(session, config, column, min_ndoc)
            for config, column in _CONFIGS
        }
        session.commit()

        for config, n in counts.items():
            typer.echo(f"  {config:<8} stoplexemes: {n}")
        typer.echo(f"Total: {sum(counts.values())} (min_ndoc={min_ndoc})")
    finally:
        session.close()


if __name__ == "__main__":
    app()
