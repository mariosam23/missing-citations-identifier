"""Stage / cut over / finalize / roll back the window-text re-embed experiment.

Background: embeddings are currently built from ``sentence_without_markers``
(the bare citing sentence). This experiment re-embeds from
``local_window_text`` (the sentence plus ~240 chars of neighbouring context,
which contains the sentence for 100% of rows) to test whether the richer unit
improves retrieval. Only the *embedded* text changes; the displayed evidence
sentence is unaffected.

Because the dense index lives in one table that both ``/recommend`` and the eval
harness read, the swap is destructive and is split into stages so the live index
is empty for the shortest possible window:

    prepare   (safe, run now)      snapshot embeddings → repoint
                                    context_text_for_embedding to the window.
                                    The live table + index are untouched.
    cutover   (destructive)        drop the HNSW index + TRUNCATE embeddings.
                                    Run this immediately before the Colab embed.
    --- run notebooks/colab_embed_contexts.ipynb on Colab (reads the repointed
        context_text_for_embedding; no flag needed) ---
    finalize  (after Colab)        recreate the HNSW index over the new vectors.
    rollback  (any time)           restore sentence embeddings from the snapshot,
                                    repoint back to the sentence, recreate index.

A full ``pg_dump`` backup is the ultimate safety net; the in-DB snapshot table
just makes rollback fast (a copy, not a restore).

Usage::

    python -m scripts.prepare_window_reembed prepare
    python -m scripts.prepare_window_reembed cutover --yes
    python -m scripts.prepare_window_reembed finalize
    python -m scripts.prepare_window_reembed rollback --yes
"""

from __future__ import annotations

import typer
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from utils.logger import logger

app = typer.Typer(add_completion=False)

EMB_TABLE = "citation_context_embeddings"
BACKUP_TABLE = "citation_context_embeddings_backup_sentence"
HNSW_INDEX = "citation_context_embedding_hnsw_idx"
# Matches the live definition (and Alembic 0011) exactly.
CREATE_HNSW_SQL = (
    f"CREATE INDEX {HNSW_INDEX} ON {EMB_TABLE} "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
)
REPOINT_TO_WINDOW = (
    "UPDATE citation_contexts "
    "SET context_text_for_embedding = "
    "COALESCE(NULLIF(trim(local_window_text), ''), sentence_without_markers)"
)
REPOINT_TO_SENTENCE = (
    "UPDATE citation_contexts "
    "SET context_text_for_embedding = sentence_without_markers"
)


def _scalar(session: Session, sql: str) -> int:
    return int(session.execute(text(sql)).scalar() or 0)


def _table_exists(session: Session, name: str) -> bool:
    return bool(
        session.execute(
            text("SELECT to_regclass(:n) IS NOT NULL"), {"n": f"public.{name}"}
        ).scalar()
    )


def _index_exists(session: Session, name: str) -> bool:
    return bool(
        session.execute(
            text("SELECT to_regclass(:n) IS NOT NULL"), {"n": f"public.{name}"}
        ).scalar()
    )


@app.command()
def prepare(force: bool = typer.Option(False, "--force", help="Overwrite an existing snapshot table.")) -> None:
    """Safe staging: snapshot embeddings + repoint to the window. Non-destructive."""
    session = get_session()
    try:
        if _table_exists(session, BACKUP_TABLE):
            if not force:
                raise typer.BadParameter(
                    f"{BACKUP_TABLE} already exists — rollback/inspect it first, "
                    f"or pass --force to overwrite."
                )
            session.execute(text(f"DROP TABLE {BACKUP_TABLE}"))
            logger.info("dropped existing %s (--force)", BACKUP_TABLE)

        live = _scalar(session, f"SELECT COUNT(*) FROM {EMB_TABLE}")
        logger.info("snapshotting %d embeddings → %s ...", live, BACKUP_TABLE)
        session.execute(text(f"CREATE TABLE {BACKUP_TABLE} AS SELECT * FROM {EMB_TABLE}"))
        backed = _scalar(session, f"SELECT COUNT(*) FROM {BACKUP_TABLE}")
        if backed != live:
            raise RuntimeError(f"snapshot mismatch: {backed} != {live}")

        logger.info("repointing context_text_for_embedding → local_window_text ...")
        repointed = session.execute(text(REPOINT_TO_WINDOW)).rowcount

        session.commit()
        same = _scalar(
            session,
            "SELECT COUNT(*) FROM citation_contexts "
            "WHERE context_text_for_embedding IS NOT DISTINCT FROM sentence_without_markers",
        )
        typer.echo(
            f"prepared: snapshot={backed} rows, repointed={repointed} rows; "
            f"{same} rows still equal the bare sentence (no window available).\n"
            f"Live index untouched. Next: `cutover --yes` right before the Colab embed."
        )
    finally:
        session.close()


@app.command()
def cutover(yes: bool = typer.Option(False, "--yes", help="Confirm the destructive truncate.")) -> None:
    """Destructive: drop the HNSW index and TRUNCATE embeddings. Run just before Colab."""
    session = get_session()
    try:
        if not _table_exists(session, BACKUP_TABLE):
            raise typer.BadParameter(
                f"no {BACKUP_TABLE} snapshot — run `prepare` first."
            )
        live = _scalar(session, f"SELECT COUNT(*) FROM {EMB_TABLE}")
        if not yes:
            raise typer.BadParameter(
                f"this will DROP {HNSW_INDEX} and TRUNCATE {EMB_TABLE} "
                f"({live} rows). Re-run with --yes to proceed."
            )
        session.execute(text(f"DROP INDEX IF EXISTS {HNSW_INDEX}"))
        session.execute(text(f"TRUNCATE {EMB_TABLE}"))
        session.commit()
        typer.echo(
            f"cutover done: index dropped, {EMB_TABLE} truncated (was {live} rows).\n"
            f"Now run notebooks/colab_embed_contexts.ipynb on Colab, then `finalize`."
        )
    finally:
        session.close()


@app.command()
def finalize() -> None:
    """After the Colab embed: recreate the HNSW index over the new vectors."""
    session = get_session()
    try:
        rows = _scalar(session, f"SELECT COUNT(*) FROM {EMB_TABLE}")
        if rows == 0:
            raise typer.BadParameter(
                f"{EMB_TABLE} is empty — run the Colab embed before finalize."
            )
        if _index_exists(session, HNSW_INDEX):
            typer.echo(f"{HNSW_INDEX} already exists; nothing to do ({rows} rows).")
            return
        logger.info("building %s over %d vectors (this can take a few minutes)...", HNSW_INDEX, rows)
        session.execute(text(CREATE_HNSW_SQL))
        session.commit()
        typer.echo(f"finalize done: {HNSW_INDEX} rebuilt over {rows} embeddings.")
    finally:
        session.close()


@app.command()
def rollback(yes: bool = typer.Option(False, "--yes", help="Confirm restoring sentence embeddings.")) -> None:
    """Restore sentence embeddings from the snapshot and repoint back."""
    session = get_session()
    try:
        if not _table_exists(session, BACKUP_TABLE):
            raise typer.BadParameter(f"no {BACKUP_TABLE} to restore from.")
        backed = _scalar(session, f"SELECT COUNT(*) FROM {BACKUP_TABLE}")
        if not yes:
            raise typer.BadParameter(
                f"this will replace {EMB_TABLE} with the {backed}-row snapshot and "
                f"repoint to the sentence. Re-run with --yes."
            )
        session.execute(text(f"DROP INDEX IF EXISTS {HNSW_INDEX}"))
        session.execute(text(f"TRUNCATE {EMB_TABLE}"))
        session.execute(text(f"INSERT INTO {EMB_TABLE} SELECT * FROM {BACKUP_TABLE}"))
        session.execute(text(REPOINT_TO_SENTENCE))
        logger.info("rebuilding %s ...", HNSW_INDEX)
        session.execute(text(CREATE_HNSW_SQL))
        session.commit()
        restored = _scalar(session, f"SELECT COUNT(*) FROM {EMB_TABLE}")
        typer.echo(
            f"rolled back: {restored} sentence embeddings restored, index rebuilt, "
            f"context_text_for_embedding repointed to the sentence.\n"
            f"Drop {BACKUP_TABLE} manually once you are satisfied."
        )
    finally:
        session.close()


if __name__ == "__main__":
    app()
