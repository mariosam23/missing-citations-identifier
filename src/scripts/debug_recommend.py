"""Context-level retrieval inspector for a single query.

Runs encode → dense + sparse retrieve → *context-level* RRF fusion → group →
score and prints a table of the top-K candidates with the *components* of their
score exposed (``mean_top_3``, distinct-citers bonus, popularity penalty), plus
the top-3 evidence sentences. Optionally pins a target paper (by ID or title
substring) so its rank is reported even when it falls outside the top-K.

NOTE — this is no longer a mirror of what ``/recommend`` ranks. Since the
prod/eval unification, ``/recommend`` (and the ``hybrid_rrf`` eval variant) rank
via *paper-level* RRF in ``pipeline.retrieval.hybrid``; the ordering and score
breakdown printed here are the *context-level* aggregator (preserved as the
``hybrid_context`` ablation). This tool is kept deliberately context-level
because its value is the per-branch context provenance and the score-component
breakdown, which only exist in that path — use it to inspect retrieval coverage
and the famous-paper question, not to predict the served ranking.

The ``src(d/s/b)`` column and the ``[d|s|b]`` tag on each evidence line make
the Phase 6 fusion observable: how many of a paper's contexts came from the
dense branch, the sparse branch, or both. The header line also reports the
dense/sparse/fused counts and the dense∩sparse overlap.

The output is the diagnostic for the §31.2 famous-paper-everywhere question:
"why isn't BERT ranked #1 for *We use BERT to encode sentences*?" — the
breakdown tells you whether the answer is low ``mean_top_3``, the popularity
penalty, missing retrieval, or something else.

Usage::

    python -m scripts.debug_recommend \\
        "We use BERT to encode sentences before computing similarity." \\
        --pin-title BERT
    python -m scripts.debug_recommend "..." --pin-paper-id 11531 --top-k 30
    python -m scripts.debug_recommend "..." --top-n 2000
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import typer
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from database.postgres.tables.papers import Paper
from pipeline.embedding.embedder import encode_query
from pipeline.retrieval.aggregate import (
    DISTINCT_CITERS_WEIGHT,
    POPULARITY_PENALTY_WEIGHT,
    PaperAggregate,
    compute_features,
    group_by_paper,
)
from pipeline.retrieval.dense import (
    DEFAULT_TOP_N,
    ContextSource,
    RetrievedContext,
    retrieve_dense,
)
from pipeline.retrieval.fusion import reciprocal_rank_fusion
from pipeline.retrieval.sparse import retrieve_sparse
from utils.logger import logger

# Windows cp1252 stdout can't encode Greek letters / em-dashes that appear in
# sentences and titles. Force UTF-8 so this diagnostic doesn't crash mid-table.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)

DEFAULT_TOP_K = 25
SENTENCE_PREVIEW_CHARS = 140


@dataclass(slots=True, frozen=True)
class _Row:
    rank: int
    paper_id: int
    title: str
    first_author: str
    year: int | None
    mean_top_3: float
    distinct_citers: int
    global_count: int
    score: float
    distinct_bonus: float
    popularity_penalty: float
    # Phase 6: how many of this paper's fused contexts came from each branch.
    n_dense: int
    n_sparse: int
    n_both: int


@app.command()
def main(
    query: str = typer.Argument(..., help="Query sentence to recommend for."),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Rows to print."),
    top_n: int = typer.Option(
        DEFAULT_TOP_N, help="Contexts pulled from dense retrieval."
    ),
    pin_paper_id: int | None = typer.Option(
        None, help="Report rank/score of this paper_id even if below top-K."
    ),
    pin_title: str | None = typer.Option(
        None,
        help="Substring match (case-insensitive) on canonical_title for the pin.",
    ),
    target_year: int | None = typer.Option(
        None, help="Same temporal filter as /recommend."
    ),
    show_evidence: bool = typer.Option(
        True, help="Print top-3 evidence sentences for each row."
    ),
) -> None:
    """Run the recommend pipeline and dump a score-breakdown report."""
    logger.info("query=%r top_k=%d top_n=%d", query, top_k, top_n)

    with get_session() as session:
        query_embedding = encode_query(query)
        dense_ctxs = retrieve_dense(
            session,
            query_embedding,
            top_n=top_n,
            target_year=target_year,
        )
        sparse_ctxs = retrieve_sparse(
            session,
            query,
            top_n=top_n,
            target_year=target_year,
        )
        fused = reciprocal_rank_fusion([dense_ctxs, sparse_ctxs], top_n=top_n)
        overlap = len(
            {c.context_id for c in dense_ctxs}
            & {c.context_id for c in sparse_ctxs}
        )
        typer.echo(
            f"\nRetrieval — dense={len(dense_ctxs)} sparse={len(sparse_ctxs)} "
            f"fused={len(fused)} overlap={overlap}"
        )
        typer.echo(
            "  (context-level inspector — ordering/score below is the "
            "pre-unification aggregator, NOT what /recommend now serves; "
            "see pipeline.retrieval.hybrid)"
        )
        if not sparse_ctxs:
            typer.echo(
                "  (sparse branch empty — query produced no lexemes, or no "
                "context matched any of them; fusion degraded to dense-only)"
            )
        if not fused:
            typer.echo("Empty result; nothing to score.")
            raise typer.Exit(code=0)

        aggregates = group_by_paper(fused)
        compute_features(session, aggregates)
        ranked = sorted(
            aggregates.values(), key=lambda a: a.score, reverse=True
        )

        paper_by_id = _hydrate_papers(session, [a.cited_paper_id for a in ranked])
        rows = [_to_row(rank, agg, paper_by_id) for rank, agg in enumerate(ranked, 1)]

        _print_table(rows[:top_k])
        if show_evidence:
            _print_evidence(ranked[:top_k], paper_by_id)

        pinned = _resolve_pin(session, ranked, paper_by_id, pin_paper_id, pin_title)
        if pinned is not None:
            pinned_rank, pinned_agg = pinned
            if pinned_rank <= top_k:
                typer.echo(
                    f"\nPinned paper already shown above at rank {pinned_rank}."
                )
            else:
                typer.echo("\nPinned paper (below top-K cutoff):")
                _print_table([_to_row(pinned_rank, pinned_agg, paper_by_id)])
                if show_evidence:
                    _print_evidence([pinned_agg], paper_by_id)

        _print_summary(ranked, top_k)


def _source_counts(contexts: list[RetrievedContext]) -> tuple[int, int, int]:
    """Return ``(dense_only, sparse_only, both)`` counts over fused contexts."""
    n_dense = sum(1 for c in contexts if c.source is ContextSource.DENSE)
    n_sparse = sum(1 for c in contexts if c.source is ContextSource.SPARSE)
    n_both = sum(1 for c in contexts if c.source is ContextSource.BOTH)
    return n_dense, n_sparse, n_both


def _to_row(
    rank: int, agg: PaperAggregate, paper_by_id: dict[int, Paper]
) -> _Row:
    paper = paper_by_id.get(agg.cited_paper_id)
    title = (paper.canonical_title if paper else "?") or "?"
    author = (paper.first_author if paper else "?") or "?"
    year = paper.year if paper else None
    distinct_bonus = DISTINCT_CITERS_WEIGHT * math.log1p(agg.distinct_citing_papers)
    popularity_ratio = agg.global_context_count / max(agg.distinct_citing_papers, 1)
    popularity_penalty = POPULARITY_PENALTY_WEIGHT * math.log1p(popularity_ratio)
    n_dense, n_sparse, n_both = _source_counts(agg.contexts)
    return _Row(
        rank=rank,
        paper_id=agg.cited_paper_id,
        title=title,
        first_author=author,
        year=year,
        mean_top_3=agg.mean_top_3_similarity,
        distinct_citers=agg.distinct_citing_papers,
        global_count=agg.global_context_count,
        score=agg.score,
        distinct_bonus=distinct_bonus,
        popularity_penalty=popularity_penalty,
        n_dense=n_dense,
        n_sparse=n_sparse,
        n_both=n_both,
    )


def _hydrate_papers(session: Session, paper_ids: list[int]) -> dict[int, Paper]:
    if not paper_ids:
        return {}
    rows = session.execute(
        select(Paper).where(Paper.paper_id.in_(paper_ids))
    ).scalars().all()
    return {p.paper_id: p for p in rows}


def _print_table(rows: list[_Row]) -> None:
    header = (
        f"{'#':>3}  {'paper_id':>8}  {'mean3':>6}  {'distinct':>8}  {'global':>6}  "
        f"{'+bonus':>6}  {'-penal':>6}  {'src(d/s/b)':>11}  {'score':>7}  who"
    )
    typer.echo("\n" + header)
    typer.echo("-" * len(header))
    for r in rows:
        who = _truncate(
            f"{r.first_author} ({r.year or 'n.d.'}) — {r.title}", 80
        )
        src_mix = f"{r.n_dense}/{r.n_sparse}/{r.n_both}"
        typer.echo(
            f"{r.rank:>3}  {r.paper_id:>8}  {r.mean_top_3:>6.3f}  "
            f"{r.distinct_citers:>8}  {r.global_count:>6}  "
            f"{r.distinct_bonus:>+6.3f}  {-r.popularity_penalty:>+6.3f}  "
            f"{src_mix:>11}  {r.score:>+7.3f}  {who}"
        )


def _print_evidence(
    aggregates: list[PaperAggregate], paper_by_id: dict[int, Paper]
) -> None:
    typer.echo("\nTop-3 evidence per candidate:")
    for rank, agg in enumerate(aggregates, 1):
        paper = paper_by_id.get(agg.cited_paper_id)
        title = (paper.canonical_title if paper else "?") or "?"
        typer.echo(f"\n  #{rank} [{agg.cited_paper_id}] {_truncate(title, 100)}")
        for ev in agg.top_evidence(3):
            typer.echo(
                f"    [{ev.source.value:>6}] sim={ev.similarity:.3f} | "
                f"{_truncate(ev.sentence, SENTENCE_PREVIEW_CHARS)}"
            )


def _resolve_pin(
    session: Session,
    ranked: list[PaperAggregate],
    paper_by_id: dict[int, Paper],
    pin_paper_id: int | None,
    pin_title: str | None,
) -> tuple[int, PaperAggregate] | None:
    if pin_paper_id is None and not pin_title:
        return None

    target_id = pin_paper_id
    if target_id is None and pin_title:
        target_id = _lookup_paper_id_by_title(session, pin_title)
        if target_id is None:
            typer.echo(
                f"\nNo paper matches title substring {pin_title!r}. Skipping pin.",
                err=True,
            )
            return None

    for rank, agg in enumerate(ranked, 1):
        if agg.cited_paper_id == target_id:
            return rank, agg

    assert target_id is not None
    paper = paper_by_id.get(target_id) or _hydrate_papers(session, [target_id]).get(
        target_id
    )
    title = paper.canonical_title if paper else "?"
    typer.echo(
        f"\nPinned paper_id={target_id} ({title!r}) is NOT in the retrieved "
        f"top-{len(ranked)} cited papers (i.e. no context for it surfaced in "
        f"the dense top-N). Increase --top-n or check coverage."
    )
    return None


def _lookup_paper_id_by_title(session: Session, substring: str) -> int | None:
    rows = session.execute(
        select(Paper.paper_id, Paper.canonical_title)
        .where(Paper.canonical_title.ilike(f"%{substring}%"))
        .limit(5)
    ).all()
    if not rows:
        return None
    if len(rows) > 1:
        typer.echo(
            f"\nWarning: {len(rows)} papers match {substring!r}; using the first:",
            err=True,
        )
        for pid, title in rows:
            typer.echo(f"  paper_id={pid}  {title!r}", err=True)
    return int(rows[0][0])


def _print_summary(ranked: list[PaperAggregate], top_k: int) -> None:
    if not ranked:
        return
    typer.echo(
        f"\nSummary: scored {len(ranked)} distinct papers; "
        f"top score = {ranked[0].score:+.3f}; "
        f"score at rank {top_k} = "
        f"{ranked[min(top_k, len(ranked)) - 1].score:+.3f}."
    )


def _truncate(value: str, max_chars: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


if __name__ == "__main__":
    app()
