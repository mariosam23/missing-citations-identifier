"""Grid-search the aggregator scoring hyperparameters on a split — no GPU, no re-embed.

Dense retrieval returns the same rank-ordered contexts regardless of how they are
later scored, so we encode each query once, run ``retrieve_dense`` once, cache the
contexts, and then re-score the cache under every ``(distinct_weight, top_m,
score_top_n)`` combination in memory. The scoring loop touches no DB (the
popularity penalty stays at 0), so the whole grid evaluates in seconds.

`ef_search` changes *which* contexts HNSW returns, so it cannot be swept from the
cache — pass ``--ef-search`` and re-run the script per value to find the knee.

Usage::

    python -m scripts.sweep_scoring                       # default grid on val
    python -m scripts.sweep_scoring --ef-search 200       # re-retrieve deeper
    python -m scripts.sweep_scoring --split-name test     # confirm on test
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import typer
from sqlalchemy import text
from tqdm import tqdm

from database.postgres.engine import get_session
from evaluation.dataset import DEFAULT_SPLIT_DIR, load_split, materialise_queries
from evaluation.metrics import compute_all_metrics
from pipeline.embedding.embedder import encode_texts
from pipeline.retrieval.aggregate import compute_features, group_by_paper, rank_papers
from pipeline.retrieval.dense import RetrievedContext, retrieve_dense
from utils.logger import logger

app = typer.Typer(add_completion=False)

KS: tuple[int, ...] = (1, 5, 10, 20)
_MRR_KEY = f"mrr@{max(KS)}"
_TRACKED = [f"recall@{k}" for k in KS] + [_MRR_KEY]


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _score_cache(
    cache: list[tuple[int, list[RetrievedContext]]],
    *,
    distinct_weight: float,
    top_m: int,
    score_top_n: int,
    top_k: int,
    session,
) -> dict[str, float]:
    """Re-score every cached query under one config; return mean metrics."""
    totals = {key: 0.0 for key in _TRACKED}
    for gold, contexts in cache:
        ranked: list[int] = []
        sub = contexts[:score_top_n]
        if sub:
            aggregates = group_by_paper(sub)
            compute_features(
                session,
                aggregates,
                distinct_weight=distinct_weight,
                top_m=top_m,
                penalty_weight=0.0,
            )
            ranked = [a.cited_paper_id for a in rank_papers(aggregates, top_k=top_k)]
        metrics = compute_all_metrics(ranked, gold, ks=KS)
        for key in _TRACKED:
            totals[key] += metrics[key]
    n = max(len(cache), 1)
    return {key: totals[key] / n for key in _TRACKED}


@app.command()
def main(
    split_path: str = typer.Option(
        str(DEFAULT_SPLIT_DIR / "split_42.json"), help="Eval split JSON."
    ),
    split_name: str = typer.Option("val", help="Which partition: val or test."),
    retrieve_top_n: int = typer.Option(
        1000, help="Contexts pulled per query (cache depth)."
    ),
    distinct_weights: str = typer.Option(
        "0,0.1,0.2,0.3,0.5,0.7", help="Comma-separated distinct-citers weights."
    ),
    top_ms: str = typer.Option("1,3,5,10", help="Comma-separated mean-top-m cutoffs."),
    score_top_ns: str = typer.Option(
        "200,500,1000", help="Comma-separated scoring depths (≤ retrieve_top_n)."
    ),
    top_k: int = typer.Option(20, help="Ranking depth handed to the metrics."),
    ef_search: int = typer.Option(100, help="HNSW ef_search for this run."),
    output_path: str = typer.Option(
        "data/eval/reports/scoring_sweep_{split}_ef{ef}.json",
        help="Report path; {split}/{ef} are substituted.",
    ),
) -> None:
    session = get_session()
    if ef_search:
        # SET does not accept bind parameters; ef_search is int-validated by typer.
        session.execute(text(f"SET hnsw.ef_search = {int(ef_search)}"))
        logger.info("hnsw.ef_search set to %d", ef_search)

    split = load_split(Path(split_path), strict=False, session=session)
    paper_ids = (
        split.val_paper_ids if split_name == "val" else split.test_paper_ids
    )
    queries, skipped = materialise_queries(
        session, paper_ids, target_year=None, require_reachable=True
    )
    if not queries:
        logger.error("no queries materialised — check the split / DB")
        raise typer.Exit(code=1)
    logger.info("materialised %d queries (skipped %d unreachable)", len(queries), skipped)

    t0 = time.perf_counter()
    matrix = encode_texts(
        [q.sentence for q in queries], show_progress_bar=True, is_query=True
    )
    logger.info("encoded %d queries in %.1fs", len(queries), time.perf_counter() - t0)

    t0 = time.perf_counter()
    cache: list[tuple[int, list[RetrievedContext]]] = []
    for i, q in enumerate(tqdm(queries, desc="retrieve", unit="q")):
        contexts = retrieve_dense(
            session,
            matrix[i],
            top_n=retrieve_top_n,
            exclude_citing_paper_id=q.citing_paper_id,
        )
        cache.append((q.gold_paper_id, contexts))
    logger.info("retrieved %d queries in %.1fs", len(cache), time.perf_counter() - t0)

    grid = list(
        itertools.product(
            _parse_floats(distinct_weights),
            _parse_ints(top_ms),
            _parse_ints(score_top_ns),
        )
    )
    logger.info("scoring %d configs over the cache...", len(grid))
    results: list[dict[str, float]] = []
    for dw, tm, stn in grid:
        means = _score_cache(
            cache,
            distinct_weight=dw,
            top_m=tm,
            score_top_n=stn,
            top_k=top_k,
            session=session,
        )
        row = {"distinct_weight": dw, "top_m": tm, "score_top_n": stn}
        row.update({k: round(v, 4) for k, v in means.items()})
        results.append(row)

    results.sort(key=lambda r: r["recall@20"], reverse=True)

    out = output_path.format(split=split_name, ef=ef_search)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(
            {
                "split": split_name,
                "ef_search": ef_search,
                "retrieve_top_n": retrieve_top_n,
                "num_queries": len(queries),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", out)

    print(f"\nTop 12 configs by recall@20 ({split_name}, ef_search={ef_search}):")
    print(f"  {'dw':>4} {'top_m':>5} {'top_n':>5} | {'r@5':>6} {'r@10':>6} {'r@20':>6} {'mrr':>6}")
    for r in results[:12]:
        print(
            f"  {r['distinct_weight']:>4} {r['top_m']:>5} {r['score_top_n']:>5} | "
            f"{r['recall@5']:>6} {r['recall@10']:>6} {r['recall@20']:>6} {r[_MRR_KEY]:>6}"
        )
    session.close()


if __name__ == "__main__":
    app()
