"""Stratified evaluation: retrieval metrics bucketed by gold-paper citation density.

The single most important validation of the co-citation hypothesis: if
"similar sentences cite the same paper" works, recall must climb as the gold
paper gains more *independent* citers in the corpus (more evidence to match
against). This script runs a retrieval variant over a split exactly like
``scripts.evaluate``, but tags every query with its gold paper's distinct-citer
count and reports per-bucket aggregates instead of one global number.

Buckets are by the gold paper's corpus-wide distinct citing papers
(``2``, ``3-5``, ``6-10``, ``10+``). The query's own citing paper is excluded
from the candidate pool at retrieval time (same leakage guard as the main
harness), so a "2 citers" gold paper offers exactly one other citer as evidence.

Usage::

    python -m scripts.stratified_eval --variant dense_only --split test
    python -m scripts.stratified_eval --variant dense_only --split val --workers 4
"""

from __future__ import annotations

import importlib
import itertools
import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import typer
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from database.postgres.engine import get_session
from evaluation.dataset import (
    DEFAULT_SEED,
    DEFAULT_SPLIT_DIR,
    EvalQuery,
    load_split,
    materialise_queries,
)
from evaluation.metrics import compute_all_metrics
from evaluation.runner import KS
from utils.logger import logger

# Windows cp1252 stdout fix (same as scripts.evaluate / debug_recommend).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)

_VARIANT_REGISTRY: dict[str, str] = {
    "dense_only": "evaluation.variants.dense_only.DenseOnly",
    "sparse_only": "evaluation.variants.sparse_only.SparseOnly",
    "hybrid_rrf": "evaluation.variants.hybrid_rrf.HybridRRF",
}

EVAL_DEFAULT_TOP_N = 200

# Bucket edges on the gold paper's corpus-wide distinct-citer count. Ordered;
# first matching label wins. Mirrors the §2 corpus breakdown so per-bucket
# query counts line up with the dataset statistics.
_BUCKET_ORDER: tuple[str, ...] = ("2", "3-5", "6-10", "10+")


def _bucket_for(n_citers: int) -> str:
    """Map a distinct-citer count to its density bucket label."""
    if n_citers <= 2:
        return "2"
    if n_citers <= 5:
        return "3-5"
    if n_citers <= 10:
        return "6-10"
    return "10+"


def _load_variant_class(name: str) -> type:
    if name not in _VARIANT_REGISTRY:
        available = ", ".join(sorted(_VARIANT_REGISTRY))
        raise typer.BadParameter(f"Unknown variant {name!r}. Available: {available}")
    module_path, class_name = _VARIANT_REGISTRY[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _set_ef_search(session: Session, ef_search: int | None) -> None:
    """Override per-session ``hnsw.ef_search`` (engine default is 100)."""
    if ef_search:
        # SET takes no bind params; ef_search is int-validated by the caller.
        session.execute(text(f"SET hnsw.ef_search = {int(ef_search)}"))


def _citer_counts(session: Session, gold_ids: list[int]) -> dict[int, int]:
    """Map each gold paper id to its corpus-wide distinct citing-paper count."""
    if not gold_ids:
        return {}
    rows = session.execute(
        text(
            """
            SELECT cited_paper_id, COUNT(DISTINCT citing_paper_id) AS n
            FROM citation_contexts
            WHERE cited_paper_id = ANY(:ids)
              AND cited_paper_id IS NOT NULL
            GROUP BY cited_paper_id
            """
        ),
        {"ids": gold_ids},
    ).all()
    return {int(r[0]): int(r[1]) for r in rows}


def _batch_embed(queries: list[EvalQuery]) -> list[np.ndarray]:
    """Encode all query sentences in one GPU pass (asymmetric query mode)."""
    from pipeline.embedding.embedder import encode_texts

    texts = [q.sentence for q in queries]
    logger.info("batch-encoding %d queries...", len(texts))
    t0 = time.perf_counter()
    matrix = encode_texts(texts, show_progress_bar=True, is_query=True)
    logger.info("encoded in %.1fs", time.perf_counter() - t0)
    return [matrix[i] for i in range(matrix.shape[0])]


def _evaluate(
    variant_cls: type,
    queries: list[EvalQuery],
    embeddings: list[np.ndarray],
    *,
    top_n: int,
    top_k: int,
    target_year: int | None,
    workers: int,
    ef_search: int | None,
) -> list[dict[str, float]]:
    """Score every query, returning per-query metric dicts in query order.

    Threaded with one Session + Variant per worker (mirrors EvalRunner). The
    result list is index-aligned with ``queries`` so density buckets attach
    unambiguously.
    """
    n = len(queries)
    results: list[dict[str, float] | None] = [None] * n
    counter = itertools.count()
    progress = tqdm(total=n, desc=variant_cls.name, unit="q")

    def score_one(variant: object, idx: int) -> dict[str, float]:
        q = queries[idx]
        ranked = variant.candidates(  # type: ignore[attr-defined]
            q.sentence,
            target_year=target_year,
            exclude_citing_paper_id=q.citing_paper_id,
            top_k=top_k,
            query_embedding=embeddings[idx],
        )
        return compute_all_metrics(ranked, q.gold_paper_id, ks=KS)

    def worker() -> None:
        sess = get_session()
        _set_ef_search(sess, ef_search)
        variant = variant_cls(sess, top_n=top_n)
        try:
            while True:
                i = next(counter)
                if i >= n:
                    break
                results[i] = score_one(variant, i)
                progress.update(1)
        finally:
            sess.close()

    if workers <= 1:
        seq_session = get_session()
        _set_ef_search(seq_session, ef_search)
        variant = variant_cls(seq_session, top_n=top_n)
        for i in range(n):
            results[i] = score_one(variant, i)
            progress.update(1)
    else:
        threads = [
            threading.Thread(target=worker, daemon=True) for _ in range(workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    progress.close()

    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} queries failed to score (indices {missing[:5]}...). "
            "Re-run, optionally with --workers 1."
        )
    return [r for r in results if r is not None]


def _aggregate(
    per_query: list[dict[str, float]], buckets: list[str]
) -> dict[str, dict[str, float]]:
    """Mean each metric within every bucket plus an 'all' row."""
    metric_keys = list(per_query[0].keys()) if per_query else []
    agg: dict[str, dict[str, float]] = {}

    def mean_over(indices: list[int]) -> dict[str, float]:
        out = {"n": float(len(indices))}
        for m in metric_keys:
            out[m] = (
                sum(per_query[i][m] for i in indices) / len(indices)
                if indices
                else 0.0
            )
        return out

    for label in _BUCKET_ORDER:
        idxs = [i for i, b in enumerate(buckets) if b == label]
        if idxs:
            agg[label] = mean_over(idxs)
    agg["all"] = mean_over(list(range(len(per_query))))
    return agg


def _print_table(agg: dict[str, dict[str, float]]) -> None:
    cols = ("n", "hit@1", "recall@5", "recall@10", "recall@20", "mrr@20", "ndcg@20")
    header = f"{'density':>8}" + "".join(f"{c:>11}" for c in cols)
    typer.echo("\n" + header)
    typer.echo("-" * len(header))
    for label in (*_BUCKET_ORDER, "all"):
        if label not in agg:
            continue
        row = f"{label:>8}"
        for c in cols:
            v = agg[label].get(c, 0.0)
            row += f"{int(v):>11}" if c == "n" else f"{v:>11.4f}"
        typer.echo(row)


@app.command()
def main(
    variant: str = typer.Option("dense_only", "--variant", help="Variant name."),
    split_name: str = typer.Option("test", "--split", help="test or val."),
    top_k: int = typer.Option(20, "--top-k", help="Retrieval depth for metrics."),
    top_n: int = typer.Option(
        EVAL_DEFAULT_TOP_N, "--top-n", help="Contexts per retrieval branch."
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed", help="Split seed."),
    target_year: int | None = typer.Option(
        None, "--target-year", help="Historical mode: filter candidate pool."
    ),
    workers: int = typer.Option(4, "--workers", help="Parallel DB workers."),
    ef_search: int | None = typer.Option(
        None,
        "--ef-search",
        help="Override hnsw.ef_search per session (engine default is 100).",
    ),
    split_dir: str = typer.Option(str(DEFAULT_SPLIT_DIR), "--split-dir"),
    output: str | None = typer.Option(None, "--output", help="JSON report path."),
    strict: bool = typer.Option(True, "--strict/--relax"),
) -> None:
    """Run a variant and report recall stratified by gold-paper citation density."""
    split_path = Path(split_dir) / f"split_{seed}.json"
    if not split_path.exists():
        typer.echo(f"Split file not found: {split_path}", err=True)
        raise typer.Exit(code=1)

    variant_cls = _load_variant_class(variant)
    session = get_session()
    try:
        split = load_split(split_path, strict=strict, session=session)
        paper_ids = (
            split.test_paper_ids if split_name == "test" else split.val_paper_ids
        )

        queries, unreachable = materialise_queries(
            session, paper_ids, target_year=target_year, require_reachable=True
        )
        if not queries:
            typer.echo("No reachable queries materialised.", err=True)
            raise typer.Exit(code=1)

        counts = _citer_counts(
            session, sorted({q.gold_paper_id for q in queries})
        )
        buckets = [_bucket_for(counts.get(q.gold_paper_id, 0)) for q in queries]

        embeddings = _batch_embed(queries)
        per_query = _evaluate(
            variant_cls,
            queries,
            embeddings,
            top_n=top_n,
            top_k=top_k,
            target_year=target_year,
            workers=workers,
            ef_search=ef_search,
        )

        agg = _aggregate(per_query, buckets)

        typer.echo(
            f"\nVariant: {variant}  Split: {split_name}  ef_search={ef_search or 100}  "
            f"({len(paper_ids)} citing papers, {len(queries)} reachable queries, "
            f"{unreachable} unreachable skipped)"
        )
        _print_table(agg)

        report = {
            "variant_name": variant,
            "split_name": split_name,
            "seed": seed,
            "top_k": top_k,
            "top_n": top_n,
            "ef_search": ef_search or 100,
            "target_year": target_year,
            "num_citing_papers": len(paper_ids),
            "num_queries": len(queries),
            "num_unreachable_skipped": unreachable,
            "timestamp": datetime.now(UTC).isoformat(),
            "buckets": agg,
        }
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        out_path = (
            Path(output)
            if output
            else DEFAULT_SPLIT_DIR
            / "reports"
            / f"stratified_{variant}_{split_name}_{today}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"\nReport saved to {out_path}")
    finally:
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ignoring session close error: %s", exc)


if __name__ == "__main__":
    app()
