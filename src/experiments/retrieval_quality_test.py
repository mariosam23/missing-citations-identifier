"""End-to-end retrieval quality tests against the live Qdrant index.

Runs four checks, each diagnosing a different failure mode:

1. **Self-retrieval sanity**   — query with a paper's own abstract; the same
   paper should hit rank 1. Catches catastrophic indexing/encoding bugs.

2. **Citation-graph recall**   — for source papers in Postgres that cite >=3
   in-corpus targets, query with (title + abstract) and measure Recall@10/50/100
   against the known cited papers. This is the metric that matters for the
   missing-citation use case.

3. **Cosine score distribution** — top-1 dense cosine across the citation
   queries; tells you whether the model is confident or hedging.

4. **Context-window sensitivity** — vary the query text shape (title / first
   sentence / abstract / title+abstract) for one paper and compare recall +
   target rank. Answers "does adding/removing context help?".

Run from the ``src/`` directory:

    python -m experiments.retrieval_quality_test
"""

import argparse
import random
import statistics
import sys
from typing import Any

from sqlalchemy import text

from database.postgres.engine import get_session
from utils import config, logger


def _build_retriever():
    """Construct HybridRetriever with project-default models."""
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding

    from database.qdrant import create_qdrant_client
    from pipeline import HybridRetriever

    print("[setup] loading dense model...", flush=True)
    dense = SentenceTransformer(config.DENSE_MODEL)
    print("[setup] loading sparse model...", flush=True)
    sparse = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
    print("[setup] connecting to qdrant...", flush=True)
    client = create_qdrant_client(config.QDRANT_URL)

    return HybridRetriever(
        qdrant_client=client,
        dense_model=dense,
        sparse_model=sparse,
        collection=config.QDRANT_COLLECTION_NAME,
        prefetch_limit=200,
    )


def _sample_self_retrieval_papers(n: int) -> list[dict[str, Any]]:
    """Pick n random papers with non-empty title + abstract."""
    with get_session() as s:
        rows = s.execute(text("""
            SELECT "paperId", title, abstract
            FROM papers
            WHERE COALESCE(title,'')   <> ''
              AND COALESCE(abstract,'') <> ''
            ORDER BY random()
            LIMIT :n
        """), {"n": n}).all()
    return [{"paper_id": r[0], "title": r[1], "abstract": r[2]} for r in rows]


def _sample_citation_eval_papers(n: int, min_cites: int = 5, max_cites: int = 20) -> list[dict[str, Any]]:
    """Pick n source papers whose in-corpus cited targets are usable."""
    with get_session() as s:
        rows = s.execute(text("""
            SELECT ps."paperId", ps.title, ps.abstract,
                   array_agg(c.target_paper_id) AS targets
            FROM citations c
            JOIN papers ps ON ps."paperId" = c.source_paper_id
            JOIN papers pt ON pt."paperId" = c.target_paper_id
            WHERE COALESCE(ps.title,'')    <> ''
              AND COALESCE(ps.abstract,'') <> ''
              AND COALESCE(pt.title,'')    <> ''
            GROUP BY ps."paperId", ps.title, ps.abstract
            HAVING COUNT(*) BETWEEN :lo AND :hi
            ORDER BY random()
            LIMIT :n
        """), {"n": n, "lo": min_cites, "hi": max_cites}).all()
    return [
        {"paper_id": r[0], "title": r[1], "abstract": r[2], "cited": list(r[3])}
        for r in rows
    ]


def test_self_retrieval(retriever, n: int = 30) -> None:
    """Query each paper with its own abstract; same paper should rank 1."""
    print("\n" + "=" * 70)
    print(f"TEST 1: Self-retrieval sanity (n={n})")
    print("=" * 70)
    papers = _sample_self_retrieval_papers(n)
    if not papers:
        print("  [skip] no eligible papers")
        return

    queries = [p["abstract"] for p in papers]
    results_per_query = retriever.retrieve_batch(queries, top_k=10)

    rank_1 = 0
    rank_top10 = 0
    miss_examples = []
    for paper, results in zip(papers, results_per_query):
        ranks = [r.paper_id for r in results]
        if ranks and ranks[0] == paper["paper_id"]:
            rank_1 += 1
            rank_top10 += 1
        elif paper["paper_id"] in ranks:
            rank_top10 += 1
        else:
            miss_examples.append((paper["paper_id"], paper["title"][:60]))

    print(f"  Recall@1  : {rank_1}/{n} = {100*rank_1/n:.1f}%")
    print(f"  Recall@10 : {rank_top10}/{n} = {100*rank_top10/n:.1f}%")
    if miss_examples:
        print("  Misses (paper not in top-10 of its own abstract query):")
        for pid, title in miss_examples[:5]:
            print(f"    {pid}  {title!r}")
    if rank_1 < 0.8 * n:
        print("  ⚠ Self-retrieval below 80% — indexing or encoding may be broken.")


def _recall_at(retrieved_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for pid in retrieved_ids[:k] if pid in gold)
    return hits / len(gold)


def test_citation_recall(retriever, n: int = 50) -> dict[str, float]:
    """For each source paper, query (title + abstract); score Recall@k vs cited."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Citation-graph recall (n={n} source papers)")
    print("=" * 70)
    papers = _sample_citation_eval_papers(n)
    if not papers:
        print("  [skip] no eligible source papers")
        return {}

    queries = [f"{p['title']}. {p['abstract']}" for p in papers]
    results_per_query = retriever.retrieve_batch(queries, top_k=100)
    cosine_responses = retriever.probe_dense_cosine_batch(queries, top_k=10)

    recalls_10, recalls_50, recalls_100 = [], [], []
    top1_cosines = []
    per_paper_target_ranks: list[int | None] = []

    for paper, results, cos_results in zip(papers, results_per_query, cosine_responses):
        retrieved_ids = [r.paper_id for r in results]
        # exclude the source paper itself if it appears (would be a trivial hit)
        retrieved_ids = [pid for pid in retrieved_ids if pid != paper["paper_id"]]
        gold = set(paper["cited"]) - {paper["paper_id"]}
        recalls_10.append(_recall_at(retrieved_ids, gold, 10))
        recalls_50.append(_recall_at(retrieved_ids, gold, 50))
        recalls_100.append(_recall_at(retrieved_ids, gold, 100))

        if cos_results:
            top1_cosines.append(cos_results[0].score)

        ranks = [i + 1 for i, pid in enumerate(retrieved_ids) if pid in gold]
        per_paper_target_ranks.append(min(ranks) if ranks else None)

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    hit_at_50 = sum(1 for r in per_paper_target_ranks if r is not None and r <= 50)
    hit_at_100 = sum(1 for r in per_paper_target_ranks if r is not None and r <= 100)

    print(f"  Mean Recall@10  : {_avg(recalls_10)*100:.1f}%")
    print(f"  Mean Recall@50  : {_avg(recalls_50)*100:.1f}%")
    print(f"  Mean Recall@100 : {_avg(recalls_100)*100:.1f}%")
    print(f"  At-least-one-hit @50  : {hit_at_50}/{n} = {100*hit_at_50/n:.1f}%")
    print(f"  At-least-one-hit @100 : {hit_at_100}/{n} = {100*hit_at_100/n:.1f}%")
    if top1_cosines:
        print(f"  Top-1 dense cosine    : mean={statistics.mean(top1_cosines):.3f}  "
              f"median={statistics.median(top1_cosines):.3f}  "
              f"min={min(top1_cosines):.3f}  max={max(top1_cosines):.3f}")

    print("\n  Per-paper best-target rank (first 10):")
    for paper, rank in list(zip(papers, per_paper_target_ranks))[:10]:
        rank_str = str(rank) if rank else "miss"
        print(f"    {paper['paper_id']}  best_rank={rank_str:>5}  "
              f"|cited|={len(paper['cited'])}  {paper['title'][:50]!r}")

    return {
        "recall@10": _avg(recalls_10),
        "recall@50": _avg(recalls_50),
        "recall@100": _avg(recalls_100),
    }


def test_negative_query(retriever) -> None:
    """Off-domain text should produce low cosine, not match academic content."""
    print("\n" + "=" * 70)
    print("TEST 3: Negative / off-domain queries (sanity check)")
    print("=" * 70)
    queries = [
        "the cat sat on the mat and watched the rain",
        "buy two get one free at the grocery store this weekend",
        "asdf qwerty zxcv lorem ipsum dolor sit amet",
    ]
    cos = retriever.probe_dense_cosine_batch(queries, top_k=1)
    for q, results in zip(queries, cos):
        if results:
            print(f"  cos={results[0].score:.3f}  {q!r}  -> {results[0].title[:55]!r}")
    print("  (off-domain queries should have notably lower top-1 cosine than real papers)")


def test_context_window(retriever) -> None:
    """For one source paper, vary query text and compare recall + best rank."""
    print("\n" + "=" * 70)
    print("TEST 4: Context-window sensitivity (1 paper, 4 query shapes)")
    print("=" * 70)
    papers = _sample_citation_eval_papers(1, min_cites=8, max_cites=20)
    if not papers:
        print("  [skip] no eligible source paper")
        return
    p = papers[0]
    abstract = p["abstract"]
    first_sentence = abstract.split(". ")[0] + "."
    title_only = p["title"]
    title_plus_abs = f"{p['title']}. {abstract}"

    variants = [
        ("title only",       title_only),
        ("first sentence",   first_sentence),
        ("abstract only",    abstract),
        ("title + abstract", title_plus_abs),
    ]
    queries = [v[1] for v in variants]
    results_per_query = retriever.retrieve_batch(queries, top_k=100)
    cosine_per_query = retriever.probe_dense_cosine_batch(queries, top_k=1)

    gold = set(p["cited"]) - {p["paper_id"]}
    print(f"  Source: {p['paper_id']}  |cited|={len(p['cited'])}  title={p['title'][:60]!r}")
    print(f"  {'shape':<18} {'len':>5} {'top1_cos':>9} {'R@10':>6} {'R@50':>6} {'R@100':>6} {'best_rank':>10}")
    for (label, q), results, cos in zip(variants, results_per_query, cosine_per_query):
        retrieved = [r.paper_id for r in results if r.paper_id != p["paper_id"]]
        r10 = _recall_at(retrieved, gold, 10) * 100
        r50 = _recall_at(retrieved, gold, 50) * 100
        r100 = _recall_at(retrieved, gold, 100) * 100
        ranks = [i + 1 for i, pid in enumerate(retrieved) if pid in gold]
        best = min(ranks) if ranks else None
        cos_score = cos[0].score if cos else 0.0
        print(f"  {label:<18} {len(q):>5} {cos_score:>9.3f} "
              f"{r10:>5.0f}% {r50:>5.0f}% {r100:>5.0f}% {str(best):>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-self", type=int, default=30, help="Self-retrieval sample size.")
    parser.add_argument("--n-cite", type=int, default=50, help="Citation-recall sample size.")
    parser.add_argument("--skip", choices=["none", "self", "cite", "neg", "ctx"], default="none")
    args = parser.parse_args()

    random.seed(args.seed)
    retriever = _build_retriever()

    if args.skip != "self":
        test_self_retrieval(retriever, n=args.n_self)
    if args.skip != "cite":
        test_citation_recall(retriever, n=args.n_cite)
    if args.skip != "neg":
        test_negative_query(retriever)
    if args.skip != "ctx":
        test_context_window(retriever)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
