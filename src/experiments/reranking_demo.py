"""Stage 4B reranking demo experiment.

Runs the Stage 4A hybrid retriever to get a candidate pool, then applies the
Stage 4B cross-encoder reranker and prints before/after rankings.

Usage
-----
    python -m src.experiments.reranking_demo --num-queries 3 --candidate-k 30 --top-k 10
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with timestamps and level prefixes."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    sys.stdout.flush()


def load_retrieval_demo_helpers():
    """Load Stage 4A demo helpers only when the experiment actually runs."""
    try:
        from experiments.retrieval_demo import (
            format_result_table,
            initialize_retriever,
            load_acl_arc_queries,
        )
    except ImportError as e:
        raise RuntimeError(f"Failed to import retrieval demo helpers: {e}") from e

    return format_result_table, initialize_retriever, load_acl_arc_queries


def initialize_reranker(model_name: str):
    """Initialize the cross-encoder reranker."""
    print(f"[*] Loading reranker model: {model_name}...", flush=True)
    print("    (Downloading/loading may take a while on first run)", flush=True)
    try:
        from pipeline.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=model_name)
        # Force model load here so initialization time is reported separately.
        reranker._get_model()
        print("    OK reranker loaded", flush=True)
        return reranker
    except Exception as e:
        raise RuntimeError(f"Failed to initialize reranker: {e}") from e


def format_rank_changes(hybrid_results: list, reranked_results: list, top_k: int) -> str:
    """Summarize how reranking moved papers within the candidate pool."""
    original_ranks = {
        result.paper_id: rank
        for rank, result in enumerate(hybrid_results, start=1)
    }

    lines = []
    lines.append("Rank movement after cross-encoder reranking")
    lines.append("-" * 100)
    lines.append(
        f"{'New':<5} {'Old':<5} {'Delta':<7} {'Paper ID':<15} "
        f"{'Rerank Score':>12} {'Title':<45}"
    )
    lines.append("-" * 100)

    for new_rank, result in enumerate(reranked_results[:top_k], start=1):
        old_rank = original_ranks.get(result.paper_id)
        if old_rank is None:
            old_display = "?"
            delta_display = "?"
        else:
            old_display = str(old_rank)
            delta = old_rank - new_rank
            delta_display = f"{delta:+d}"

        title = result.title[:43]
        if len(result.title) > 43:
            title += ".."

        lines.append(
            f"{new_rank:<5} {old_display:<5} {delta_display:<7} "
            f"{result.paper_id[:12]:<15} {result.score:>12.4f} {title:<45}"
        )

    lines.append("")
    return "\n".join(lines)


def main(
    top_k: int = 10,
    candidate_k: int = 30,
    num_queries: int = 3,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> int:
    """Run the Stage 4B reranking demo."""
    if candidate_k < top_k:
        print("ERROR: candidate-k must be greater than or equal to top-k", flush=True)
        return 1

    print("\n" + "=" * 60, flush=True)
    print("  Stage 4B Cross-Encoder Reranking Demo", flush=True)
    print("=" * 60 + "\n", flush=True)

    setup_logging(level=logging.WARNING)

    try:
        format_result_table, initialize_retriever, load_acl_arc_queries = (
            load_retrieval_demo_helpers()
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}\n", flush=True)
        return 1

    try:
        print("[Step 1/4] Initializing retriever...\n", flush=True)
        retriever, num_papers = initialize_retriever()
        print(f"\nOK retriever ready ({num_papers} papers indexed)\n", flush=True)
    except RuntimeError as e:
        print(f"\nERROR: {e}\n", flush=True)
        return 1

    try:
        print("[Step 2/4] Initializing reranker...\n", flush=True)
        reranker = initialize_reranker(model_name)
        print("", flush=True)
    except RuntimeError as e:
        print(f"\nERROR: {e}\n", flush=True)
        return 1

    print("[Step 3/4] Loading queries...", flush=True)
    try:
        queries = load_acl_arc_queries(num_queries=num_queries)
        print(f"OK loaded {len(queries)} queries\n", flush=True)
    except Exception as e:
        print(f"ERROR: failed to load queries: {e}\n", flush=True)
        return 1

    print(
        f"[Step 4/4] Retrieving {candidate_k} candidates and reranking top {top_k}...\n",
        flush=True,
    )

    total_retrieval_time = 0.0
    total_rerank_time = 0.0
    completed_queries = 0

    for i, query_dict in enumerate(queries, start=1):
        query_text = query_dict.get("text", "")
        if not query_text:
            print(f"  Query {i}: SKIPPED (no text)", flush=True)
            continue

        query_metadata: dict[str, Any] = {
            "label": query_dict.get("label", "N/A"),
            "section": query_dict.get("section", "N/A"),
            "citation_worthy": query_dict.get("citation_worthy", "N/A"),
        }

        print(f"  Query {i}: Hybrid retrieval...", end="", flush=True)
        retrieval_start = time.perf_counter()
        try:
            hybrid_results = retriever.retrieve(query_text, top_k=candidate_k)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue
        retrieval_elapsed = time.perf_counter() - retrieval_start
        print(f" {len(hybrid_results)} candidates in {retrieval_elapsed:.2f}s", flush=True)

        print(f"  Query {i}: Cross-encoder reranking...", end="", flush=True)
        rerank_start = time.perf_counter()
        try:
            reranked_results = reranker.rerank(query_text, hybrid_results, top_k=top_k)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue
        rerank_elapsed = time.perf_counter() - rerank_start
        print(f" {len(reranked_results)} results in {rerank_elapsed:.2f}s", flush=True)

        print("\nHybrid top-k before reranking")
        print(format_result_table(hybrid_results[:top_k], query_text, query_metadata, top_k=top_k))
        print("Cross-encoder top-k after reranking")
        print(format_result_table(reranked_results, query_text, query_metadata, top_k=top_k))
        print(format_rank_changes(hybrid_results, reranked_results, top_k=top_k))

        total_retrieval_time += retrieval_elapsed
        total_rerank_time += rerank_elapsed
        completed_queries += 1

    print("=" * 60, flush=True)
    print(f"Completed queries: {completed_queries}/{len(queries)}", flush=True)
    if completed_queries:
        print(
            f"Avg hybrid retrieval time: {total_retrieval_time / completed_queries:.2f}s/query",
            flush=True,
        )
        print(
            f"Avg reranking time: {total_rerank_time / completed_queries:.2f}s/query",
            flush=True,
        )
    print("=" * 60 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of final reranked results to display (default: 10)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
        help="Number of hybrid candidates to rerank (default: 30)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=3,
        help="Number of ACL-ARC sample queries to run (default: 3)",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder model name (default: BAAI/bge-reranker-v2-m3)",
    )
    args = parser.parse_args()

    exit_code = main(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        num_queries=args.num_queries,
        model_name=args.model_name,
    )
    sys.exit(exit_code)
