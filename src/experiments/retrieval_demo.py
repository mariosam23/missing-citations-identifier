"""Retrieval Demo Experiment.

Demonstrates hybrid retrieval (dense E5 + sparse SPLADE + RRF fusion) using
queries from the ACL-ARC evaluation dataset. Loads sample sentences and shows
top-k results from the indexed paper collection in Qdrant.

Usage
-----
    python -m src.experiments.retrieval_demo [--top-k 10] [--num-queries 5]

Examples
--------
    # Show top 10 results for 5 sample queries
    python -m src.experiments.retrieval_demo

    # Show top 20 results for 3 queries
    python -m src.experiments.retrieval_demo --top-k 20 --num-queries 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from utils import logger as project_logger
logger = project_logger.getChild(__name__)
project_logger.info("Imported retrieval_demo module")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with timestamps and level prefixes."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing config
    )
    # Also ensure stdout is flushed
    sys.stdout.flush()


def load_acl_arc_queries(
    num_queries: int = 5,
    dataset_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load sample queries from ACL-ARC test dataset.

    Parameters
    ----------
    num_queries:
        Number of queries to load (first N examples).
    dataset_path:
        Path to test.jsonl. If None, uses default location relative to workspace.

    Returns
    -------
    list[dict]
        List of query dicts with keys: example_id, text, label, section, etc.
    """
    if dataset_path is None:
        # Find the dataset relative to this file
        current_dir = Path(__file__).parent.parent.parent  # src/experiments -> missing-citations-identifier
        dataset_path = current_dir / "eval" / "acl_arc_dataset" / "test.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"ACL-ARC dataset not found at {dataset_path}\n"
            "Expected location: eval/acl_arc_dataset/test.jsonl"
        )

    queries = []
    with dataset_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_queries:
                break
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    logger.info(f"Loaded {len(queries)} queries from {dataset_path.name}")
    return queries


def format_result_table(
    results: list,
    query_text: str,
    query_metadata: dict[str, Any],
    top_k: int = 10,
) -> str:
    """Format retrieval results as a human-readable table.

    Parameters
    ----------
    results:
        List of RetrievalResult objects from retriever.retrieve().
    query_text:
        The original query text (for display).
    query_metadata:
        Metadata about the query (label, section, etc).
    top_k:
        Maximum results to display in table.

    Returns
    -------
    str
        Formatted table as a multi-line string.
    """
    lines = []

    # Header with query info
    lines.append("=" * 100)
    lines.append(f"Query: {query_text[:80]}")
    if len(query_text) > 80:
        lines.append(f"        {query_text[80:160]}")

    # Metadata
    label = query_metadata.get("label", "N/A")
    section = query_metadata.get("section", "N/A")
    citation_worthy = query_metadata.get("citation_worthy", "N/A")
    lines.append(
        f"Metadata: label={label!r}, section={section!r}, "
        f"citation_worthy={citation_worthy}"
    )
    lines.append("=" * 100)

    # Results table header
    lines.append(
        f"{'Rank':<5} {'Paper ID':<15} {'Score':>8} {'Year':>6} "
        f"{'Venue':<25} {'Title':<40}"
    )
    lines.append("-" * 100)

    # Results rows (limited to top_k)
    for rank, result in enumerate(results[:top_k], start=1):
        paper_id = result.paper_id[:12]  # Truncate long IDs
        score = result.score
        year = result.year if result.year else "?"
        venue = (result.venue or "")[:23]  # Truncate venue
        title = result.title[:38]  # Truncate title
        if len(result.title) > 38:
            title += ".."

        lines.append(
            f"{rank:<5} {paper_id:<15} {score:>8.4f} {year:>6} "
            f"{venue:<25} {title:<40}"
        )

    # Summary stats
    if results:
        scores = [r.score for r in results]
        lines.append("-" * 100)
        lines.append(
            f"Summary: retrieved {len(results)} papers, "
            f"score range [{min(scores):.4f}, {max(scores):.4f}], "
            f"avg {sum(scores) / len(scores):.4f}"
        )
    else:
        lines.append("[No results]")

    lines.append("")
    return "\n".join(lines)


def initialize_retriever() -> tuple[Any, int]:
    """Initialize and verify the hybrid retriever.

    Returns
    -------
    tuple[Any, int]
        (retriever instance, total papers in collection)

    Raises
    ------
    RuntimeError
        If Qdrant is unavailable or collection is empty.
    """
    # Import here to avoid hanging on module load
    print("[*] Importing Qdrant client...", flush=True)
    try:
        from qdrant_client import QdrantClient
        print("    ✓ Qdrant imported", flush=True)
    except ImportError as e:
        raise RuntimeError(f"Failed to import qdrant_client: {e}") from e

    print("[*] Importing sentence transformers (this may take 10-30s)...", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        print("    ✓ Sentence transformers imported", flush=True)
    except ImportError as e:
        raise RuntimeError(f"Failed to import sentence_transformers: {e}") from e

    print("[*] Importing FastEmbed...", flush=True)
    try:
        from fastembed import SparseTextEmbedding
        print("    ✓ FastEmbed imported", flush=True)
    except ImportError as e:
        raise RuntimeError(f"Failed to import fastembed: {e}") from e

    print("[*] Importing retriever and config...", flush=True)
    try:
        from pipeline.retriever import HybridRetriever
        from utils.config import config
        print("    ✓ Retriever and config imported", flush=True)
    except ImportError as e:
        raise RuntimeError(f"Failed to import local modules: {e}") from e

    print(f"[*] Connecting to Qdrant at {config.QDRANT_URL}...", flush=True)
    try:
        client = QdrantClient(url=config.QDRANT_URL or "http://localhost:6333")
        # Verify connection
        collections = client.get_collections()
        print(f"    ✓ Connected (found {len(collections.collections)} collections)", flush=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Qdrant at {config.QDRANT_URL}\n"
            f"Ensure Qdrant is running (docker-compose up) and accessible.\n"
            f"Error: {e}"
        ) from e

    print(f"[*] Loading dense model: {config.DENSE_MODEL}...", flush=True)
    print("    (Downloading/loading may take 30-60s on first run)", flush=True)
    try:
        dense = SentenceTransformer(config.DENSE_MODEL)
        print("    ✓ Dense model loaded", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load dense model: {e}") from e

    print(f"[*] Loading sparse model: {config.SPARSE_MODEL}...", flush=True)
    try:
        sparse = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
        print("    ✓ Sparse model loaded", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load sparse model: {e}") from e

    print("[*] Creating HybridRetriever instance...", flush=True)
    # Create retriever
    try:
        retriever = HybridRetriever(
            qdrant_client=client,
            dense_model=dense,
            sparse_model=sparse,
            collection=config.QDRANT_COLLECTION_NAME,
            prefetch_limit=50,
        )
        print("    ✓ Retriever initialized", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize retriever: {e}") from e

    # Check collection exists and has data
    print(f"[*] Checking collection '{config.QDRANT_COLLECTION_NAME}'...", flush=True)
    try:
        collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
        num_papers = collection_info.points_count or 0
        print(f"    ✓ Collection has {num_papers} papers", flush=True)
        if num_papers == 0:
            raise RuntimeError(
                f"Collection '{config.QDRANT_COLLECTION_NAME}' is empty. "
                "Please index papers first (see src/indexer.py)."
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to access collection '{config.QDRANT_COLLECTION_NAME}': {e}"
        ) from e

    return retriever, int(num_papers)


def main(top_k: int = 10, num_queries: int = 5) -> int:
    """Run the retrieval demo.

    Parameters
    ----------
    top_k:
        Number of top results to display per query.
    num_queries:
        Number of queries to run.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    print("\n" + "="*60, flush=True)
    print("  Retrieval Demo Experiment", flush=True)
    print("="*60 + "\n", flush=True)
    
    setup_logging(level=logging.WARNING)  # Keep logging quiet, use print for main output

    # Initialize
    try:
        print("[Step 1/3] Initializing retriever...\n", flush=True)
        retriever, num_papers = initialize_retriever()
        print(f"\n✓ Retriever ready ({num_papers} papers indexed)\n", flush=True)
    except RuntimeError as e:
        print(f"\n✗ ERROR: {e}\n", flush=True)
        return 1

    # Load queries
    print("[Step 2/3] Loading queries...", flush=True)
    try:
        queries = load_acl_arc_queries(num_queries=num_queries)
        print(f"✓ Loaded {len(queries)} queries\n", flush=True)
    except Exception as e:
        print(f"✗ ERROR: Failed to load queries: {e}\n", flush=True)
        return 1

    # Execute retrievals
    print(f"[Step 3/3] Executing retrievals (top_k={top_k})...\n", flush=True)

    total_time = 0.0
    total_results = 0

    for i, query_dict in enumerate(queries, start=1):
        query_text = query_dict.get("text", "")
        if not query_text:
            print(f"  Query {i}: SKIPPED (no text)", flush=True)
            continue

        query_metadata = {
            "label": query_dict.get("label", "N/A"),
            "section": query_dict.get("section", "N/A"),
            "citation_worthy": query_dict.get("citation_worthy", "N/A"),
        }

        # Execute retrieval
        print(f"  Query {i}: Retrieving...", end="", flush=True)
        start = time.perf_counter()
        try:
            results = retriever.retrieve(query_text, top_k=top_k)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue
        elapsed = time.perf_counter() - start
        print(f" {len(results)} results in {elapsed:.2f}s", flush=True)

        # Display results
        table = format_result_table(results, query_text, query_metadata, top_k=top_k)
        print(table)

        total_time += elapsed
        total_results += len(results)

    # Summary
    print("="*60, flush=True)
    print(f"Completed: {total_results} total results from {len(queries)} queries", flush=True)
    print(f"Total time: {total_time:.2f}s (avg {total_time / len(queries) if queries else 0:.2f}s/query)", flush=True)
    print("="*60 + "\n", flush=True)
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
        help="Number of top results to display per query (default: 10)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of queries to run (default: 5)",
    )
    args = parser.parse_args()
    project_logger.info(f"Starting retrieval_demo with top_k={args.top_k} num_queries={args.num_queries}")

    exit_code = main(top_k=args.top_k, num_queries=args.num_queries)
    sys.exit(exit_code)
