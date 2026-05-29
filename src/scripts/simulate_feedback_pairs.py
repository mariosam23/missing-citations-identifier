"""Simulate user feedback by generating (query, positive, hard_negative) triplets.

Queries are citation contexts.
Positive is another citation context for the same cited_paper_id.
Hard negative is a citation context retrieved by the baseline model that belongs to a different cited_paper_id.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from sqlalchemy import text
from tqdm import tqdm

from database.postgres.engine import get_session
from pipeline.embedding.embedder import get_embedder
from pipeline.retrieval.dense import retrieve_dense
from utils.logger import logger

app = typer.Typer(add_completion=False)


def _load_excluded_citing_papers(split_path: str) -> set[int]:
    if not Path(split_path).exists():
        logger.warning(f"Split file {split_path} not found. Proceeding with empty exclusion set.")
        return set()
    with open(split_path, encoding="utf-8") as f:
        data = json.load(f)
    val_ids = data.get("val_paper_ids", [])
    test_ids = data.get("test_paper_ids", [])
    return set(val_ids + test_ids)


@app.command()
def main(
    limit: int = typer.Option(5000, help="Maximum number of triplets to generate"),
    output_path: str = typer.Option("data/finetune/simulated_pool.jsonl", help="Output JSONL path"),
    split_path: str = typer.Option("data/eval/split_42.json", help="Path to evaluation split JSON"),
    max_pairs_per_paper: int = typer.Option(50, help="Limit examples per cited_paper_id"),
) -> None:
    logger.info("simulate_feedback_pairs started")

    excluded_ids = _load_excluded_citing_papers(split_path)
    logger.info(f"Loaded {len(excluded_ids)} excluded citing_paper_ids from {split_path}")

    session = get_session()
    embedder = get_embedder()

    # Extract candidate pairs (a, b) citing the same paper.
    sql = text("""
        SELECT a.context_id, a.sentence_without_markers, a.citing_paper_id,
               b.context_id, b.sentence_without_markers, b.citing_paper_id,
               a.cited_paper_id
        FROM citation_contexts a
        JOIN citation_contexts b
          ON a.cited_paper_id = b.cited_paper_id
         AND a.context_id < b.context_id
        WHERE a.cited_paper_id IS NOT NULL
    """)

    logger.info("Executing pair extraction query... (this might take a few seconds)")
    rows = session.execute(sql).all()
    logger.info(f"Fetched {len(rows)} potential positive pairs")

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    triplets_generated = 0
    paper_counts: dict[int, int] = {}

    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="Mining hard negatives"):
            if triplets_generated >= limit:
                break

            a_ctx_id, a_sent, a_citing, b_ctx_id, b_sent, b_citing, cited_id = row

            # Leakage guard
            if a_citing in excluded_ids or b_citing in excluded_ids:
                continue

            # Diversity cap
            if paper_counts.get(cited_id, 0) >= max_pairs_per_paper:
                continue

            if not a_sent or not b_sent:
                continue

            # Simulate baseline retrieval to find a hard negative
            # We encode `a` and find a highly ranked context that cites a DIFFERENT paper.
            try:
                q_emb = embedder.encode_query(a_sent)
                retrieved = retrieve_dense(session, q_emb, top_n=100)
                
                hard_negative_sent = None
                for res in retrieved:
                    # Skip the query context itself, any duplicate text,
                    # and contexts citing the same paper (not a negative).
                    if (
                        res.context_id != a_ctx_id
                        and res.cited_paper_id != cited_id
                        and res.sentence
                        and res.sentence != a_sent
                    ):
                        hard_negative_sent = res.sentence
                        break
                
                if hard_negative_sent:
                    record = {
                        "query": a_sent,
                        "pos": b_sent,
                        "neg": hard_negative_sent,
                        "cited_paper_id": cited_id
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    
                    paper_counts[cited_id] = paper_counts.get(cited_id, 0) + 1
                    triplets_generated += 1

            except Exception as e:
                logger.error(f"Error processing context_id {a_ctx_id}: {e}")
                continue

    logger.info(f"simulate_feedback_pairs finished! Wrote {triplets_generated} triplets to {output_path}")

if __name__ == "__main__":
    app()
