"""Diagnose why Stage 4 recall@10 is ~8%.

Replays the same 100 hide-and-seek examples used by the Stage 4 experiment,
then for each example:

1. Checks whether each held-out target is actually present in the Qdrant
   collection (by stable UUID id derived from the paper_id).
2. Runs a deep retrieval (top-k = 200) and records the rank of every
   held-out target. Targets retrieved beyond rank 200 are recorded as None.
3. Logs the source/target text quality (presence + length of abstract).

Outputs one JSON file with per-example diagnostics and a printed summary
of:
- index containment rate (what fraction of held-out targets are even in
  Qdrant)
- recall@K curve for K in [10, 50, 100, 200]
- median / quartile / "missed" rank distribution
- breakdown by query/target abstract availability
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def stable_uuid(paper_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_id))


def main() -> int:
    from sqlalchemy import select
    from sqlalchemy.orm import aliased

    from database.postgres.engine import get_session
    from database.postgres.tables import Citation, Paper
    from utils.config import config

    SEED = 42
    MAX_EXAMPLES = 100
    HIDE_FRACTION = 0.3
    MIN_REFS = 2
    DEEP_K = 200
    K_CURVE = (1, 5, 10, 20, 50, 100, 200)

    # ---- load examples (mirror stage4_experiments.load_db_hide_seek_examples) ----
    rng = random.Random(SEED)
    target_paper = aliased(Paper)
    refs_by_source: dict[str, list[str]] = defaultdict(list)
    source_rows: dict[str, tuple[str | None, str | None, Any]] = {}

    row_limit = max(MAX_EXAMPLES * max(MIN_REFS, 1) * 20, 1000)
    stmt = (
        select(
            Paper.paperId,
            Paper.title,
            Paper.abstract,
            Citation.target_paper_id,
            Paper.publication_date,
        )
        .join(Citation, Citation.source_paper_id == Paper.paperId)
        .join(target_paper, target_paper.paperId == Citation.target_paper_id)
        .where(Paper.paperId.is_not(None))
        .where(Citation.target_paper_id.is_not(None))
        .order_by(Paper.paperId)
        .limit(row_limit)
    )
    with get_session() as sess:
        rows = sess.execute(stmt).all()

    for source_id, title, abstract, target_id, pub_date in rows:
        sid = str(source_id)
        source_rows[sid] = (title, abstract, pub_date)
        refs_by_source[sid].append(str(target_id))

    examples: list[dict[str, Any]] = []
    for source_id, refs in refs_by_source.items():
        unique = list(dict.fromkeys(refs))
        if len(unique) < MIN_REFS:
            continue
        title, abstract, _ = source_rows[source_id]
        n = len(unique)
        k = max(1, min(n, int(round(HIDE_FRACTION * n))))
        hidden = list(rng.sample(unique, k))
        query_parts = [p.strip() for p in (title or "", abstract or "") if p and p.strip()]
        query = ". ".join(query_parts).strip()[:1800].strip()
        if not query:
            continue
        examples.append(
            {
                "example_id": source_id,
                "query": query,
                "title": title or "",
                "abstract": abstract or "",
                "hidden": hidden,
                "indexed_refs": unique,
            }
        )
        if len(examples) >= MAX_EXAMPLES:
            break

    print(f"Built {len(examples)} examples", flush=True)

    # ---- gather metadata for held-out targets ----
    all_targets: set[str] = set()
    for ex in examples:
        all_targets.update(ex["hidden"])
    target_meta: dict[str, dict[str, Any]] = {}
    with get_session() as sess:
        rows2 = sess.execute(
            select(Paper.paperId, Paper.title, Paper.abstract).where(
                Paper.paperId.in_(list(all_targets))
            )
        ).all()
    for pid, t, ab in rows2:
        target_meta[str(pid)] = {
            "title": t or "",
            "abstract": ab or "",
            "in_papers_table": True,
        }
    for tid in all_targets:
        target_meta.setdefault(tid, {"title": "", "abstract": "", "in_papers_table": False})

    # ---- check Qdrant containment ----
    print("Connecting to Qdrant...", flush=True)
    from qdrant_client import QdrantClient

    qc = QdrantClient(url=config.QDRANT_URL or "http://localhost:6333")
    coll = config.QDRANT_COLLECTION_NAME or "papers"
    n_points = qc.get_collection(coll).points_count
    print(f"Qdrant collection {coll!r} has {n_points} points", flush=True)

    target_uuids = {tid: stable_uuid(tid) for tid in all_targets}
    uuid_list = list(target_uuids.values())
    in_qdrant: set[str] = set()
    for i in range(0, len(uuid_list), 200):
        chunk = uuid_list[i : i + 200]
        recs = qc.retrieve(collection_name=coll, ids=chunk, with_payload=False, with_vectors=False)
        in_qdrant.update(str(r.id) for r in recs)
    contains = {tid: (uid in in_qdrant) for tid, uid in target_uuids.items()}
    n_targets = len(all_targets)
    n_in_qdrant = sum(contains.values())
    print(
        f"Held-out targets in Qdrant: {n_in_qdrant}/{n_targets} "
        f"({100 * n_in_qdrant / max(n_targets, 1):.1f}%)",
        flush=True,
    )

    # Also: how many target papers are present in papers table at all?
    n_in_pg = sum(1 for v in target_meta.values() if v["in_papers_table"])
    print(
        f"Held-out targets in papers table: {n_in_pg}/{n_targets} "
        f"({100 * n_in_pg / max(n_targets, 1):.1f}%)",
        flush=True,
    )

    # Stratify abstract availability for source queries and held-out targets
    src_with_abs = sum(1 for ex in examples if len(ex["abstract"]) > 50)
    print(
        f"Source queries with usable abstract (>50 chars): "
        f"{src_with_abs}/{len(examples)} ({100*src_with_abs/len(examples):.1f}%)",
        flush=True,
    )
    tgt_with_abs = sum(1 for v in target_meta.values() if len(v["abstract"]) > 50)
    print(
        f"Held-out targets with usable abstract (>50 chars): "
        f"{tgt_with_abs}/{n_targets} ({100*tgt_with_abs/n_targets:.1f}%)",
        flush=True,
    )

    # ---- initialize the same retriever the experiment uses ----
    print("Loading dense + sparse models (this may take 30-60s)...", flush=True)
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding

    from pipeline import HybridRetriever

    dense = SentenceTransformer(config.DENSE_MODEL)
    sparse = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
    retriever = HybridRetriever(
        qdrant_client=qc,
        dense_model=dense,
        sparse_model=sparse,
        collection=coll,
        prefetch_limit=max(DEEP_K, 50),
    )

    # ---- run deep retrieval and record ranks ----
    per_example: list[dict[str, Any]] = []
    target_ranks: list[int | None] = []  # one entry per (example, hidden) pair

    t0 = time.perf_counter()
    for i, ex in enumerate(examples, 1):
        results = retriever.retrieve(ex["query"], top_k=DEEP_K)
        rank_of: dict[str, int | None] = {}
        retrieved_ids = [r.paper_id for r in results]
        retrieved_pos = {pid: pos for pos, pid in enumerate(retrieved_ids, start=1)}
        for h in ex["hidden"]:
            rank_of[h] = retrieved_pos.get(h)
            target_ranks.append(rank_of[h])
        per_example.append(
            {
                "example_id": ex["example_id"],
                "n_hidden": len(ex["hidden"]),
                "src_has_abstract": len(ex["abstract"]) > 50,
                "ranks": [
                    {
                        "target": h,
                        "rank": rank_of[h],
                        "in_qdrant": contains.get(h, False),
                        "tgt_has_abstract": len(target_meta[h]["abstract"]) > 50,
                    }
                    for h in ex["hidden"]
                ],
                "top1_score": results[0].score if results else 0.0,
                "top1_id": results[0].paper_id if results else None,
                "top1_title": (results[0].title if results else "") or "",
            }
        )
        if i % 10 == 0:
            print(f"  ... {i}/{len(examples)} queries done ({time.perf_counter()-t0:.1f}s)", flush=True)

    # ---- aggregate ----
    n_pairs = len(target_ranks)
    found = [r for r in target_ranks if r is not None]
    missed = n_pairs - len(found)

    print()
    print("=" * 72)
    print("DIAGNOSIS SUMMARY")
    print("=" * 72)
    print(f"# examples:          {len(examples)}")
    print(f"# (example,target) pairs: {n_pairs}")
    print(f"# pairs hidden target ALSO in Qdrant: "
          f"{sum(1 for ex in per_example for r in ex['ranks'] if r['in_qdrant'])} / {n_pairs}")
    print(
        f"# pairs found anywhere in top-{DEEP_K}: "
        f"{len(found)} ({100 * len(found) / n_pairs:.1f}%)"
    )
    print(f"# pairs MISSED past top-{DEEP_K}:    {missed} ({100*missed/n_pairs:.1f}%)")
    print()

    # rank distribution among found
    if found:
        print("Rank distribution among FOUND held-out targets:")
        print(f"  median:          {statistics.median(found)}")
        print(f"  mean:            {statistics.mean(found):.1f}")
        print(f"  quartiles:       p25={int(statistics.quantiles(found, n=4)[0])}, "
              f"p50={int(statistics.quantiles(found, n=4)[1])}, "
              f"p75={int(statistics.quantiles(found, n=4)[2])}")
        print(f"  min / max:       {min(found)} / {max(found)}")
    print()

    # recall curve
    print(f"Recall curve (per-pair: target found at rank <= K, K up to {DEEP_K}):")
    for k in K_CURVE:
        hits = sum(1 for r in found if r <= k)
        print(f"  K={k:>4}:  recall = {hits/n_pairs:.4f}")

    # stratify by abstract availability
    print()
    print("Recall@10 stratified by source query / target abstract availability:")
    buckets: dict[tuple[bool, bool], list[int | None]] = defaultdict(list)
    for ex in per_example:
        sq = ex["src_has_abstract"]
        for r in ex["ranks"]:
            buckets[(sq, r["tgt_has_abstract"])].append(r["rank"])
    for (sq, tq), rs in sorted(buckets.items()):
        n = len(rs)
        hit10 = sum(1 for x in rs if x is not None and x <= 10)
        hit200 = sum(1 for x in rs if x is not None)
        label = (
            f"src_abs={'Y' if sq else 'N'} tgt_abs={'Y' if tq else 'N'}"
        )
        print(
            f"  {label}: n={n:>4}  recall@10={hit10/n:.4f}  recall@200={hit200/n:.4f}"
        )

    # write per-example file
    out_path = ROOT / "eval" / "stage4" / "diagnose_recall.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "max_examples": len(examples),
                    "deep_k": DEEP_K,
                    "seed": SEED,
                    "qdrant_points": n_points,
                    "n_pairs": n_pairs,
                    "n_hidden_in_qdrant": sum(1 for ex in per_example for r in ex["ranks"] if r["in_qdrant"]),
                    "n_found": len(found),
                    "n_missed": missed,
                    "src_with_abstract": src_with_abs,
                    "tgt_with_abstract": tgt_with_abs,
                },
                "recall_curve": {str(k): sum(1 for r in found if r <= k) / n_pairs for k in K_CURVE},
                "per_example": per_example,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
