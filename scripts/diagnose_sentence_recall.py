"""Deep-rank diagnostic for the sentence-level benchmark.

Mirrors ``scripts/diagnose_recall.py`` but for the sentence-level dataset:
loads ``eval/sentence_dataset/test.jsonl``, runs each row through the
hybrid retriever at top-k=200, and reports:

* containment of each target in Qdrant
* recall@K curve (K = 1, 5, 10, 50, 100, 200)
* rank distribution of FOUND targets
* split by single- vs multi-facet
* a side-by-side comparison of two query strategies:
    - ``query_text`` (citation-stripped sentence)
    - ``query_text`` plus the surrounding sentence(s) when the row carries
      ``raw_text`` plus stored context (we synthesise context by reading
      the JSONL and concatenating consecutive same-paper rows).

The point is to answer two questions before running Stage 4 in earnest:
  1. Is the benchmark too hard for THIS retriever (i.e., recall@200 is
     already low)?
  2. Does enriching the query with adjacent sentences move the needle?
"""

from __future__ import annotations

import json
import statistics
import sys
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def stable_uuid(paper_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_id))


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient

    from pipeline import HybridRetriever
    from utils.config import config

    DEEP_K = 200
    K_CURVE = (1, 5, 10, 50, 100, 200)

    jsonl = ROOT / "eval" / "sentence_dataset" / "test.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(rows)} sentence examples", flush=True)

    # Group rows by source paper so we can build a 1-sentence context window.
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["citing_paper_path"]].append(r)
    # Keep the JSONL order within each paper.
    contextual: dict[str, str] = {}
    for paper, paper_rows in by_paper.items():
        for i, r in enumerate(paper_rows):
            prev_txt = paper_rows[i - 1]["query_text"] if i > 0 else ""
            next_txt = paper_rows[i + 1]["query_text"] if i + 1 < len(paper_rows) else ""
            contextual[r["sentence_id"]] = " ".join(
                p for p in (prev_txt, r["query_text"], next_txt) if p
            )

    # Containment check
    print("Connecting to Qdrant ...", flush=True)
    qc = QdrantClient(url=config.QDRANT_URL or "http://localhost:6333")
    coll = config.QDRANT_COLLECTION_NAME or "papers"
    n_points = qc.get_collection(coll).points_count
    print(f"Qdrant {coll!r} has {n_points} points", flush=True)

    all_targets: set[str] = {t for r in rows for t in r["indexed_reference_ids"]}
    target_uuids = {tid: stable_uuid(tid) for tid in all_targets}
    in_qdrant: set[str] = set()
    uuid_list = list(target_uuids.values())
    for i in range(0, len(uuid_list), 200):
        recs = qc.retrieve(
            collection_name=coll,
            ids=uuid_list[i : i + 200],
            with_payload=False,
            with_vectors=False,
        )
        in_qdrant.update(str(r.id) for r in recs)
    contains = {tid: (uid in in_qdrant) for tid, uid in target_uuids.items()}
    print(
        f"Targets in Qdrant: {sum(contains.values())}/{len(contains)} "
        f"({100 * sum(contains.values()) / max(len(contains), 1):.1f}%)",
        flush=True,
    )

    # Init retriever
    print("Loading dense + sparse models ...", flush=True)
    dense = SentenceTransformer(config.DENSE_MODEL)
    sparse = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
    retriever = HybridRetriever(
        qdrant_client=qc,
        dense_model=dense,
        sparse_model=sparse,
        collection=coll,
        prefetch_limit=DEEP_K,
    )

    def deep_ranks(query: str, target_ids: list[str]) -> list[int | None]:
        results = retriever.retrieve(query, top_k=DEEP_K)
        pos = {r.paper_id: idx + 1 for idx, r in enumerate(results)}
        return [pos.get(t) for t in target_ids]

    # Run both query strategies
    pairs_sent: list[int | None] = []
    pairs_ctx: list[int | None] = []
    rows_per_pair: list[dict] = []
    for r in rows:
        targets = r["indexed_reference_ids"]
        ranks_sent = deep_ranks(r["query_text"], targets)
        ranks_ctx = deep_ranks(contextual[r["sentence_id"]], targets)
        for t, rs, rc in zip(targets, ranks_sent, ranks_ctx):
            pairs_sent.append(rs)
            pairs_ctx.append(rc)
            rows_per_pair.append(
                {
                    "sentence_id": r["sentence_id"],
                    "target": t,
                    "rank_sentence": rs,
                    "rank_context": rc,
                    "is_multi_facet": r["is_multi_facet"],
                    "section": r["section"],
                    "in_qdrant": contains.get(t, False),
                }
            )

    n_pairs = len(pairs_sent)
    found_sent = [r for r in pairs_sent if r is not None]
    found_ctx = [r for r in pairs_ctx if r is not None]

    print()
    print("=" * 72)
    print("RANK DIAGNOSIS")
    print("=" * 72)
    print(f"# (sentence, target) pairs: {n_pairs}")
    print(f"# pairs with target in Qdrant: {sum(1 for p in rows_per_pair if p['in_qdrant'])}")
    print()
    print(f"sentence-only:   recall@200 = {len(found_sent) / n_pairs:.4f}  "
          f"({len(found_sent)} / {n_pairs})")
    print(f"sentence+ctx:    recall@200 = {len(found_ctx) / n_pairs:.4f}  "
          f"({len(found_ctx)} / {n_pairs})")
    print()

    if found_sent:
        print(f"sentence-only   median rank (found): {statistics.median(found_sent)}  "
              f"min={min(found_sent)} max={max(found_sent)}")
    if found_ctx:
        print(f"sentence+ctx    median rank (found): {statistics.median(found_ctx)}  "
              f"min={min(found_ctx)} max={max(found_ctx)}")
    print()
    print(f"Recall curve (per-pair, K up to {DEEP_K}):")
    print(f"  {'K':>5}  {'sentence':>10}  {'+context':>10}")
    for k in K_CURVE:
        a = sum(1 for r in pairs_sent if r is not None and r <= k) / n_pairs
        b = sum(1 for r in pairs_ctx if r is not None and r <= k) / n_pairs
        print(f"  {k:>5}  {a:>10.4f}  {b:>10.4f}")

    print()
    multi_pairs = [p for p in rows_per_pair if p["is_multi_facet"]]
    single_pairs = [p for p in rows_per_pair if not p["is_multi_facet"]]
    for label, pool in [("MULTI ", multi_pairs), ("SINGLE", single_pairs)]:
        if not pool:
            continue
        n = len(pool)
        s10 = sum(1 for p in pool if p["rank_sentence"] is not None and p["rank_sentence"] <= 10) / n
        s200 = sum(1 for p in pool if p["rank_sentence"] is not None) / n
        c10 = sum(1 for p in pool if p["rank_context"] is not None and p["rank_context"] <= 10) / n
        c200 = sum(1 for p in pool if p["rank_context"] is not None) / n
        print(f"{label} n={n:>3} recall@10 sentence={s10:.4f} +ctx={c10:.4f}  "
              f"recall@200 sentence={s200:.4f} +ctx={c200:.4f}")

    out_path = ROOT / "eval" / "sentence_dataset" / "diagnose_sentence_recall.json"
    out_path.write_text(
        json.dumps(
            {
                "n_pairs": n_pairs,
                "n_in_qdrant": sum(1 for p in rows_per_pair if p["in_qdrant"]),
                "recall_curve_sentence": {
                    str(k): sum(1 for r in pairs_sent if r is not None and r <= k) / n_pairs
                    for k in K_CURVE
                },
                "recall_curve_context": {
                    str(k): sum(1 for r in pairs_ctx if r is not None and r <= k) / n_pairs
                    for k in K_CURVE
                },
                "per_pair": rows_per_pair,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
