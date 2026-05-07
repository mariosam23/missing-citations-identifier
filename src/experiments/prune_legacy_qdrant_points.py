"""Remove legacy Qdrant paper points that predate the ``abstract`` payload field.

Earlier indexing stored payloads without an ``abstract`` key. After a full
re-index with stable ``uuid5`` point ids, those legacy rows remain as *extra*
points (different ids, same ``paper_id``), inflating collection size and
polluting hybrid search.

This script scrolls the collection, finds points whose payload does **not**
contain the key ``abstract``, and deletes them in batches.

Usage (from repo root, ``PYTHONPATH=src``)::

    # Preview only (default): counts and exits
    python -m experiments.prune_legacy_qdrant_points

    # Actually delete
    python -m experiments.prune_legacy_qdrant_points --apply

Runtime (rough): one full scroll over ~150k points is often well under a
minute on localhost; ~65k deletes in batches of 256 is on the order of a few
hundred HTTP calls (typically a few minutes total, disk-bound). Use
``--scroll-batch`` / ``--delete-batch`` to tune.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from database.qdrant import create_qdrant_client
from utils import logger
from utils.config import config


def _is_legacy_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return "abstract" not in payload


def main(
    *,
    apply: bool,
    scroll_batch: int,
    delete_batch: int,
) -> int:
    if not (config.QDRANT_URL or "").strip():
        logger.error("QDRANT_URL is not set (check .env).")
        return 1

    coll = (config.QDRANT_COLLECTION_NAME or "papers").strip() or "papers"
    client = create_qdrant_client(config.QDRANT_URL)

    scanned = 0
    legacy = 0
    deleted = 0
    pending: list[Any] = []

    t0 = time.perf_counter()

    def flush_delete() -> None:
        nonlocal deleted, pending
        if not pending or not apply:
            pending = []
            return
        client.delete(collection_name=coll, points_selector=pending, wait=True)
        deleted += len(pending)
        logger.info("Deleted %d legacy points (running total=%d).", len(pending), deleted)
        pending = []

    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=coll,
            limit=scroll_batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            scanned += 1
            if _is_legacy_payload(p.payload):
                legacy += 1
                pending.append(p.id)
                if len(pending) >= delete_batch:
                    flush_delete()
        if offset is None:
            break

    flush_delete()

    elapsed = time.perf_counter() - t0
    after = client.count(collection_name=coll, exact=True).count

    print(f"Collection {coll!r} @ {config.QDRANT_URL!r}")
    print(f"  Scanned points:        {scanned}")
    print(f"  Legacy (no abstract):  {legacy}")
    print(f"  Deleted this run:    {deleted}")
    print(f"  Count after (exact): {after}")
    print(f"  Elapsed:             {elapsed:.1f}s")
    if not apply and legacy:
        print("\nDry-run only. Re-run with --apply to delete legacy points.", flush=True)

    logger.info(
        "prune_legacy_qdrant_points: scanned=%d legacy=%d deleted=%d apply=%s in %.1fs",
        scanned,
        legacy,
        deleted,
        apply,
        elapsed,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletes (default is dry-run: count only).",
    )
    parser.add_argument(
        "--scroll-batch",
        type=int,
        default=512,
        metavar="N",
        help="Points per scroll page (default: 512).",
    )
    parser.add_argument(
        "--delete-batch",
        type=int,
        default=256,
        metavar="N",
        help="Point ids per delete request (default: 256).",
    )
    args = parser.parse_args()
    if args.scroll_batch < 1 or args.delete_batch < 1:
        print("ERROR: batch sizes must be >= 1.", flush=True)
        sys.exit(1)

    sys.exit(
        main(
            apply=args.apply,
            scroll_batch=args.scroll_batch,
            delete_batch=args.delete_batch,
        )
    )
