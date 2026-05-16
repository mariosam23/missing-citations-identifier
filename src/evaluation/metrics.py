"""IR evaluation metrics for single-relevant retrieval.

All functions expect:
- ``ranked``: an ordered list of candidate ``paper_id``s (best first).
- ``gold``: the single ground-truth ``paper_id``.
- ``k``: the depth cutoff.

With ``|G| = 1`` (single relevant document), Hit@K and Recall@K are
identical; both are reported for clarity in the output tables.
"""

from __future__ import annotations

import math


def hit_at_k(ranked: list[int], gold: int, k: int) -> float:
    """1.0 if ``gold`` appears in ``ranked[:k]``, else 0.0."""
    return 1.0 if gold in ranked[:k] else 0.0


def recall_at_k(ranked: list[int], gold: int, k: int) -> float:
    """Same as :func:`hit_at_k` when ``|G| = 1``."""
    return hit_at_k(ranked, gold, k)


def reciprocal_rank(ranked: list[int], gold: int, k: int) -> float:
    """``1 / position(gold)`` if found in ``ranked[:k]``, else 0.0."""
    for i, candidate in enumerate(ranked[:k], start=1):
        if candidate == gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[int], gold: int, k: int) -> float:
    """NDCG@K with binary relevance and single-relevant.

    IDCG = 1.0 (the gold at position 1 gives ``1/log2(2) = 1``).
    DCG  = ``1/log2(1+pos)`` if the gold appears in ``ranked[:k]``.
    """
    for i, candidate in enumerate(ranked[:k], start=1):
        if candidate == gold:
            return 1.0 / math.log2(1.0 + i)
    return 0.0


# -----------------------------------------------------------------------
# Aggregate helpers
# -----------------------------------------------------------------------

def compute_all_metrics(
    ranked: list[int],
    gold: int,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Compute all metrics at every ``k`` for a single query.

    Returns a flat dict, e.g. ``{"hit@1": 0.0, "hit@5": 1.0, ...,
    "mrr@10": 0.2, "ndcg@10": 0.431}``.
    """
    result: dict[str, float] = {}
    for k in ks:
        result[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        result[f"recall@{k}"] = recall_at_k(ranked, gold, k)
    # MRR and NDCG reported at the highest k only.
    max_k = max(ks)
    result[f"mrr@{max_k}"] = reciprocal_rank(ranked, gold, max_k)
    result[f"ndcg@{max_k}"] = ndcg_at_k(ranked, gold, max_k)
    return result
