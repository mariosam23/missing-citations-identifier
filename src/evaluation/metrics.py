"""Retrieval metrics and paired bootstrap statistics for evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


def recall_at_k(ranked_ids: Sequence[T], relevant_ids: Iterable[T], k: int) -> float:
    """|relevant ∩ top-k| / |relevant|. Returns 0.0 if there are no relevant items."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
 
    top = set(ranked_ids[:k])
 
    return len(rel & top) / len(rel)


def precision_at_k(ranked_ids: Sequence[T], relevant_ids: Iterable[T], k: int) -> float:
    """|relevant ∩ top-k| / k."""
    if k <= 0:
        return 0.0
    rel = set(relevant_ids)
    top = set(ranked_ids[:k])
    return len(rel & top) / k


def mrr(ranked_ids: Sequence[T], relevant_ids: Iterable[T]) -> float:
    """Mean reciprocal rank of the first hit (1-indexed). 0.0 if no hit."""
    rel = set(relevant_ids)
    for i, pid in enumerate(ranked_ids, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[T], relevant_ids: Iterable[T], k: int) -> float:
    """Binary nDCG@k: relevance 1 iff id is in the relevant set.

    Only the first occurrence of each id in the ranked prefix contributes gain, so
    duplicate ids cannot inflate DCG above the ideal for unique relevant documents.
    """
    if k <= 0:
        return 0.0
    
    rel = set(relevant_ids)
    dcg = 0.0
    seen_rel: set[T] = set()
    
    for i, pid in enumerate(ranked_ids[:k], start=1):
        if pid in rel and pid not in seen_rel:
            dcg += 1.0 / math.log2(i + 1)
            seen_rel.add(pid)

    num_rel = min(len(rel), k)
    if num_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def per_example_retrieval_metrics(
    ranked_ids: Sequence[T],
    relevant_ids: Iterable[T],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute recall@k, precision@k for each k, MRR, and nDCG@10 for one ranked list."""
    out: dict[str, float] = {}
    rel = set(relevant_ids)
    for kk in ks:
        out[f"recall@{kk}"] = recall_at_k(ranked_ids, rel, kk)
        out[f"precision@{kk}"] = precision_at_k(ranked_ids, rel, kk)
    out["mrr"] = mrr(ranked_ids, rel)
    out["ndcg@10"] = ndcg_at_k(ranked_ids, rel, 10)
    return out


def mean_metrics(rows: Sequence[Mapping[str, float]], skip_keys: frozenset[str] | None = None) -> dict[str, float]:
    """Average numeric columns across per-example metric dicts."""
    if not rows:
        return {}
    skip = skip_keys or frozenset()
    keys = [k for k in rows[0] if k not in skip]
    out: dict[str, float] = {}
    n = len(rows)
    for key in keys:
        total = 0.0
        for row in rows:
            total += float(row[key])
        out[key] = total / n
    return out


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value_two_sided: float
    n_boot: int


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> PairedBootstrapResult:
    """Paired bootstrap on per-example scores.

    Confidence interval: percentile interval for the mean of (a - b) over resampled indices.

    Two-sided p-value (approximate): null-shift bootstrap. Center differences so the mean is 0,
    resample, and estimate how often a bootstrap mean is at least as extreme as the observed mean.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    n = len(a)
    if n == 0:
        return PairedBootstrapResult(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, n_boot)

    rng = random.Random(seed)
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    mean_diff = mean_a - mean_b

    boot_means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _j in range(n):
            idx = rng.randrange(n)
            s += diffs[idx]
        boot_means.append(s / n)

    boot_means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    ci_low = boot_means[lo_idx]
    ci_high = boot_means[hi_idx]

    centered = [d - mean_diff for d in diffs]
    extreme = 0
    for _ in range(n_boot):
        s = 0.0
        for _j in range(n):
            idx = rng.randrange(n)
            s += centered[idx]
        bm = s / n
        if abs(bm) >= abs(mean_diff):
            extreme += 1
    p_value = min(1.0, (extreme + 1) / (n_boot + 1))

    return PairedBootstrapResult(
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value_two_sided=p_value,
        n_boot=n_boot,
    )
