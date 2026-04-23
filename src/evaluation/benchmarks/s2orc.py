"""S2ORC-style hide-and-seek rows → :class:`BenchmarkExample` for retrieval evaluation."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import BenchmarkExample


def reference_coverage(indexed_count: int, total_reference_count: int) -> float:
    """Fraction of a paper's references that appear in the index (OpenAlex / Qdrant)."""
    if total_reference_count <= 0:
        return 1.0 if indexed_count > 0 else 0.0
    return indexed_count / total_reference_count


def random_hidden_subset(
    indexed_reference_ids: Sequence[str],
    *,
    hide_fraction: float = 0.3,
    rng: random.Random | None = None,
) -> frozenset[str]:
    """Choose approximately ``hide_fraction`` of *indexed* references to hold out.

    Uses unique ids; order of ``indexed_reference_ids`` is ignored.
    """
    if hide_fraction < 0:
        raise ValueError("hide_fraction must be non-negative")
  
    r = rng or random.Random()
    uniq = list(dict.fromkeys(str(x) for x in indexed_reference_ids))
    n = len(uniq)
  
    if n == 0:
        return frozenset()
  
    if hide_fraction == 0:
        return frozenset()
  
    k = max(1, min(n, int(round(hide_fraction * n))))
    return frozenset(r.sample(uniq, k))


def row_to_benchmark_example(
    row: Mapping[str, Any],
    *,
    hidden: frozenset[str],
    query_key: str = "query_text",
    id_key: str = "sentence_id",
    citing_key: str = "citing_paper_id",
    section_key: str = "section",
) -> BenchmarkExample:
    """Build a single :class:`BenchmarkExample` from a row dict and a precomputed hidden set."""
    q = row.get(query_key) or row.get("sentence") or row.get("text") or ""
    sid = str(row.get(id_key) or row.get("example_id") or row.get("id") or "")
    cite = row.get(citing_key)
    ex_id = sid if sid else (f"{citing_key}:{cite}" if cite is not None else "unknown")
    section = row.get(section_key)
    base_keys = {query_key, "sentence", "text", id_key, "example_id", "id", section_key, citing_key}
    meta = {k: v for k, v in row.items() if k not in base_keys}

    if cite is not None:
        meta = {**meta, citing_key: cite}

    return BenchmarkExample(
        example_id=str(ex_id),
        query_text=str(q).strip(),
        hidden_paper_ids=hidden,
        section=str(section) if section is not None else None,
        citation_intent=None,
        is_multi_facet=row.get("is_multi_facet") if isinstance(row.get("is_multi_facet"), bool) else None,
        metadata=meta,
    )


def iter_hide_seek_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    rng: random.Random | None = None,
    hide_fraction: float = 0.3,
    min_coverage: float = 0.7,
    indexed_key: str = "indexed_reference_ids",
    total_key: str = "total_reference_count",
) -> Iterator[BenchmarkExample]:
    """Yield retrieval examples from pre-aggregated rows (JSONL / DB export).

    Each row should provide:
    - ``indexed_key``: list of reference paper ids present in the retrieval index
    - ``total_key`` (optional): total references on the citing paper; if omitted,
      coverage is treated as 1.0 when there is at least one indexed id.

    Rows failing the coverage gate are skipped. Hidden ids are sampled per row.
    """
    r = rng or random.Random()
    for row in rows:
        raw_idx = row.get(indexed_key) or row.get("indexed_references") or []
        if not isinstance(raw_idx, list):
            continue
        indexed = [str(x) for x in raw_idx]
        total_raw = row.get(total_key)
        total = int(total_raw) if total_raw is not None else len(indexed)
        cov = reference_coverage(len(indexed), total)
        if cov < min_coverage:
            continue
        hidden = random_hidden_subset(indexed, hide_fraction=hide_fraction, rng=r)
        yield row_to_benchmark_example(row, hidden=hidden)


def load_hide_seek_jsonl(path: str | Path, *, seed: int = 42) -> list[BenchmarkExample]:
    """Load JSONL (one object per line) and build examples with random hide splits."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return list(iter_hide_seek_examples(rows, rng=random.Random(seed)))
