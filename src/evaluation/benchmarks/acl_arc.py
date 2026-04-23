"""ACL-ARC adapter: load labeled sentences for citation-function classifier evaluation."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from entities import CitationIntent

from .common import ClassifierEvalExample


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())


# Fine-grained ACL-ARC style labels → coarse ``CitationIntent`` used in this project.
ACL_ARC_LABEL_TO_INTENT: dict[str, CitationIntent] = {
    "background": CitationIntent.BACKGROUND,
    "basis": CitationIntent.BACKGROUND,
    "motivation": CitationIntent.BACKGROUND,
    "compare": CitationIntent.METHOD,
    "contrast": CitationIntent.METHOD,
    "model": CitationIntent.METHOD,
    "investigation": CitationIntent.METHOD,
    "method": CitationIntent.METHOD,
    "result": CitationIntent.RESULT,
    "outcome": CitationIntent.RESULT,
    "other": CitationIntent.BACKGROUND,
}


def map_acl_arc_label(raw: str) -> CitationIntent | None:
    """Map a dataset label string to :class:`CitationIntent`, if possible."""
    key = _norm_label(raw).lstrip("_")
    if key in ACL_ARC_LABEL_TO_INTENT:
        return ACL_ARC_LABEL_TO_INTENT[key]
 
    for suffix in ("_background", "_method", "_result"):
        if key.endswith(suffix):
            stem = key[: -len(suffix)].lstrip("_")
            if stem in ACL_ARC_LABEL_TO_INTENT:
                return ACL_ARC_LABEL_TO_INTENT[stem]
 
    return ACL_ARC_LABEL_TO_INTENT.get(key.lstrip("_"))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_acl_arc_examples(
    rows: Iterator[Mapping[str, Any]],
    *,
    text_keys: tuple[str, ...] = ("text", "sentence", "string", "citation_string"),
    label_keys: tuple[str, ...] = ("label", "gold", "y", "citation_class"),
    id_keys: tuple[str, ...] = ("id", "example_id", "idx"),
) -> Iterator[ClassifierEvalExample]:
    """Yield classifier examples from generic row dicts (JSONL / HuggingFace export)."""
    for i, row in enumerate(rows):
        text = ""
     
        for k in text_keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break
     
        label_raw = ""
        for k in label_keys:
            v = row.get(k)
            if v is not None and v != "":
                label_raw = str(v).strip()
                break
     
        ex_id = ""
        for k in id_keys:
            v = row.get(k)
            if v is not None and str(v).strip():
                ex_id = str(v).strip()
                break
     
        if not ex_id:
            ex_id = f"acl-arc-{i}"
     
        intent = map_acl_arc_label(label_raw) if label_raw else None
        section = row.get("section")
     
        yield ClassifierEvalExample(
            example_id=ex_id,
            text=text,
            gold_label=label_raw or "UNKNOWN",
            citation_intent=intent,
            citation_worthy=row.get("citation_worthy") if isinstance(row.get("citation_worthy"), bool) else None,
            section=str(section) if section is not None else None,
            metadata={k: v for k, v in row.items() if k not in set(text_keys) | set(label_keys) | set(id_keys)},
        )


def load_acl_arc_jsonl(path: str | Path) -> list[ClassifierEvalExample]:
    """Load ACL-ARC-style JSONL (one JSON object per line)."""
    return list(iter_acl_arc_examples(_iter_jsonl(Path(path))))


def load_acl_arc_tsv(
    path: str | Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
    id_column: str | None = "id",
    dialect: str = "excel-tab",
) -> list[ClassifierEvalExample]:
    """Load a TSV/CSV with configurable column names."""
    p = Path(path)
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows: list[dict[str, Any]] = []
        for row in reader:
            d: dict[str, Any] = dict(row)
            if id_column and id_column in d:
                d["example_id"] = d[id_column]
            d["text"] = d.get(text_column, "")
            d["label"] = d.get(label_column, "")
            rows.append(d)
    return list(iter_acl_arc_examples(iter(rows), text_keys=("text",), label_keys=("label",), id_keys=("example_id",)))
