"""SciCite adapter: load labeled citing sentences for classifier evaluation."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from entities import CitationIntent

from .common import ClassifierEvalExample


def _norm(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())


# SciCite / extended labels → project ``CitationIntent``.
SCICITE_LABEL_TO_INTENT: dict[str, CitationIntent] = {
    "background": CitationIntent.BACKGROUND,
    "bg": CitationIntent.BACKGROUND,
    "method": CitationIntent.METHOD,
    "compare": CitationIntent.METHOD,
    "comparison": CitationIntent.METHOD,
    "result": CitationIntent.RESULT,
    "results": CitationIntent.RESULT,
    "implicit": CitationIntent.BACKGROUND,
    "none": CitationIntent.BACKGROUND,
}


def map_scicite_label(raw: str) -> CitationIntent | None:
    """Map a SciCite-style label to :class:`CitationIntent`, if possible."""
    key = _norm(raw)
    if key in SCICITE_LABEL_TO_INTENT:
        return SCICITE_LABEL_TO_INTENT[key]
    return None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_scicite_examples(
    rows: Iterator[Mapping[str, Any]],
    *,
    text_keys: tuple[str, ...] = ("text", "string", "sentence", "citing_string"),
    label_keys: tuple[str, ...] = ("label", "cite_type", "gold", "y", "class"),
    id_keys: tuple[str, ...] = ("example_id", "id", "cite_id"),
) -> Iterator[ClassifierEvalExample]:
    """Yield classifier examples from SciCite-style row dicts."""
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
            ex_id = f"scicite-{i}"
        intent = map_scicite_label(label_raw) if label_raw else None
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


def load_scicite_jsonl(path: str | Path) -> list[ClassifierEvalExample]:
    """Load SciCite-style JSONL (one JSON object per line)."""
    return list(iter_scicite_examples(_iter_jsonl(Path(path))))


def load_scicite_tsv(
    path: str | Path,
    *,
    text_column: str = "string",
    label_column: str = "label",
    id_column: str | None = None,
    dialect: str = "excel-tab",
) -> list[ClassifierEvalExample]:
    """Load SciCite TSV (default column names match common releases)."""
    p = Path(path)
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows: list[dict[str, Any]] = []
        for row in reader:
            d = dict(row)
            if id_column and id_column in d:
                d["example_id"] = d[id_column]
            d["text"] = d.get(text_column, "")
            d["label"] = d.get(label_column, "")
            rows.append(d)
    id_keys: tuple[str, ...] = ("example_id",) if id_column else ("example_id", "id", "cite_id")
    return list(iter_scicite_examples(iter(rows), text_keys=("text",), label_keys=("label",), id_keys=id_keys))
