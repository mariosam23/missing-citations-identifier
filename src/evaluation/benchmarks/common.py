"""Shared benchmark types for retrieval (hide-and-seek) and classifier evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from entities import CitationIntent


@dataclass
class BenchmarkExample:
    """One hide-and-seek style retrieval query with ground-truth hidden references.

    ``query_text`` is what the retriever scores against (e.g. stripped sentence + context).
    ``hidden_paper_ids`` are positive labels (references that were held out).
    Stratification fields are optional; they populate bucketed aggregates in ``EvaluationResult``.
    """

    example_id: str
    query_text: str
    hidden_paper_ids: frozenset[str]
    section: str | None = None
    citation_intent: CitationIntent | None = None
    is_multi_facet: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierEvalExample:
    """One sentence-level example for citation intent / worthiness classifiers (ACL-ARC, SciCite)."""

    example_id: str
    text: str
    gold_label: str
    citation_intent: CitationIntent | None = None
    citation_worthy: bool | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
