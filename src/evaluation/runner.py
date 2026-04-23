"""Retrieval evaluation harness: drives a predict function over benchmark examples."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from entities import CitationIntent

from .benchmarks.common import BenchmarkExample
from .metrics import mean_metrics, per_example_retrieval_metrics


class PredictFn(Protocol):
    def __call__(self, example: BenchmarkExample) -> list[str]: ...


@dataclass
class EvaluationResult:
    """Aggregate and stratified retrieval metrics."""

    n_examples: int
    n_evaluated: int
    overall: dict[str, float]
    per_section: dict[str, dict[str, float]]
    per_intent: dict[str, dict[str, float]]
    per_multi_facet: dict[str, dict[str, float]]
    per_example: list[dict[str, Any]]


def _intent_key(intent: CitationIntent | None) -> str:
    if intent is None:
        return "UNKNOWN"
    return intent.name


def _facet_key(is_multi: bool | None) -> str:
    if is_multi is None:
        return "unknown"
    return "multi" if is_multi else "single"


def _bucket_rows(
    rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        hid = row.get("hidden_count", 0)
        if not isinstance(hid, int) or hid <= 0:
            continue
        m = {
            k: float(v)
            for k, v in row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "hidden_count"
        }
        groups[key_fn(row)].append(m)
    return {k: mean_metrics(v) for k, v in groups.items() if v}


class RetrievalEvaluator:
    """Runs `predict` on each example and aggregates retrieval metrics."""

    def __init__(self, ks: Sequence[int] = (1, 5, 10)):
        self.ks = tuple(ks)

    def evaluate(
        self,
        examples: Iterable[BenchmarkExample],
        predict: PredictFn,
    ) -> EvaluationResult:
        """Evaluate `predict(example) -> ranked paper ids` against hidden references."""
        per_example: list[dict[str, Any]] = []
        metric_rows: list[dict[str, float]] = []

        for ex in examples:
            hidden = ex.hidden_paper_ids
            hidden_n = len(hidden)
            ranked = list(predict(ex))
            base: dict[str, Any] = {
                "example_id": ex.example_id,
                "hidden_count": hidden_n,
                "section": ex.section,
                "citation_intent": _intent_key(ex.citation_intent),
                "is_multi_facet": ex.is_multi_facet,
            }

            if hidden_n == 0:
                per_example.append({**base, "skipped": True})
                continue

            m = per_example_retrieval_metrics(ranked, hidden, ks=self.ks)
            per_example.append({**base, **m, "skipped": False})
            metric_rows.append(m)

        overall = mean_metrics(metric_rows) if metric_rows else {}

        per_section = _bucket_rows(
            per_example,
            lambda r: str(r.get("section") or "UNKNOWN"),
        )
        per_intent = _bucket_rows(
            per_example,
            lambda r: str(r.get("citation_intent") or "UNKNOWN"),
        )
        per_multi_facet = _bucket_rows(
            per_example,
            lambda r: _facet_key(r.get("is_multi_facet")),
        )

        return EvaluationResult(
            n_examples=len(per_example),
            n_evaluated=len(metric_rows),
            overall=overall,
            per_section=per_section,
            per_intent=per_intent,
            per_multi_facet=per_multi_facet,
            per_example=per_example,
        )
