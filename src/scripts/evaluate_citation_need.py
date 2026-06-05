"""Evaluate the binary LLM citation-need identifier.

Reuses the weak-supervision dataset already mined from the GROBID TEI corpus
(``eval/missing_citations/tei_citeworth.jsonl``) and collapses its labels to the
binary "should this sentence cite?" task:

* ``MISSING_CITATION``     -> gold positive (a real cited sentence, marker stripped)
* ``NOT_CITATION_WORTHY``  -> gold negative (substantive but uncited prose)
* ``HAS_CITATION``         -> excluded from precision/recall; used only to check
  the local prefilter never re-flags an already-cited sentence.

The negatives here are genuine *uncited* sentences (mined from the absence of a
TEI citation marker), NOT ``citation_contexts`` rows with ``cited_paper_id IS
NULL`` — those are *unresolved citations*, i.e. positives, and would silently
corrupt the metrics. The weak negatives still carry label noise (some uncited
sentences are in fact citation-worthy), which caps achievable precision.

The few-shot examples live in the prompt and are hand-written, so no scored row
ever appears in-context: there is no train/test leakage to control for.

Real Gemini API calls are made, batched (``--batch-size`` targets per call) to
keep the request count low under restrictive free-tier quotas. Pass ``--limit``
while iterating. Run from the repo root:
``python -m scripts.evaluate_citation_need run --limit 200 --batch-size 10``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from pipeline.citation_need import (
    CitationNeedIdentifier,
    CitationNeedJudgement,
    CitationNeedQuery,
)
from pipeline.missing_citations.detector import CitationNeedLabel
from utils.logger import logger

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)


@app.callback()
def _main() -> None:
    """Binary LLM citation-need identifier evaluation (CiteWorth-style)."""


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    _PROJECT_ROOT / "eval" / "missing_citations" / "tei_citeworth.jsonl"
)
DEFAULT_OUTPUT = (
    _PROJECT_ROOT / "data" / "eval" / "reports" / "citation_need_binary.json"
)
_THRESHOLDS: tuple[float, ...] = (0.0, 0.5, 0.7, 0.9)


@dataclass(frozen=True, slots=True)
class _Row:
    sentence_id: str
    text: str
    gold_label: CitationNeedLabel
    previous_sentence: str | None
    next_sentence: str | None


@dataclass(frozen=True, slots=True)
class _Prediction:
    sentence_id: str
    gold_positive: bool
    needs_citation: bool
    confidence: float
    sent: bool
    answered: bool


@app.command()
def run(
    dataset: str = typer.Option(str(DEFAULT_DATASET), "--dataset"),
    output: str = typer.Option(str(DEFAULT_OUTPUT), "--output"),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Cap the number of rows scored (API cost)."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Override the Gemini model name."
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Headline operating threshold on self-reported confidence.",
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        min=1,
        help="Targets packed into one LLM call (fewer calls = less quota use).",
    ),
    filter_inputs: bool = typer.Option(
        True,
        "--filter/--no-filter",
        help=(
            "Apply the pre-LLM input-sanitization filter (clean residue + skip "
            "non-prose). Use --no-filter for the unfiltered baseline."
        ),
    ),
    sleep: float = typer.Option(
        0.0,
        "--sleep",
        min=0.0,
        help="Seconds to wait between batches (throttle free-tier rate limits).",
    ),
) -> None:
    """Score the identifier on the binary citation-need task."""
    rows = _load_rows(Path(dataset), limit=limit)
    identifier = CitationNeedIdentifier(
        model_name=model, enable_sanitizer=filter_inputs
    )

    queries = [
        CitationNeedQuery(
            target=row.text,
            previous=row.previous_sentence,
            next=row.next_sentence,
        )
        for row in rows
    ]
    judgements = _classify_in_batches(
        identifier, queries, batch_size=batch_size, sleep=sleep
    )

    predictions: list[_Prediction] = []
    cited_flagged = 0
    cited_total = 0
    for row, judgement in zip(rows, judgements, strict=True):
        if row.gold_label == CitationNeedLabel.HAS_CITATION:
            cited_total += 1
            if judgement.needs_citation:
                cited_flagged += 1
            continue
        predictions.append(
            _Prediction(
                sentence_id=row.sentence_id,
                gold_positive=row.gold_label
                == CitationNeedLabel.MISSING_CITATION,
                needs_citation=judgement.needs_citation,
                confidence=judgement.confidence,
                sent=judgement.sent,
                answered=judgement.answered,
            )
        )

    sent = sum(1 for j in judgements if j.sent)
    answered = sum(1 for j in judgements if j.answered)
    if sent and not answered:
        raise typer.BadParameter(
            f"all {sent} sent targets returned no decision (the LLM never "
            "answered) — the run is invalid, not a 0.0 result. Check API "
            "quota/billing and the model name; see the logged errors above. "
            "No report written."
        )

    report = _build_report(
        predictions,
        headline_threshold=threshold,
        cited_flagged=cited_flagged,
        cited_total=cited_total,
        model_name=identifier.model_name,
        dataset=str(dataset),
        filter_inputs=filter_inputs,
    )
    out_path = _write_report(report, Path(output))
    _print_report(report)
    typer.echo(f"\nReport saved to {out_path}")


def _classify_in_batches(
    identifier: CitationNeedIdentifier,
    queries: list[CitationNeedQuery],
    *,
    batch_size: int,
    sleep: float,
) -> list[CitationNeedJudgement]:
    """Classify in chunks so a throttle can be applied between LLM calls."""
    if not sleep:
        return identifier.classify(queries, batch_size=batch_size)

    judgements: list[CitationNeedJudgement] = []
    for start in range(0, len(queries), batch_size):
        if start:
            time.sleep(sleep)
        chunk = queries[start : start + batch_size]
        judgements.extend(identifier.classify(chunk, batch_size=batch_size))
    return judgements


def _load_rows(path: Path, *, limit: int | None) -> list[_Row]:
    rows: list[_Row] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                gold_label = CitationNeedLabel(raw["gold_label"])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} has invalid gold_label"
                ) from exc
            rows.append(
                _Row(
                    sentence_id=str(raw["sentence_id"]),
                    text=str(raw["text"]),
                    gold_label=gold_label,
                    previous_sentence=_optional_str(raw.get("previous_sentence")),
                    next_sentence=_optional_str(raw.get("next_sentence")),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_report(
    predictions: list[_Prediction],
    *,
    headline_threshold: float,
    cited_flagged: int,
    cited_total: int,
    model_name: str,
    dataset: str,
    filter_inputs: bool,
) -> dict[str, Any]:
    by_threshold = {
        f"{t:.2f}": _metrics_at(predictions, t) for t in _THRESHOLDS
    }
    return {
        "settings": {
            "model": model_name,
            "temperature": 0.0,
            "dataset": dataset,
            "headline_threshold": headline_threshold,
            "input_filter": filter_inputs,
            "num_scored": len(predictions),
            "num_unmatched": sum(
                1 for p in predictions if p.sent and not p.answered
            ),
            "num_prefiltered": sum(1 for p in predictions if not p.sent),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "headline": _metrics_at(predictions, headline_threshold),
        "by_threshold": by_threshold,
        "already_cited": {
            "total": cited_total,
            "wrongly_flagged": cited_flagged,
            "false_flag_rate": _safe_div(cited_flagged, cited_total),
        },
        "predictions": [
            {
                "sentence_id": p.sentence_id,
                "gold_positive": p.gold_positive,
                "needs_citation": p.needs_citation,
                "confidence": p.confidence,
                "sent": p.sent,
                "answered": p.answered,
            }
            for p in predictions
        ],
    }


def _metrics_at(predictions: list[_Prediction], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for p in predictions:
        predicted_positive = p.needs_citation and p.confidence >= threshold
        if p.gold_positive and predicted_positive:
            tp += 1
        elif p.gold_positive and not predicted_positive:
            fn += 1
        elif not p.gold_positive and predicted_positive:
            fp += 1
        else:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _print_report(report: dict[str, Any]) -> None:
    settings = report["settings"]
    headline = report["headline"]
    cited = report["already_cited"]
    typer.echo(f"Model: {settings['model']} (temperature 0)")
    typer.echo(
        f"Input filter: {'on' if settings.get('input_filter') else 'off'} "
        f"(prefiltered without an LLM call: {settings['num_prefiltered']})"
    )
    typer.echo(
        f"Scored: {settings['num_scored']} "
        f"(unmatched: {settings['num_unmatched']})"
    )
    typer.echo(
        f"Headline @ conf>={headline['threshold']:.2f}: "
        f"precision={headline['precision']:.4f} "
        f"recall={headline['recall']:.4f} "
        f"f1={headline['f1']:.4f} "
        f"fpr={headline['false_positive_rate']:.4f}"
    )
    typer.echo(
        "False-flag rate on already-cited sentences: "
        f"{cited['false_flag_rate']:.4f} "
        f"({cited['wrongly_flagged']}/{cited['total']})"
    )
    typer.echo("Recall / precision by threshold:")
    for label, metrics in report["by_threshold"].items():
        typer.echo(
            f"  conf>={label}: recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"fpr={metrics['false_positive_rate']:.4f}"
        )


def _write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("saved citation-need eval report to %s", path)
    return path


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    app()
