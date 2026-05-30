"""Evaluate the deterministic missing-citation detector.

Two MVP evidence sources are supported:

* ``manual``: labelled JSONL rows such as ``eval/novel_work_dataset``.
* ``synthetic``: citation-deletion examples sampled from ``citation_contexts``.

The script intentionally evaluates the rule-based detector and the existing
``hybrid_context`` recommender path; it does not train a classifier.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from pipeline.embedding.embedder import encode_query
from pipeline.missing_citations.detector import CitationNeedLabel, scan_document
from pipeline.retrieval.dense import DEFAULT_TOP_N
from pipeline.retrieval.hybrid import hybrid_rank
from utils.logger import logger

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL_DATASET = (
    _PROJECT_ROOT / "eval" / "novel_work_dataset" / "dataset.jsonl"
)
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "eval" / "reports"
_KS: tuple[int, ...] = (5, 10, 20)


@dataclass(frozen=True, slots=True)
class ManualRow:
    sentence_id: str
    text: str
    gold_label: CitationNeedLabel
    previous_sentence: str | None
    next_sentence: str | None


@app.command()
def manual(
    dataset: str = typer.Option(
        str(DEFAULT_MANUAL_DATASET),
        "--dataset",
        help="JSONL file with text, gold_label, and optional neighbours.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Where to write the JSON report.",
    ),
) -> None:
    """Evaluate detector labels on a manually labelled sentence dataset."""
    rows = _load_manual_rows(Path(dataset))
    pairs: list[tuple[CitationNeedLabel, CitationNeedLabel]] = []

    for row in rows:
        predicted = _predict_manual_row(row)
        pairs.append((row.gold_label, predicted))

    report = _classification_report(pairs)
    out_path = _write_report(
        report,
        Path(output) if output else DEFAULT_OUTPUT_DIR / "missing_manual.json",
    )
    _print_classification_report(report)
    typer.echo(f"\nReport saved to {out_path}")


@app.command()
def synthetic(
    limit: int = typer.Option(200, "--limit", min=1, help="Max contexts to sample."),
    top_n: int = typer.Option(
        200,
        "--top-n",
        min=1,
        help="Contexts per dense/sparse retrieval branch.",
    ),
    top_k: int = typer.Option(
        20,
        "--top-k",
        min=1,
        help="Maximum recommendation depth; recall@5/10/20 is capped by this.",
    ),
    target_year: int | None = typer.Option(
        None,
        "--target-year",
        help="Filter candidate papers to this publication year or earlier.",
    ),
    exclude_source_paper: bool = typer.Option(
        True,
        "--exclude-source-paper/--allow-source-paper",
        help="Exclude contexts from the citing paper that supplied the example.",
    ),
    leak_free: bool = typer.Option(
        True,
        "--leak-free/--allow-verbatim",
        help="Drop verbatim duplicate contexts from recommendation candidates.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Where to write the JSON report.",
    ),
) -> None:
    """Run synthetic citation-deletion detection plus end-to-end recall."""
    with get_session() as session:
        rows = _load_synthetic_rows(session, limit=limit)
        report = _evaluate_synthetic_rows(
            session,
            rows,
            top_n=top_n,
            top_k=top_k,
            target_year=target_year,
            exclude_source_paper=exclude_source_paper,
            leak_free=leak_free,
        )

    out_path = _write_report(
        report,
        Path(output) if output else DEFAULT_OUTPUT_DIR / "missing_synthetic.json",
    )
    _print_synthetic_report(report)
    typer.echo(f"\nReport saved to {out_path}")


def _load_manual_rows(path: Path) -> list[ManualRow]:
    rows: list[ManualRow] = []
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
                ManualRow(
                    sentence_id=str(raw["sentence_id"]),
                    text=str(raw["text"]),
                    gold_label=gold_label,
                    previous_sentence=_optional_str(raw.get("previous_sentence")),
                    next_sentence=_optional_str(raw.get("next_sentence")),
                )
            )
    return rows


def _predict_manual_row(row: ManualRow) -> CitationNeedLabel:
    document, target_start, target_end = _build_manual_document(row)
    detections = scan_document(document, max_sentences=3)
    for detection in detections:
        sentence = detection.sentence
        if sentence.start_offset <= target_start and sentence.end_offset >= target_end:
            return detection.label
        if sentence.text.strip() == row.text.strip():
            return detection.label

    fallback = scan_document(row.text, max_sentences=1)
    if fallback:
        return fallback[0].label
    return CitationNeedLabel.NOT_CITATION_WORTHY


def _build_manual_document(row: ManualRow) -> tuple[str, int, int]:
    parts: list[str] = []
    if row.previous_sentence:
        parts.append(row.previous_sentence)
    target_start = sum(len(part) + 1 for part in parts)
    parts.append(row.text)
    target_end = target_start + len(row.text)
    if row.next_sentence:
        parts.append(row.next_sentence)
    return " ".join(parts), target_start, target_end


def _classification_report(
    pairs: list[tuple[CitationNeedLabel, CitationNeedLabel]],
) -> dict[str, Any]:
    labels = [label.value for label in CitationNeedLabel]
    matrix: dict[str, dict[str, int]] = {
        gold: {predicted: 0 for predicted in labels} for gold in labels
    }
    for gold, predicted in pairs:
        matrix[gold.value][predicted.value] += 1

    missing = CitationNeedLabel.MISSING_CITATION.value
    has_citation = CitationNeedLabel.HAS_CITATION.value
    true_positive = matrix[missing][missing]
    false_positive = sum(
        matrix[gold][missing] for gold in labels if gold != missing
    )
    false_negative = sum(
        matrix[missing][predicted] for predicted in labels if predicted != missing
    )
    cited_total = sum(matrix[has_citation].values())

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "num_items": len(pairs),
        "confusion_matrix": matrix,
        "missing_citation": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "false_positive_rate_on_already_cited": _safe_div(
            matrix[has_citation][missing],
            cited_total,
        ),
    }


def _load_synthetic_rows(
    session: Session,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT context_id,
                   sentence_without_markers,
                   cited_paper_id,
                   citing_paper_id,
                   citing_year
            FROM citation_contexts
            WHERE cited_paper_id IS NOT NULL
              AND sentence_without_markers IS NOT NULL
              AND sentence_without_markers != ''
            ORDER BY context_id
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).all()
    return [
        {
            "context_id": int(row[0]),
            "sentence": str(row[1]),
            "gold_paper_id": int(row[2]),
            "citing_paper_id": int(row[3]) if row[3] is not None else None,
            "citing_year": int(row[4]) if row[4] is not None else None,
        }
        for row in rows
    ]


def _evaluate_synthetic_rows(
    session: Session,
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    top_k: int,
    target_year: int | None,
    exclude_source_paper: bool,
    leak_free: bool,
) -> dict[str, Any]:
    detected = 0
    detector_labels: Counter[str] = Counter()
    hits = {k: 0 for k in _KS if k <= top_k}
    hits_on_detected = {k: 0 for k in hits}

    for row in rows:
        sentence = str(row["sentence"])
        detections = scan_document(sentence, max_sentences=1)
        label = (
            detections[0].label
            if detections
            else CitationNeedLabel.NOT_CITATION_WORTHY
        )
        detector_labels[label.value] += 1
        if label != CitationNeedLabel.MISSING_CITATION:
            continue

        detected += 1
        ranked = _recommend_for_synthetic_row(
            session,
            row,
            top_n=top_n,
            top_k=top_k,
            target_year=target_year,
            exclude_source_paper=exclude_source_paper,
            leak_free=leak_free,
        )
        for k in hits:
            if int(row["gold_paper_id"]) in ranked[:k]:
                hits[k] += 1
                hits_on_detected[k] += 1

    total = len(rows)
    return {
        "num_items": total,
        "detector_label_counts": dict(detector_labels),
        "detector_recall": _safe_div(detected, total),
        "end_to_end_recall": {
            f"recall@{k}": _safe_div(value, total) for k, value in hits.items()
        },
        "recommendation_recall_on_detected": {
            f"recall@{k}": _safe_div(value, detected)
            for k, value in hits_on_detected.items()
        },
        "settings": {
            "top_n": top_n,
            "top_k": top_k,
            "target_year": target_year,
            "exclude_source_paper": exclude_source_paper,
            "leak_free": leak_free,
        },
    }


def _recommend_for_synthetic_row(
    session: Session,
    row: dict[str, Any],
    *,
    top_n: int,
    top_k: int,
    target_year: int | None,
    exclude_source_paper: bool,
    leak_free: bool,
) -> list[int]:
    sentence = str(row["sentence"])
    query_embedding = encode_query(sentence)
    ranked = hybrid_rank(
        session,
        sentence,
        query_embedding,
        top_n=top_n,
        top_k=top_k,
        target_year=target_year,
        exclude_citing_paper_id=(
            int(row["citing_paper_id"])
            if exclude_source_paper and row["citing_paper_id"] is not None
            else None
        ),
        exclude_sentence=sentence if leak_free else None,
    )
    return [aggregate.cited_paper_id for aggregate in ranked]


def _print_classification_report(report: dict[str, Any]) -> None:
    missing = report["missing_citation"]
    typer.echo(f"Items: {report['num_items']}")
    typer.echo(
        "MISSING_CITATION "
        f"precision={missing['precision']:.4f} "
        f"recall={missing['recall']:.4f} "
        f"f1={missing['f1']:.4f}"
    )
    typer.echo(
        "False-positive rate on already-cited sentences: "
        f"{report['false_positive_rate_on_already_cited']:.4f}"
    )


def _print_synthetic_report(report: dict[str, Any]) -> None:
    typer.echo(f"Items: {report['num_items']}")
    typer.echo(f"Detector recall: {report['detector_recall']:.4f}")
    typer.echo("End-to-end recall:")
    for metric, value in report["end_to_end_recall"].items():
        typer.echo(f"  {metric}: {value:.4f}")


def _write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("saved missing-citation eval report to %s", path)
    return path


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    app()
