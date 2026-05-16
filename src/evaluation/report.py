"""Evaluation report dataclass and JSON serialization.

A report captures the variant name, split metadata, per-query metric vectors,
and aggregate statistics (mean, median, percentiles). Reports are stored as
JSON in ``data/eval/reports/`` so they can be diffed across variants.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from utils.logger import logger

DEFAULT_REPORT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "reports"
)


@dataclass(slots=True)
class Report:
    """Aggregate evaluation output from a single variant run."""

    variant_name: str
    split_name: str
    num_citing_papers: int
    num_queries: int
    target_year: int | None
    top_k: int
    num_unreachable_skipped: int = 0
    require_reachable: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    # Per-query metric lists (one value per query).
    per_query: dict[str, list[float]] = field(default_factory=dict)

    # Aggregate statistics (populated by ``compute_aggregates``).
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)

    def compute_aggregates(self) -> None:
        """Populate ``aggregates`` from ``per_query`` vectors."""
        for metric_name, values in self.per_query.items():
            if not values:
                self.aggregates[metric_name] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "p25": 0.0,
                    "p75": 0.0,
                    "p95": 0.0,
                }
                continue

            sorted_vals = sorted(values)
            n = len(sorted_vals)
            self.aggregates[metric_name] = {
                "mean": statistics.mean(sorted_vals),
                "median": statistics.median(sorted_vals),
                "p25": sorted_vals[max(0, int(n * 0.25))],
                "p75": sorted_vals[min(n - 1, int(n * 0.75))],
                "p95": sorted_vals[min(n - 1, int(n * 0.95))],
            }


# -----------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------

def save_report(report: Report, path: Path | None = None) -> Path:
    """Write the report to JSON; return the path."""
    if path is None:
        DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        path = DEFAULT_REPORT_DIR / f"{report.variant_name}_{today}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant_name": report.variant_name,
        "split_name": report.split_name,
        "num_citing_papers": report.num_citing_papers,
        "num_queries": report.num_queries,
        "num_unreachable_skipped": report.num_unreachable_skipped,
        "require_reachable": report.require_reachable,
        "target_year": report.target_year,
        "top_k": report.top_k,
        "timestamp": report.timestamp,
        "per_query": report.per_query,
        "aggregates": report.aggregates,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved report to %s", path)
    return path


def load_report(path: Path) -> Report:
    """Deserialize a report from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    report = Report(
        variant_name=raw["variant_name"],
        split_name=raw["split_name"],
        num_citing_papers=raw["num_citing_papers"],
        num_queries=raw["num_queries"],
        num_unreachable_skipped=raw.get("num_unreachable_skipped", 0),
        require_reachable=raw.get("require_reachable", True),
        target_year=raw.get("target_year"),
        top_k=raw["top_k"],
        timestamp=raw.get("timestamp", ""),
        per_query=raw.get("per_query", {}),
        aggregates=raw.get("aggregates", {}),
    )
    return report
