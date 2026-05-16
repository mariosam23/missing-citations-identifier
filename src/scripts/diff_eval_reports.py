"""Compare two evaluation reports and print a delta table.

Optionally computes bootstrap confidence intervals (1000 resamples)
for each metric delta.

Usage::

    python -m src.scripts.diff_eval_reports \\
        data/eval/reports/dense_only_2026-05-15.json \\
        data/eval/reports/hybrid_rrf_2026-05-15.json
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import typer

from evaluation.report import load_report

# Windows cp1252 stdout fix.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def _bootstrap_ci(
    values_a: list[float],
    values_b: list[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a (1-alpha) bootstrap CI for ``mean(B) - mean(A)``.

    Paired bootstrap: resample indices, compute delta on each resample.
    """
    if len(values_a) != len(values_b):
        # Different query counts → can't pair. Return NaN.
        nan = float("nan")
        return (nan, nan)

    n = len(values_a)
    if n == 0:
        return (0.0, 0.0)

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        mean_a = statistics.mean(values_a[i] for i in indices)
        mean_b = statistics.mean(values_b[i] for i in indices)
        deltas.append(mean_b - mean_a)

    deltas.sort()
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2))
    return (deltas[lo_idx], deltas[min(hi_idx, len(deltas) - 1)])


@app.command()
def main(
    report_a: str = typer.Argument(
        ..., help="Path to the baseline report JSON."
    ),
    report_b: str = typer.Argument(
        ..., help="Path to the comparison report JSON."
    ),
    bootstrap: bool = typer.Option(
        True, "--bootstrap/--no-bootstrap",
        help="Compute bootstrap 95% CIs for deltas.",
    ),
) -> None:
    """Print a side-by-side delta table comparing two evaluation reports."""
    a = load_report(Path(report_a))
    b = load_report(Path(report_b))

    typer.echo(
        f"\n{'Metric':<14}  {'Baseline':>10}  {'Comparison':>10}  "
        f"{'Delta':>8}  {'95% CI':>22}"
    )
    typer.echo(
        f"{'':>14}  {a.variant_name:>10}  {b.variant_name:>10}"
    )
    typer.echo("-" * 72)

    all_metrics = sorted(
        set(a.aggregates.keys()) | set(b.aggregates.keys())
    )

    for metric in all_metrics:
        mean_a = a.aggregates.get(metric, {}).get("mean", 0.0)
        mean_b = b.aggregates.get(metric, {}).get("mean", 0.0)
        delta = mean_b - mean_a
        sign = "+" if delta >= 0 else ""

        ci_str = ""
        if bootstrap:
            vals_a = a.per_query.get(metric, [])
            vals_b = b.per_query.get(metric, [])
            if vals_a and vals_b:
                lo, hi = _bootstrap_ci(vals_a, vals_b)
                ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
            else:
                ci_str = "n/a"

        typer.echo(
            f"{metric:<14}  {mean_a:>10.4f}  {mean_b:>10.4f}  "
            f"{sign}{delta:>7.4f}  {ci_str:>22}"
        )

    typer.echo(
        f"\nBaseline:   {a.variant_name} ({a.num_queries} queries)\n"
        f"Comparison: {b.variant_name} ({b.num_queries} queries)"
    )


if __name__ == "__main__":
    app()
