"""Run an evaluation variant over a split and write a JSON report.

Usage::

    python -m src.scripts.evaluate --variant hybrid_rrf --split test --top-k 20
    python -m src.scripts.evaluate --variant dense_only --split val --target-year 2022
    python -m src.scripts.evaluate \\
        --variant hybrid_rrf --split test --workers 4 --top-n 200 \\
        --output data/eval/reports/hybrid_rrf_2026-05-15.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from database.postgres.engine import get_session
from evaluation.dataset import DEFAULT_SEED, DEFAULT_SPLIT_DIR, load_split
from evaluation.report import Report, save_report
from evaluation.runner import EvalRunner
from utils.logger import logger

# Windows cp1252 stdout fix (same as debug_recommend.py).
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)

_VARIANT_REGISTRY: dict[str, str] = {
    "dense_only": "evaluation.variants.dense_only.DenseOnly",
    "sparse_only": "evaluation.variants.sparse_only.SparseOnly",
    "hybrid_rrf": "evaluation.variants.hybrid_rrf.HybridRRF",
}

# Eval default: 200 contexts per branch is plenty for top-20 paper ranking
# and avoids the full top-1000 scan that makes sparse retrieval slow.
EVAL_DEFAULT_TOP_N = 200


def _load_variant_class(name: str) -> type:
    """Dynamically import and return the variant class."""
    if name not in _VARIANT_REGISTRY:
        available = ", ".join(sorted(_VARIANT_REGISTRY))
        raise typer.BadParameter(
            f"Unknown variant {name!r}. Available: {available}"
        )

    module_path, class_name = _VARIANT_REGISTRY[name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@app.command()
def main(
    variant: str = typer.Option(
        ..., "--variant", help="Variant name (dense_only, sparse_only, hybrid_rrf)."
    ),
    split_name: str = typer.Option(
        "test", "--split", help="Which split to evaluate on (test or val)."
    ),
    top_k: int = typer.Option(20, "--top-k", help="Retrieval depth for metrics."),
    top_n: int = typer.Option(
        EVAL_DEFAULT_TOP_N, "--top-n",
        help="Contexts per retrieval branch (lower = faster; 200 is safe for top-20).",
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed", help="Split seed."),
    target_year: int | None = typer.Option(
        None, "--target-year", help="Historical mode: filter candidate pool."
    ),
    workers: int = typer.Option(
        4, "--workers", help="Parallel DB workers (each gets its own session)."
    ),
    split_dir: str = typer.Option(
        str(DEFAULT_SPLIT_DIR), "--split-dir", help="Directory containing split JSON."
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output path for the JSON report."
    ),
    strict: bool = typer.Option(
        True, "--strict/--relax",
        help="Error on missing split members (--relax to skip them).",
    ),
    include_unreachable: bool = typer.Option(
        False, "--include-unreachable",
        help=(
            "Include queries whose gold paper has no other citer in the corpus."
            " By default these are skipped because retrieval excludes the"
            " citing paper's own contexts, making the gold unreachable."
        ),
    ),
) -> None:
    """Evaluate a retrieval variant and write a JSON report."""
    split_path = Path(split_dir) / f"split_{seed}.json"
    if not split_path.exists():
        typer.echo(
            f"Split file not found: {split_path}\n"
            f"Run: python -m src.scripts.build_eval_split --seed {seed}",
            err=True,
        )
        raise typer.Exit(code=1)

    variant_cls = _load_variant_class(variant)

    # The main session sits idle for the duration of the retrieval phase
    # (workers use their own sessions via variant_factory). Over a long
    # tunnel like ngrok TCP, that idle connection often gets dropped
    # mid-flight — and the ROLLBACK on __exit__ then crashes the script
    # right after the report is computed. We close defensively so a dead
    # main connection cannot trash an already-finished evaluation.
    session = get_session()
    try:
        split = load_split(
            split_path, strict=strict, session=session
        )

        variant_instance = variant_cls(session, top_n=top_n)

        def variant_factory(sess):  # type: ignore[no-untyped-def]
            return variant_cls(sess, top_n=top_n)

        runner = EvalRunner(
            variant_instance,
            split,
            session,
            variant_factory=variant_factory,
            split_name=split_name,
            target_year=target_year,
            top_k=top_k,
            workers=workers,
            require_reachable=not include_unreachable,
        )

        report = runner.run()

        _print_report(report)

        out_path = Path(output) if output else None
        saved = save_report(report, path=out_path)
        typer.echo(f"\nReport saved to {saved}")
    finally:
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ignoring session close error: %s", exc)


def _print_report(report: Report) -> None:
    """Print a formatted summary table to stdout."""
    skipped_note = (
        f", {report.num_unreachable_skipped} unreachable skipped"
        if report.require_reachable and report.num_unreachable_skipped
        else ""
    )
    typer.echo(
        f"\nVariant: {report.variant_name}\n"
        f"Split:   {report.split_name} "
        f"({report.num_citing_papers} citing papers, "
        f"{report.num_queries} queries{skipped_note})\n"
        f"Reachable-only filter: {report.require_reachable}\n"
        f"Target year filter: {report.target_year or 'none'}\n"
    )

    if not report.aggregates:
        typer.echo("No metrics computed (empty query set).")
        return

    # Column header.
    metrics = sorted(report.aggregates.keys())
    header = f"{'':>10}" + "".join(f"  {m:>12}" for m in metrics)
    typer.echo(header)
    typer.echo("-" * len(header))

    for stat in ("mean", "p25", "median", "p75", "p95"):
        row = f"{stat:>10}"
        for m in metrics:
            value = report.aggregates[m].get(stat, 0.0)
            row += f"  {value:>12.4f}"
        typer.echo(row)


if __name__ == "__main__":
    app()
