"""Build a deterministic train/val/test split of citing papers.

One-time script; the output ``data/eval/split_{seed}.json`` is checked
into git so every reviewer and CI run uses the same partition.

Usage::

    python -m src.scripts.build_eval_split --seed 42
    python -m src.scripts.build_eval_split --seed 42 --test-frac 0.10 --val-frac 0.10
"""

from __future__ import annotations

import sys

import typer

from database.postgres.engine import get_session
from evaluation.dataset import (
    DEFAULT_SEED,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
    build_split,
    save_split,
)
from utils.logger import logger

# Windows cp1252 stdout can't encode the "→" in the summary line; force UTF-8
# (same fix as scripts.evaluate / scripts.debug_recommend).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed for the split."),
    val_frac: float = typer.Option(
        DEFAULT_VAL_FRAC, "--val-frac", help="Fraction of papers for validation."
    ),
    test_frac: float = typer.Option(
        DEFAULT_TEST_FRAC, "--test-frac", help="Fraction of papers for test."
    ),
    output_dir: str = typer.Option(
        str(DEFAULT_SPLIT_DIR), "--output-dir", help="Directory for split JSON."
    ),
) -> None:
    """Build and save a citing-paper-level eval split."""
    from pathlib import Path

    out = Path(output_dir)
    with get_session() as session:
        split = build_split(
            session,
            seed=seed,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        path = save_split(split, directory=out)

    logger.info("done — split written to %s", path)
    typer.echo(
        f"Split written: train={len(split.train_paper_ids)} "
        f"val={len(split.val_paper_ids)} "
        f"test={len(split.test_paper_ids)} → {path}"
    )


if __name__ == "__main__":
    app()
