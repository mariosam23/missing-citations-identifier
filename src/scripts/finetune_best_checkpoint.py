"""Train-to-step and save a fine-tuned model for DB-backed evaluation.

Given a learning curve JSONL produced by ``incremental_finetune.py``, this
script identifies the best step (by a chosen metric), retrains the base model
to that exact checkpoint, saves the resulting model, and prints instructions
for running the DB-backed evaluation with it.

Usage::

    python -m scripts.finetune_best_checkpoint
    python -m scripts.finetune_best_checkpoint --metric val_cosine_accuracy@1
    python -m scripts.finetune_best_checkpoint --step 500  # force a step
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import typer
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

from utils.config import config
from utils.logger import logger

app = typer.Typer(add_completion=False)

BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


def _load_model() -> SentenceTransformer:
    """Load the base embedder with production query asymmetry."""
    model = SentenceTransformer(
        config.EMBEDDER_MODEL_NAME,
        trust_remote_code=True,
        truncate_dim=config.EMBEDDER_DIM,
    )
    model.prompts["query"] = BGE_QUERY_PROMPT
    return model


@app.command()
def main(
    curve_path: str = typer.Option(
        "data/eval/reports/learning_curve.jsonl",
        help="Path to learning_curve.jsonl",
    ),
    pool_path: str = typer.Option(
        "data/finetune/simulated_pool.jsonl",
        help="Simulated triplets path",
    ),
    output_model: str = typer.Option(
        "models/bge-large-finetuned-best",
        help="Output path for the saved model",
    ),
    metric: str = typer.Option(
        "val_cosine_accuracy@1",
        help="Metric to optimise (pick the step with the best value)",
    ),
    step: int | None = typer.Option(
        None, help="Force a specific step instead of auto-selecting",
    ),
    batch_size: int = typer.Option(8, help="Training batch size"),
    grad_accum_steps: int = typer.Option(4, help="Gradient accumulation steps"),
) -> None:
    """Train to the best learning-curve step and save the model."""
    # 1. Identify best step
    curve_file = Path(curve_path)
    if not curve_file.exists():
        logger.error("Learning curve not found: %s", curve_file)
        raise typer.Exit(code=1)

    records: list[dict] = []
    with curve_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["step"])

    if step is not None:
        target_step = step
        logger.info("Forced step: %d", target_step)
    else:
        # Find the best *fine-tuned* step (skip step 0 = baseline)
        fine_tuned = [r for r in records if r["step"] > 0]
        if not fine_tuned:
            logger.error("No fine-tuned steps found in %s", curve_file)
            raise typer.Exit(code=1)
        best_rec = max(fine_tuned, key=lambda r: r["metrics"].get(metric, 0.0))
        target_step = best_rec["step"]
        best_val = best_rec["metrics"].get(metric, 0.0)
        baseline_val = records[0]["metrics"].get(metric, 0.0)
        logger.info(
            "Best fine-tuned step for %s: %d (%.4f vs baseline %.4f, Δ=%+.4f)",
            metric, target_step, best_val, baseline_val, best_val - baseline_val,
        )

    # 2. Load triplets up to that step
    pool_file = Path(pool_path)
    if not pool_file.exists():
        logger.error("Pool not found: %s", pool_file)
        raise typer.Exit(code=1)

    all_triplets: list[dict] = []
    with pool_file.open(encoding="utf-8") as f:
        for line in f:
            all_triplets.append(json.loads(line))

    train_triplets = all_triplets[:target_step]
    logger.info("Training on %d triplets (step %d)", len(train_triplets), target_step)

    # 3. Train
    model = _load_model()
    train_examples = [
        InputExample(texts=[BGE_QUERY_PROMPT + t["query"], t["pos"], t["neg"]])
        for t in train_triplets
    ]
    loader: DataLoader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)  # type: ignore[arg-type]
    train_loss = MultipleNegativesRankingLoss(model)
    n_steps = len(loader) // grad_accum_steps

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=1,
        warmup_steps=max(1, int(0.1 * n_steps)),
        show_progress_bar=True,
        use_amp=True,
    )

    # 4. Save
    out_path = Path(output_model)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    logger.info("Saved fine-tuned model to %s", out_path)

    # 5. Write metadata
    meta = {
        "base_model": config.EMBEDDER_MODEL_NAME,
        "step": target_step,
        "triplets_used": len(train_triplets),
        "metric_optimised": metric,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
    }
    meta_path = out_path / "finetune_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Cleanup
    del model, train_loss, loader, train_examples
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Print instructions
    typer.echo("\n" + "=" * 70)
    typer.echo("Model saved. To run DB-backed evaluation:")
    typer.echo("=" * 70)
    typer.echo(f"""
1. Backup current embeddings:
   python -c "
   from database.postgres.engine import get_session
   from sqlalchemy import text
   s = get_session()
   s.execute(text('CREATE TABLE IF NOT EXISTS citation_context_embeddings_backup_finetune AS SELECT * FROM citation_context_embeddings'))
   s.commit()
   "

2. Truncate and re-embed with the fine-tuned model:
   EMBEDDER_MODEL_NAME={out_path} python -m scripts.embed_contexts

3. Rebuild HNSW index:
   python -c "
   from database.postgres.engine import get_session
   from sqlalchemy import text
   s = get_session()
   s.execute(text('DROP INDEX IF EXISTS citation_context_embedding_hnsw_idx'))
   s.execute(text('CREATE INDEX citation_context_embedding_hnsw_idx ON citation_context_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)'))
   s.commit()
   "

4. Run the DB-backed evaluation:
   python -m scripts.evaluate --variant dense_only --split val \\
     --output data/eval/reports/dense_only_finetune_best_val.json

5. Compare with baseline:
   python -m scripts.diff_eval_reports \\
     data/eval/reports/dense_only_2026-05-28.json \\
     data/eval/reports/dense_only_finetune_best_val.json \\
     --bootstrap

6. Restore original embeddings (if fine-tune is worse):
   python -c "
   from database.postgres.engine import get_session
   from sqlalchemy import text
   s = get_session()
   s.execute(text('TRUNCATE citation_context_embeddings'))
   s.execute(text('INSERT INTO citation_context_embeddings SELECT * FROM citation_context_embeddings_backup_finetune'))
   s.commit()
   "
""")


if __name__ == "__main__":
    app()
