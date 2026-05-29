"""Simulate continuous learning and plot a learning curve.

Loads the simulated feedback pool, fine-tunes in chunks, and evaluates at each step
using an in-memory InformationRetrievalEvaluator to avoid trashing the live database.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import typer
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sqlalchemy import text
from torch.utils.data import DataLoader

from database.postgres.engine import get_session
from utils.config import config
from utils.logger import logger

app = typer.Typer(add_completion=False)

# Production embeds queries with BGE-large's asymmetric instruction prefix
# (see pipeline/embedding/embedder.py). The model ships its "query" prompt
# empty ({'query': '', 'document': ''}), so loading it plainly here would run
# the evaluator and trainer in symmetric (prompt-free) mode — inconsistent with
# how the deployed system encodes queries. Inject the exact production string.
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


def _load_model() -> SentenceTransformer:
    """Load the base embedder, configured to match production query asymmetry.

    The injected ``query`` prompt makes both the in-memory evaluator
    (``query_prompt_name="query"``) and the manually-prefixed training anchors
    behave like the deployed encoder: queries carry the instruction prefix,
    passages stay prompt-free.
    """
    model = SentenceTransformer(
        config.EMBEDDER_MODEL_NAME,
        trust_remote_code=True,
        truncate_dim=config.EMBEDDER_DIM,
    )
    model.prompts["query"] = BGE_QUERY_PROMPT
    return model


def _load_eval_data(split_path: str, max_queries: int = 1000):
    """Load queries and corpus from DB for in-memory evaluation."""
    logger.info("Loading evaluation data from database...")
    with open(split_path, encoding="utf-8") as f:
        data = json.load(f)
    val_ids = set(data.get("val_paper_ids", []))
    test_ids = set(data.get("test_paper_ids", []))

    session = get_session()
    
    # 1. Load the corpus (all contexts NOT in val/test)
    corpus_sql = text("""
        SELECT context_id, sentence_without_markers, cited_paper_id, citing_paper_id
        FROM citation_contexts
        WHERE cited_paper_id IS NOT NULL
    """)
    rows = session.execute(corpus_sql).all()
    
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    # Organize by cited_paper_id
    corpus_by_cited: dict[int, set[str]] = {}
    
    logger.info("Building in-memory corpus...")
    query_count = 0
    for row in rows:
        ctx_id, text_str, cited_id, citing_id = row.context_id, row.sentence_without_markers, row.cited_paper_id, row.citing_paper_id
        if not text_str:
            continue
            
        ctx_id_str = str(ctx_id)
        if citing_id in val_ids:
            if query_count < max_queries:
                # Raw text only — the evaluator applies the model's
                # built-in query prompt via query_prompt_name.
                queries[ctx_id_str] = text_str
                # We will map it to relevant docs later
                relevant_docs[ctx_id_str] = cited_id
                query_count += 1
        elif citing_id not in test_ids:
            corpus[ctx_id_str] = text_str
            if cited_id not in corpus_by_cited:
                corpus_by_cited[cited_id] = set()
            corpus_by_cited[cited_id].add(ctx_id_str)
            
    # Resolve relevant_docs
    final_relevant_docs = {}
    final_queries = {}
    for qid, cited_id in relevant_docs.items():
        rel_docs = corpus_by_cited.get(cited_id, set())
        if rel_docs:
            final_relevant_docs[qid] = rel_docs
            final_queries[qid] = queries[qid]
            
    logger.info(f"Loaded {len(final_queries)} validation queries and {len(corpus)} corpus documents.")
    session.close()
    return final_queries, corpus, final_relevant_docs


@app.command()
def main(
    pool_path: str = typer.Option("data/finetune/simulated_pool.jsonl", help="Simulated triplets path"),
    split_path: str = typer.Option("data/eval/split_42.json", help="Eval split path"),
    output_path: str = typer.Option("data/eval/reports/learning_curve.jsonl", help="Output learning curve"),
    chunk_size: int = typer.Option(500, help="Number of triplets per training step"),
    batch_size: int = typer.Option(8, help="Per-device batch size (8 safe for 6GB VRAM)"),
    grad_accum_steps: int = typer.Option(4, help="Gradient accumulation steps (effective_bs = batch_size * grad_accum_steps)"),
) -> None:
    # 1. Load Data
    triplets = []
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            triplets.append(json.loads(line))
            
    logger.info(f"Loaded {len(triplets)} triplets from {pool_path}")
    
    # 2. Build In-Memory Evaluator
    val_queries, val_corpus, val_relevant_docs = _load_eval_data(split_path)
    
    evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        relevant_docs=val_relevant_docs,
        mrr_at_k=[20],
        accuracy_at_k=[1, 5, 10, 20],
        precision_recall_at_k=[1, 5, 10, 20],
        show_progress_bar=True,
        name="val",
        # Let the evaluator apply the model's built-in query prompt
        # (e.g. BGE-large "query" prompt) instead of baking it into
        # the query strings — avoids double-prompting.
        query_prompt_name="query",
    )
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume support: detect already-completed steps ---
    completed_steps: set[int] = set()
    out_file = Path(output_path)
    if out_file.exists() and out_file.stat().st_size > 0:
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_steps.add(json.loads(line)["step"])
        if completed_steps:
            logger.info(
                f"Resuming: found {len(completed_steps)} completed "
                f"step(s) {sorted(completed_steps)} — skipping them."
            )
    
    def _evaluate_and_log(step: int, model_instance: SentenceTransformer):
        metrics = evaluator(model_instance, output_path=str(out_dir))
        
        # SentenceTransformers evaluator returns a flat dict of metrics
        # Keys include the similarity function, e.g. 'val_cosine_accuracy@1'.
        record = {"step": step, "metrics": metrics}
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        # Log to console — keys include 'cosine_' prefix
        acc1 = metrics.get('val_cosine_accuracy@1', 0.0)
        rec20 = metrics.get('val_cosine_recall@20', 0.0)
        mrr20 = metrics.get('val_cosine_mrr@20', 0.0)
        logger.info(f"Step {step} | hit@1: {acc1:.4f} | recall@20: {rec20:.4f} | mrr@20: {mrr20:.4f}")

    # 4. Zero-shot Evaluation (Step 0)
    if 0 not in completed_steps:
        logger.info("Loading base model: %s", config.EMBEDDER_MODEL_NAME)
        model = _load_model()
        logger.info("Running baseline evaluation (Step 0)...")
        _evaluate_and_log(0, model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("Step 0 already completed — skipping baseline evaluation.")
    
    # 5. Incremental Training Loop
    # We fine-tune cumulatively: step 1 trains on chunks[0], step 2 trains on chunks[0:2], etc.
    # Actually, fine-tuning from the base model each time ensures we don't overfit to the early chunks.
    # For a true simulation, we can just reload the base model and train on the cumulative dataset.
    
    for i in range(0, len(triplets), chunk_size):
        current_pool = triplets[:i + chunk_size]
        step_num = len(current_pool)
        
        if step_num in completed_steps:
            logger.info(f"Step {step_num} already completed — skipping.")
            continue
        
        logger.info(f"--- Training Step: {step_num} Triplets ---")
        
        # Reload fresh base model to prevent catastrophic forgetting
        # of general semantics across chunks when training on small datasets.
        # Do NOT load in fp16 — use_amp handles mixed precision and
        # needs fp32 master weights for GradScaler to work.
        step_model = _load_model()

        # Asymmetric retrieval: the anchor (query) carries BGE-large's query
        # instruction prefix, exactly as production encodes queries; the
        # positive/negative passages stay prompt-free. MNRL does NOT apply
        # prompts to InputExample, so we prepend the prefix explicitly.
        train_examples = [
            InputExample(texts=[BGE_QUERY_PROMPT + t["query"], t["pos"], t["neg"]])
            for t in current_pool
        ]
        
        # Use small batch_size to fit 6GB VRAM (triplets = 3 fwd passes
        # per step) with gradient accumulation for a larger effective bs.
        loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(step_model)
        n_steps = len(loader) // grad_accum_steps
        
        step_model.fit(
            train_objectives=[(loader, train_loss)],
            epochs=1,
            warmup_steps=max(1, int(0.1 * n_steps)),
            show_progress_bar=True,
            use_amp=True,
        )
        
        logger.info(f"Evaluating Step {step_num}...")
        _evaluate_and_log(step_num, step_model)
        
        # Save the very last model
        if step_num == len(triplets):
            save_path = "models/bge-large-finetuned"
            logger.info(f"Saving final model to {save_path}...")
            step_model.save(save_path)

        # Free model and clear CUDA cache to prevent OOM from
        # memory fragmentation across iterations.
        del step_model, train_loss, loader, train_examples
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    app()
