"""Stage 4 retrieval experiments over real DB/Qdrant data.

The experiment builds hide-and-seek retrieval examples from local Postgres
papers/citations when available, evaluates Stage 4 variants, and writes
thesis-ready metrics and plots.

Usage
-----
    python -m src.experiments.stage4_experiments --source db --max-examples 100
    python -m src.experiments.stage4_experiments --source jsonl --jsonl-path eval/s2orc_rows.jsonl
"""


import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.benchmarks.common import BenchmarkExample
from evaluation.benchmarks.s2orc import load_hide_seek_jsonl, random_hidden_subset
from evaluation.metrics import PairedBootstrapResult, paired_bootstrap_ci
from evaluation.runner import EvaluationResult, RetrievalEvaluator
from pipeline.post_filters import FilterContext, PostFilterPipeline
from utils.config import config as app_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage4Config:
    source: str
    jsonl_path: Path | None
    max_examples: int
    top_k: int
    candidate_k: int
    candidates_per_subclaim: int
    seed: int
    hide_fraction: float
    min_refs: int
    min_abstract_chars: int
    model_name: str
    include_v2: bool
    decomposer_model: str
    output_dir: Path
    assets_dir: Path


@dataclass(frozen=True)
class VariantOutput:
    name: str
    result: EvaluationResult


@dataclass(frozen=True)
class Stage5RunData:
    decomposition_cache: Mapping[str, Any]
    aggregate_cache: Mapping[str, Sequence[Any]]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _short_query(title: str | None, abstract: str | None, max_chars: int = 1800) -> str:
    parts = [p.strip() for p in (title or "", abstract or "") if p and p.strip()]
    query = ". ".join(parts).strip()
    return query[:max_chars].strip()


def load_db_hide_seek_examples(
    *,
    max_examples: int = 100,
    seed: int = 42,
    hide_fraction: float = 0.3,
    min_refs: int = 2,
    min_abstract_chars: int = 50,
) -> list[BenchmarkExample]:
    """Build hide-and-seek examples from local Postgres papers/citations.

    Each example uses one citing paper's title + abstract as the retrieval query
    and hides a seeded subset of its references that also exist in the local
    paper table. The latter approximates references available to the indexed
    retrieval corpus.

    ``min_abstract_chars`` quality-gates the benchmark: source papers must have
    an abstract of at least that many characters, and only target papers with
    an abstract of at least that many characters are eligible to be hidden.
    Title-only documents make the title+abstract retrieval signal degenerate
    (see ``scripts/diagnose_recall.py``), so this gate is required for the
    experiment to have statistical power. Set to 0 to disable.
    """
    try:
        from sqlalchemy import func, select
        from sqlalchemy.orm import aliased

        from database.postgres.engine import get_session
        from database.postgres.tables import Citation, Paper
        from utils.config import config
    except Exception as e:  # pragma: no cover - depends on local DB stack
        raise RuntimeError(f"Failed to import database dependencies: {e}") from e

    if not config.DB_URL:
        raise RuntimeError("DB_URL is empty. Set DB_URL in .env before using --source db.")

    rng = random.Random(seed)
    target_paper = aliased(Paper)
    refs_by_source: dict[str, list[str]] = defaultdict(list)
    eligible_targets_by_source: dict[str, set[str]] = defaultdict(set)
    source_rows: dict[str, tuple[str | None, str | None, Any]] = {}

    row_limit = max(max_examples * max(min_refs, 1) * 40, 4000)
    stmt = (
        select(
            Paper.paperId,
            Paper.title,
            Paper.abstract,
            Citation.target_paper_id,
            Paper.publication_date,
            func.coalesce(func.length(target_paper.abstract), 0).label("target_abs_len"),
        )
        .join(Citation, Citation.source_paper_id == Paper.paperId)
        .join(target_paper, target_paper.paperId == Citation.target_paper_id)
        .where(Paper.paperId.is_not(None))
        .where(Citation.target_paper_id.is_not(None))
    )
    if min_abstract_chars > 0:
        stmt = stmt.where(func.coalesce(func.length(Paper.abstract), 0) >= min_abstract_chars)
    stmt = stmt.order_by(Paper.paperId).limit(row_limit)

    try:
        with get_session() as session:
            rows = session.execute(stmt).all()
    except Exception as e:  # pragma: no cover - depends on local DB stack
        raise RuntimeError(f"Failed to load examples from Postgres: {e}") from e

    for source_id, title, abstract, target_id, pub_date, target_abs_len in rows:
        query = _short_query(title, abstract)
        if not query:
            continue
        sid = str(source_id)
        tid = str(target_id)
        source_rows[sid] = (title, abstract, pub_date)
        refs_by_source[sid].append(tid)
        if int(target_abs_len or 0) >= min_abstract_chars:
            eligible_targets_by_source[sid].add(tid)

    examples: list[BenchmarkExample] = []
    for source_id, refs in refs_by_source.items():
        unique_refs = list(dict.fromkeys(refs))
        if len(unique_refs) < min_refs:
            continue

        eligible = [r for r in unique_refs if r in eligible_targets_by_source[source_id]]
        if not eligible:
            continue

        title, abstract, pub_date = source_rows[source_id]
        hidden = random_hidden_subset(eligible, hide_fraction=hide_fraction, rng=rng)
        if not hidden:
            continue

        query = _short_query(title, abstract)
        query_year = pub_date.year if pub_date else None
        examples.append(
            BenchmarkExample(
                example_id=source_id,
                query_text=query,
                hidden_paper_ids=hidden,
                section=None,
                citation_intent=None,
                is_multi_facet=None,
                metadata={
                    "source": "db",
                    "source_paper_id": source_id,
                    "query_year": query_year,
                    "indexed_reference_ids": unique_refs,
                    "indexed_reference_count": len(unique_refs),
                    "total_reference_count": len(unique_refs),
                    "eligible_target_count": len(eligible),
                    "min_abstract_chars": min_abstract_chars,
                    "citation_worthy": True,
                },
            )
        )
        if len(examples) >= max_examples:
            break

    if not examples:
        raise RuntimeError(
            "No DB hide-and-seek examples were created. Check that papers have "
            "title/abstract text and citations whose targets exist in papers, "
            f"and that min_abstract_chars={min_abstract_chars} is not too strict."
        )
    return examples


def load_examples(config: Stage4Config) -> list[BenchmarkExample]:
    if config.source == "db":
        return load_db_hide_seek_examples(
            max_examples=config.max_examples,
            seed=config.seed,
            hide_fraction=config.hide_fraction,
            min_refs=config.min_refs,
            min_abstract_chars=config.min_abstract_chars,
        )

    if config.jsonl_path is None:
        raise RuntimeError("--jsonl-path is required when --source jsonl")
    examples = load_hide_seek_jsonl(config.jsonl_path, seed=config.seed)
    return examples[: config.max_examples]


def initialize_retriever(prefetch_limit: int = 200):
    """Reuse the Stage 4A demo initializer to get a Qdrant-backed retriever."""
    try:
        from experiments.retrieval_demo import initialize_retriever
    except Exception as e:
        raise RuntimeError(f"Failed to import retrieval initializer: {e}") from e

    return initialize_retriever(prefetch_limit=prefetch_limit)


def initialize_reranker(model_name: str):
    try:
        from pipeline.reranker import CrossEncoderReranker
    except Exception as e:
        raise RuntimeError(f"Failed to import reranker: {e}") from e

    reranker = CrossEncoderReranker(model_name=model_name)
    reranker._get_model()
    return reranker


class _CachedDecomposer:
    """Wraps a ClaimDecomposer with a JSON disk cache.

    Why: the experiment is pinned to a free-tier Gemini model (5 req/min). A
    single 100-example V2 run hammers the rate limit and 90% of decompositions
    fall back to ``single_claim`` (the original query unchanged), which makes
    V2 indistinguishable from V1. Caching successful decompositions lets us
    pay the rate-limit cost exactly once and cleanly re-run V2 thereafter.
    """

    _SCHEMA_VERSION = 1
    # Free-tier Gemini is 5 requests/minute. 13s between actual API calls keeps
    # us under that with a small safety margin.
    _MIN_INTERVAL_S = 13.0

    def __init__(self, inner, cache_path: Path) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._last_api_call: float = 0.0
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("version") == self._SCHEMA_VERSION:
                    entries = payload.get("entries")
                    if isinstance(entries, dict):
                        self._cache = entries
                logger.info(
                    "Loaded %d cached decompositions from %s",
                    len(self._cache),
                    cache_path,
                )
            except Exception as exc:
                logger.warning("Could not load decomposition cache %s: %s", cache_path, exc)

    @staticmethod
    def _key(claim: str) -> str:
        import hashlib

        return hashlib.sha256(claim.strip().encode("utf-8")).hexdigest()

    def decompose(self, claim: str):
        from entities import AggregationStrategy, Decomposition, Subclaim

        key = self._key(claim)
        cached = self._cache.get(key)
        if cached and len(cached.get("subclaims", [])) > 1:
            return Decomposition(
                original_text=claim,
                subclaims=tuple(
                    Subclaim(text=str(s["text"]), importance=float(s["importance"]))
                    for s in cached["subclaims"]
                ),
                aggregation=AggregationStrategy(str(cached.get("aggregation", "WEIGHTED")).upper()),
            )

        # Rate-limit pace: ensure at least _MIN_INTERVAL_S since the last live
        # API call (cache hits don't count).
        now = time.perf_counter()
        wait = self._MIN_INTERVAL_S - (now - self._last_api_call)
        if self._last_api_call > 0 and wait > 0:
            time.sleep(wait)
        self._last_api_call = time.perf_counter()

        decomp = self._inner.decompose(claim)
        # Only cache non-trivial decompositions; a single-subclaim fallback was
        # almost certainly a rate-limit failure and we want to retry next time.
        if len(decomp.subclaims) > 1:
            self._cache[key] = {
                "claim": claim,
                "subclaims": [
                    {"text": s.text, "importance": s.importance} for s in decomp.subclaims
                ],
                "aggregation": decomp.aggregation.value
                if hasattr(decomp.aggregation, "value")
                else str(decomp.aggregation),
            }
            self._dirty = True
            self._flush()
        return decomp

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(
                {"version": self._SCHEMA_VERSION, "entries": self._cache},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._dirty = False


def initialize_decomposer(model_name: str, cache_path: Path | None = None):
    try:
        from pipeline.claim_decomposer import ClaimDecomposer
    except Exception as e:
        raise RuntimeError(f"Failed to import claim decomposer: {e}") from e

    inner = ClaimDecomposer(model=model_name)
    if cache_path is None:
        return inner
    return _CachedDecomposer(inner, cache_path)


def _paper_ids(results: Sequence[Any]) -> list[str]:
    return [str(result.paper_id) for result in results]


def _example_query_year(example: BenchmarkExample) -> int | None:
    query_year = example.metadata.get("query_year")
    return int(query_year) if query_year is not None else None


def _apply_filters(
    candidates: list[Any],
    example: BenchmarkExample,
    pipeline: PostFilterPipeline | None,
) -> list[Any]:
    if pipeline is None or not candidates:
        return candidates

    source_id = example.metadata.get("source_paper_id")
    query_year = example.metadata.get("query_year")
    
    indexed_refs = set(example.metadata.get("indexed_reference_ids", []))
    hidden_refs = example.hidden_paper_ids
    already_cited = frozenset(indexed_refs - hidden_refs)

    context = FilterContext(
        source_paper_id=str(source_id) if source_id else None,
        query_year=int(query_year) if query_year is not None else None,
        already_cited_ids=already_cited,
    )
    return pipeline.apply(candidates, context)

def make_cached_hybrid_predictor(
    retriever,
    *,
    candidate_k: int,
    top_k: int,
    cache: dict[str, list[Any]],
    filter_pipeline: PostFilterPipeline | None = None,
) -> Callable[[BenchmarkExample], list[str]]:
    def predict(example: BenchmarkExample) -> list[str]:
        if example.example_id not in cache:
            cache[example.example_id] = retriever.retrieve(
                example.query_text,
                top_k=candidate_k,
                max_year=_example_query_year(example),
            )
        candidates = cache[example.example_id]
        if filter_pipeline:
            candidates = _apply_filters(candidates, example, filter_pipeline)
        return _paper_ids(candidates[:top_k])

    return predict


def make_rerank_predictor(
    retriever,
    reranker,
    *,
    candidate_k: int,
    top_k: int,
    candidate_cache: dict[str, list[Any]],
    rerank_cache: dict[str, list[Any]],
    filter_pipeline: PostFilterPipeline | None = None,
) -> Callable[[BenchmarkExample], list[str]]:
    def predict(example: BenchmarkExample) -> list[str]:
        if example.example_id not in candidate_cache:
            candidate_cache[example.example_id] = retriever.retrieve(
                example.query_text,
                top_k=candidate_k,
                max_year=_example_query_year(example),
            )
        if example.example_id not in rerank_cache:
            rerank_cache[example.example_id] = reranker.rerank(
                example.query_text,
                candidate_cache[example.example_id],
                top_k=candidate_k,
            )
        candidates = rerank_cache[example.example_id]
        if filter_pipeline:
            candidates = _apply_filters(candidates, example, filter_pipeline)
        return _paper_ids(candidates[:top_k])

    return predict


def make_decomposed_predictor(
    decomposer,
    decomposed_retriever,
    *,
    candidate_k: int,
    top_k: int,
    decomposition_cache: dict[str, Any],
    aggregate_cache: dict[str, list[Any]],
    filter_pipeline: PostFilterPipeline | None = None,
    final_reranker=None,
    final_rerank_k: int = 100,
) -> Callable[[BenchmarkExample], list[str]]:
    """Predictor for V2 (decomposition).

    The optional ``final_reranker`` reranks the fused candidate list with the
    *full* query (not the per-subclaim text). This keeps V2 comparable to V1
    (which also reranks with the full query) — the only difference is the
    candidate set fed to the reranker (decomposed-and-fused vs. hybrid).
    """

    def predict(example: BenchmarkExample) -> list[str]:
        if example.example_id not in decomposition_cache:
            decomposition_cache[example.example_id] = decomposer.decompose(example.query_text)
        if example.example_id not in aggregate_cache:
            fused = decomposed_retriever.retrieve_and_aggregate(
                decomposition_cache[example.example_id],
                top_k=final_rerank_k,
                max_year=_example_query_year(example),
            )
            if final_reranker is not None and fused:
                fused = final_reranker.rerank(
                    example.query_text,
                    fused,
                    top_k=final_rerank_k,
                )
            aggregate_cache[example.example_id] = fused
        candidates = aggregate_cache[example.example_id]
        if filter_pipeline:
            candidates = _apply_filters(candidates, example, filter_pipeline)
        return _paper_ids(candidates[:top_k])

    return predict


def _logging_predict(name: str, predict, total: int, every: int = 5):
    """Wrap a predict fn so it prints a progress line every ``every`` examples.

    Stage 4 evaluation runs are silent for several minutes on CPU; surfacing
    a per-query rate makes it possible to tell ``in-progress`` from ``stuck``
    without resorting to ``tasklist``.
    """
    counter = {"n": 0, "started": time.perf_counter()}

    def wrapped(example: BenchmarkExample) -> list[str]:
        out = predict(example)
        counter["n"] += 1
        n = counter["n"]
        if n % every == 0 or n == total:
            elapsed = time.perf_counter() - counter["started"]
            rate = n / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{name}] {n}/{total} examples ({rate:.2f}/s, "
                f"elapsed {elapsed:.1f}s)",
                flush=True,
            )
        return out

    return wrapped



def evaluate_variants(
    examples: Sequence[BenchmarkExample],
    retriever,
    reranker,
    *,
    top_k: int,
    candidate_k: int,
    candidates_per_subclaim: int = 30,
    decomposer=None,
    filter_pipeline: PostFilterPipeline | None = None,
) -> tuple[list[VariantOutput], Stage5RunData | None]:
    evaluator = RetrievalEvaluator(ks=(1, 5, 10))
    hybrid_cache: dict[str, list[Any]] = {}
    candidate_cache: dict[str, list[Any]] = {}
    rerank_cache: dict[str, list[Any]] = {}
    n = len(examples)

    v0_predict = make_cached_hybrid_predictor(
        retriever, candidate_k=candidate_k, top_k=top_k, cache=hybrid_cache, filter_pipeline=filter_pipeline
    )
    v1_predict = make_rerank_predictor(
        retriever,
        reranker,
        candidate_k=candidate_k,
        top_k=top_k,
        candidate_cache=candidate_cache,
        rerank_cache=rerank_cache,
        filter_pipeline=filter_pipeline,
    )

    print(f"Evaluating V0 (hybrid) on {n} examples ...", flush=True)
    v0_out = evaluator.evaluate(examples, _logging_predict("V0", v0_predict, n))
    print(f"Evaluating V1 (hybrid + rerank) on {n} examples ...", flush=True)
    v1_out = evaluator.evaluate(examples, _logging_predict("V1", v1_predict, n))
    outputs = [VariantOutput("V0", v0_out), VariantOutput("V1", v1_out)]
    stage5_run: Stage5RunData | None = None
    if decomposer is not None:
        from pipeline.aggregator import DecomposedRetriever

        decomposition_cache: dict[str, Any] = {}
        aggregate_cache: dict[str, list[Any]] = {}
        # Per-subclaim retrieval has no per-subclaim rerank: the aggregator
        # fuses ranked lists, then we (optionally) rerank the fused top-K once
        # with the FULL query inside ``make_decomposed_predictor`` if needed.
        # Per-subclaim rerank was ~5x more rerank passes per query and the
        # bge-reranker is the dominant CPU cost.
        decomposed_retriever = DecomposedRetriever(
            retriever,
            reranker=None,
            candidates_per_subclaim=candidates_per_subclaim,
        )
        v2_predict = make_decomposed_predictor(
            decomposer,
            decomposed_retriever,
            candidate_k=candidates_per_subclaim,
            top_k=top_k,
            decomposition_cache=decomposition_cache,
            aggregate_cache=aggregate_cache,
            filter_pipeline=filter_pipeline,
            final_reranker=reranker,
            final_rerank_k=candidate_k,
        )
        print(
            f"Evaluating V2 (decomposition, "
            f"{candidates_per_subclaim}/subclaim, final rerank top-{candidate_k}) "
            f"on {n} examples ...",
            flush=True,
        )
        v2_out = evaluator.evaluate(examples, _logging_predict("V2", v2_predict, n))
        outputs.append(VariantOutput("V2", v2_out))
        stage5_run = Stage5RunData(
            decomposition_cache=decomposition_cache,
            aggregate_cache=aggregate_cache,
        )
    return outputs, stage5_run


def _safe_metric(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _rows_by_id(result: EvaluationResult) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["example_id"]): row
        for row in result.per_example
        if not row.get("skipped", False)
    }


def compute_bootstrap(
    baseline: EvaluationResult,
    variant: EvaluationResult,
    *,
    seed: int,
    metrics: Sequence[str] = ("recall@10", "mrr", "ndcg@10"),
) -> dict[str, PairedBootstrapResult]:
    baseline_rows = _rows_by_id(baseline)
    variant_rows = _rows_by_id(variant)
    shared_ids = sorted(set(baseline_rows) & set(variant_rows))
    out: dict[str, PairedBootstrapResult] = {}
    for metric in metrics:
        variant_scores = [_safe_metric(variant_rows[eid], metric) for eid in shared_ids]
        baseline_scores = [_safe_metric(baseline_rows[eid], metric) for eid in shared_ids]
        out[metric] = paired_bootstrap_ci(variant_scores, baseline_scores, seed=seed)
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_per_example_csv(
    path: Path,
    outputs: Sequence[VariantOutput],
    stage5_run: Stage5RunData | None = None,
    filter_pipeline: PostFilterPipeline | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for output in outputs:
        for row in output.result.per_example:
            eid = str(row.get("example_id", ""))
            decomposition = _decomposition_for(
                eid,
                stage5_run.decomposition_cache if stage5_run is not None else None,
            )
            rows.append(
                {
                    "variant": output.name,
                    **row,
                    "subclaim_count": len(_decomposition_subclaims(decomposition))
                    if decomposition is not None
                    else "",
                    "subclaims": _subclaims_json(decomposition)
                    if decomposition is not None
                    else "",
                    "aggregation_strategy": _decomposition_strategy(decomposition),
                }
            )

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bootstrap_markdown(
    path: Path,
    bootstrap: Mapping[str, PairedBootstrapResult],
    *,
    variant_name: str,
    baseline_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Paired Bootstrap: {variant_name} vs {baseline_name}",
        "",
        f"| Metric | {variant_name} Mean | {baseline_name} Mean | Diff | 95% CI | p-value |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, result in bootstrap.items():
        lines.append(
            f"| {metric} | {result.mean_a:.4f} | {result.mean_b:.4f} | "
            f"{result.mean_diff:.4f} | [{result.ci_low:.4f}, {result.ci_high:.4f}] | "
            f"{result.p_value_two_sided:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decomposition_for(
    example_id: str,
    decompositions: Mapping[str, Any] | None,
) -> Any | None:
    if decompositions is None:
        return None
    return decompositions.get(str(example_id))


def _decomposition_subclaims(decomposition: Any | None) -> list[Any]:
    if decomposition is None:
        return []
    raw = getattr(decomposition, "subclaims", [])
    return list(raw) if raw is not None else []


def _decomposition_strategy(decomposition: Any | None) -> str:
    strategy = getattr(decomposition, "aggregation", None)
    value = getattr(strategy, "value", strategy)
    return str(value) if value is not None else ""


def _subclaims_json(decomposition: Any | None) -> str:
    subclaims = [
        {
            "text": str(getattr(subclaim, "text", "")),
            "importance": float(getattr(subclaim, "importance", 0.0)),
        }
        for subclaim in _decomposition_subclaims(decomposition)
    ]
    return json.dumps(subclaims, ensure_ascii=False)


def stage5_facet_label(
    example_id: str,
    is_multi_facet: Any,
    decompositions: Mapping[str, Any] | None,
) -> str:
    """Return single/multi facet using benchmark labels, then decomposition fallback."""
    if isinstance(is_multi_facet, bool):
        return "multi" if is_multi_facet else "single"
    decomposition = _decomposition_for(example_id, decompositions)
    subclaim_count = len(_decomposition_subclaims(decomposition))
    if subclaim_count > 0:
        return "multi" if subclaim_count > 1 else "single"
    return "unknown"


def plot_bootstrap_comparisons(
    outputs: Mapping[str, VariantOutput],
    out_dir: Path,
) -> None:
    if "V1" not in outputs or "V2" not in outputs:
        return
    bootstrap = compute_bootstrap(
        outputs["V0"].result,
        outputs["V1"].result,
        seed=42,
    )
    write_bootstrap_markdown(
        out_dir / "bootstrap_v1_vs_v0.md",
        bootstrap,
        variant_name="V1",
        baseline_name="V0",
    )
    bootstrap_v2 = compute_bootstrap(
        outputs["V0"].result,
        outputs["V2"].result,
        seed=42,
    )
    write_bootstrap_markdown(
        out_dir / "bootstrap_v2_vs_v0.md",
        bootstrap_v2,
        variant_name="V2",
        baseline_name="V0",
    )


def build_stage5_stratified_metrics(
    outputs_by_name: Mapping[str, VariantOutput],
    decompositions: Mapping[str, Any] | None,
    *,
    metrics: Sequence[str] = ("recall@10", "ndcg@10"),
) -> list[dict[str, Any]]:
    """Build thesis table: V1/V2 metrics by single/multi facet bucket."""
    rows: list[dict[str, Any]] = []
    for variant_name in ("V1", "V2"):
        output = outputs_by_name.get(variant_name)
        if output is None:
            continue
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in output.result.per_example:
            if row.get("skipped", False):
                continue
            example_id = str(row.get("example_id", ""))
            facet = stage5_facet_label(
                example_id,
                row.get("is_multi_facet"),
                decompositions,
            )
            if facet in {"single", "multi"}:
                groups[facet].append(row)

        for facet in ("single", "multi"):
            facet_rows = groups.get(facet, [])
            out: dict[str, Any] = {
                "variant": variant_name,
                "facet": facet,
                "n": len(facet_rows),
            }
            for metric in metrics:
                values = [_safe_metric(row, metric) for row in facet_rows]
                out[metric] = sum(values) / len(values) if values else 0.0
            rows.append(out)
    return rows


def build_stage5_subclaim_histogram(
    decompositions: Mapping[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for decomposition in decompositions.values():
        counts[str(len(_decomposition_subclaims(decomposition)))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def build_stage5_aggregation_heatmap(
    v2_output: VariantOutput,
    *,
    metrics: Sequence[str] = ("recall@10", "ndcg@10", "mrr"),
    strategy: str = "WEIGHTED",
) -> list[dict[str, Any]]:
    row: dict[str, Any] = {"aggregation_strategy": strategy}
    for metric in metrics:
        row[metric] = float(v2_output.result.overall.get(metric, 0.0))
    return [row]


def _example_by_id(examples: Sequence[BenchmarkExample]) -> dict[str, BenchmarkExample]:
    return {str(example.example_id): example for example in examples}


def select_stage5_qualitative_examples(
    examples: Sequence[BenchmarkExample],
    decompositions: Mapping[str, Any],
    aggregate_cache: Mapping[str, Sequence[Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    by_id = _example_by_id(examples)
    selected: list[dict[str, Any]] = []
    for example_id in sorted(decompositions):
        example = by_id.get(str(example_id))
        if example is None:
            continue
        decomposition = decompositions[example_id]
        facet = stage5_facet_label(
            str(example_id),
            example.is_multi_facet,
            decompositions,
        )
        results = list(aggregate_cache.get(str(example_id), []))
        if facet != "multi" or not results:
            continue
        selected.append(
            {
                "example_id": str(example_id),
                "query": example.query_text,
                "subclaims": [
                    {
                        "text": str(getattr(subclaim, "text", "")),
                        "importance": float(getattr(subclaim, "importance", 0.0)),
                    }
                    for subclaim in _decomposition_subclaims(decomposition)
                ],
                "retrieved_papers": [
                    {
                        "paper_id": str(getattr(result, "paper_id", "")),
                        "title": str(getattr(result, "title", "")),
                        "score": float(getattr(result, "score", 0.0)),
                    }
                    for result in results[:5]
                ],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_stage5_qualitative_markdown(
    path: Path,
    examples: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 5 Qualitative Multi-Facet Examples",
        "",
        "| Example | Sentence / query | Subclaims | Retrieved papers |",
        "|---|---|---|---|",
    ]
    for item in examples:
        subclaims = "<br>".join(
            f"{idx}. {subclaim['text']} ({subclaim['importance']:.2f})"
            for idx, subclaim in enumerate(item.get("subclaims", []), start=1)
        )
        papers = "<br>".join(
            f"{idx}. {paper['title']} [{paper['paper_id']}]"
            for idx, paper in enumerate(item.get("retrieved_papers", []), start=1)
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(str(item.get("example_id", ""))),
                    _markdown_cell(str(item.get("query", ""))),
                    _markdown_cell(subclaims),
                    _markdown_cell(papers),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage5_analysis(
    *,
    config: Stage4Config,
    examples: Sequence[BenchmarkExample],
    outputs_by_name: Mapping[str, VariantOutput],
    stage5_run: Stage5RunData,
) -> dict[str, str]:
    v2_output = outputs_by_name["V2"]
    stratified = build_stage5_stratified_metrics(
        outputs_by_name,
        stage5_run.decomposition_cache,
    )
    histogram = build_stage5_subclaim_histogram(stage5_run.decomposition_cache)
    heatmap = build_stage5_aggregation_heatmap(v2_output)
    qualitative = select_stage5_qualitative_examples(
        examples,
        stage5_run.decomposition_cache,
        stage5_run.aggregate_cache,
    )

    analysis_path = config.output_dir / "stage5_analysis.json"
    qualitative_path = config.assets_dir / "stage5_qualitative_examples.md"
    write_json(
        analysis_path,
        {
            "stratified_v1_v2": stratified,
            "subclaim_histogram": histogram,
            "aggregation_heatmap": heatmap,
            "qualitative_examples": qualitative,
        },
    )
    write_stage5_qualitative_markdown(qualitative_path, qualitative)
    generate_stage5_visuals(
        config.assets_dir,
        stratified=stratified,
        histogram=histogram,
        heatmap=heatmap,
    )
    return {
        "analysis_json": str(analysis_path),
        "qualitative_examples_md": str(qualitative_path),
        "stratified_bar": str(config.assets_dir / "stage5_v1_v2_stratified_bar.png"),
        "aggregation_heatmap": str(config.assets_dir / "stage5_aggregation_heatmap.png"),
        "subclaim_histogram": str(config.assets_dir / "stage5_subclaim_histogram.png"),
    }


def write_outputs(
    config: Stage4Config,
    *,
    examples: Sequence[BenchmarkExample],
    num_papers: int,
    elapsed_s: float,
    outputs: Sequence[VariantOutput],
    stage5_run: Stage5RunData | None = None,
    filter_pipeline: PostFilterPipeline | None = None,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {output.name: output.result.overall for output in outputs}
    outputs_by_name = {output.name: output for output in outputs}
    bootstrap_v1_vs_v0 = compute_bootstrap(
        outputs_by_name["V0"].result,
        outputs_by_name["V1"].result,
        seed=config.seed,
    )
    bootstrap_json = {metric: asdict(result) for metric, result in bootstrap_v1_vs_v0.items()}
    bootstrap_v2_vs_v1 = None
    bootstrap_v2_vs_v0 = None
    if "V2" in outputs_by_name:
        bootstrap_v2_vs_v1 = compute_bootstrap(
            outputs_by_name["V1"].result,
            outputs_by_name["V2"].result,
            seed=config.seed,
        )
        bootstrap_v2_vs_v0 = compute_bootstrap(
            outputs_by_name["V0"].result,
            outputs_by_name["V2"].result,
            seed=config.seed,
        )

    write_json(
        config.output_dir / "metrics.json",
        {
            "config": {
                "source": config.source,
                "jsonl_path": str(config.jsonl_path) if config.jsonl_path else None,
                "max_examples": config.max_examples,
                "top_k": config.top_k,
                "candidate_k": config.candidate_k,
                "seed": config.seed,
                "hide_fraction": config.hide_fraction,
                "min_refs": config.min_refs,
                "min_abstract_chars": config.min_abstract_chars,
                "candidates_per_subclaim": config.candidates_per_subclaim,
                "model_name": config.model_name,
                "include_v2": config.include_v2,
                "decomposer_model": config.decomposer_model,
            },
            "n_examples": len(examples),
            "n_indexed_papers": num_papers,
            "elapsed_s": elapsed_s,
            "variants": metrics,
            "bootstrap_v1_vs_v0": bootstrap_json,
            "bootstrap_v2_vs_v1": {
                metric: asdict(result)
                for metric, result in (bootstrap_v2_vs_v1 or {}).items()
            },
            "bootstrap_v2_vs_v0": {
                metric: asdict(result)
                for metric, result in (bootstrap_v2_vs_v0 or {}).items()
            },
        },
    )
    write_json(config.output_dir / "bootstrap_v1_vs_v0.json", bootstrap_json)
    write_bootstrap_markdown(
        config.output_dir / "bootstrap_v1_vs_v0.md",
        bootstrap_v1_vs_v0,
        variant_name="V1",
        baseline_name="V0",
    )
    if bootstrap_v2_vs_v1 is not None:
        write_json(
            config.output_dir / "bootstrap_v2_vs_v1.json",
            {metric: asdict(result) for metric, result in bootstrap_v2_vs_v1.items()},
        )
        write_bootstrap_markdown(
            config.output_dir / "bootstrap_v2_vs_v1.md",
            bootstrap_v2_vs_v1,
            variant_name="V2",
            baseline_name="V1",
        )
    if bootstrap_v2_vs_v0 is not None:
        write_json(
            config.output_dir / "bootstrap_v2_vs_v0.json",
            {metric: asdict(result) for metric, result in bootstrap_v2_vs_v0.items()},
        )
        write_bootstrap_markdown(
            config.output_dir / "bootstrap_v2_vs_v0.md",
            bootstrap_v2_vs_v0,
            variant_name="V2",
            baseline_name="V0",
        )
    stage5_artifacts: dict[str, str] = {}
    if stage5_run is not None and "V2" in outputs_by_name:
        stage5_artifacts = write_stage5_analysis(
            config=config,
            examples=examples,
            outputs_by_name=outputs_by_name,
            stage5_run=stage5_run,
        )
    write_json(
        config.output_dir / "summary.json",
        {
            "n_examples": len(examples),
            "n_evaluated": {output.name: output.result.n_evaluated for output in outputs},
            "output_dir": str(config.output_dir),
            "assets_dir": str(config.assets_dir),
            "stage5_artifacts": stage5_artifacts,
        },
    )
    write_per_example_csv(config.output_dir / "per_example.csv", outputs, stage5_run)
    generate_visuals(config.assets_dir, outputs)


def _metric_values(result: EvaluationResult, metric: str) -> list[float]:
    return [
        _safe_metric(row, metric)
        for row in result.per_example
        if not row.get("skipped", False)
    ]


def generate_visuals(
    assets_dir: Path,
    outputs: Sequence[VariantOutput],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional plotting dependency
        logger.warning("Skipping plots because matplotlib is unavailable: %s", e)
        return

    assets_dir.mkdir(parents=True, exist_ok=True)
    metric_names = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    variant_names = [output.name for output in outputs]

    plt.figure(figsize=(10, 6))
    x = list(range(len(variant_names)))
    width = 0.18
    for offset, metric in enumerate(metric_names):
        values = [output.result.overall.get(metric, 0.0) for output in outputs]
        positions = [i + (offset - 1.5) * width for i in x]
        plt.bar(positions, values, width=width, label=metric)
    plt.xticks(x, variant_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Stage 4 Retrieval Metrics Across Variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "stage4_metrics_bar.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [
            _metric_values(outputs[0].result, "recall@10"),
            _metric_values(outputs[-1].result, "recall@10"),
        ],
        tick_labels=[outputs[0].name, outputs[-1].name],
    )
    plt.ylabel("Per-query Recall@10")
    plt.title("Stage 4 Recall@10 Distribution")
    plt.tight_layout()
    plt.savefig(assets_dir / "stage4_recall10_boxplot.png")
    plt.close()


def generate_stage5_visuals(
    assets_dir: Path,
    *,
    stratified: Sequence[Mapping[str, Any]],
    histogram: Mapping[str, int],
    heatmap: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional plotting dependency
        logger.warning("Skipping Stage 5 plots because matplotlib is unavailable: %s", e)
        return

    assets_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["recall@10", "ndcg@10"]
    facets = ["single", "multi"]
    variants = ["V1", "V2"]
    lookup = {
        (str(row.get("variant")), str(row.get("facet"))): row
        for row in stratified
    }
    labels = [f"{facet}\n{metric}" for facet in facets for metric in metrics]
    x = list(range(len(labels)))
    width = 0.36

    plt.figure(figsize=(9, 5))
    for variant_index, variant in enumerate(variants):
        values = [
            float(lookup.get((variant, facet), {}).get(metric, 0.0))
            for facet in facets
            for metric in metrics
        ]
        positions = [i + (variant_index - 0.5) * width for i in x]
        plt.bar(positions, values, width=width, label=variant)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Stage 5 V1 vs V2 by Facet Complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets_dir / "stage5_v1_v2_stratified_bar.png")
    plt.close()

    if heatmap:
        metric_names = [key for key in heatmap[0] if key != "aggregation_strategy"]
        strategy_names = [str(row.get("aggregation_strategy", "")) for row in heatmap]
        matrix = [
            [float(row.get(metric, 0.0)) for metric in metric_names]
            for row in heatmap
        ]
        plt.figure(figsize=(max(5, len(metric_names) * 1.5), max(2.5, len(strategy_names) * 0.8)))
        image = plt.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(image, fraction=0.046, pad=0.04)
        plt.xticks(range(len(metric_names)), metric_names)
        plt.yticks(range(len(strategy_names)), strategy_names)
        for row_index, row_values in enumerate(matrix):
            for col_index, value in enumerate(row_values):
                plt.text(col_index, row_index, f"{value:.3f}", ha="center", va="center")
        plt.title("Stage 5 Aggregation Strategy Comparison")
        plt.tight_layout()
        plt.savefig(assets_dir / "stage5_aggregation_heatmap.png")
        plt.close()

    if histogram:
        counts = sorted((int(k), int(v)) for k, v in histogram.items())
        xs = [item[0] for item in counts]
        ys = [item[1] for item in counts]
        plt.figure(figsize=(7, 4.5))
        plt.bar(xs, ys, width=0.75)
        plt.xticks(xs)
        plt.xlabel("# subclaims per sentence")
        plt.ylabel("Sentences")
        plt.title("Stage 5 Subclaim Count Distribution")
        plt.tight_layout()
        plt.savefig(assets_dir / "stage5_subclaim_histogram.png")
        plt.close()


def print_summary(outputs: Sequence[VariantOutput], output_dir: Path, assets_dir: Path) -> None:
    print("\nStage 4 results")
    print("-" * 72)
    print(f"{'Variant':<8} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>10} {'nDCG@10':>10}")
    print("-" * 72)
    for output in outputs:
        overall = output.result.overall
        print(
            f"{output.name:<8} "
            f"{overall.get('recall@5', 0.0):>10.4f} "
            f"{overall.get('recall@10', 0.0):>10.4f} "
            f"{overall.get('mrr', 0.0):>10.4f} "
            f"{overall.get('ndcg@10', 0.0):>10.4f}"
        )
    print("-" * 72)
    print(f"Outputs: {output_dir}")
    print(f"Figures: {assets_dir}")


def run(config: Stage4Config) -> int:
    if config.candidate_k < config.top_k:
        print("ERROR: --candidate-k must be greater than or equal to --top-k", flush=True)
        return 1

    start = time.perf_counter()
    try:
        examples = load_examples(config)
        print(f"Loaded {len(examples)} examples from {config.source}", flush=True)

        print(
            f"Initializing Qdrant-backed hybrid retriever (prefetch={config.candidate_k})...",
            flush=True,
        )
        retriever, num_papers = initialize_retriever(prefetch_limit=config.candidate_k)
        print(f"Retriever ready ({num_papers} papers indexed)", flush=True)

        print(f"Loading reranker model: {config.model_name}", flush=True)
        reranker = initialize_reranker(config.model_name)
        print("Reranker ready", flush=True)

        decomposer = None
        if config.include_v2:
            cache_path = config.output_dir / "decomposition_cache.json"
            print(
                f"Initializing claim decomposer: {config.decomposer_model} "
                f"(disk cache: {cache_path})",
                flush=True,
            )
            decomposer = initialize_decomposer(config.decomposer_model, cache_path=cache_path)
            print("Claim decomposer ready", flush=True)

        filter_pipeline = PostFilterPipeline()

        outputs, stage5_run = evaluate_variants(
            examples,
            retriever,
            reranker,
            top_k=config.top_k,
            candidate_k=config.candidate_k,
            candidates_per_subclaim=config.candidates_per_subclaim,
            decomposer=decomposer,
            filter_pipeline=filter_pipeline,
        )
        elapsed_s = time.perf_counter() - start
        write_outputs(
            config,
            examples=examples,
            num_papers=num_papers,
            elapsed_s=elapsed_s,
            outputs=outputs,
            stage5_run=stage5_run,
            filter_pipeline=filter_pipeline,
        )
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        logger.exception("Stage 4 experiment failed")
        return 1

    print_summary(outputs, config.output_dir, config.assets_dir)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> Stage4Config:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", choices=("db", "jsonl"), default="db")
    parser.add_argument("--jsonl-path", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help=(
            "Candidates pulled from hybrid retrieval before reranking. Set high "
            "enough to give the reranker real material to work on; "
            "see scripts/diagnose_recall.py for the recall vs K curve."
        ),
    )
    parser.add_argument(
        "--candidates-per-subclaim",
        type=int,
        default=30,
        help=(
            "Per-subclaim retrieval budget for V2 (decomposition). Smaller than "
            "candidate_k because each query yields ~3-5 subclaims and the "
            "aggregator merges all of their candidate lists. The dominant CPU "
            "cost is the cross-encoder reranker; this knob is the main one for "
            "controlling V2 wall time."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hide-fraction", type=float, default=0.3)
    parser.add_argument("--min-refs", type=int, default=2)
    parser.add_argument(
        "--min-abstract-chars",
        type=int,
        default=50,
        help=(
            "Minimum abstract length (chars) required for both source paper and "
            "held-out targets. Title-only documents make the title+abstract "
            "retrieval signal degenerate. Set to 0 to disable."
        ),
    )
    parser.add_argument("--model-name", default=app_config.RERANKER_MODEL)
    parser.add_argument(
        "--include-v2",
        action="store_true",
        help="Include V2 (Claim Decomposition) in evaluation",
    )
    parser.add_argument("--decomposer-model", default=app_config.DECOMPOSER_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "eval" / "stage4")
    parser.add_argument("--assets-dir", type=Path, default=Path(__file__).resolve().parents[2] / "thesis" / "assets")
    args = parser.parse_args(argv)

    return Stage4Config(
        source=args.source,
        jsonl_path=args.jsonl_path,
        max_examples=args.max_examples,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        candidates_per_subclaim=args.candidates_per_subclaim,
        seed=args.seed,
        hide_fraction=args.hide_fraction,
        min_refs=args.min_refs,
        min_abstract_chars=args.min_abstract_chars,
        model_name=args.model_name,
        include_v2=args.include_v2,
        decomposer_model=args.decomposer_model,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging(logging.WARNING)
    config = parse_args(argv)

    # Fail fast on missing required settings rather than blowing up mid-run.
    required = ["QDRANT_URL"]
    if config.source == "db":
        required.append("DB_URL")
    if config.include_v2:
        required.append("GEMINI_API_KEY")
    try:
        app_config.validate_required(*required)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2

    return run(config)


if __name__ == "__main__":
    sys.exit(main())
