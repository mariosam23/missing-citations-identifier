"""Evaluation runner: variant × split → Report.

Performance optimisations over the naive per-query loop:

1. **Batch embeddings** — all query sentences are encoded in one GPU call
   via ``encode_texts`` before the query loop starts.
2. **Per-thread DB sessions** — when ``workers > 1`` each thread gets its
   own SQLAlchemy ``Session`` and ``Variant`` instance so queries hit
   Postgres in parallel without sharing a single connection.
3. **Reduced retrieval depth** — eval variants default to ``top_n=200``
   instead of 1000; the top-20 paper ranking is insensitive to contexts
   beyond position ~200.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from evaluation.dataset import EvalQuery, Split, materialise_queries
from evaluation.metrics import compute_all_metrics
from evaluation.report import Report
from utils.logger import logger


class Variant(Protocol):
    """Pluggable retrieval strategy for evaluation."""

    name: str

    def candidates(
        self,
        query: str,
        *,
        target_year: int | None,
        exclude_citing_paper_id: int,
        top_k: int,
        query_embedding: np.ndarray | None = None,
    ) -> list[int]: ...


KS: tuple[int, ...] = (1, 5, 10, 20)


class EvalRunner:
    """Run a variant over a split and produce a report."""

    def __init__(
        self,
        variant: Variant,
        split: Split,
        session: Session,
        *,
        variant_factory: Callable[[Session], Variant] | None = None,
        split_name: str = "test",
        target_year: int | None = None,
        top_k: int = 20,
        workers: int = 1,
        require_reachable: bool = True,
    ) -> None:
        self._variant = variant
        self._split = split
        self._session = session
        self._variant_factory = variant_factory
        self._split_name = split_name
        self._target_year = target_year
        self._top_k = top_k
        self._workers = max(1, workers)
        self._require_reachable = require_reachable

    def run(self) -> Report:
        """Execute evaluation and return a populated report."""
        paper_ids = (
            self._split.test_paper_ids
            if self._split_name == "test"
            else self._split.val_paper_ids
        )

        queries, unreachable_skipped = materialise_queries(
            self._session,
            paper_ids,
            target_year=self._target_year,
            require_reachable=self._require_reachable,
        )

        if not queries:
            logger.warning("no queries materialised — empty report")
            report = Report(
                variant_name=self._variant.name,
                split_name=self._split_name,
                num_citing_papers=len(paper_ids),
                num_queries=0,
                num_unreachable_skipped=unreachable_skipped,
                require_reachable=self._require_reachable,
                target_year=self._target_year,
                top_k=self._top_k,
            )
            report.compute_aggregates()
            return report

        logger.info(
            "running %s over %d queries (split=%s, top_k=%d, workers=%d)",
            self._variant.name,
            len(queries),
            self._split_name,
            self._top_k,
            self._workers,
        )

        embeddings = self._batch_embed(queries)
        t0 = time.perf_counter()

        if self._workers <= 1:
            all_metrics = self._eval_sequential(queries, embeddings)
        else:
            all_metrics = self._eval_parallel(queries, embeddings)

        elapsed = time.perf_counter() - t0
        logger.info(
            "evaluation complete in %.1fs (%.3fs/query)",
            elapsed,
            elapsed / len(queries),
        )

        per_query: dict[str, list[float]] = {}
        for m in all_metrics:
            for key, value in m.items():
                per_query.setdefault(key, []).append(value)

        report = Report(
            variant_name=self._variant.name,
            split_name=self._split_name,
            num_citing_papers=len(paper_ids),
            num_queries=len(queries),
            num_unreachable_skipped=unreachable_skipped,
            require_reachable=self._require_reachable,
            target_year=self._target_year,
            top_k=self._top_k,
            per_query=per_query,
        )
        report.compute_aggregates()
        return report

    # ------------------------------------------------------------------

    def _batch_embed(self, queries: list[EvalQuery]) -> list[np.ndarray]:
        from pipeline.embedding.embedder import encode_texts

        texts = [q.sentence for q in queries]
        logger.info("batch-encoding %d queries on GPU...", len(texts))
        t0 = time.perf_counter()
        matrix = encode_texts(texts, show_progress_bar=True, is_query=True)
        elapsed = time.perf_counter() - t0
        logger.info(
            "batch-encoding done in %.1fs (%.0f queries/s)",
            elapsed,
            len(texts) / max(elapsed, 0.001),
        )
        return [matrix[i] for i in range(matrix.shape[0])]

    def _score_one(
        self,
        variant: Variant,
        query: EvalQuery,
        embedding: np.ndarray,
    ) -> dict[str, float]:
        ranked = variant.candidates(
            query.sentence,
            target_year=self._target_year,
            exclude_citing_paper_id=query.citing_paper_id,
            top_k=self._top_k,
            query_embedding=embedding,
        )
        return compute_all_metrics(ranked, query.gold_paper_id, ks=KS)

    def _eval_sequential(
        self,
        queries: list[EvalQuery],
        embeddings: list[np.ndarray],
    ) -> list[dict[str, float]]:
        return [
            self._score_one(self._variant, q, emb)
            for q, emb in tqdm(
                zip(queries, embeddings, strict=True),
                total=len(queries),
                desc=self._variant.name,
                unit="q",
            )
        ]

    def _eval_parallel(
        self,
        queries: list[EvalQuery],
        embeddings: list[np.ndarray],
    ) -> list[dict[str, float]]:
        """Evaluate in parallel with per-thread DB sessions."""
        if self._variant_factory is None:
            raise ValueError(
                "variant_factory is required when workers > 1 "
                "(each thread needs its own Session + Variant)"
            )

        from database.postgres.engine import get_session

        n = len(queries)
        results: list[dict[str, float] | None] = [None] * n
        counter = itertools.count()
        progress = tqdm(total=n, desc=self._variant.name, unit="q")

        def worker() -> None:
            session = get_session()
            variant = self._variant_factory(session)  # type: ignore[misc]
            try:
                while True:
                    i = next(counter)
                    if i >= n:
                        break
                    results[i] = self._score_one(
                        variant, queries[i], embeddings[i]
                    )
                    progress.update(1)
            finally:
                session.close()

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(self._workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        progress.close()

        return [r for r in results if r is not None]
