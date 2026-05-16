"""Eval split construction and loading.

The split unit is the **citing paper** (not individual contexts): splitting
on contexts would leak ~30% of the signal because contexts from the same
paper share topic, style, and cited-author distributions.

Split storage: ``data/eval/split_{seed}.json`` — plain JSON checked into
git so reviewers reproduce the same numbers.

Each test/val query is a ``(sentence, gold_paper_id, citing_paper_id)``
triple derived from a ``citation_contexts`` row where
``cited_paper_id IS NOT NULL``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from utils.logger import logger

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.80
DEFAULT_VAL_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.10

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPLIT_DIR = _PROJECT_ROOT / "data" / "eval"


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One evaluation query: a sentence whose gold answer is known."""

    sentence: str
    gold_paper_id: int
    citing_paper_id: int
    citing_year: int | None


@dataclass(slots=True)
class Split:
    """A frozen train/val/test partition of citing papers + derived queries."""

    seed: int
    train_paper_ids: list[int] = field(default_factory=list)
    val_paper_ids: list[int] = field(default_factory=list)
    test_paper_ids: list[int] = field(default_factory=list)

    # Queries are built lazily from the DB; not stored in JSON.
    val_queries: list[EvalQuery] = field(default_factory=list)
    test_queries: list[EvalQuery] = field(default_factory=list)


# -----------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------

def build_split(
    session: Session,
    *,
    seed: int = DEFAULT_SEED,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> Split:
    """Create a deterministic train/val/test split of citing papers.

    Only papers with at least one resolved citation context
    (``cited_paper_id IS NOT NULL``) are included.
    """
    rows = session.execute(
        text(
            """
            SELECT DISTINCT citing_paper_id
            FROM citation_contexts
            WHERE cited_paper_id IS NOT NULL
              AND citing_paper_id IS NOT NULL
            ORDER BY citing_paper_id
            """
        )
    ).all()
    all_ids = [int(r[0]) for r in rows]
    logger.info(
        "build_split: %d citing papers with resolved citations", len(all_ids)
    )

    rng = random.Random(seed)
    rng.shuffle(all_ids)

    n = len(all_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_ids = sorted(all_ids[:n_test])
    val_ids = sorted(all_ids[n_test : n_test + n_val])
    train_ids = sorted(all_ids[n_test + n_val :])

    logger.info(
        "split sizes: train=%d val=%d test=%d (seed=%d)",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        seed,
    )

    return Split(
        seed=seed,
        train_paper_ids=train_ids,
        val_paper_ids=val_ids,
        test_paper_ids=test_ids,
    )


# -----------------------------------------------------------------------
# Persist / load
# -----------------------------------------------------------------------

def save_split(split: Split, directory: Path = DEFAULT_SPLIT_DIR) -> Path:
    """Write ``split_{seed}.json`` to *directory*; return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"split_{split.seed}.json"
    payload = {
        "seed": split.seed,
        "train_paper_ids": split.train_paper_ids,
        "val_paper_ids": split.val_paper_ids,
        "test_paper_ids": split.test_paper_ids,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved split to %s", path)
    return path


def load_split(
    path: Path,
    *,
    strict: bool = True,
    session: Session | None = None,
) -> Split:
    """Load a split JSON and optionally verify member existence.

    When ``strict=True`` *and* ``session`` is provided, errors if any
    split member is missing from the DB. When ``strict=False``, missing
    members are silently dropped with a warning.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    split = Split(
        seed=raw["seed"],
        train_paper_ids=raw["train_paper_ids"],
        val_paper_ids=raw["val_paper_ids"],
        test_paper_ids=raw["test_paper_ids"],
    )

    if session is not None:
        _validate_split(split, session, strict=strict)

    return split


def _validate_split(
    split: Split, session: Session, *, strict: bool
) -> None:
    """Check that all split paper IDs still exist in the DB."""
    all_ids = (
        split.train_paper_ids
        + split.val_paper_ids
        + split.test_paper_ids
    )
    if not all_ids:
        return

    rows = session.execute(
        text(
            """
            SELECT DISTINCT citing_paper_id
            FROM citation_contexts
            WHERE citing_paper_id = ANY(:ids)
              AND cited_paper_id IS NOT NULL
            """
        ),
        {"ids": all_ids},
    ).all()
    existing = {int(r[0]) for r in rows}
    missing = set(all_ids) - existing

    if missing:
        msg = f"{len(missing)} split paper(s) not found in DB"
        if strict:
            raise ValueError(msg)
        logger.warning("%s — dropping them (strict=False)", msg)
        split.train_paper_ids = [
            i for i in split.train_paper_ids if i in existing
        ]
        split.val_paper_ids = [
            i for i in split.val_paper_ids if i in existing
        ]
        split.test_paper_ids = [
            i for i in split.test_paper_ids if i in existing
        ]


# -----------------------------------------------------------------------
# Query materialisation
# -----------------------------------------------------------------------

def materialise_queries(
    session: Session,
    paper_ids: list[int],
    *,
    target_year: int | None = None,
    require_reachable: bool = True,
) -> tuple[list[EvalQuery], int]:
    """Fetch eval queries for a set of citing papers.

    Each resolved citation context yields one query. When
    ``target_year`` is set, only contexts from papers whose
    ``citing_year == target_year + 1`` are included (Phase 8 temporal
    evaluation).

    When ``require_reachable`` is True (default), drops queries whose
    gold paper has no other citing paper in the corpus — retrieval
    cannot surface them because the citing paper's own contexts are
    excluded from the candidate pool. Returns the count of dropped
    queries so the runner can report it.
    """
    if not paper_ids:
        return [], 0

    base_sql = """
        SELECT cc.sentence_without_markers,
               cc.cited_paper_id,
               cc.citing_paper_id,
               cc.citing_year,
               EXISTS (
                 SELECT 1
                 FROM citation_contexts c2
                 WHERE c2.cited_paper_id = cc.cited_paper_id
                   AND c2.citing_paper_id IS DISTINCT FROM cc.citing_paper_id
               ) AS gold_has_other_citer
        FROM citation_contexts cc
        WHERE cc.citing_paper_id = ANY(:paper_ids)
          AND cc.cited_paper_id IS NOT NULL
          AND cc.sentence_without_markers IS NOT NULL
          AND cc.sentence_without_markers != ''
    """

    if target_year is not None:
        base_sql += (
            " AND cc.citing_year = :citing_year_filter"
        )

    base_sql += " ORDER BY cc.context_id"

    params: dict[str, int | list[int]] = {"paper_ids": paper_ids}
    if target_year is not None:
        params["citing_year_filter"] = target_year + 1

    rows = session.execute(text(base_sql), params).all()

    queries: list[EvalQuery] = []
    unreachable_skipped = 0
    for row in rows:
        if require_reachable and not row[4]:
            unreachable_skipped += 1
            continue
        queries.append(
            EvalQuery(
                sentence=row[0],
                gold_paper_id=int(row[1]),
                citing_paper_id=int(row[2]),
                citing_year=int(row[3]) if row[3] is not None else None,
            )
        )

    logger.info(
        "materialised %d queries from %d papers "
        "(skipped %d with unreachable gold, require_reachable=%s)",
        len(queries),
        len(paper_ids),
        unreachable_skipped,
        require_reachable,
    )
    return queries, unreachable_skipped
