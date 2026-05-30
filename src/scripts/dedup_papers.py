"""Merge duplicate paper records so retrieval and evaluation see one identity.

The corpus ingests from multiple sources (OpenAlex, Semantic Scholar), so the
same work often lands under several ``paper_id``s — e.g. id 26 and id 40398 are
both "A survey on Image Data Augmentation…", same DOI, different ``source``.
Duplicates corrupt the leave-citing-paper-out evaluation two ways:

* **Leak**: a duplicated *citing* paper's twin keeps its (identical) citation
  sentences in the candidate pool after the query's own ``citing_paper_id`` is
  excluded, so the gold is recovered by trivial string match.
* **Miss inflation**: a duplicated *cited* (gold) paper splits its evidence
  across ids; retrieval may surface the non-gold twin and score a false miss.

Merge rule (conservative — favours precision over recall of duplicates):
two papers merge iff they share a non-null **DOI**, or share both
**normalized_title and first_author** (non-empty). Linked papers are unioned
into components; the lowest ``paper_id`` is canonical. Title-only matches with
disagreeing authors are *not* merged (title collisions are real).

The merge only remaps ``citation_contexts.{cited,citing}_paper_id`` to the
canonical id — embeddings (keyed by ``context_id``) are untouched, and the
now-orphaned ``papers`` rows are left in place (harmless: no context references
them, so they never surface in retrieval). Rebuild the eval split afterwards.

Dry-run by default; pass ``--apply`` to write (inside one transaction).

Usage::

    python -m scripts.dedup_papers            # report only
    python -m scripts.dedup_papers --apply    # remap citation_contexts
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import typer
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.postgres.engine import get_session
from evaluation.dataset import DEFAULT_SPLIT_DIR
from utils.logger import logger

app = typer.Typer(add_completion=False)


class _UnionFind:
    """Minimal union-find; representative is always the smallest member id."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self._parent[hi] = lo  # keep the smaller id as root


def _build_merge_map(session: Session) -> dict[int, int]:
    """Return ``{duplicate_paper_id: canonical_paper_id}`` (canonical excluded)."""
    rows = session.execute(
        text(
            """
            SELECT paper_id, normalized_title, lower(first_author) AS fa,
                   lower(doi) AS doi
            FROM papers
            """
        )
    ).all()

    uf = _UnionFind()
    by_doi: dict[str, list[int]] = defaultdict(list)
    by_title_author: dict[tuple[str, str], list[int]] = defaultdict(list)
    for pid, ntitle, fa, doi in rows:
        uf.find(pid)  # register every paper
        if doi:
            by_doi[doi].append(pid)
        if ntitle and fa:
            by_title_author[(ntitle, fa)].append(pid)

    for group in (*by_doi.values(), *by_title_author.values()):
        first = group[0]
        for other in group[1:]:
            uf.union(first, other)

    merge_map: dict[int, int] = {}
    for pid, *_ in rows:
        canonical = uf.find(pid)
        if canonical != pid:
            merge_map[pid] = canonical
    return merge_map


def _split_members(merge_map: dict[int, int]) -> int:
    """Count how many split paper ids would be remapped (so the split is stale)."""
    path = DEFAULT_SPLIT_DIR / "split_42.json"
    if not path.exists():
        return 0
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = (
        data.get("train_paper_ids", [])
        + data.get("val_paper_ids", [])
        + data.get("test_paper_ids", [])
    )
    return sum(1 for i in ids if i in merge_map)


def _report(session: Session, merge_map: dict[int, int]) -> None:
    components = len(set(merge_map.values()))
    typer.echo("\nDuplicate-merge plan")
    typer.echo(f"  merge components (canonical papers) : {components}")
    typer.echo(f"  duplicate papers folded away         : {len(merge_map)}")
    if not merge_map:
        return

    dup_ids = list(merge_map)
    cited = session.execute(
        text(
            "SELECT count(*) FROM citation_contexts "
            "WHERE cited_paper_id = ANY(:ids)"
        ),
        {"ids": dup_ids},
    ).scalar_one()
    citing = session.execute(
        text(
            "SELECT count(*) FROM citation_contexts "
            "WHERE citing_paper_id = ANY(:ids)"
        ),
        {"ids": dup_ids},
    ).scalar_one()
    typer.echo(f"  contexts to remap (cited_paper_id)  : {cited}")
    typer.echo(f"  contexts to remap (citing_paper_id) : {citing}")
    typer.echo(f"  split_42 members remapped (stale)   : {_split_members(merge_map)}")

    typer.echo("\n  examples (duplicate -> canonical):")
    for dup, canon in list(merge_map.items())[:5]:
        title = session.execute(
            text("SELECT left(canonical_title, 60) FROM papers WHERE paper_id=:p"),
            {"p": canon},
        ).scalar_one_or_none()
        typer.echo(f"    {dup:>7} -> {canon:<7} | {title!r}")


def _write_backup(session: Session, merge_map: dict[int, int]) -> Path:
    """Snapshot the pre-remap FK state so the merge is fully reversible."""
    dup_ids = list(merge_map)
    affected = session.execute(
        text(
            "SELECT context_id, cited_paper_id, citing_paper_id "
            "FROM citation_contexts "
            "WHERE cited_paper_id = ANY(:ids) OR citing_paper_id = ANY(:ids)"
        ),
        {"ids": dup_ids},
    ).all()
    backup = {
        "merge_map": {str(o): n for o, n in merge_map.items()},
        "contexts": [
            {"context_id": cid, "cited_paper_id": cited, "citing_paper_id": citing}
            for cid, cited, citing in affected
        ],
    }
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = DEFAULT_SPLIT_DIR / f"dedup_backup_{ts}.json"
    path.write_text(json.dumps(backup), encoding="utf-8")
    typer.echo(f"  backup written: {path} ({len(affected)} contexts snapshotted)")
    return path


def _apply(session: Session, merge_map: dict[int, int]) -> None:
    """Remap citation_contexts FKs to canonical ids inside one transaction."""
    _write_backup(session, merge_map)
    session.execute(
        text(
            "CREATE TEMP TABLE paper_merge_map "
            "(old_id bigint PRIMARY KEY, new_id bigint NOT NULL) ON COMMIT DROP"
        )
    )
    session.execute(
        text("INSERT INTO paper_merge_map (old_id, new_id) VALUES (:o, :n)"),
        [{"o": o, "n": n} for o, n in merge_map.items()],
    )
    cited = session.execute(
        text(
            "UPDATE citation_contexts cc SET cited_paper_id = m.new_id "
            "FROM paper_merge_map m WHERE cc.cited_paper_id = m.old_id"
        )
    ).rowcount
    citing = session.execute(
        text(
            "UPDATE citation_contexts cc SET citing_paper_id = m.new_id "
            "FROM paper_merge_map m WHERE cc.citing_paper_id = m.old_id"
        )
    ).rowcount
    session.commit()
    logger.info("dedup applied — remapped cited=%d citing=%d", cited, citing)
    typer.echo(f"\nApplied: remapped cited={cited} citing={citing} contexts.")
    typer.echo("Rebuild the eval split now: python -m scripts.build_eval_split")


@app.command()
def main(
    apply: bool = typer.Option(
        False, "--apply", help="Write the remap (default: dry-run report only)."
    ),
) -> None:
    """Detect duplicate papers and (optionally) merge them."""
    session = get_session()
    try:
        merge_map = _build_merge_map(session)
        _report(session, merge_map)
        if not merge_map:
            typer.echo("\nNothing to merge.")
            return
        if apply:
            _apply(session, merge_map)
        else:
            typer.echo("\nDry-run only. Re-run with --apply to write.")
    finally:
        session.close()


if __name__ == "__main__":
    app()
