"""Download a sample of open-access PDFs from OpenAlex into ``papers/``.

The sentence-level retrieval benchmark needs more source PDFs to produce a
statistically powerful comparison (3 PDFs gives ~50 rows; we want hundreds).
This script:

  1. Samples N OpenAlex Work IDs from the local Postgres ``papers`` table,
     prioritising papers that are most likely to have a fetchable open PDF
     (recent, well-cited, with a usable abstract).
  2. Queries OpenAlex once per paper to grab the open-access metadata.
  3. Extracts the best PDF URL from ``best_oa_location`` /
     ``primary_location`` / ``locations`` (in that order), preferring arXiv
     when available because arXiv URLs are stable and consistently free.
  4. Downloads with a polite delay between requests and a real User-Agent;
     skips files already on disk.
  5. Saves each PDF as ``<openalex_id>.pdf`` so downstream tooling can map
     the filename back to a corpus paper-id without re-querying OpenAlex.

The script writes a JSON diagnostics file alongside the PDFs that records
why each attempted paper was kept, skipped, or failed — useful for
reporting yield rate when expanding the corpus.

Usage
-----
    python scripts/fetch_papers_from_openalex.py --n 50
    python scripts/fetch_papers_from_openalex.py --n 100 --min-cited 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


logger = logging.getLogger("fetch_papers")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def sample_candidates(
    *,
    n: int,
    min_cited: int,
    seed: int,
    require_abstract_chars: int = 200,
) -> list[dict[str, Any]]:
    """Pull a candidate pool of papers from local Postgres.

    We over-sample by ~3x because PDF availability is roughly 30-50% even
    among well-cited OA papers. The downloader stops once it has accumulated
    ``n`` successful downloads.
    """
    from sqlalchemy import func, select

    from database.postgres.engine import get_session
    from database.postgres.tables.paper import Paper

    pool_target = max(n * 4, 200)
    stmt = (
        select(Paper.paperId, Paper.title, Paper.cited_by_count, Paper.publication_date)
        .where(Paper.paperId.is_not(None))
        .where(func.coalesce(func.length(Paper.abstract), 0) >= require_abstract_chars)
        .where(Paper.cited_by_count >= min_cited)
        .order_by(func.random())
        .limit(pool_target)
    )

    with get_session() as session:
        rows = session.execute(stmt).all()

    rng = random.Random(seed)
    candidates = [
        {
            "paper_id": str(pid),
            "title": title or "",
            "cited_by_count": int(cited or 0),
            "year": pub_date.year if pub_date is not None else None,
        }
        for pid, title, cited, pub_date in rows
    ]
    rng.shuffle(candidates)
    return candidates


def fetch_openalex_work(paper_id: str, *, mailto: str, api_key: str | None) -> dict[str, Any] | None:
    """Pull a single Work record from OpenAlex by ID. Returns ``None`` on failure."""
    url = f"https://api.openalex.org/works/{paper_id}"
    params: dict[str, str | int] = {"mailto": mailto}
    if api_key:
        params["api_key"] = api_key
    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.RequestException as exc:
        logger.warning("OpenAlex request failed for %s: %s", paper_id, exc)
        return None
    if resp.status_code != 200:
        logger.warning("OpenAlex returned %s for %s", resp.status_code, paper_id)
        return None
    return resp.json()


def best_pdf_url(work: dict[str, Any]) -> tuple[str | None, str]:
    """Pick the best PDF URL from an OpenAlex Work record.

    Returns ``(url, source_label)`` where ``source_label`` is one of
    ``"best_oa"``, ``"primary"``, ``"alt_location"``, ``"arxiv_id"``, or
    ``""`` when nothing usable was found. arXiv URLs are preferred because
    they're cached and don't 403 on bots.
    """
    # 1) explicit arXiv ID — synthesise the stable URL ourselves.
    arxiv_id = (work.get("ids") or {}).get("arxiv_id")
    if arxiv_id:
        # OpenAlex sometimes stores it as a URL.
        if isinstance(arxiv_id, str):
            stripped = arxiv_id.rsplit("/", 1)[-1]
            if stripped:
                return f"https://arxiv.org/pdf/{stripped}.pdf", "arxiv_id"

    locations = []
    best_oa = work.get("best_oa_location")
    if isinstance(best_oa, dict):
        locations.append(("best_oa", best_oa))
    primary = work.get("primary_location")
    if isinstance(primary, dict):
        locations.append(("primary", primary))
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            locations.append(("alt_location", loc))

    # Prefer locations whose source landing URL is arXiv.
    def _score(item: tuple[str, dict[str, Any]]) -> int:
        _, loc = item
        url = (loc.get("pdf_url") or "").lower()
        landing = (loc.get("landing_page_url") or "").lower()
        if "arxiv.org" in url or "arxiv.org" in landing:
            return 0  # most preferred
        if loc.get("pdf_url"):
            return 1
        return 2

    locations.sort(key=_score)
    for label, loc in locations:
        url = loc.get("pdf_url")
        if url:
            return url, label

    return None, ""


def download_pdf(url: str, dest: Path, *, timeout: int = 30) -> bool:
    """Download ``url`` to ``dest`` atomically. Returns True on success."""
    headers = {
        # arxiv.org returns 403 for default Python UAs; identify ourselves.
        "User-Agent": (
            "missing-citations-thesis/0.1 "
            "(academic research; mailto:research@example.invalid)"
        ),
        "Accept": "application/pdf,*/*;q=0.8",
    }
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as resp:
            if resp.status_code != 200:
                logger.warning("HTTP %s downloading %s", resp.status_code, url)
                return False
            content_type = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                # Some servers serve HTML landing pages with a redirect; reject.
                logger.warning("Non-PDF Content-Type %r at %s", content_type, url)
                return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            # PDFs always start with "%PDF". A tiny sniff catches HTML 200s.
            with tmp.open("rb") as f:
                head = f.read(5)
            if not head.startswith(b"%PDF"):
                logger.warning("Downloaded payload at %s is not a PDF", url)
                tmp.unlink(missing_ok=True)
                return False
            tmp.replace(dest)
            return True
    except requests.RequestException as exc:
        logger.warning("Download error for %s: %s", url, exc)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="Target number of PDFs to download.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "papers")
    parser.add_argument(
        "--min-cited",
        type=int,
        default=50,
        help="Filter local DB candidates to papers with at least this many citations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--polite-delay",
        type=float,
        default=2.0,
        help="Seconds between PDF download requests (be a good arxiv citizen).",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional JSON diagnostics output (defaults to <out>/_fetch_diagnostics.json).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Cap on candidates to attempt; default unbounded (until N succeed).",
    )
    args = parser.parse_args(argv)

    setup_logging()

    from utils.config import config

    if not config.OPEN_ALEX_EMAIL:
        logger.error("OPEN_ALEX_EMAIL is not set; OpenAlex polite-pool requires it.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = args.diagnostics or args.out_dir / "_fetch_diagnostics.json"

    logger.info("Sampling candidates from local DB (min_cited=%d) ...", args.min_cited)
    candidates = sample_candidates(
        n=args.n,
        min_cited=args.min_cited,
        seed=args.seed,
    )
    logger.info("Got %d candidates", len(candidates))
    if args.max_attempts:
        candidates = candidates[: args.max_attempts]

    succeeded = 0
    attempted = 0
    counters: Counter[str] = Counter()
    per_paper: list[dict[str, Any]] = []

    for cand in candidates:
        if succeeded >= args.n:
            break
        attempted += 1

        dest = args.out_dir / f"{cand['paper_id']}.pdf"
        if dest.exists():
            counters["already_on_disk"] += 1
            per_paper.append({**cand, "status": "already_on_disk", "path": str(dest)})
            continue

        work = fetch_openalex_work(
            cand["paper_id"],
            mailto=config.OPEN_ALEX_EMAIL,
            api_key=getattr(config, "OPEN_ALEX_API_KEY", None) or None,
        )
        if work is None:
            counters["openalex_lookup_failed"] += 1
            per_paper.append({**cand, "status": "openalex_lookup_failed"})
            continue

        pdf_url, source = best_pdf_url(work)
        if not pdf_url:
            counters["no_open_pdf_url"] += 1
            per_paper.append({**cand, "status": "no_open_pdf_url", "title": work.get("title")})
            continue

        time.sleep(args.polite_delay)

        ok = download_pdf(pdf_url, dest)
        if ok:
            counters[f"downloaded_{source}"] += 1
            succeeded += 1
            per_paper.append(
                {
                    **cand,
                    "status": "downloaded",
                    "pdf_url": pdf_url,
                    "source": source,
                    "path": str(dest),
                    "title": work.get("title"),
                }
            )
            logger.info(
                "[%d/%d] %s (%s) -> %s",
                succeeded,
                args.n,
                cand["paper_id"],
                source,
                dest.name,
            )
        else:
            counters["download_failed"] += 1
            per_paper.append(
                {
                    **cand,
                    "status": "download_failed",
                    "pdf_url": pdf_url,
                    "source": source,
                }
            )

    diag_path.write_text(
        json.dumps(
            {
                "target_n": args.n,
                "attempted": attempted,
                "succeeded": succeeded,
                "counters": dict(counters),
                "per_paper": per_paper,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 70)
    logger.info(
        "Done: %d/%d PDFs downloaded after %d candidates.",
        succeeded,
        args.n,
        attempted,
    )
    logger.info("Counters: %s", dict(counters))
    logger.info("Diagnostics: %s", diag_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
