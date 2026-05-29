"""Ingest corpus: discover → download (temp) → GROBID → delete PDF → DB.

Each work in ``openalex_works.jsonl`` is processed end-to-end in a single pass:

    OpenAlex Work
        → download PDF to a NamedTemporaryFile
        → GROBID processFulltextDocument → TEI XML (cached to disk)
        → delete temp PDF
        → parse_tei() → ParsedDocument
        → DB inserts: papers + source_documents + references + citation_contexts

PDFs are never kept on disk. Only the TEI XML is cached
(``data/corpus/tei/{openalex_id}.tei.xml``, ~50–200 KB each), which makes
re-runs after a DB wipe essentially free — GROBID is not called again for
documents whose TEI already exists.

GROBID + download is the slow leg (5–60 s per paper) and is farmed out to a
``ThreadPoolExecutor``. DB writes are **single-threaded**, batched every
``BATCH_DOCS`` documents, so we don't fight for sequence values.

Restartable: works whose OpenAlex ID already appears at
``source_documents.parse_status='ok'`` are skipped.

Usage:
    python -m scripts.parse_corpus
    python -m scripts.parse_corpus --limit 50 --workers 4
"""

from __future__ import annotations

import re
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from database.postgres.engine import get_session
from database.postgres.tables.citation_contexts import CitationContext
from database.postgres.tables.papers import Paper
from database.postgres.tables.references_ import Reference
from database.postgres.tables.source_documents import SourceDocument
from entities.parsed import ParsedDocument
from pipeline.parsing.grobid_client import (
    GrobidError,
    GrobidTimeoutError,
    process_fulltext,
)
from pipeline.parsing.tei_parser import parse_tei
from pipeline.resolution.semantic_scholar_client import SemanticScholarClient
from utils.config import config
from utils.logger import logger
from utils.regex_patterns import OPENALEX_ID_PATTERN

DEFAULT_INPUT = Path("data/corpus/openalex_works.jsonl")
DEFAULT_TEI_DIR = Path("data/corpus/tei")
BATCH_DOCS = 50
DEFAULT_WORKERS = 4
PDF_MAGIC = b"%PDF-"
MAX_PDF_BYTES = 50 * 1024 * 1024

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _extract_paper_id(work: dict[str, Any]) -> str | None:
    s2_id = work.get("paperId")
    if isinstance(s2_id, str) and s2_id:
        return f"S2:{s2_id}"
    raw = work.get("id") or ""
    m = OPENALEX_ID_PATTERN.search(raw)
    return m.group(1) if m else None


_ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/\d{7})",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"']+)")
_PMC_RE = re.compile(r"(PMC\d+)")


def _iter_locations(work: dict[str, Any]) -> list[dict[str, Any]]:
    """Best/primary first, then every entry in ``locations[]`` (dedup-able)."""
    out: list[dict[str, Any]] = []
    for key in ("best_oa_location", "primary_location"):
        loc = work.get(key)
        if isinstance(loc, dict):
            out.append(loc)
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            out.append(loc)
    return out


def _work_arxiv_id(work: dict[str, Any]) -> str | None:
    """Extract an arXiv id from any location's landing-page / pdf URL."""
    for loc in _iter_locations(work):
        for key in ("pdf_url", "landing_page_url"):
            val = loc.get(key)
            if isinstance(val, str):
                m = _ARXIV_RE.search(val)
                if m:
                    return m.group(1)
    return None


def _work_doi(work: dict[str, Any]) -> str | None:
    """Return the bare DOI (``10.xxxx/...``) from the OpenAlex record."""
    raw = (work.get("ids") or {}).get("doi") or work.get("doi")
    if not isinstance(raw, str):
        return None
    m = _DOI_RE.search(raw)
    return m.group(1) if m else None


def _pmc_pdf_url(work: dict[str, Any]) -> str | None:
    pmcid_raw = (work.get("ids") or {}).get("pmcid")
    if isinstance(pmcid_raw, str):
        m = _PMC_RE.search(pmcid_raw)
        if m:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{m.group(1)}/pdf/"
    return None


def _pdf_url_candidates(work: dict[str, Any]) -> list[str]:
    """Ordered, de-duplicated PDF URLs drawn from the OpenAlex record alone.

    Priority: arXiv direct (most reliable) → every ``locations[].pdf_url``
    (best/primary first) → PMC direct → ``open_access.oa_url`` (often a landing
    page, so last). The previous version only tried best/primary + oa_url, which
    are exactly the publisher gateway links that 403 or return HTML.
    """
    urls: list[str] = []

    def _add(url: str | None) -> None:
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)

    arxiv_id = _work_arxiv_id(work)
    if arxiv_id:
        _add(f"https://arxiv.org/pdf/{arxiv_id}")
    for loc in _iter_locations(work):
        _add(loc.get("pdf_url"))
    _add(_pmc_pdf_url(work))
    _add((work.get("open_access") or {}).get("oa_url"))
    
    # Semantic Scholar fallback
    s2_oa = work.get("openAccessPdf")
    if isinstance(s2_oa, dict):
        _add(s2_oa.get("url"))
    
    return urls


# Semantic Scholar is rate-limited to 1 RPS for the whole account; serialise
# every S2 call across the download threadpool through this lock + timestamp.
_S2_LOCK = threading.Lock()
_S2_MIN_INTERVAL_S = 1.1
_s2_next_allowed = 0.0


def _s2_pdf_url(
    s2_client: SemanticScholarClient, work: dict[str, Any]
) -> str | None:
    """Throttled (≤1 RPS) S2 ``openAccessPdf`` lookup by DOI then arXiv."""
    doi = _work_doi(work)
    arxiv_id = _work_arxiv_id(work)
    if not (doi or arxiv_id):
        return None

    global _s2_next_allowed
    with _S2_LOCK:
        wait = _s2_next_allowed - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return s2_client.get_open_access_pdf(doi=doi, arxiv_id=arxiv_id)
        except httpx.HTTPError as exc:
            logger.warning("s2 openAccessPdf lookup failed: %s", exc)
            return None
        finally:
            _s2_next_allowed = time.monotonic() + _S2_MIN_INTERVAL_S


def _unpaywall_pdf_url(
    http_client: httpx.Client, doi: str | None, email: str | None
) -> str | None:
    """Resolve a working OA PDF URL for a DOI via Unpaywall.

    Unpaywall aggregates OA copies across repositories and is the most
    reliable single source for "give me a PDF for this DOI". No per-second
    rate limit (≤100k/day with the ``email`` param), so it is safe to call
    from the download threadpool.
    """
    if not (doi and email):
        return None
    try:
        r = http_client.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=30.0,
        )
        if r.status_code != 200:
            return None
        loc = (r.json() or {}).get("best_oa_location") or {}
        url = loc.get("url_for_pdf") or loc.get("url")
        return url if isinstance(url, str) and url else None
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("unpaywall lookup failed for doi=%s: %s", doi, exc)
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    # Only retry transient transport problems (timeouts, connection resets).
    # 4xx/5xx come back via raise_for_status as HTTPStatusError and must NOT be
    # retried: a 403/404 publisher link never recovers, and retrying it 3× with
    # backoff made bulk recovery ~10× slower for no benefit.
    retry=retry_if_exception_type((httpx.TransportError,)),
    reraise=True,
)
def _http_get(client: httpx.Client, url: str) -> tuple[bytes, str]:
    r = client.get(url, follow_redirects=True, timeout=30.0)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "")


def _looks_like_pdf(content: bytes, ctype: str) -> bool:
    return content.startswith(PDF_MAGIC) or "application/pdf" in ctype.lower()


def _download_pdf(
    http_client: httpx.Client, urls: list[str]
) -> tuple[bytes | None, str | None]:
    """Try each URL in order; return (pdf_bytes, None) or (None, last_error)."""
    last_error: str | None = None
    for url in urls:
        try:
            content, ctype = _http_get(http_client, url)
        except httpx.HTTPError as exc:
            last_error = f"http:{exc}"
            continue
        if len(content) > MAX_PDF_BYTES:
            last_error = f"oversize:{len(content)}"
            continue
        if not _looks_like_pdf(content, ctype):
            last_error = f"not_pdf:{ctype[:64]}"
            continue
        return content, None
    return None, last_error


# ---------------------------------------------------------------------------
# Per-work ingestion (runs in a thread)
# ---------------------------------------------------------------------------

@dataclass
class WorkResult:
    openalex_id: str
    parsed: ParsedDocument | None
    tei_path: Path | None
    status: str          # 'ok' | 'timeout' | 'error' | 'no_pdf' | 'tei_parse_error'
    error: str | None


def _ingest_one(
    work: dict[str, Any],
    tei_dir: Path,
    http_client: httpx.Client,
    s2_client: SemanticScholarClient | None = None,
    unpaywall_email: str | None = None,
) -> WorkResult:
    """Download → GROBID → delete → parse TEI. Runs in a thread pool worker."""
    paper_id = _extract_paper_id(work) or ""
    tei_path = tei_dir / f"{paper_id}.tei.xml"

    # TEI cache hit: skip download + GROBID entirely.
    if tei_path.exists() and tei_path.stat().st_size > 0:
        try:
            tei_xml = tei_path.read_text(encoding="utf-8")
            parsed = parse_tei(
                tei_xml,
                openalex_id=paper_id,
                tei_path=tei_path,
            )
            return WorkResult(paper_id, parsed, tei_path, "ok", None)
        except Exception as exc:  # noqa: BLE001
            return WorkResult(paper_id, None, tei_path, "tei_parse_error", str(exc))

    # Try the OpenAlex-derived candidates first (free, no extra API calls).
    pdf_content, last_error = _download_pdf(http_client, _pdf_url_candidates(work))

    # Fallback 1: Unpaywall (free, DOI-keyed, no per-second limit). Most of the
    # failures are publisher-gateway 403s; Unpaywall often has a repository copy.
    if pdf_content is None and unpaywall_email:
        up_url = _unpaywall_pdf_url(http_client, _work_doi(work), unpaywall_email)
        if up_url:
            content, up_error = _download_pdf(http_client, [up_url])
            if content is not None:
                pdf_content = content
            else:
                last_error = f"unpaywall:{up_error}"

    # Fallback 2: Semantic Scholar openAccessPdf (throttled to 1 RPS; off by
    # default because the shared pool 429s readily — enable with --use-s2).
    if pdf_content is None and s2_client is not None:
        s2_url = _s2_pdf_url(s2_client, work)
        if s2_url:
            content, s2_error = _download_pdf(http_client, [s2_url])
            if content is not None:
                pdf_content = content
            else:
                last_error = f"s2:{s2_error}"

    if pdf_content is None:
        return WorkResult(
            paper_id, None, None, "no_pdf", last_error or "no url candidates"
        )

    # Write to a temp file so GROBID can POST it, then delete immediately.
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(pdf_content)
        del pdf_content  # release memory before GROBID call

        try:
            tei_xml = process_fulltext(tmp_path)
        except GrobidTimeoutError as exc:
            return WorkResult(paper_id, None, None, "timeout", str(exc))
        except GrobidError as exc:
            return WorkResult(paper_id, None, None, "error", str(exc))
        finally:
            tmp_path.unlink(missing_ok=True)

        tei_path.write_text(tei_xml, encoding="utf-8")
        parsed = parse_tei(
            tei_xml,
            openalex_id=paper_id,
            tei_path=tei_path,
        )
        return WorkResult(paper_id, parsed, tei_path, "ok", None)

    except Exception as exc:  # noqa: BLE001
        return WorkResult(paper_id, None, None, "tei_parse_error", str(exc))


# ---------------------------------------------------------------------------
# DB writers (single-threaded, called in the main thread)
# ---------------------------------------------------------------------------

def _quick_normalize(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()


def _upsert_paper(session: Session, parsed: ParsedDocument) -> Paper:
    paper_id = parsed.openalex_id
    if paper_id.startswith("S2:"):
        paper_url = f"https://www.semanticscholar.org/paper/{paper_id[3:]}"
        source = "semanticscholar"
    else:
        paper_url = f"https://openalex.org/{paper_id}"
        source = "openalex"

    existing = session.execute(
        select(Paper).where(Paper.url == paper_url)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    paper = Paper(
        canonical_title=parsed.canonical_title or paper_id,
        normalized_title=_quick_normalize(parsed.canonical_title or ""),
        authors={"list": parsed.authors} if parsed.authors else None,
        first_author=parsed.first_author,
        year=parsed.year,
        venue=parsed.venue,
        doi=parsed.doi,
        arxiv_id=parsed.arxiv_id,
        url=paper_url,
        source=source,
        abstract=parsed.abstract,
    )
    session.add(paper)
    session.flush()
    return paper


def _persist(
    session: Session, result: WorkResult
) -> tuple[int, int]:
    """Insert one successfully parsed document. Returns (n_refs, n_contexts)."""
    assert result.parsed is not None  # caller checks
    assert result.tei_path is not None
    parsed = result.parsed

    paper = _upsert_paper(session, parsed)

    session.add(
        SourceDocument(
            paper_id=paper.paper_id,
            source_path=None,  # PDF was deleted; tei_path is in raw_metadata
            source_type="pdf",
            parse_status="ok",
            raw_metadata={
                "tei_path": str(result.tei_path),
                "openalex_id": parsed.openalex_id,
            },
        )
    )
    session.flush()

    ref_id_by_key: dict[str, int] = {}
    for ex in parsed.references:
        ref = Reference(
            citing_paper_id=paper.paper_id,
            raw_reference_text=ex.raw_reference_text,
            ref_key=ex.ref_key,
            parsed_title=ex.parsed_title,
            parsed_authors=(
                {"list": ex.parsed_authors} if ex.parsed_authors else None
            ),
            parsed_first_author=ex.parsed_first_author,
            parsed_year=ex.parsed_year,
            parsed_venue=ex.parsed_venue,
            doi=ex.doi,
            arxiv_id=ex.arxiv_id,
        )
        session.add(ref)
        session.flush()
        ref_id_by_key[ex.ref_key] = ref.reference_id

    for ctx in parsed.contexts:
        session.add(
            CitationContext(
                citing_paper_id=paper.paper_id,
                cited_paper_id=None,
                reference_id=(
                    ref_id_by_key.get(ctx.ref_key) if ctx.ref_key else None
                ),
                section_name=ctx.section_name,
                section_type=ctx.section_type,
                paragraph_index=ctx.paragraph_index,
                sentence_index=ctx.sentence_index,
                sentence_with_markers=ctx.sentence_with_markers,
                sentence_without_markers=ctx.sentence_without_markers,
                left_context=ctx.left_context,
                right_context=ctx.right_context,
                marker_text=ctx.marker_text,
                marker_start_char=ctx.marker_start_char,
                marker_end_char=ctx.marker_end_char,
                citation_group_id=ctx.citation_group_id,
                citation_group_size=ctx.citation_group_size,
                local_window_text=ctx.local_window_text,
                context_text_for_embedding=ctx.context_text_for_embedding,
                citing_year=paper.year,
            )
        )

    return len(parsed.references), len(parsed.contexts)


def _record_failure(
    session: Session, openalex_id: str, status: str, error: str | None
) -> None:
    session.add(
        SourceDocument(
            paper_id=None,
            source_path=None,
            source_type="pdf",
            parse_status=status,
            raw_metadata={"openalex_id": openalex_id, "error": error},
        )
    )


def _flush_batch(session: Session, batch: list[WorkResult], dry_run: bool = False) -> None:
    try:
        total_refs = total_ctx = 0
        for result in batch:
            refs, ctxs = _persist(session, result)
            total_refs += refs
            total_ctx += ctxs
        if not dry_run:
            session.commit()
            logger.info(
                "batch committed — docs=%d refs=%d contexts=%d",
                len(batch),
                total_refs,
                total_ctx,
            )
        else:
            session.flush()
            logger.info(
                "DRY RUN: batch flushed (not committed) — docs=%d refs=%d contexts=%d",
                len(batch),
                total_refs,
                total_ctx,
            )
    except Exception:
        session.rollback()
        logger.exception("batch failed; rolled back %d documents", len(batch))
        raise


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _load_done_ids(session: Session) -> set[str]:
    rows = session.execute(
        select(SourceDocument.raw_metadata).where(
            SourceDocument.parse_status == "ok"
        )
    ).all()
    ids: set[str] = set()
    for (meta,) in rows:
        if meta and isinstance(meta, dict):
            oid = meta.get("openalex_id")
            if oid:
                ids.add(oid)
    return ids


@app.command()
def main(
    input_path: Path = typer.Option(DEFAULT_INPUT, "--input"),
    tei_dir: Path = typer.Option(DEFAULT_TEI_DIR, "--tei-dir"),
    workers: int = typer.Option(DEFAULT_WORKERS, "--workers"),
    limit: int | None = typer.Option(None, "--limit"),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help=(
            "Delete prior non-'ok' source_documents (pure status rows,"
            " paper_id NULL) before the run so failed works are re-attempted"
            " cleanly without accumulating duplicate failure rows."
        ),
    ),
    use_unpaywall: bool = typer.Option(
        True,
        "--use-unpaywall/--no-unpaywall",
        help=(
            "Fall back to Unpaywall (by DOI) when OpenAlex PDF links fail."
            " Free, needs OPEN_ALEX_EMAIL; the most reliable OA-PDF resolver."
        ),
    ),
    use_s2: bool = typer.Option(
        False,
        "--use-s2/--no-s2",
        help=(
            "Also fall back to Semantic Scholar openAccessPdf (throttled to"
            " 1 RPS). Off by default: the shared pool 429s readily and is slow"
            " for bulk recovery; Unpaywall covers the same DOIs more reliably."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run normally but explicitly rollback the database transaction at the end.",
    ),
) -> None:
    """Download + parse + ingest OpenAlex Works into the citation context DB."""
    import json

    if not input_path.exists():
        raise typer.BadParameter(f"input not found: {input_path}")
    tei_dir.mkdir(parents=True, exist_ok=True)

    session = get_session()
    s2_client = SemanticScholarClient() if use_s2 else None
    try:
        if retry_failed:
            deleted = session.execute(
                text(
                    "DELETE FROM source_documents WHERE parse_status <> 'ok'"
                )
            ).rowcount
            if not dry_run:
                session.commit()
            logger.info(
                "retry-failed: cleared %d prior non-ok source_documents.",
                deleted,
            )

        done_ids = _load_done_ids(session)
        logger.info("Skipping %d works already at parse_status=ok.", len(done_ids))

        works: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                work = json.loads(line)
                oid = _extract_paper_id(work)
                if oid and oid not in done_ids:
                    works.append(work)
                    if limit is not None and len(works) >= limit:
                        break

        logger.info("Ingesting %d works with %d workers.", len(works), workers)

        # Browser-like UA: several publishers 403 a bot UA but serve the same
        # OA PDF to a browser. mailto stays in a custom header for politeness.
        http_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.5",
            "From": config.OPEN_ALEX_EMAIL or "",
        }
        unpaywall_email = config.OPEN_ALEX_EMAIL if use_unpaywall else None

        ok_count = failed_count = 0
        ok_batch: list[WorkResult] = []

        with (
            httpx.Client(headers=http_headers) as http_client,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            future_to_id: dict[Future[WorkResult], str] = {
                pool.submit(
                    _ingest_one,
                    work,
                    tei_dir,
                    http_client,
                    s2_client,
                    unpaywall_email,
                ): (_extract_paper_id(work) or "")
                for work in works
            }

            for fut in as_completed(future_to_id):
                result = fut.result()
                if result.status != "ok" or result.parsed is None:
                    logger.warning(
                        "fail %s status=%s error=%s",
                        result.openalex_id,
                        result.status,
                        result.error,
                    )
                    _record_failure(
                        session, result.openalex_id, result.status, result.error
                    )
                    failed_count += 1
                    if failed_count % BATCH_DOCS == 0 and not dry_run:
                        session.commit()
                    continue

                ok_batch.append(result)
                if len(ok_batch) >= BATCH_DOCS:
                    _flush_batch(session, ok_batch, dry_run=dry_run)
                    ok_count += len(ok_batch)
                    ok_batch.clear()

        if ok_batch:
            _flush_batch(session, ok_batch, dry_run=dry_run)
            ok_count += len(ok_batch)

        if dry_run:
            session.rollback()
            logger.info("DRY RUN: rolled back all changes.")
        else:
            session.commit()
        logger.info(
            "ingest finished — ok=%d failed=%d",
            ok_count,
            failed_count,
        )
    finally:
        session.close()
        if s2_client is not None:
            s2_client.close()


if __name__ == "__main__":
    app()
