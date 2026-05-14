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
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer
from sqlalchemy import select
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

def _extract_openalex_id(work: dict[str, Any]) -> str | None:
    raw = work.get("id") or ""
    m = OPENALEX_ID_PATTERN.search(raw)
    return m.group(1) if m else None


def _pdf_url_candidates(work: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for loc_key in ("best_oa_location", "primary_location"):
        loc = work.get(loc_key) or {}
        if loc.get("pdf_url"):
            urls.append(loc["pdf_url"])
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        urls.append(oa["oa_url"])
    return urls


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError,)),
    reraise=True,
)
def _http_get(client: httpx.Client, url: str) -> tuple[bytes, str]:
    r = client.get(url, follow_redirects=True, timeout=60.0)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "")


def _looks_like_pdf(content: bytes, ctype: str) -> bool:
    return content.startswith(PDF_MAGIC) or "application/pdf" in ctype.lower()


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
) -> WorkResult:
    """Download → GROBID → delete → parse TEI. Runs in a thread pool worker."""
    openalex_id = _extract_openalex_id(work) or ""
    tei_path = tei_dir / f"{openalex_id}.tei.xml"

    # TEI cache hit: skip download + GROBID entirely.
    if tei_path.exists() and tei_path.stat().st_size > 0:
        try:
            tei_xml = tei_path.read_text(encoding="utf-8")
            parsed = parse_tei(
                tei_xml,
                openalex_id=openalex_id,
                tei_path=tei_path,
            )
            return WorkResult(openalex_id, parsed, tei_path, "ok", None)
        except Exception as exc:  # noqa: BLE001
            return WorkResult(openalex_id, None, tei_path, "tei_parse_error", str(exc))

    # Try each PDF URL candidate.
    urls = _pdf_url_candidates(work)
    if not urls:
        return WorkResult(openalex_id, None, None, "no_pdf", "no url candidates")

    pdf_content: bytes | None = None
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
        pdf_content = content
        break

    if pdf_content is None:
        return WorkResult(openalex_id, None, None, "no_pdf", last_error)

    # Write to a temp file so GROBID can POST it, then delete immediately.
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(pdf_content)
        del pdf_content  # release memory before GROBID call

        try:
            tei_xml = process_fulltext(tmp_path)
        except GrobidTimeoutError as exc:
            return WorkResult(openalex_id, None, None, "timeout", str(exc))
        except GrobidError as exc:
            return WorkResult(openalex_id, None, None, "error", str(exc))
        finally:
            tmp_path.unlink(missing_ok=True)

        tei_path.write_text(tei_xml, encoding="utf-8")
        parsed = parse_tei(
            tei_xml,
            openalex_id=openalex_id,
            tei_path=tei_path,
        )
        return WorkResult(openalex_id, parsed, tei_path, "ok", None)

    except Exception as exc:  # noqa: BLE001
        return WorkResult(openalex_id, None, None, "tei_parse_error", str(exc))


# ---------------------------------------------------------------------------
# DB writers (single-threaded, called in the main thread)
# ---------------------------------------------------------------------------

def _quick_normalize(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()


def _upsert_paper(session: Session, parsed: ParsedDocument) -> Paper:
    openalex_url = f"https://openalex.org/{parsed.openalex_id}"
    existing = session.execute(
        select(Paper).where(Paper.url == openalex_url)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    paper = Paper(
        canonical_title=parsed.canonical_title or parsed.openalex_id,
        normalized_title=_quick_normalize(parsed.canonical_title or ""),
        authors={"list": parsed.authors} if parsed.authors else None,
        first_author=parsed.first_author,
        year=parsed.year,
        venue=parsed.venue,
        doi=parsed.doi,
        arxiv_id=parsed.arxiv_id,
        url=openalex_url,
        source="openalex",
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


def _flush_batch(session: Session, batch: list[WorkResult]) -> None:
    try:
        total_refs = total_ctx = 0
        for result in batch:
            refs, ctxs = _persist(session, result)
            total_refs += refs
            total_ctx += ctxs
        session.commit()
        logger.info(
            "batch committed — docs=%d refs=%d contexts=%d",
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
) -> None:
    """Download + parse + ingest OpenAlex Works into the citation context DB."""
    import json

    if not input_path.exists():
        raise typer.BadParameter(f"input not found: {input_path}")
    tei_dir.mkdir(parents=True, exist_ok=True)

    session = get_session()
    try:
        done_ids = _load_done_ids(session)
        logger.info("Skipping %d works already at parse_status=ok.", len(done_ids))

        works: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                work = json.loads(line)
                oid = _extract_openalex_id(work)
                if oid and oid not in done_ids:
                    works.append(work)
                    if limit is not None and len(works) >= limit:
                        break

        logger.info("Ingesting %d works with %d workers.", len(works), workers)

        http_headers = {
            "User-Agent": (
                f"missing-citations-identifier"
                f" (mailto:{config.OPEN_ALEX_EMAIL})"
                if config.OPEN_ALEX_EMAIL
                else "missing-citations-identifier"
            ),
            "Accept": "application/pdf,*/*;q=0.5",
        }

        ok_count = failed_count = 0
        ok_batch: list[WorkResult] = []

        with (
            httpx.Client(headers=http_headers) as http_client,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            future_to_id: dict[Future[WorkResult], str] = {
                pool.submit(_ingest_one, work, tei_dir, http_client): (
                    _extract_openalex_id(work) or ""
                )
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
                    if failed_count % BATCH_DOCS == 0:
                        session.commit()
                    continue

                ok_batch.append(result)
                if len(ok_batch) >= BATCH_DOCS:
                    _flush_batch(session, ok_batch)
                    ok_count += len(ok_batch)
                    ok_batch.clear()

        if ok_batch:
            _flush_batch(session, ok_batch)
            ok_count += len(ok_batch)

        session.commit()
        logger.info(
            "ingest finished — ok=%d failed=%d",
            ok_count,
            failed_count,
        )
    finally:
        session.close()


if __name__ == "__main__":
    app()
