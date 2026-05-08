"""Build a sentence-level citation-retrieval benchmark from a folder of PDFs.

For every PDF:
  1. GROBID parses it into a ``ParsedPaper`` with ``bibliography[bibkey]=raw``
     and section text containing ``[CITE:bX]`` markers.
  2. Sentences are extracted; each one carries the list of bibkeys cited in it.
  3. Each bibliography entry is resolved (DOI → OpenAlex search) to the local
     OpenAlex Work ID it corresponds to. Unresolved or non-corpus refs drop
     out — they cannot be evaluated.
  4. Each sentence with at least one resolved + indexed target becomes one
     row of the JSONL benchmark, in the schema expected by
     ``evaluation.benchmarks.s2orc.iter_hide_seek_examples``.

Usage
-----
    python scripts/build_sentence_benchmark.py \
        --pdf-dir papers \
        --out eval/sentence_dataset/test.jsonl \
        --max-pdfs 50

The output JSONL row is:

    {
      "sentence_id":          "<paper-stem>:s<sec>:<idx>",
      "query_text":           "<retrieval text, citation markers stripped>",
      "indexed_reference_ids":["W12345…", ...],
      "total_reference_count": <int>,
      "section":              "Methods",
      "is_multi_facet":       <bool>,        # >1 cited target paper in the sentence
      "citing_paper_path":    "<pdf>",
      "raw_text":             "<sentence with markers>",
      "resolution_methods":   ["exact_doi", "openalex", ...],
    }

Hide-and-seek loaders treat ``indexed_reference_ids`` as the positive set,
so for sentence-level eval one runs Stage 4 with ``--hide-fraction 1.0`` to
hide every cited target.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline.pdf_parser import GrobidPDFParser  # noqa: E402
from pipeline.reference_resolver import ReferenceResolver  # noqa: E402
from pipeline.sentence_extractor import extract_sentences  # noqa: E402

logger = logging.getLogger("build_sentence_benchmark")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def load_qdrant_id_set(collection: str | None = None) -> set[str]:
    """Pull the full set of ``paper_id`` payloads from the Qdrant collection.

    The local ``papers.paperId`` column already matches the Qdrant payload
    ``paper_id`` field; we use the Qdrant set directly so the benchmark only
    contains targets the retriever can actually surface.
    """
    from utils.config import config
    from qdrant_client import QdrantClient

    coll = collection or config.QDRANT_COLLECTION_NAME or "papers"
    client = QdrantClient(url=config.QDRANT_URL or "http://localhost:6333")

    ids: set[str] = set()
    next_offset: Any = None
    while True:
        points, next_offset = client.scroll(
            collection_name=coll,
            limit=4096,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            pid = payload.get("paper_id")
            if pid:
                ids.add(str(pid))
        if next_offset is None:
            break
    return ids


def resolve_bibliography(
    bibliography: dict[str, str],
    resolver: ReferenceResolver,
    indexed_ids: set[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Resolve each bibkey to a local Work ID, dropping refs not in the index.

    Returns
    -------
    resolved_to_paper:
        ``{bibkey: paper_id}`` for entries we *can* score.
    methods:
        ``{bibkey: method}`` for diagnostics ("exact_doi" / "openalex" / ...).
    counters:
        Aggregate counts: total / resolved / in_index / dropped.
    """
    resolved_to_paper: dict[str, str] = {}
    methods: dict[str, str] = {}
    counters: dict[str, int] = Counter()

    for bibkey, raw in bibliography.items():
        counters["total"] += 1
        result = resolver.resolve(raw)
        methods[bibkey] = result.method or "unresolved"
        if not result.is_resolved or not result.resolved_paper_id:
            counters["unresolved"] += 1
            continue
        counters["resolved"] += 1
        if result.resolved_paper_id not in indexed_ids:
            counters["resolved_but_not_in_index"] += 1
            continue
        counters["in_index"] += 1
        resolved_to_paper[bibkey] = result.resolved_paper_id

    return resolved_to_paper, methods, counters


def _strip_context(raw: str | None) -> str:
    """Drop GROBID markers and the natural-language citation regex from a
    neighbouring sentence so context windows feed clean text to the encoder.
    """
    if not raw:
        return ""
    from pipeline.sentence_extractor import _strip_citation_artifacts

    return _strip_citation_artifacts(raw)


def sentence_to_row(
    *,
    sentence_index: int,
    section_index: int,
    paper_stem: str,
    sentence,
    resolved_to_paper: dict[str, str],
    methods: dict[str, str],
    context_window: int = 0,
) -> dict[str, Any] | None:
    """Convert one ``SentenceRecord`` to a JSONL benchmark row, or skip it.

    ``context_window`` controls how many neighbouring sentences (each side)
    are concatenated into ``query_text``. Diagnostic runs found that
    ``context_window=1`` roughly doubles recall@200 on this corpus because a
    bare academic sentence is often too short to discriminate. The raw
    sentence text and the bibkeys remain attached to the row regardless of
    window size, so downstream consumers can re-derive the bare-sentence
    variant without rebuilding from PDFs.
    """
    if not sentence.has_citation or not sentence.cited_bibkeys:
        return None

    target_ids: list[str] = []
    used_methods: list[str] = []
    seen: set[str] = set()
    for bk in sentence.cited_bibkeys:
        pid = resolved_to_paper.get(bk)
        if not pid or pid in seen:
            continue
        target_ids.append(pid)
        used_methods.append(methods.get(bk, "unknown"))
        seen.add(pid)

    if not target_ids:
        return None

    sentence_text = (sentence.retrieval_text or sentence.text or "").strip()
    if not sentence_text or len(sentence_text.split()) < 4:
        return None

    if context_window > 0:
        # Currently the SentenceRecord carries one neighbour each side; that
        # matches context_window=1. Larger windows would require structural
        # changes to the extractor, so we cap to what's available.
        prev_text = _strip_context(sentence.previous_sentence)
        next_text = _strip_context(sentence.next_sentence)
        query_text = " ".join(p for p in (prev_text, sentence_text, next_text) if p)
    else:
        query_text = sentence_text

    return {
        "sentence_id": f"{paper_stem}:s{section_index}:{sentence_index}",
        "query_text": query_text,
        "sentence_text": sentence_text,
        "raw_text": sentence.text,
        "indexed_reference_ids": target_ids,
        "total_reference_count": len(sentence.cited_bibkeys),
        "section": sentence.section,
        "is_multi_facet": len(target_ids) > 1,
        "citing_paper_path": paper_stem,
        "resolution_methods": used_methods,
        "context_window": context_window,
    }


def process_pdf(
    pdf_path: Path,
    *,
    grobid_url: str,
    resolver: ReferenceResolver,
    indexed_ids: set[str],
    context_window: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse one PDF and return (rows, per-paper diagnostics)."""
    diag: dict[str, int] = Counter()
    paper = GrobidPDFParser(str(pdf_path), grobid_url=grobid_url).parse()
    diag["bibliography_size"] = len(paper.bibliography)

    resolved_to_paper, methods, counters = resolve_bibliography(
        paper.bibliography, resolver, indexed_ids
    )
    for k, v in counters.items():
        diag[f"refs_{k}"] = v

    rows: list[dict[str, Any]] = []
    sections = list(paper.sections.items())
    for section_index, (_, _section_text) in enumerate(sections):
        # ``extract_sentences`` consumes the whole paper, so we only call it
        # once and then keep all sentences regardless of section_index. The
        # section_index in the row is just a stable sentence-id component.
        pass

    sentences = extract_sentences(paper)
    diag["sentences_total"] = len(sentences)
    diag["sentences_with_citation"] = sum(1 for s in sentences if s.has_citation)
    diag["sentences_with_bibkeys"] = sum(1 for s in sentences if s.cited_bibkeys)

    paper_stem = pdf_path.stem
    for i, sentence in enumerate(sentences):
        row = sentence_to_row(
            sentence_index=i,
            section_index=0,
            paper_stem=paper_stem,
            sentence=sentence,
            resolved_to_paper=resolved_to_paper,
            methods=methods,
            context_window=context_window,
        )
        if row is not None:
            rows.append(row)

    diag["rows_emitted"] = len(rows)
    diag["rows_multi_facet"] = sum(1 for r in rows if r["is_multi_facet"])
    return rows, diag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "papers")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval" / "sentence_dataset" / "test.jsonl",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional JSON file capturing per-PDF processing stats.",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Cap on number of PDFs to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help=(
            "Neighbouring sentences (each side) to concatenate into "
            "``query_text``. Diagnostic runs found 1 nearly doubles "
            "recall@200 vs the bare sentence. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection to use as the corpus filter; defaults to config.",
    )
    args = parser.parse_args(argv)

    setup_logging()

    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error("No PDFs found under %s", args.pdf_dir)
        return 1
    if args.max_pdfs:
        pdf_paths = pdf_paths[: args.max_pdfs]
    logger.info("Processing %d PDFs from %s", len(pdf_paths), args.pdf_dir)

    logger.info("Loading Qdrant paper-id set ...")
    indexed_ids = load_qdrant_id_set(args.collection)
    logger.info("Qdrant index has %d unique paper_ids", len(indexed_ids))

    resolver = ReferenceResolver()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, Any] = {"per_pdf": {}, "totals": Counter()}

    t0 = time.perf_counter()
    with args.out.open("w", encoding="utf-8") as out_fh:
        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                rows, diag = process_pdf(
                    pdf_path,
                    grobid_url=args.grobid_url,
                    resolver=resolver,
                    indexed_ids=indexed_ids,
                    context_window=args.context_window,
                )
            except Exception as exc:
                logger.exception("Failed to process %s: %s", pdf_path.name, exc)
                diagnostics["per_pdf"][pdf_path.name] = {"error": str(exc)}
                continue

            for row in rows:
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            diagnostics["per_pdf"][pdf_path.name] = dict(diag)
            for k, v in diag.items():
                diagnostics["totals"][k] += v

            elapsed = time.perf_counter() - t0
            logger.info(
                "[%d/%d] %s -> %d rows (bib=%d resolved=%d in_index=%d, %.1fs total)",
                i,
                len(pdf_paths),
                pdf_path.name,
                diag.get("rows_emitted", 0),
                diag.get("bibliography_size", 0),
                diag.get("refs_resolved", 0),
                diag.get("refs_in_index", 0),
                elapsed,
            )

    diagnostics["totals"] = dict(diagnostics["totals"])
    diagnostics["resolver_stats"] = dict(resolver.stats)
    diag_path = args.diagnostics or args.out.with_suffix(".diagnostics.json")
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("Done. Wrote %s (rows=%d)", args.out, diagnostics["totals"].get("rows_emitted", 0))
    logger.info("Diagnostics: %s", diag_path)
    logger.info("Resolver methods: %s", resolver.stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
