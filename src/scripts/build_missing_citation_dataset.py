"""Build a labelled citation-need dataset from the cached GROBID TEI corpus.

Weak-supervision (CiteWorth-style) labelling, fully reproducible from
``data/corpus/tei/**`` — no human annotation:

* A sentence that GROBID linked to a bibliography entry demonstrably *needed* a
  citation. It yields two rows:
  - ``HAS_CITATION``     — the sentence with its ``[CITE:bN]`` sentinels rewritten
    as real ``\\cite{bN}`` markers (tests "never flag an already-cited sentence").
  - ``MISSING_CITATION`` — the same sentence with markers removed (tests recall
    of genuinely citation-worthy text).
* A substantive prose sentence with no citation marker is a (weak) negative:
  ``NOT_CITATION_WORTHY``. These are noisy — some uncited sentences are in fact
  citation-worthy — which caps achievable precision; the limitation is recorded
  in ``docs/eval-results.md``.

Output is JSONL compatible with ``scripts.evaluate_missing_citations manual``.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer
from lxml import etree  # type: ignore[import-untyped]

from pipeline.extraction.sentences import split_sentences
from pipeline.missing_citations.detector import CitationNeedLabel
from pipeline.parsing.tei_parser import (
    NS,
    _Marker,
    _classify_section,
    _reconstruct_paragraph,
    _strip_markers_for_embedding,
)
from utils.logger import logger

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEI_DIR = _PROJECT_ROOT / "data" / "corpus" / "tei"
DEFAULT_OUTPUT = (
    _PROJECT_ROOT / "eval" / "missing_citations" / "tei_citeworth.jsonl"
)
_SENTINEL_PATTERN = re.compile(r"\[CITE:([^\]]+)\]")
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_MIN_NEGATIVE_WORDS = 8
_MIN_POSITIVE_WORDS = 6
_PER_PAPER_CAP = 12  # keep any one paper from dominating a label


@dataclass(frozen=True, slots=True)
class _Row:
    sentence_id: str
    text: str
    gold_label: CitationNeedLabel
    previous_sentence: str | None
    next_sentence: str | None
    section: str | None
    source: str


@app.command()
def build(
    tei_dir: str = typer.Option(str(DEFAULT_TEI_DIR), "--tei-dir"),
    output: str = typer.Option(str(DEFAULT_OUTPUT), "--output"),
    per_label: int = typer.Option(
        400, "--per-label", min=10, help="Target rows per gold label."
    ),
    max_files: int = typer.Option(
        400, "--max-files", min=1, help="Cap TEI files scanned (sorted, stable)."
    ),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Mine TEI prose into a balanced citation-need JSONL."""
    tei_paths = sorted(Path(tei_dir).rglob("*.tei.xml"))[:max_files]
    if not tei_paths:
        raise typer.BadParameter(f"no .tei.xml files under {tei_dir}")

    by_label: dict[CitationNeedLabel, list[_Row]] = defaultdict(list)
    for tei_path in tei_paths:
        for row in _rows_from_tei(tei_path):
            by_label[row.gold_label].append(row)

    rng = random.Random(seed)
    selected: list[_Row] = []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        selected.extend(rows[:per_label])
        logger.info("label=%s mined=%d kept=%d", label.value, len(rows), min(len(rows), per_label))

    rng.shuffle(selected)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(_serialise(row), ensure_ascii=False) + "\n")

    counts = Counter(row.gold_label.value for row in selected)
    typer.echo(f"Wrote {len(selected)} rows to {out_path}")
    for label_value, count in sorted(counts.items()):
        typer.echo(f"  {label_value}: {count}")


def _rows_from_tei(tei_path: Path) -> list[_Row]:
    try:
        root = etree.fromstring(tei_path.read_bytes())
    except etree.XMLSyntaxError as exc:
        logger.warning("skip %s: %s", tei_path.name, exc)
        return []

    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return []

    paper_id = tei_path.name.removesuffix(".tei.xml")
    per_paper: Counter[CitationNeedLabel] = Counter()
    rows: list[_Row] = []
    paragraph_index = 0
    for div in body.findall(".//tei:div", NS):
        head_elem = div.find("tei:head", NS)
        head = "".join(head_elem.itertext()).strip() if head_elem is not None else None
        section_type = _classify_section(head)
        for paragraph in div.findall("tei:p", NS):
            text, markers = _reconstruct_paragraph(paragraph)
            if text:
                rows.extend(
                    _rows_from_paragraph(
                        text,
                        markers,
                        paper_id=paper_id,
                        paragraph_index=paragraph_index,
                        section_type=section_type,
                        per_paper=per_paper,
                    )
                )
            paragraph_index += 1
    return rows


def _rows_from_paragraph(
    text: str,
    markers: list[_Marker],
    *,
    paper_id: str,
    paragraph_index: int,
    section_type: str | None,
    per_paper: Counter[CitationNeedLabel],
) -> list[_Row]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    spans = _sentence_spans(text, sentences)

    rows: list[_Row] = []
    for i, ((s_start, s_end), raw) in enumerate(zip(spans, sentences, strict=False)):
        has_marker = any(s_start <= m.start < s_end for m in markers)
        prev_clean = _clean(sentences[i - 1]) if i > 0 else None
        next_clean = _clean(sentences[i + 1]) if i + 1 < len(sentences) else None
        base_id = f"{paper_id}:p{paragraph_index}:s{i}"

        if has_marker:
            cited = _sentinels_to_cite(raw)
            missing = _strip_markers_for_embedding(raw)
            if _word_count(missing) < _MIN_POSITIVE_WORDS:
                continue
            if _take(per_paper, CitationNeedLabel.HAS_CITATION):
                rows.append(
                    _Row(f"{base_id}:cited", cited, CitationNeedLabel.HAS_CITATION,
                         prev_clean, next_clean, section_type, f"tei:{base_id}")
                )
            if _take(per_paper, CitationNeedLabel.MISSING_CITATION):
                rows.append(
                    _Row(f"{base_id}:missing", missing, CitationNeedLabel.MISSING_CITATION,
                         prev_clean, next_clean, section_type, f"tei:{base_id}")
                )
        else:
            clean = _clean(raw)
            if (
                _word_count(clean) >= _MIN_NEGATIVE_WORDS
                and "[CITE:" not in clean
                and _take(per_paper, CitationNeedLabel.NOT_CITATION_WORTHY)
            ):
                rows.append(
                    _Row(f"{base_id}:neg", clean, CitationNeedLabel.NOT_CITATION_WORTHY,
                         prev_clean, next_clean, section_type, f"tei:{base_id}")
                )
    return rows


def _sentence_spans(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sent in sentences:
        idx = text.find(sent, cursor)
        if idx < 0:
            idx = cursor
        spans.append((idx, idx + len(sent)))
        cursor = idx + len(sent)
    return spans


def _sentinels_to_cite(sentence: str) -> str:
    return _SENTINEL_PATTERN.sub(lambda m: f"\\cite{{{m.group(1)}}}", sentence).strip()


def _clean(sentence: str) -> str:
    return _strip_markers_for_embedding(sentence)


def _word_count(sentence: str) -> int:
    return len(_WORD_PATTERN.findall(sentence))


def _take(per_paper: Counter[CitationNeedLabel], label: CitationNeedLabel) -> bool:
    if per_paper[label] >= _PER_PAPER_CAP:
        return False
    per_paper[label] += 1
    return True


def _serialise(row: _Row) -> dict[str, object]:
    return {
        "sentence_id": row.sentence_id,
        "text": row.text,
        "gold_label": row.gold_label.value,
        "previous_sentence": row.previous_sentence,
        "next_sentence": row.next_sentence,
        "section": row.section,
        "source": row.source,
    }


if __name__ == "__main__":
    app()
