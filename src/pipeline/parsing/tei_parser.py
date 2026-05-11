"""Parse GROBID TEI XML into ``ParsedDocument``.

Why this module is the riskiest single step in Phase 1:

Every ``citation_contexts`` row's ``reference_id`` ultimately depends on the
``ref_key`` we record here. If we lose the link between the in-text
``<ref target="#bN">`` marker and the bibliography ``<biblStruct xml:id="bN">``,
the whole downstream retrieval pipeline becomes useless. So we never try to
regex citations out of rendered text — we trust GROBID's structural linkage.

Pipeline per document:

1. Parse ``<teiHeader>`` for paper-level metadata (title, year, authors, DOI).
2. Walk ``<listBibl>/<biblStruct>`` → ``bib_index: dict[ref_key, Extracted-
   Reference]``. ``xml:id`` is the only key we will ever match on.
3. Walk ``<body>//<div>``: ``<head>`` gives section name; ``<p>`` is the
   paragraph unit.
4. Reconstruct paragraph text by iterating ``paragraph.iter()`` and
   substituting every ``<ref type="bibr" target="#bN">`` with the sentinel
   ``[CITE:bN]`` (see ``GROBID_CITE_MARKER_PATTERN``). Track ``(char_offset,
   ref_key, marker_text)`` as we go.
5. Sentence-segment that text via pysbd.
6. For each sentence: find markers whose char offsets fall inside the
   sentence span, emit one ``ExtractedContext`` per marker. Joint citations
   (e.g. ``[1, 2, 3]``) share a ``citation_group_id`` + ``citation_group_size``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]

from entities.parsed import (
    ExtractedContext,
    ExtractedReference,
    ParsedDocument,
)
from pipeline.extraction.sentences import split_sentences
from utils.logger import logger
from utils.regex_patterns import (
    DOUBLED_PUNCTUATION_PATTERN,
    EMPTY_PAREN_PATTERN,
    GROBID_CITE_MARKER_PATTERN,
    ORPHAN_SEMICOLON_PATTERN,
    RESIDUAL_AUTHOR_YEAR_PATTERN,
    SPACE_AFTER_OPEN_PAREN_PATTERN,
    SPACE_BEFORE_PUNCTUATION_PATTERN,
)

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"tei": TEI_NS}
XML_ID = f"{{{XML_NS}}}id"

# Heuristic section_type classifier — lowercase substring → canonical bucket.
# Inspection-of-`<head>` only; nothing fancy. Used by the recommender to bias
# toward (e.g.) related-work over methods for background-style queries.
_SECTION_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("abstract", "abstract"),
    ("introduction", "introduction"),
    ("background", "background"),
    ("related", "related_work"),
    ("prior", "related_work"),
    ("literature", "related_work"),
    ("method", "methods"),
    ("approach", "methods"),
    ("model", "methods"),
    ("experiment", "experiments"),
    ("evaluation", "experiments"),
    ("result", "results"),
    ("discussion", "discussion"),
    ("analysis", "discussion"),
    ("conclusion", "conclusion"),
    ("limitation", "limitations"),
    ("future", "conclusion"),
)


@dataclass(slots=True)
class _Marker:
    """Mutable in-paragraph marker accumulator. Internal to this module."""

    ref_key: str
    marker_text: str
    start: int
    end: int


def _classify_section(head: str | None) -> str | None:
    if not head:
        return None
    h = head.lower()
    for needle, label in _SECTION_TYPE_RULES:
        if needle in h:
            return label
    return None


def _text(elem: etree._Element | None) -> str | None:
    if elem is None:
        return None
    s = "".join(elem.itertext()).strip()
    return s or None


def _first_text(parent: etree._Element, xpath: str) -> str | None:
    found = parent.find(xpath, NS)
    return _text(found)


def _parse_header(root: etree._Element) -> dict[str, object]:
    """Pull paper-level metadata out of ``<teiHeader>``."""
    header = root.find(".//tei:teiHeader", NS)
    if header is None:
        return {}

    title = _first_text(header, ".//tei:titleStmt/tei:title")

    authors: list[str] = []
    for pers in header.findall(".//tei:sourceDesc//tei:author/tei:persName", NS):
        forename = _first_text(pers, "tei:forename")
        surname = _first_text(pers, "tei:surname")
        full = " ".join(p for p in (forename, surname) if p)
        if full:
            authors.append(full)

    year: int | None = None
    date_elem = header.find(".//tei:publicationStmt/tei:date", NS)
    if date_elem is None:
        date_elem = header.find(".//tei:sourceDesc//tei:imprint/tei:date", NS)
    if date_elem is not None:
        when = date_elem.get("when") or _text(date_elem) or ""
        match = re.search(r"(\d{4})", when)
        if match:
            year = int(match.group(1))

    doi: str | None = None
    for idno in header.findall(".//tei:idno", NS):
        kind = (idno.get("type") or "").lower()
        if kind == "doi":
            doi = (_text(idno) or "").lower() or None
            break

    abstract = _first_text(header, ".//tei:profileDesc/tei:abstract")

    venue = _first_text(header, ".//tei:sourceDesc//tei:monogr/tei:title")

    return {
        "canonical_title": title,
        "authors": authors,
        "first_author": authors[0] if authors else None,
        "year": year,
        "doi": doi,
        "abstract": abstract,
        "venue": venue,
    }


def _parse_bibliography(root: etree._Element) -> dict[str, ExtractedReference]:
    """Walk ``<listBibl>/<biblStruct>``. Key = ``xml:id`` (e.g. ``b12``)."""
    bib: dict[str, ExtractedReference] = {}
    for entry in root.findall(".//tei:back//tei:listBibl/tei:biblStruct", NS):
        ref_key = entry.get(XML_ID)
        if not ref_key:
            continue

        title = _first_text(entry, ".//tei:title")
        venue = _first_text(entry, ".//tei:monogr/tei:title")

        authors: list[str] = []
        for pers in entry.findall(".//tei:author/tei:persName", NS):
            forename = _first_text(pers, "tei:forename")
            surname = _first_text(pers, "tei:surname")
            full = " ".join(p for p in (forename, surname) if p)
            if full:
                authors.append(full)

        parsed_year: int | None = None
        date_elem = entry.find(".//tei:imprint/tei:date", NS)
        if date_elem is not None:
            when = date_elem.get("when") or _text(date_elem) or ""
            match = re.search(r"(\d{4})", when)
            if match:
                parsed_year = int(match.group(1))

        doi: str | None = None
        arxiv_id: str | None = None
        for idno in entry.findall(".//tei:idno", NS):
            kind = (idno.get("type") or "").lower()
            value = (_text(idno) or "").strip()
            if not value:
                continue
            if kind == "doi" and not doi:
                doi = value.lower()
            elif kind in {"arxiv", "arxivid"} and not arxiv_id:
                arxiv_id = value

        raw_text = _text(entry)

        bib[ref_key] = ExtractedReference(
            ref_key=ref_key,
            raw_reference_text=raw_text,
            parsed_title=title,
            parsed_authors=authors,
            parsed_first_author=authors[0] if authors else None,
            parsed_year=parsed_year,
            parsed_venue=venue,
            doi=doi,
            arxiv_id=arxiv_id,
        )
    return bib


def _reconstruct_paragraph(
    paragraph: etree._Element,
) -> tuple[str, list[_Marker]]:
    """Flatten a ``<p>`` element to plain text + a marker offset list.

    Substitutes every ``<ref type="bibr" target="#bN">…</ref>`` with the
    ``[CITE:bN]`` sentinel and records its character offsets. Handles ``<ref>``
    elements nested arbitrarily deep inside inline markup (``<hi>``, ``<s>``,
    formula containers, etc.) rather than only direct children of ``<p>``.
    """
    parts: list[str] = []
    markers: list[_Marker] = []
    cursor = 0

    def _append(text: str) -> None:
        nonlocal cursor
        if not text:
            return
        parts.append(text)
        cursor += len(text)

    def _walk(elem: etree._Element) -> None:
        """Depth-first walk; handles text / tail at every level."""
        tag = etree.QName(elem).localname
        is_bibr = tag == "ref" and (elem.get("type") or "").lower() == "bibr"

        if is_bibr:
            target = (elem.get("target") or "").lstrip("#")
            inner = "".join(elem.itertext())
            marker_text = inner.strip() or None
            if target:
                sentinel = f"[CITE:{target}]"
                start = cursor
                _append(sentinel)
                end = cursor
                markers.append(
                    _Marker(
                        ref_key=target,
                        marker_text=marker_text or sentinel,
                        start=start,
                        end=end,
                    )
                )
            else:
                _append(inner)
            # Do NOT recurse — itertext() already captured all descendant text.
        else:
            if elem.text:
                _append(elem.text)
            for child in elem:
                _walk(child)
                if child.tail:
                    _append(child.tail)

    if paragraph.text:
        _append(paragraph.text)
    for child in paragraph:
        _walk(child)
        if child.tail:
            _append(child.tail)

    return "".join(parts), markers


def _strip_markers_for_embedding(sentence: str) -> str:
    """Produce the marker-free sentence used for embedding/BM25."""
    cleaned = GROBID_CITE_MARKER_PATTERN.sub("", sentence)
    cleaned = RESIDUAL_AUTHOR_YEAR_PATTERN.sub("", cleaned)
    cleaned = EMPTY_PAREN_PATTERN.sub("", cleaned)
    cleaned = ORPHAN_SEMICOLON_PATTERN.sub("(", cleaned)
    cleaned = DOUBLED_PUNCTUATION_PATTERN.sub(r"\1", cleaned)
    cleaned = SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r"\1", cleaned)
    cleaned = SPACE_AFTER_OPEN_PAREN_PATTERN.sub("(", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _emit_contexts_for_paragraph(
    paragraph_text: str,
    markers: list[_Marker],
    *,
    paragraph_index: int,
    section_name: str | None,
    section_type: str | None,
    bib_index: dict[str, ExtractedReference],
) -> list[ExtractedContext]:
    """Given a reconstructed paragraph + markers, emit one context per marker."""
    if not markers:
        return []

    sentences = split_sentences(paragraph_text)
    if not sentences:
        return []

    # Build (start, end) spans for each sentence in the paragraph by walking
    # through `paragraph_text` and consuming sentence strings in order.
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sent in sentences:
        idx = paragraph_text.find(sent, cursor)
        if idx < 0:
            # Sentence couldn't be located (whitespace drift); fall back to
            # cursor and accept some misalignment rather than dropping it.
            idx = cursor
        spans.append((idx, idx + len(sent)))
        cursor = idx + len(sent)

    # Group markers that fall in the same sentence — joint citations share a
    # citation_group_id.
    contexts: list[ExtractedContext] = []
    for sent_i, ((s_start, s_end), sentence_text) in enumerate(zip(spans, sentences, strict=False)):
        in_sentence = [
            m for m in markers if s_start <= m.start < s_end
        ]
        if not in_sentence:
            continue

        group_size = len(in_sentence)
        group_id = (
            f"p{paragraph_index}s{sent_i}g"
            f"{in_sentence[0].start}-{in_sentence[-1].end}"
            if group_size > 1
            else None
        )
        sentence_clean = _strip_markers_for_embedding(sentence_text)
        # Left/right context = neighboring sentences if they exist.
        left = sentences[sent_i - 1] if sent_i > 0 else None
        right = sentences[sent_i + 1] if sent_i + 1 < len(sentences) else None
        local_window = " ".join(
            s for s in (left, sentence_clean, right) if s
        ).strip() or None

        for m in in_sentence:
            ref = bib_index.get(m.ref_key)
            contexts.append(
                ExtractedContext(
                    section_name=section_name,
                    section_type=section_type,
                    paragraph_index=paragraph_index,
                    sentence_index=sent_i,
                    sentence_with_markers=sentence_text,
                    sentence_without_markers=sentence_clean,
                    left_context=left,
                    right_context=right,
                    marker_text=m.marker_text,
                    marker_start_char=m.start - s_start,
                    marker_end_char=m.end - s_start,
                    citation_group_id=group_id,
                    citation_group_size=group_size,
                    ref_key=m.ref_key if ref is not None else None,
                    local_window_text=local_window,
                    context_text_for_embedding=sentence_clean,
                )
            )
    return contexts


def parse_tei(
    tei_xml: str | bytes,
    *,
    openalex_id: str,
    tei_path: Path,
    pdf_path: Path | None = None,
) -> ParsedDocument:
    """Parse one TEI document into a ``ParsedDocument``."""
    if isinstance(tei_xml, str):
        tei_bytes = tei_xml.encode("utf-8")
    else:
        tei_bytes = tei_xml

    root = etree.fromstring(tei_bytes)
    header_meta = _parse_header(root)
    bib_index = _parse_bibliography(root)

    contexts: list[ExtractedContext] = []
    paragraph_counter = 0
    body = root.find(".//tei:text/tei:body", NS)
    if body is not None:
        for div in body.findall(".//tei:div", NS):
            head = _first_text(div, "tei:head")
            section_type = _classify_section(head)
            for paragraph in div.findall("tei:p", NS):
                text, markers = _reconstruct_paragraph(paragraph)
                if not text:
                    paragraph_counter += 1
                    continue
                contexts.extend(
                    _emit_contexts_for_paragraph(
                        text,
                        markers,
                        paragraph_index=paragraph_counter,
                        section_name=head,
                        section_type=section_type,
                        bib_index=bib_index,
                    )
                )
                paragraph_counter += 1

    logger.info(
        "tei parsed openalex=%s refs=%d contexts=%d",
        openalex_id,
        len(bib_index),
        len(contexts),
    )

    return ParsedDocument(
        openalex_id=openalex_id,
        canonical_title=header_meta.get("canonical_title"),  # type: ignore[arg-type]
        authors=header_meta.get("authors") or [],  # type: ignore[arg-type]
        first_author=header_meta.get("first_author"),  # type: ignore[arg-type]
        year=header_meta.get("year"),  # type: ignore[arg-type]
        venue=header_meta.get("venue"),  # type: ignore[arg-type]
        doi=header_meta.get("doi"),  # type: ignore[arg-type]
        abstract=header_meta.get("abstract"),  # type: ignore[arg-type]
        tei_path=str(tei_path),
        pdf_path=str(pdf_path) if pdf_path else None,
        references=list(bib_index.values()),
        contexts=contexts,
    )
