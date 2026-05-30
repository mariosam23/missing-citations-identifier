"""Rule-based first-stage detector for potentially missing citations.

The detector is intentionally deterministic. It does not try to decide the
exact missing reference; it only finds uncited sentences that are likely worth
running through the existing citation recommender.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class CitationNeedLabel(StrEnum):
    """Sentence-level citation-need labels returned by the scan pipeline."""

    HAS_CITATION = "HAS_CITATION"
    COVERED_BY_BLOCK = "COVERED_BY_BLOCK"
    MISSING_CITATION = "MISSING_CITATION"
    NOT_CITATION_WORTHY = "NOT_CITATION_WORTHY"


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """One sentence in the scanned document, with absolute offsets."""

    sentence_id: str
    text: str
    start_offset: int
    end_offset: int
    paragraph_index: int
    sentence_index: int
    section_type: str | None


@dataclass(frozen=True, slots=True)
class Detection:
    """Rule-based citation-need decision for a single sentence."""

    sentence: SentenceSpan
    label: CitationNeedLabel
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _OffsetSegment:
    """Maps a contiguous run of paragraph-block text to document offsets.

    Within one segment, block and original positions advance one-for-one. The
    gaps between segments absorb stripped indentation and the original line
    breaks (CRLF or LF) that the block normalises to a single newline, so a
    sentence offset is never thrown off by how the source was wrapped.
    """

    block_start: int
    orig_start: int
    length: int


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    paragraph_index: int
    section_type: str | None
    segments: tuple[_OffsetSegment, ...]

    def map_offset(self, block_pos: int) -> int:
        """Translate a position in ``text`` to an absolute document offset."""
        for segment in self.segments:
            block_end = segment.block_start + segment.length
            if segment.block_start <= block_pos <= block_end:
                return segment.orig_start + (block_pos - segment.block_start)
        last = self.segments[-1]
        return last.orig_start + last.length


@dataclass(frozen=True, slots=True)
class _CueScore:
    score: float
    reasons: tuple[str, ...]


MIN_MEANINGFUL_WORDS = 6
MISSING_RULE_THRESHOLD = 0.35
COVERED_RULE_THRESHOLD = 0.55

_PARAGRAPH_PATTERN = re.compile(r"\S(?:.*?)(?=\n\s*\n|\Z)", re.DOTALL)
_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=(?:[`'\"(\[])?[A-Z0-9])"
)
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_LATEX_CITE_PATTERN = re.compile(
    r"\\(?:[A-Za-z]*cite[A-Za-z]*|autocite|parencite|textcite|footcite)"
    r"\*?(?:\s*\[[^\]]*\]){0,2}\s*\{[^{}]+\}",
    re.IGNORECASE,
)
_MARKDOWN_CITE_PATTERN = re.compile(
    r"\[@[A-Za-z0-9_:.#/&%$+-]+(?:\s*;\s*@[A-Za-z0-9_:.#/&%$+-]+)*\]"
)
_AUTHOR_YEAR_PATTERN = re.compile(
    r"(\b[A-Z][A-Za-z-]+(?:\s+et\s+al\.)?\s*\(\d{4}[a-z]?\))|"
    r"(\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?\))"
)

_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*#{1,6}\s+\S+"),
    re.compile(r"^\s*\\(?:chapter|section|subsection|subsubsection)\*?\{"),
)
_LATEX_HEADING_LINE = re.compile(
    r"^\s*\\(?:chapter|section|subsection|subsubsection)\*?\{([^}]+)\}",
    re.IGNORECASE,
)
_LATEX_DOCUMENT_MARKERS = re.compile(r"\\documentclass\b|\\begin\{document\}")
_STRUCTURAL_LINE_PATTERN = re.compile(
    r"^\s*(?:"
    r"\\(?:begin|end|caption|label|item|ref|eqref)\b|"
    r"\$\$|\\\[|\\\]|%|[-*]\s*$"
    r")"
)

_OWN_WORK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:we|this paper|this work|our work)\s+"
        r"(?:propose|present|introduce|develop|show|demonstrate|evaluate|"
        r"find|contribute|report|obtain|achieve)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:our|the proposed)\s+"
        r"(?:method|approach|model|framework|system|algorithm|results?|"
        r"experiments?|contributions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:in this paper|in this work),?\s+we\b",
        re.IGNORECASE,
    ),
)

_STRONG_PRIOR_CUES: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"\bprevious work\b", re.IGNORECASE), 0.45, "previous work cue"),
    (re.compile(r"\bprior work\b", re.IGNORECASE), 0.45, "prior work cue"),
    (re.compile(r"\bexisting (?:work|methods?|approaches?)\b", re.IGNORECASE), 0.4, "existing-methods cue"),
    (re.compile(r"\bhas been shown\b", re.IGNORECASE), 0.4, "evidence claim"),
    (re.compile(r"\bhave been shown\b", re.IGNORECASE), 0.4, "evidence claim"),
    (re.compile(r"\bstate[- ]of[- ]the[- ]art\b", re.IGNORECASE), 0.35, "state-of-the-art claim"),
    (re.compile(r"\bwidely used\b", re.IGNORECASE), 0.35, "widely-used claim"),
    (re.compile(r"\bcommonly used\b", re.IGNORECASE), 0.35, "commonly-used claim"),
    (re.compile(r"\bfollowing\b", re.IGNORECASE), 0.35, "following prior work"),
    (re.compile(r"\bbased on\b", re.IGNORECASE), 0.3, "based-on cue"),
    (re.compile(r"\binspired by\b", re.IGNORECASE), 0.35, "inspired-by cue"),
    (re.compile(r"\bbuilding on\b", re.IGNORECASE), 0.35, "building-on cue"),
)

_METHOD_USE_PATTERN = re.compile(
    r"\b(?:we|our model|our system|the model|the system)\s+"
    r"(?:use|uses|adopt|adopts|employ|employs|rely on|relies on|"
    r"fine-tune|fine-tunes)\b",
    re.IGNORECASE,
)
_COMPARISON_PATTERN = re.compile(
    r"\b(?:compared with|compared to|unlike|similar to|outperforms|"
    r"surpasses)\b",
    re.IGNORECASE,
)
_SURVEY_PATTERN = re.compile(
    r"\b(?:approaches|methods|techniques|models|systems)\s+"
    r"(?:for|to|that)\b",
    re.IGNORECASE,
)
_NAMED_ARTIFACT_PATTERN = re.compile(
    r"\b(?:BERT|RoBERTa|GPT|Transformer|WordPiece|ELMo|CNN|LSTM|RNN|"
    r"GLUE|SQuAD|ImageNet|MNIST|CIFAR|BLEU|ROUGE|F1|BM25|TF-IDF|"
    r"LoRA|SPECTER|SciBERT)\b"
)

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


def scan_document(text: str, *, max_sentences: int = 250) -> list[Detection]:
    """Return rule-based citation-need decisions for up to ``max_sentences``.

    The returned detections include all labels. The API layer filters to
    actionable ``MISSING_CITATION`` rows after retrieval confidence is known.
    """
    if max_sentences <= 0:
        return []

    sentences = list(iter_sentence_spans(text, max_sentences=max_sentences))
    explicit_citation_ids = {
        s.sentence_id for s in sentences if has_explicit_citation(s.text)
    }

    detections: list[Detection] = []
    for sentence in sentences:
        stripped = sentence.text.strip()

        if sentence.sentence_id in explicit_citation_ids:
            detections.append(
                Detection(
                    sentence=sentence,
                    label=CitationNeedLabel.HAS_CITATION,
                    confidence=1.0,
                    reasons=("explicit citation marker",),
                )
            )
            continue

        if _is_not_citation_worthy(stripped):
            detections.append(
                Detection(
                    sentence=sentence,
                    label=CitationNeedLabel.NOT_CITATION_WORTHY,
                    confidence=0.95,
                    reasons=("structural, short, or own-contribution sentence",),
                )
            )
            continue

        cue_score = _score_citation_need(sentence)
        if (
            cue_score.score < COVERED_RULE_THRESHOLD
            and _is_near_cited_sentence(sentence, sentences, explicit_citation_ids)
        ):
            detections.append(
                Detection(
                    sentence=sentence,
                    label=CitationNeedLabel.COVERED_BY_BLOCK,
                    confidence=0.75,
                    reasons=("nearby sentence in the same paragraph is cited",),
                )
            )
            continue

        if cue_score.score >= MISSING_RULE_THRESHOLD:
            detections.append(
                Detection(
                    sentence=sentence,
                    label=CitationNeedLabel.MISSING_CITATION,
                    confidence=cue_score.score,
                    reasons=cue_score.reasons,
                )
            )
            continue

        detections.append(
            Detection(
                sentence=sentence,
                label=CitationNeedLabel.NOT_CITATION_WORTHY,
                confidence=max(0.05, 1.0 - cue_score.score),
                reasons=cue_score.reasons or ("no citation-need cue",),
            )
        )

    return detections


def iter_sentence_spans(
    text: str,
    *,
    max_sentences: int = 250,
) -> list[SentenceSpan]:
    """Split a document into sentence spans while preserving offsets."""
    spans: list[SentenceSpan] = []
    for paragraph in _iter_paragraphs(text):
        if len(spans) >= max_sentences:
            break
        for sentence_index, start, _end, sentence in _split_paragraph_sentences(
            paragraph.text
        ):
            if len(spans) >= max_sentences:
                break
            stripped = sentence.strip()
            if not stripped:
                continue
            trim_left = len(sentence) - len(sentence.lstrip())
            trim_right = len(sentence.rstrip())
            start_offset = paragraph.map_offset(start + trim_left)
            end_offset = paragraph.map_offset(start + trim_right)
            spans.append(
                SentenceSpan(
                    sentence_id=(
                        f"p{paragraph.paragraph_index}:s{sentence_index}"
                    ),
                    text=text[start_offset:end_offset],
                    start_offset=start_offset,
                    end_offset=end_offset,
                    paragraph_index=paragraph.paragraph_index,
                    sentence_index=sentence_index,
                    section_type=paragraph.section_type,
                )
            )
    return spans


def has_explicit_citation(sentence: str) -> bool:
    """Return True when the sentence already contains an inline citation."""
    return any(
        pattern.search(sentence)
        for pattern in (
            _LATEX_CITE_PATTERN,
            _MARKDOWN_CITE_PATTERN,
            _AUTHOR_YEAR_PATTERN,
        )
    )


def _iter_paragraphs(text: str) -> list[_Paragraph]:
    if _looks_like_latex(text):
        return _iter_latex_prose_paragraphs(text)
    return _iter_plain_paragraphs(text)


def _looks_like_latex(text: str) -> bool:
    return _LATEX_DOCUMENT_MARKERS.search(text) is not None


def _iter_plain_paragraphs(text: str) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    current_section: str | None = None
    paragraph_index = 0

    for match in _PARAGRAPH_PATTERN.finditer(text):
        block = match.group(0)
        stripped = block.strip()
        if not stripped:
            continue
        heading = _extract_heading(stripped)
        if heading is not None:
            current_section = _classify_section(heading)
            continue
        if _is_structural_block(stripped):
            continue
        paragraphs.append(
            _Paragraph(
                text=block,
                paragraph_index=paragraph_index,
                section_type=current_section,
                segments=(
                    _OffsetSegment(
                        block_start=0,
                        orig_start=match.start(),
                        length=len(block),
                    ),
                ),
            )
        )
        paragraph_index += 1

    return paragraphs


def _iter_latex_prose_paragraphs(text: str) -> list[_Paragraph]:
    """Extract prose lines from LaTeX while skipping commands and preamble noise."""
    paragraphs: list[_Paragraph] = []
    current_section: str | None = None
    paragraph_index = 0
    prose_lines: list[tuple[int, str]] = []

    def flush_prose() -> None:
        nonlocal paragraph_index, prose_lines
        if not prose_lines:
            return
        block_parts: list[str] = []
        segments: list[_OffsetSegment] = []
        block_cursor = 0
        for orig_start, content in prose_lines:
            if block_parts:
                block_cursor += 1  # the "\n" joining this line to the prior one
            segments.append(
                _OffsetSegment(
                    block_start=block_cursor,
                    orig_start=orig_start,
                    length=len(content),
                )
            )
            block_parts.append(content)
            block_cursor += len(content)
        paragraphs.append(
            _Paragraph(
                text="\n".join(block_parts),
                paragraph_index=paragraph_index,
                section_type=current_section,
                segments=tuple(segments),
            )
        )
        paragraph_index += 1
        prose_lines = []

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = offset + (len(line) - len(line.lstrip()))
        offset += len(line)

        if not stripped:
            flush_prose()
            continue

        heading_match = _LATEX_HEADING_LINE.match(stripped)
        if heading_match is not None:
            flush_prose()
            current_section = _classify_section(heading_match.group(1))
            continue

        if _is_latex_structural_line(stripped):
            flush_prose()
            continue

        prose_lines.append((line_start, stripped))

    flush_prose()
    return paragraphs


def _is_latex_structural_line(line: str) -> bool:
    if line.startswith("%"):
        return True
    if not line.startswith("\\"):
        return False
    return not _LATEX_HEADING_LINE.match(line)


def _split_paragraph_sentences(
    paragraph: str,
) -> list[tuple[int, int, int, str]]:
    boundaries = [0]
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(paragraph):
        boundaries.append(match.end())
    boundaries.append(len(paragraph))

    out: list[tuple[int, int, int, str]] = []
    for index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:], strict=False)
    ):
        sentence = paragraph[start:end]
        if sentence.strip():
            out.append((index, start, end, sentence))
    return out


def _extract_heading(block: str) -> str | None:
    first_line = block.splitlines()[0].strip()
    for pattern in _HEADING_PATTERNS:
        if not pattern.search(first_line):
            continue
        latex_match = re.search(r"\{([^{}]+)\}", first_line)
        if latex_match:
            return latex_match.group(1)
        return first_line.lstrip("#").strip()
    return None


def _classify_section(heading: str | None) -> str | None:
    if not heading:
        return None
    normalized = heading.lower()
    for needle, label in _SECTION_TYPE_RULES:
        if needle in normalized:
            return label
    return None


def _is_structural_block(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return True
    if block.startswith("```") or block.endswith("```"):
        return True
    structural = sum(1 for line in lines if _STRUCTURAL_LINE_PATTERN.search(line))
    return structural == len(lines)


def _is_not_citation_worthy(sentence: str) -> bool:
    words = _WORD_PATTERN.findall(sentence)
    if len(words) < MIN_MEANINGFUL_WORDS:
        return True
    if _STRUCTURAL_LINE_PATTERN.search(sentence):
        return True
    if _extract_heading(sentence) is not None:
        return True
    return any(pattern.search(sentence) for pattern in _OWN_WORK_PATTERNS)


def _score_citation_need(sentence: SentenceSpan) -> _CueScore:
    text = sentence.text
    score = 0.0
    reasons: list[str] = []

    for pattern, weight, reason in _STRONG_PRIOR_CUES:
        if pattern.search(text):
            score += weight
            reasons.append(reason)

    if _METHOD_USE_PATTERN.search(text):
        score += 0.25
        reasons.append("method-use cue")

    if _COMPARISON_PATTERN.search(text):
        score += 0.25
        reasons.append("comparison cue")

    if _SURVEY_PATTERN.search(text):
        score += 0.2
        reasons.append("survey-style sentence")

    if _NAMED_ARTIFACT_PATTERN.search(text):
        score += 0.2
        reasons.append("named model/dataset/metric")

    if sentence.section_type in {"introduction", "background", "related_work"}:
        score += 0.1
        reasons.append(f"{sentence.section_type} section")

    if "?" in text:
        score -= 0.15

    return _CueScore(score=min(1.0, max(0.0, score)), reasons=tuple(reasons))


def _is_near_cited_sentence(
    sentence: SentenceSpan,
    all_sentences: list[SentenceSpan],
    explicit_citation_ids: set[str],
) -> bool:
    for other in all_sentences:
        if other.paragraph_index != sentence.paragraph_index:
            continue
        if other.sentence_id not in explicit_citation_ids:
            continue
        if abs(other.sentence_index - sentence.sentence_index) <= 1:
            return True
    return False
