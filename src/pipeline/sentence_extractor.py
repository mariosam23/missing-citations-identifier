import re
import string

from utils.model_manager import get_sentence_nlp
from entities.parsed_paper import ParsedPaper
from entities.sentence_record import SentenceRecord
from utils.regex_patterns import (
    CITATION_PATTERN,
    WHITESPACE_CLEANUP_PATTERN,
    BULLET_MARKER_PATTERN,
    BARE_PUNCTUATION_PATTERN,
    HEADING_LIKE_SENTENCE_PATTERN,
    ORPHAN_SEMICOLON_PATTERN,
    EMPTY_PAREN_PATTERN,
    DOUBLED_PUNCTUATION_PATTERN,
    SPACE_BEFORE_PUNCTUATION_PATTERN,
    SPACE_AFTER_OPEN_PAREN_PATTERN,
    GROBID_CITE_MARKER_PATTERN,
    RESIDUAL_AUTHOR_YEAR_PATTERN,
    _SENTENCE_END_PUNCTUATION,
    HYPHENATION_PATTERN,
    GROBID_FUSED_HYPHEN_PATTERN,
    BIBLIOGRAPHY_SURNAME_PATTERN,
    BIBLIOGRAPHY_YEAR_PATTERN,
    INTEXT_AUTHOR_YEAR_PATTERN,
    KNOWN_DEHYPHENATION_FIXES,
)


# ------------------------------------------------------------------
# PDF Text Cleanup
# ------------------------------------------------------------------

# Pre-compile a single regex for known dehyphenation fixes.  The pattern
# matches any of the fused tokens (case-insensitive, word-boundary).
if KNOWN_DEHYPHENATION_FIXES:
    _DEHYPHENATION_RE: re.Pattern[str] | None = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in KNOWN_DEHYPHENATION_FIXES) + r")\b",
        re.IGNORECASE,
    )
else:
    _DEHYPHENATION_RE = None


def _fix_fused_compounds(text: str) -> str:
    """Re-insert hyphens/spaces into known fused compound words.

    GROBID / PDF extractors strip hyphens from line breaks, producing
    tokens like ``lefttoright`` or ``taskspecific``. This replaces them
    with the correct form using a known-word lookup.
    """
    if _DEHYPHENATION_RE is None:
        return text
    return _DEHYPHENATION_RE.sub(
        lambda m: KNOWN_DEHYPHENATION_FIXES.get(m.group(0).lower(), m.group(0)),
        text,
    )


def clean_text(text: str) -> str:
    """Basic text cleanup: fix hyphens, fused compounds, collapse whitespace."""
    text = HYPHENATION_PATTERN.sub(r"\1\2", text)
    # Fix GROBID artifacts like "left-toright" -> look up the fused form
    # (without the hyphen) in the known-compounds table.
    def _fix_fused_hyphen(m: re.Match) -> str:
        fused = (m.group(1) + m.group(2)).lower()
        if fused in KNOWN_DEHYPHENATION_FIXES:
            return KNOWN_DEHYPHENATION_FIXES[fused]
        return m.group(0)  # leave unchanged if not a known compound
    text = GROBID_FUSED_HYPHEN_PATTERN.sub(_fix_fused_hyphen, text)
    text = _fix_fused_compounds(text)
    text = WHITESPACE_CLEANUP_PATTERN.sub(" ", text).strip()
    return text


def is_noise(text: str) -> bool:
    """Filter out noise sentences (headings, bare punctuation, very short lines)."""
    text = text.strip()
    if len(text.split()) < 4:
        return True
    if BARE_PUNCTUATION_PATTERN.match(text):
        return True
    if HEADING_LIKE_SENTENCE_PATTERN.match(text):
        return True

    # Filter out sentences that contain too much punctuation/noise
    punct_count = sum(1 for c in text if c in string.punctuation)
    if len(text) > 0 and (punct_count / len(text)) > 0.25:
        return True

    # Filter out sentences that are mostly repeated dots/noise
    if text.count("..") > 2:
        return True

    return False


def _balance_parens(text: str) -> str:
    """Clip an unbalanced parenthetical residue from a sentence.

    GROBID's interleaving of ``<ref>`` markers with author-year text can leave
    spaCy with a sentence boundary inside a parenthesised citation, producing
    an opening ``(`` without a matching ``)``. Rather than ship that as a
    retrieval query, we clip from the dangling open paren to end-of-sentence
    while preserving any sentence-final punctuation. A symmetric clip handles
    the rarer ``")"`` -orphan case.
    """
    open_count = text.count("(")
    close_count = text.count(")")
    if open_count > close_count:
        last_open = text.rfind("(")
        if last_open != -1:
            trailing_punct = ""
            stripped = text.rstrip()
            if stripped and stripped[-1] in _SENTENCE_END_PUNCTUATION:
                trailing_punct = stripped[-1]
            text = text[:last_open].rstrip() + trailing_punct
    elif close_count > open_count:
        first_close = text.find(")")
        if first_close != -1:
            text = text[:first_close] + text[first_close + 1 :]
    return text


def _strip_citation_artifacts(text: str) -> str:
    """Remove GROBID markers, natural-language citations, and residual glue.

    Order matters:
      1. ``[CITE:bX]`` markers (cheap and unambiguous).
      2. Full natural-language citations (parenthesised or narrative).
      3. Bare author-year residues left behind by GROBID's ``<ref>`` tails
         (``"Smith et al., 2020a"`` between two markers).
      4. Empty parens, orphan semicolons, doubled punctuation, and the
         whitespace they create.
      5. Unbalanced-paren clip when a citation got split across spaCy
         sentence boundaries.

    Returning a clean retrieval-ready sentence means downstream embeddings see
    only the *substantive* claim, not citation chrome.
    """
    out = GROBID_CITE_MARKER_PATTERN.sub("", text)
    out = CITATION_PATTERN.sub("", out)
    out = RESIDUAL_AUTHOR_YEAR_PATTERN.sub("", out)
    out = ORPHAN_SEMICOLON_PATTERN.sub("(", out)
    out = EMPTY_PAREN_PATTERN.sub("", out)
    out = _balance_parens(out)
    out = DOUBLED_PUNCTUATION_PATTERN.sub(r"\1", out)
    out = SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r"\1", out)
    out = SPACE_AFTER_OPEN_PAREN_PATTERN.sub("(", out)
    return clean_text(out)


# ------------------------------------------------------------------
# Author-Year → Bibkey Resolution
# ------------------------------------------------------------------

def _build_author_year_index(bibliography: dict[str, str]) -> dict[tuple[str, str], list[str]]:
    """Build a (surname_lower, year) → [bibkey, …] lookup from the bibliography.

    Multiple entries may share the same first-author + year (e.g. Peters 2018a
    vs Peters 2018b); we keep them all so we can at least attribute a superset.

    To handle both ``"Surname, First. 2020."`` and ``"First Surname, … 2020."``
    formats we grab *every* capitalised word before the first comma/period and
    index all of them (the surname is always among them).
    """
    index: dict[tuple[str, str], list[str]] = {}
    for bibkey, raw_text in bibliography.items():
        year_m = BIBLIOGRAPHY_YEAR_PATTERN.search(raw_text)
        if not year_m:
            continue
        year = year_m.group(1)

        # All capitalised tokens before the first comma or period.
        for surname_m in BIBLIOGRAPHY_SURNAME_PATTERN.finditer(raw_text):
            # Stop after the first year to avoid matching title words.
            if surname_m.start() > year_m.start():
                break
            surname = surname_m.group(1).lower()
            key = (surname, year)
            if key not in index:
                index[key] = []
            if bibkey not in index[key]:
                index[key].append(bibkey)
    return index


def _resolve_author_year_citations(
    text: str,
    author_year_index: dict[tuple[str, str], list[str]],
) -> list[str]:
    """Find (Author, Year) citations in *text* and return matching bibkeys.

    This complements the ``[CITE:bX]`` extraction: author-year patterns that
    GROBID didn't link to a ``<ref target=…>`` now get resolved via the
    bibliography reverse-index.
    """
    resolved: list[str] = []
    for m in INTEXT_AUTHOR_YEAR_PATTERN.finditer(text):
        surname = m.group("surname").lower()
        year = m.group("year")
        # Strip trailing letter (e.g. "2018a" → "2018") for index lookup
        year_base = year[:4]
        for key in [(surname, year_base)]:
            if key in author_year_index:
                resolved.extend(author_year_index[key])
    return resolved


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_sentences(parsed_paper: ParsedPaper) -> list[SentenceRecord]:
    """Split all sections into clean, annotated sentences.

    Each sentence's ``cited_bibkeys`` reflects both GROBID-linked
    ``[CITE:bX]`` markers and author-year citations resolved against the
    bibliography. ``retrieval_text`` strips all citation chrome so it's
    safe to feed straight into a sentence encoder.
    """
    nlp = get_sentence_nlp()
    records: list[SentenceRecord] = []

    # Build the reverse index once for the whole paper.
    author_year_index = _build_author_year_index(
        parsed_paper.bibliography if parsed_paper.bibliography else {}
    )

    sections_to_process = {"Abstract": parsed_paper.abstract} if parsed_paper.abstract else {}
    sections_to_process.update(parsed_paper.sections)

    for section_name, section_text in sections_to_process.items():
        if not section_text or not section_text.strip():
            continue

        # Apply PDF-artifact cleanup *before* sentence splitting.
        section_text = clean_text(section_text)

        doc = nlp(section_text)

        # Raw sentences from spacy
        raw_sents = [sent.text.strip() for sent in doc.sents]

        # Clean and filter (noise filter runs on the citation-stripped form so
        # short sentences like "BERT [CITE:b3]." still register their
        # underlying length).
        # orphan_bibkeys[i] collects citations from noise-filtered fragments
        # that immediately follow valid_sents[i] — e.g. when spaCy splits a
        # sentence at a line-break artefact and the trailing "(Author, Year)."
        # lands in a separate fragment that is then noise-filtered away.
        valid_sents: list[str] = []
        orphan_bibkeys: list[list[str]] = []
        for s in raw_sents:
            cleaned = clean_text(BULLET_MARKER_PATTERN.sub("", s))
            stripped = _strip_citation_artifacts(cleaned)
            if not stripped or is_noise(stripped):
                if valid_sents:
                    g = GROBID_CITE_MARKER_PATTERN.findall(cleaned)
                    a = _resolve_author_year_citations(cleaned, author_year_index)
                    orphan_bibkeys[-1].extend(g + a)
                continue
            valid_sents.append(cleaned)
            orphan_bibkeys.append([])

        total_sents = len(valid_sents)
        if total_sents == 0:
            continue

        for i, sent_text in enumerate(valid_sents):
            # --- Unified citation extraction pass ---
            # 1. GROBID [CITE:bX] markers (always authoritative).
            grobid_bibkeys = GROBID_CITE_MARKER_PATTERN.findall(sent_text)

            # 2. Author-year citations resolved via bibliography index.
            author_year_bibkeys = _resolve_author_year_citations(
                sent_text, author_year_index
            )

            # 3. Citations recovered from noise-filtered trailing fragments.
            trailing_bibkeys = orphan_bibkeys[i]

            # Merge and deduplicate, preserving order.
            all_bibkeys = list(
                dict.fromkeys(grobid_bibkeys + author_year_bibkeys + trailing_bibkeys)
            )

            has_grobid_marker = bool(grobid_bibkeys)
            has_natural_cite = bool(CITATION_PATTERN.search(sent_text))
            has_cite = (
                has_grobid_marker
                or has_natural_cite
                or bool(author_year_bibkeys)
                or bool(trailing_bibkeys)
            )

            retrieval_text = _strip_citation_artifacts(sent_text)

            pos = i / max(total_sents - 1, 1) if total_sents > 1 else 0.0

            prev_sent = valid_sents[i - 1] if i > 0 else None
            next_sent = valid_sents[i + 1] if i < total_sents - 1 else None

            record = SentenceRecord(
                text=sent_text,
                section=section_name,
                position_in_section=pos,
                has_citation=has_cite,
                citation_intent=None,
                retrieval_text=retrieval_text,
                previous_sentence=prev_sent,
                next_sentence=next_sent,
                cited_bibkeys=all_bibkeys,
            )
            records.append(record)

    return records
