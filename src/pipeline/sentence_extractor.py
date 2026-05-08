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
    _SENTENCE_END_PUNCTUATION
)


def clean_text(text: str) -> str:
    """Basic text cleanup."""
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


def extract_sentences(parsed_paper: ParsedPaper) -> list[SentenceRecord]:
    """Split all sections into clean, annotated sentences.

    Each sentence's ``cited_bibkeys`` reflects the GROBID-linked
    ``[CITE:bX]`` markers that fell inside it. ``retrieval_text`` strips
    those markers (and the legacy natural-language citation regex) so it's
    safe to feed straight into a sentence encoder.
    """
    nlp = get_sentence_nlp()
    records: list[SentenceRecord] = []

    sections_to_process = {"Abstract": parsed_paper.abstract} if parsed_paper.abstract else {}
    sections_to_process.update(parsed_paper.sections)

    for section_name, section_text in sections_to_process.items():
        if not section_text or not section_text.strip():
            continue

        doc = nlp(section_text)

        # Raw sentences from spacy
        raw_sents = [sent.text.strip() for sent in doc.sents]

        # Clean and filter (noise filter runs on the citation-stripped form so
        # short sentences like "BERT [CITE:b3]." still register their
        # underlying length).
        valid_sents: list[str] = []
        for s in raw_sents:
            cleaned = clean_text(BULLET_MARKER_PATTERN.sub("", s))
            stripped = _strip_citation_artifacts(cleaned)
            if not stripped or is_noise(stripped):
                continue
            valid_sents.append(cleaned)

        total_sents = len(valid_sents)
        if total_sents == 0:
            continue

        for i, sent_text in enumerate(valid_sents):
            cited_bibkeys = GROBID_CITE_MARKER_PATTERN.findall(sent_text)
            has_grobid_marker = bool(cited_bibkeys)
            has_natural_cite = bool(CITATION_PATTERN.search(sent_text))
            has_cite = has_grobid_marker or has_natural_cite

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
                cited_bibkeys=list(dict.fromkeys(cited_bibkeys)),
            )
            records.append(record)

    return records
