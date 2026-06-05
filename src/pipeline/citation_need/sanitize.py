"""Pre-LLM input sanitization for the citation-need identifier.

The weak-supervision sentences fed to the identifier are mined from GROBID TEI by
stripping ``[CITE:bN]`` markers. That stripping leaves residue that both confuses
the model and *leaks the gold label*: orphaned ``]``/``)`` brackets (``"...
techniques ] that..."``), dangling ``(e.g.,`` openers, leading figure-number
tokens (``"2c Interestingly..."``), and truncated trailing connectives
(``"...the two equations (1) and."``). Non-prose lines (numbered list items,
captions, equation lines) slip in too.

This module is a pure, deterministic pre-processing layer used *before* any API
call. It does two things:

* :func:`clean_for_llm` removes citation-stripping residue so the model judges
  clean prose. It only deletes punctuation residue and collapses whitespace —
  it never rewrites words.
* :func:`is_classifiable` rejects targets that are not genuine prose claims, so
  they are decided "no citation" without spending an LLM call (also a quota win).

Both are conservative by design: a borderline real sentence should pass through
rather than be silently dropped. Nothing here changes the prompt wording or the
rule cue-scoring; it only manipulates the input text.
"""

from __future__ import annotations

import re

from pipeline.missing_citations.detector import _STRUCTURAL_LINE_PATTERN
from utils.config import config

# A word for the meaningful-length floor: a letter-led alphanumeric token.
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

# Residue left behind when an in-text citation marker is stripped.
_ORPHAN_BRACKET_GROUP = re.compile(r"\s*\[\s*\]")  # empty "[ ]"
_ORPHAN_PAREN_GROUP = re.compile(r"\(\s*(?:e\.g\.|cf\.|see|i\.e\.)?\s*[,;]?\s*\)")
# A dangling open "(e.g.," / "(" with nothing useful before the next break.
_DANGLING_OPEN_PAREN = re.compile(r"\(\s*(?:e\.g\.|cf\.|see|i\.e\.)\s*[,;]?(?=\s|$)")
# A lone "]" or ")" that lost its partner during stripping, kept tight to the
# preceding word: "techniques ] that" -> "techniques that"; "in 2004 in ]." ->
# handled by the dangling-connective rule below.
_ORPHAN_CLOSE_BRACKET = re.compile(r"(?<=\w)\s+[\])](?=\s|[.,;:]|$)")
# A connective left pointing at a now-deleted marker: "...published in ]." or
# "...based on )." -> drop the trailing "in ]"/"on )".
_DANGLING_CONNECTIVE = re.compile(
    r"\b(?:in|on|by|of|from|see|cf)\s*[\])](?=\s*[.,;:]?\s*$)",
    re.IGNORECASE,
)
# A leading figure/table sub-panel locator such as "2c " or "Fig. 2c " glued to
# the front of a sentence by the parser, when a real capitalized clause follows.
# Requires a digit+letter suffix ("2c", "3a") so plain "3. " enumeration markers
# are left for the list-item rejection in ``is_classifiable``.
_LEADING_FIGURE_TOKEN = re.compile(
    r"^\s*(?:fig\.?|figure|table|tab\.?|eq\.?|equation)?\s*\d+[a-z](?:\)|\.)?\s+(?=[A-Z])",
    re.IGNORECASE,
)

_MULTISPACE = re.compile(r"[ \t ]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")

# Non-prose line shapes that never warrant a citation in the "claim" sense.
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*•‣◦])\s")
_NON_ALPHA = re.compile(r"[^A-Za-z]")
_LOWERCASE_WORD = re.compile(r"\b[a-z]{2,}\b")

# Above this fraction of non-alphabetic characters a "sentence" is effectively
# an equation / table row / identifier dump, not prose.
_MAX_NON_ALPHA_FRACTION = 0.5


def clean_for_llm(text: str) -> str:
    """Return ``text`` with citation-stripping residue removed.

    Only punctuation residue and redundant whitespace are touched; the wording is
    preserved. Idempotent.
    """
    cleaned = text
    cleaned = _DANGLING_CONNECTIVE.sub("", cleaned)
    cleaned = _ORPHAN_BRACKET_GROUP.sub("", cleaned)
    cleaned = _ORPHAN_PAREN_GROUP.sub("", cleaned)
    cleaned = _DANGLING_OPEN_PAREN.sub("", cleaned)
    cleaned = _ORPHAN_CLOSE_BRACKET.sub("", cleaned)
    cleaned = _LEADING_FIGURE_TOKEN.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _MULTISPACE.sub(" ", cleaned)
    return cleaned.strip()


def is_classifiable(text: str, *, min_words: int | None = None) -> bool:
    """Return True when ``text`` is genuine prose worth an LLM decision.

    Rejects (so the caller decides "no citation" without an API call): too-short
    text, numbered/bulleted list items, structural/equation/caption lines, lines
    dominated by non-alphabetic characters, and fragments with no lowercase word.
    Expects already-:func:`clean_for_llm`-ed text but is safe on raw input too.
    """
    floor = min_words if min_words is not None else config.CITATION_NEED_MIN_WORDS
    stripped = text.strip()
    if not stripped:
        return False
    if len(_WORD_PATTERN.findall(stripped)) < floor:
        return False
    if _LIST_ITEM_PATTERN.match(stripped):
        return False
    if _STRUCTURAL_LINE_PATTERN.search(stripped):
        return False
    if not _LOWERCASE_WORD.search(stripped):
        return False
    non_alpha = len(_NON_ALPHA.findall(stripped))
    if non_alpha / len(stripped) > _MAX_NON_ALPHA_FRACTION:
        return False
    return True
