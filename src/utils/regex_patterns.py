import re

# ==========================================
# Sentence Extraction Patterns
# ==========================================
CITATION_PATTERN = re.compile(
    r"\[\s*\d+(?:\s*[-,;]\s*\d+)*\s*\]"  # e.g. [1], [1, 2], [3-5], [1; 4]
    r"|"
    r"\("  # e.g. (Smith, 2020), (Smith & Lee, 2020; Jones et al., 2021)
    r"(?:"
    r"[A-Z][A-Za-z'`-]+"
    r"(?:\s+(?:et al\.|&|and)\s+[A-Z][A-Za-z'`-]+|\s+et al\.)?"
    r"(?:(?:,\s*)+)\d{4}[a-z]?"  # allow multiple commas due to PDF parsing occasionally producing ", , 2018"
    r"(?:\s*;\s*[A-Z][A-Za-z'`-]+(?:\s+(?:et al\.|&|and)\s+[A-Z][A-Za-z'`-]+|\s+et al\.)?(?:(?:,\s*)+)\d{4}[a-z]?)*"
    r")"
    r"\)"
    r"|"
    # narrative citation: e.g. Smith (2020), Smith et al. (2020), Smith and Jones (2020)
    r"[A-Z][A-Za-z'`-]+"
    r"(?:\s+(?:et al\.|&|and)\s+[A-Z][A-Za-z'`-]+|\s+et al\.)?"
    r"\s*"
    r"\(\s*\d{4}[a-z]?\s*\)"
)

BULLET_MARKER_PATTERN = re.compile(r"[•●▪■◦]")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
INLINE_FOOTNOTE_PATTERN = re.compile(r"\bfoot_\d+\b")
LEADING_FOOTNOTE_GLUE_PATTERN = re.compile(r"^\s*\d+(?=[A-Z])")
LEADING_FOOTNOTE_SPACED_PATTERN = re.compile(r"^\s*\d{1,2}\s+(?=[A-Z])")
LEADING_SECTION_NUMBER_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)+\s+")
BARE_PUNCTUATION_PATTERN = re.compile(r"^[^A-Za-z0-9]+$")
HEADING_LIKE_SENTENCE_PATTERN = re.compile(
    r"^(?:[A-Z][A-Za-z0-9'&/-]*)(?:\s+[A-Z][A-Za-z0-9'&/-]*){0,7}:?$"
)
PUNCTUATION_SPACING_PATTERN = re.compile(r"\s+([,.;:!?])")
OPEN_PAREN_SPACING_PATTERN = re.compile(r"\(\s+")
CLOSE_PAREN_SPACING_PATTERN = re.compile(r"\s+\)")

# ==========================================
# PDF Parser (Nougat / Markdown) Patterns
# ==========================================
MARKDOWN_TITLE_PATTERN = re.compile(r"^#\s+(.+?)(?=\n#|\n\n)", re.MULTILINE | re.DOTALL)
WHITESPACE_CLEANUP_PATTERN = re.compile(r"\s+")

PYMUPDF_ABSTRACT_PATTERN = re.compile(
    r"(?i)\bAbstract\b[\s\n]+(.*?)(?=\n\s*(?:1\.?\s+Introduction|I\.\s+Introduction|Introduction)\b)",
    re.DOTALL,
)

_SENTENCE_END_PUNCTUATION = ".?!"

# Pattern matching the `[CITE:bX]` markers injected by GrobidPDFParser. The
# bibkey is captured so we can attribute each marker back to a bibliography
# entry. Matches GROBID's standard ``b<digits>`` bibkey format and is
# deliberately strict so it won't false-match arbitrary bracketed text in the
# body.
GROBID_CITE_MARKER_PATTERN = re.compile(r"\[CITE:(b\d+)\]")

# GROBID often emits ``<ref target="#b1">Smith, 2020</ref>`` followed by an
# unwrapped tail like ``"; Peters et al., 2018a"`` between siblings. After we
# replace the ``<ref>`` with ``[CITE:b1]`` the tail leaks into the retrieval
# text. This pattern matches a bare author-year (``Smith, 2020``,
# ``Smith and Lee, 2020``, ``Smith et al., 2020a``) anywhere — without the
# enclosing parens that the main CITATION_PATTERN requires — so we can scrub
# the residue.
RESIDUAL_AUTHOR_YEAR_PATTERN = re.compile(
    r"(?<![A-Za-z])"
    r"[A-Z][A-Za-z'`-]+"
    r"(?:\s+(?:et al\.|&|and)\s+[A-Z][A-Za-z'`-]+|\s+et al\.)?"
    r",?\s*\(?\d{4}[a-z]?\)?"
)

# After removing markers + author-year residue, we can be left with empty
# parenthesised groups (``()``, ``( ; )``, ``(;)``) and stray punctuation
# clusters. These cleanup patterns finish the job.
EMPTY_PAREN_PATTERN = re.compile(r"\(\s*[;,\s]*\s*\)")
ORPHAN_SEMICOLON_PATTERN = re.compile(r"\(\s*;+\s*")
DOUBLED_PUNCTUATION_PATTERN = re.compile(r"\s*([;,])\s*\1+")
SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r"\s+([.,;:!?\)])")
SPACE_AFTER_OPEN_PAREN_PATTERN = re.compile(r"\(\s+")

MARKDOWN_HEADED_ABSTRACT_PATTERN = re.compile(
    r"(?i)^#{1,6}\s*abstract\s*\n(.*?)(?=^#{1,6}\s[^#]|\Z)",
    re.MULTILINE | re.DOTALL,
)

MARKDOWN_BOLD_ABSTRACT_PATTERN = re.compile(
    r"(?i)\*\*abstract\*\*[:\.\s]*(.*?)(?=\n\n#{1,6}\s|\n\n\*\*\d|\Z)",
    re.DOTALL,
)

MARKDOWN_PLAIN_ABSTRACT_PATTERN = re.compile(
    r"(?i)(?:^|\n)abstract[:\.\s]*\n+(.*?)(?=\n\n#{1,6}\s|\n\n\*\*\d|\Z)",
    re.DOTALL,
)

MARKDOWN_UNMARKED_ABSTRACT_PATTERN = re.compile(
    r"\n\+\+\+\n\n(.*?)(?=\n#{1,6}\s)",
    re.DOTALL,
)

MARKDOWN_FALLBACK_ABSTRACT_PATTERN = re.compile(
    r"^#\s+.+?\n\n(?:.*?\n\n)*((?:[A-Z].{80,}(?:\n\n(?!#).{80,})*))\s*\n#{1,6}\s",
    re.DOTALL,
)

MARKDOWN_REFERENCES_SECTION_PATTERN = re.compile(
    r"(?i)^#{1,6}\s*references?\s*\n(.*?)(?=^#{1,6}\s[^#]|\Z)",
    re.MULTILINE | re.DOTALL,
)

MARKDOWN_REFERENCE_ENTRY_SPLIT_PATTERN = re.compile(r"\n(?=\*\s|\[?\d+\]?[\.\s])")
MARKDOWN_REFERENCE_CLEAN_PREFIX1_PATTERN = re.compile(r"^\*\s+")
MARKDOWN_REFERENCE_CLEAN_PREFIX2_PATTERN = re.compile(r"^\[?\d+\]?[\.\s]+")

MARKDOWN_SECTION_SPLIT_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+)$")
MARKDOWN_SKIP_SECTION_PATTERN = re.compile(r"(?i)^(abstract|references?)$")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")

# Basic DOI regex pattern. Look for 10.NNNN/....
DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")

# Match OpenAlex Work IDs in either bare (W123…) or URL form. The local
# ``papers.paperId`` column stores them in bare form so we strip the prefix.
OPENALEX_ID_PATTERN = re.compile(r"(?:openalex\.org/)?(W\d{6,})", re.IGNORECASE)

