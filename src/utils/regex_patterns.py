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
