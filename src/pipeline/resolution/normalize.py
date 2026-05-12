"""Normalizers for identifiers and bibliographic strings.

All functions are pure (no I/O, no DB) and safe in tight loops.
"""

from __future__ import annotations

import re
import unicodedata

_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_NONALPHANUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

_DOI_URL_PREFIXES: tuple[str, ...] = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)

_ARXIV_URL_PREFIXES: tuple[str, ...] = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
    "https://arxiv.org/e-print/",
    "http://arxiv.org/e-print/",
)


def normalize_doi(doi: str) -> str:
    """Return the canonical bare DOI — lowercase, no URL prefix.

    Examples::

        normalize_doi("https://doi.org/10.1145/1234") → "10.1145/1234"
        normalize_doi("DOI:10.1145/1234")             → "10.1145/1234"
        normalize_doi("10.1145/1234")                 → "10.1145/1234"
    """
    doi = doi.strip().lower()
    for prefix in _DOI_URL_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Return the canonical bare arXiv ID — no prefix, no version suffix.

    Handles both old-style (``cs/0102004``) and new-style (``1234.5678``)
    identifiers, full URL forms, and ``arXiv:`` / ``arxiv:`` prefixes.

    Examples::

        normalize_arxiv_id("https://arxiv.org/abs/1706.03762") → "1706.03762"
        normalize_arxiv_id("arXiv:1706.03762v3")               → "1706.03762"
        normalize_arxiv_id("1706.03762v1")                     → "1706.03762"
    """
    arxiv_id = arxiv_id.strip()
    lower = arxiv_id.lower()

    for prefix in _ARXIV_URL_PREFIXES:
        if lower.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
            break

    if arxiv_id.lower().startswith("arxiv:"):
        arxiv_id = arxiv_id[6:]

    if arxiv_id.lower().endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]

    arxiv_id = _ARXIV_VERSION_RE.sub("", arxiv_id)
    return arxiv_id.strip()


def normalize_title(title: str) -> str:
    """Return a normalised title for Python-side comparison and fuzzy matching.

    Pipeline: NFKD → ASCII → lowercase → strip non-alphanum → collapse spaces.

    This is NOT compatible with the ``papers.normalized_title`` DB column
    (which keeps punctuation). Use ``db_normalize_title`` for DB lookups.
    """
    nfkd = unicodedata.normalize("NFKD", title)
    ascii_str = nfkd.encode("ascii", errors="ignore").decode("ascii")
    lower = ascii_str.lower()
    stripped = _NONALPHANUM_RE.sub(" ", lower)
    return _MULTI_SPACE_RE.sub(" ", stripped).strip()


def db_normalize_title(title: str) -> str:
    """Return the simple normalization stored in ``papers.normalized_title``.

    Matches ``parse_corpus._quick_normalize``: lowercase + whitespace collapse,
    punctuation preserved. Use this when querying the indexed DB column.
    """
    return _MULTI_SPACE_RE.sub(" ", title.lower()).strip()


def normalize_author(author: str) -> str:
    """Return a bare ASCII surname for author comparison.

    Handles both ``Smith, John`` and ``John Smith`` formats. Strips accents
    via NFKD so ``Müller`` and ``Muller`` compare equal.
    """
    if "," in author:
        surname = author.split(",", 1)[0].strip()
    else:
        parts = author.strip().split()
        surname = parts[-1] if parts else author

    nfkd = unicodedata.normalize("NFKD", surname)
    ascii_str = nfkd.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", ascii_str.lower())
