"""Pure-function BibTeX generation from a ``Paper`` ORM row.

Responsibilities
-----------------
- Detect the best ``@entry_type`` from ``paper.venue`` / ``paper.source``.
- Format the JSONB ``authors`` blob into BibTeX canonical
  ``Last, First and Last, First`` form.
- Escape Unicode → LaTeX via ``pylatexenc``.
- Assemble a complete BibTeX entry string.

Design notes
~~~~~~~~~~~~
* We intentionally do **not** wrap titles in ``{...}`` braces — that
  defeats the BibTeX style file's casing rules.
* Only fields present in the ``papers`` schema are emitted.  Volume /
  number / pages are out of scope (Phase 5 spec §113).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pylatexenc.latexencode import unicode_to_latex

from database.postgres.tables.papers import Paper

# ── Entry-type detection ──────────────────────────────────────────────

_PROCEEDINGS_TOKENS: frozenset[str] = frozenset(
    {"proceedings", "conference", "workshop", "symposium"}
)
_JOURNAL_TOKENS: frozenset[str] = frozenset(
    {"journal", "transactions", "letters"}
)
# Well-known conference series whose venue names lack the word "proceedings".
_CONFERENCE_PREFIXES: tuple[str, ...] = (
    "advances in ",  # NeurIPS: "Advances in Neural Information Processing Systems"
)

EntryType = Literal["article", "inproceedings", "misc"]


def detect_entry_type(paper: Paper) -> EntryType:
    """Heuristic entry-type classification from venue / source strings."""
    venue = (paper.venue or "").lower()
    source = (paper.source or "").lower()

    if any(tok in venue for tok in _PROCEEDINGS_TOKENS):
        return "inproceedings"
    if any(venue.startswith(pfx) for pfx in _CONFERENCE_PREFIXES):
        return "inproceedings"
    if any(tok in venue for tok in _JOURNAL_TOKENS):
        return "article"
    if not venue and (paper.arxiv_id or "arxiv" in source):
        return "misc"
    if venue:
        return "article"  # safest default for unknown peer-reviewed
    return "misc"


# ── Author formatting ────────────────────────────────────────────────

def _flip_name(name: str) -> str:
    """Convert ``First Last`` → ``Last, First``; leave ``Last, First`` as-is."""
    name = name.strip()
    if "," in name:
        return name  # already in BibTeX order
    parts = name.rsplit(" ", 1)
    if len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    return name  # single-token name


def _escape_author(name: str) -> str:
    """Narrow LaTeX escape that preserves ``,`` and ``and`` (BibTeX-significant)."""
    # Only escape characters that are problematic in LaTeX but NOT
    # the comma or the word "and" which carry BibTeX semantics.
    result: list[str] = []
    for ch in name:
        if ch in (",", " "):
            result.append(ch)
        elif ch == "&":
            result.append(r"\&")
        elif ch == "#":
            result.append(r"\#")
        elif ch == "%":
            result.append(r"\%")
        elif ch == "_":
            result.append(r"\_")
        elif ord(ch) > 127:
            result.append(unicode_to_latex(ch))
        else:
            result.append(ch)
    return "".join(result)


def format_authors(paper: Paper) -> str:
    """Build ``Last, First and Last, First and ...`` from the JSONB blob."""
    names = _extract_name_strings(paper)
    if not names:
        return _escape_author(paper.first_author or "Anonymous")
    bibtex_names = [_escape_author(_flip_name(n)) for n in names]
    return " and ".join(bibtex_names)


def _extract_name_strings(paper: Paper) -> list[str]:
    """Best-effort flatten of the JSONB ``authors`` blob."""
    blob: Any = paper.authors
    if not isinstance(blob, dict):
        return []
    raw = blob.get("list")
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            out.append(entry.strip())
        elif isinstance(entry, dict):
            # Handle ``{family, given}`` shape (Semantic Scholar)
            family = entry.get("family", "")
            given = entry.get("given", "")
            if family and given:
                out.append(f"{given} {family}")
            elif family:
                out.append(family)
            else:
                # OpenAlex shape: ``{name: "..."}`` or ``{display_name: "..."}``
                name = entry.get("name") or entry.get("display_name") or ""
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
    return out


# ── LaTeX escaping ────────────────────────────────────────────────────

def escape_latex(value: str) -> str:
    """Full Unicode → LaTeX escaping for titles and venue strings."""
    return unicode_to_latex(value)


# ── Main entry point ─────────────────────────────────────────────────

def paper_to_bibtex(paper: Paper, citation_key: str) -> str:
    """Render a complete BibTeX entry for *paper*.

    Parameters
    ----------
    paper:
        The ORM ``Paper`` row with metadata.
    citation_key:
        The ``surname-year-word`` key, already disambiguated.

    Returns
    -------
    str
        A ready-to-append BibTeX entry (always ends with ``\\n``).
    """
    entry_type = detect_entry_type(paper)
    authors = format_authors(paper)
    title = escape_latex(paper.canonical_title)
    year = str(paper.year) if paper.year else ""

    fields: list[tuple[str, str]] = [
        ("title", f"{{{title}}}"),
        ("author", f"{{{authors}}}"),
    ]
    if year:
        fields.append(("year", f"{{{year}}}"))

    # Venue field depends on entry type.
    venue = paper.venue or ""
    if entry_type == "article" and venue:
        fields.append(("journal", f"{{{escape_latex(venue)}}}"))
    elif entry_type == "inproceedings" and venue:
        fields.append(("booktitle", f"{{{escape_latex(venue)}}}"))

    # Identifiers.
    if paper.doi:
        fields.append(("doi", f"{{{paper.doi}}}"))
    if entry_type == "misc" and paper.arxiv_id:
        # Normalise to bare ID for the eprint field.
        arxiv_clean = re.sub(r"^https?://arxiv\.org/abs/", "", paper.arxiv_id)
        fields.append(("eprint", f"{{{arxiv_clean}}}"))
        fields.append(("archivePrefix", "{arXiv}"))
    if paper.url:
        fields.append(("url", f"{{{paper.url}}}"))

    body = ",\n".join(f"  {k} = {v}" for k, v in fields)
    # Sanitise key: only alphanums, hyphens, underscores, colons.
    safe_key = _sanitise_key(citation_key)
    return f"@{entry_type}{{{safe_key},\n{body}\n}}\n"


_SAFE_KEY_RE = re.compile(r"[^a-zA-Z0-9_:\-]")


def _sanitise_key(key: str) -> str:
    """Strip characters that are invalid inside a BibTeX citation key."""
    nfkd = unicodedata.normalize("NFKD", key)
    ascii_key = nfkd.encode("ascii", "ignore").decode("ascii")
    return _SAFE_KEY_RE.sub("", ascii_key)
