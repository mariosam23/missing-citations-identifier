import copy
import math
import random
from dataclasses import dataclass

from entities.sentence_record import SentenceRecord
from utils.regex_patterns import (
    CITATION_PATTERN,
    GROBID_CITE_MARKER_PATTERN,
    EMPTY_PAREN_PATTERN,
    ORPHAN_SEMICOLON_PATTERN,
    DOUBLED_PUNCTUATION_PATTERN,
    SPACE_BEFORE_PUNCTUATION_PATTERN,
    SPACE_AFTER_OPEN_PAREN_PATTERN,
)


@dataclass
class CitationRemovalResult:
    records: list[SentenceRecord]
    # Indices into `records` where citations were stripped — used as ground truth
    # for evaluating whether the pipeline correctly flags these sentences.
    removed_indices: list[int]


def _strip_citation_markers(text: str) -> str:
    """Remove all citation markers from raw sentence text and tidy up residual punctuation."""
    text = GROBID_CITE_MARKER_PATTERN.sub("", text)
    text = CITATION_PATTERN.sub("", text)
    text = EMPTY_PAREN_PATTERN.sub("", text)
    text = ORPHAN_SEMICOLON_PATTERN.sub("(", text)
    text = DOUBLED_PUNCTUATION_PATTERN.sub(r"\1", text)
    text = SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_PAREN_PATTERN.sub("(", text)
    return " ".join(text.split())


def _erase_citation(record: SentenceRecord) -> SentenceRecord:
    """Return a deep copy of *record* with all citation information removed."""
    rec = copy.copy(record)
    # Prefer the already-stripped retrieval_text; fall back to regex scrubbing.
    stripped = record.retrieval_text if record.retrieval_text is not None else _strip_citation_markers(record.text)
    rec.text = stripped
    rec.retrieval_text = stripped
    rec.has_citation = False
    rec.cited_bibkeys = []
    rec.citation_intent = None
    rec.citation_state = None
    rec.worthiness_score = None
    rec.urgency_score = None
    return rec


def remove_random_citations(
    records: list[SentenceRecord],
    n: int | None = None,
    fraction: float | None = None,
    seed: int | None = None,
) -> CitationRemovalResult:
    """Randomly strip citations from a subset of sentences that originally have one.

    Exactly one of *n* or *fraction* must be provided.

    Args:
        records:  Input sentence records (not mutated).
        n:        Exact number of cited sentences to strip.
        fraction: Fraction of cited sentences to strip (0.0 – 1.0].
        seed:     Optional RNG seed for reproducibility.

    Returns:
        A CitationRemovalResult whose ``records`` list is a copy of the input
        with citations erased for the selected sentences, and ``removed_indices``
        listing which positions were altered (usable as ground truth).
    """
    if (n is None) == (fraction is None):
        raise ValueError("Provide exactly one of `n` or `fraction`.")
    if fraction is not None and not (0.0 < fraction <= 1.0):
        raise ValueError("`fraction` must be in the range (0.0, 1.0].")

    eligible = [i for i, r in enumerate(records) if r.has_citation]
    if not eligible:
        return CitationRemovalResult(records=list(records), removed_indices=[])

    if n is None:
        n = max(1, math.ceil(fraction * len(eligible)))

    n = min(n, len(eligible))

    rng = random.Random(seed)
    chosen = sorted(rng.sample(eligible, n))

    chosen_set = set(chosen)
    result = [
        _erase_citation(r) if i in chosen_set else r
        for i, r in enumerate(records)
    ]

    return CitationRemovalResult(records=result, removed_indices=chosen)