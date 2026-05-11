"""Sentence segmentation via ``pysbd``.

pysbd handles the citation residue (``[CITE:b12]`` markers, residual
``Smith et al., 2020`` fragments) better than spaCy's default punkt-style
segmenter and is dramatically lighter.

We provide a tiny wrapper so callers don't have to know about the segmenter
options or the ``language='en'`` argument that ``pysbd.Segmenter`` requires.
"""

from __future__ import annotations

from functools import lru_cache

import pysbd  # type: ignore[import-untyped]


@lru_cache(maxsize=1)
def _segmenter() -> pysbd.Segmenter:
    """Build a cached segmenter. ``clean=False`` preserves offsets in input."""
    return pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    """Return non-empty sentences from ``text``."""
    if not text or not text.strip():
        return []
    raw = _segmenter().segment(text)
    return [s.strip() for s in raw if s and s.strip()]
