"""Pipeline component exports.

Imports are resolved lazily so optional dependencies for one stage do not block
importing components from another stage.
"""

from __future__ import annotations

__all__ = ["GrobidPDFParser", "CrossEncoderReranker", "extract_sentences"]


def __getattr__(name: str):
    if name == "GrobidPDFParser":
        from .pdf_parser import GrobidPDFParser

        return GrobidPDFParser
    if name == "CrossEncoderReranker":
        from .reranker import CrossEncoderReranker

        return CrossEncoderReranker
    if name == "extract_sentences":
        from .sentence_extractor import extract_sentences

        return extract_sentences
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
