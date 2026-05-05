"""Pipeline component exports.

Imports are resolved lazily so optional dependencies for one stage do not block
importing components from another stage.
"""


__all__ = [
    "ClaimDecomposer",
    "DecomposedRetriever",
    "GrobidPDFParser",
    "CrossEncoderReranker",
    "HybridRetriever",
    "UrgencyScorer",
    "weighted_rrf_aggregate",
    "extract_sentences",
]


def __getattr__(name: str):
    if name == "ClaimDecomposer":
        from .claim_decomposer import ClaimDecomposer

        return ClaimDecomposer
    if name == "DecomposedRetriever":
        from .aggregator import DecomposedRetriever

        return DecomposedRetriever
    if name == "GrobidPDFParser":
        from .pdf_parser import GrobidPDFParser

        return GrobidPDFParser
    if name == "CrossEncoderReranker":
        from .reranker import CrossEncoderReranker

        return CrossEncoderReranker
    if name == "HybridRetriever":
        from database.qdrant import HybridRetriever

        return HybridRetriever
    if name == "UrgencyScorer":
        from .urgency_scorer import UrgencyScorer

        return UrgencyScorer
    if name == "weighted_rrf_aggregate":
        from .aggregator import weighted_rrf_aggregate

        return weighted_rrf_aggregate
    if name == "extract_sentences":
        from .sentence_extractor import extract_sentences

        return extract_sentences
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
