"""Database ORM table definitions."""

from .citation import Citation
from .paper import Paper
from .corpus_paper import CorpusPaper

__all__ = [
    "Citation",
    "CorpusPaper",
    "Paper",
]
