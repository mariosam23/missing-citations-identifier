"""Data contracts for Stage 5 claim decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence


class AggregationStrategy(str, Enum):
    """How per-subclaim retrieval results should be merged."""

    WEIGHTED = "WEIGHTED"


@dataclass(frozen=True)
class Subclaim:
    """One atomic claim extracted from a larger citation-worthy sentence."""

    text: str
    importance: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Subclaim text must not be empty")
        if self.importance < 0:
            raise ValueError("Subclaim importance must be non-negative")


@dataclass(frozen=True)
class Decomposition:
    """A decomposed claim plus the strategy used to aggregate retrieval results."""

    original_text: str
    subclaims: Sequence[Subclaim]
    aggregation: AggregationStrategy = AggregationStrategy.WEIGHTED

    def __post_init__(self) -> None:
        if not self.original_text.strip():
            raise ValueError("Decomposition original_text must not be empty")
        if not self.subclaims:
            raise ValueError("Decomposition must contain at least one subclaim")
        if not isinstance(self.subclaims, tuple):
            object.__setattr__(self, "subclaims", tuple(self.subclaims))
