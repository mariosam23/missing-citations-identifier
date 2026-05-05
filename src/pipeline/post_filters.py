"""Post-filters for retrieval candidates (Stage 6)."""


from dataclasses import dataclass, field
from collections.abc import Sequence

from entities.retrieval_result import RetrievalResult


@dataclass
class FilterContext:
    """Metadata needed by post-filters."""
    source_paper_id: str | None = None
    query_year: int | None = None
    already_cited_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass
class FilterStats:
    """Statistics tracked during filter application."""
    before_count: int = 0
    after_count: int = 0
    removed_count: int = 0


class FilterBase:
    """Base class for all post-filters."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def apply(self, candidates: Sequence[RetrievalResult], context: FilterContext) -> list[RetrievalResult]:
        """Apply the filter, returning a new list of kept candidates."""
        raise NotImplementedError


class TemporalFilter(FilterBase):
    """Drops candidates published after the query year."""

    def apply(self, candidates: Sequence[RetrievalResult], context: FilterContext) -> list[RetrievalResult]:
        if context.query_year is None:
            return list(candidates)

        kept = []
        for candidate in candidates:
            if candidate.year is None:
                kept.append(candidate)
            elif candidate.year <= context.query_year:
                kept.append(candidate)
        return kept


class ReferenceDedupFilter(FilterBase):
    """Drops candidates that are already visibly cited."""

    def apply(self, candidates: Sequence[RetrievalResult], context: FilterContext) -> list[RetrievalResult]:
        if not context.already_cited_ids:
            return list(candidates)

        return [c for c in candidates if str(c.paper_id) not in context.already_cited_ids]


class SelfCitationFilter(FilterBase):
    """Drops candidates that are the source paper itself."""

    def apply(self, candidates: Sequence[RetrievalResult], context: FilterContext) -> list[RetrievalResult]:
        if not context.source_paper_id:
            return list(candidates)

        source_id = str(context.source_paper_id)
        return [c for c in candidates if str(c.paper_id) != source_id]


class SpecificityFilter(FilterBase):
    """Prunes low-score candidates conservatively."""

    def apply(self, candidates: Sequence[RetrievalResult], context: FilterContext) -> list[RetrievalResult]:
        if len(candidates) < 3:
            return list(candidates)

        # Assuming candidates are already sorted by score descending, but we'll find the max just in case
        top_score = max((c.score for c in candidates), default=0.0)
        
        if top_score <= 0:
            return list(candidates)

        threshold = 0.20 * top_score
        
        # Keep top 3 always, then prune others below threshold
        # Re-sort to guarantee top 3 are the highest score
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        kept = []
        for i, candidate in enumerate(sorted_candidates):
            if i < 3:
                kept.append(candidate)
            elif candidate.score >= threshold:
                kept.append(candidate)
        return kept


class PostFilterPipeline:
    """Applies a sequence of filters to retrieval candidates."""

    def __init__(self, filters: Sequence[FilterBase] | None = None):
        self.filters = filters if filters is not None else [
            TemporalFilter(),
            ReferenceDedupFilter(),
            SelfCitationFilter(),
            SpecificityFilter(),
        ]
        self.stats: dict[str, FilterStats] = {f.name: FilterStats() for f in self.filters}

    def apply(self, candidates: Sequence[RetrievalResult], context: FilterContext) -> list[RetrievalResult]:
        current = list(candidates)
        for f in self.filters:
            before = len(current)
            current = f.apply(current, context)
            after = len(current)
            
            self.stats[f.name].before_count += before
            self.stats[f.name].after_count += after
            self.stats[f.name].removed_count += (before - after)

        return current
