import unittest

from entities.retrieval_result import RetrievalResult
from pipeline.post_filters import (
    FilterContext,
    TemporalFilter,
    ReferenceDedupFilter,
    SelfCitationFilter,
    SpecificityFilter,
    PostFilterPipeline
)


class TestPostFilters(unittest.TestCase):
    def test_temporal_filter(self):
        filter_ = TemporalFilter()
        candidates = [
            RetrievalResult("1", "A", 0.9, year=2020),
            RetrievalResult("2", "B", 0.8, year=2025), # future
            RetrievalResult("3", "C", 0.7, year=None),  # missing year
            RetrievalResult("4", "D", 0.6, year=2024),
        ]
        context = FilterContext(query_year=2024)
        filtered = filter_.apply(candidates, context)
        
        self.assertEqual(len(filtered), 3)
        self.assertNotIn("2", [c.paper_id for c in filtered])
        
        # Test no query year
        context = FilterContext(query_year=None)
        self.assertEqual(len(filter_.apply(candidates, context)), 4)

    def test_reference_dedup_filter(self):
        filter_ = ReferenceDedupFilter()
        candidates = [
            RetrievalResult("1", "A", 0.9),
            RetrievalResult("2", "B", 0.8), # visible
            RetrievalResult("3", "C", 0.7), # hidden, but dedup filter doesn't know it's hidden, context shouldn't have it
        ]
        context = FilterContext(already_cited_ids=frozenset({"2", "99"}))
        filtered = filter_.apply(candidates, context)
        
        self.assertEqual(len(filtered), 2)
        self.assertNotIn("2", [c.paper_id for c in filtered])
        self.assertIn("3", [c.paper_id for c in filtered])
        
    def test_self_citation_filter(self):
        filter_ = SelfCitationFilter()
        candidates = [
            RetrievalResult("source123", "Self", 0.9),
            RetrievalResult("2", "Other", 0.8),
        ]
        context = FilterContext(source_paper_id="source123")
        filtered = filter_.apply(candidates, context)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].paper_id, "2")
        
    def test_specificity_filter(self):
        filter_ = SpecificityFilter()
        context = FilterContext()
        
        # Test < 3 candidates
        candidates = [
            RetrievalResult("1", "A", 0.9),
            RetrievalResult("2", "B", 0.1),
        ]
        self.assertEqual(len(filter_.apply(candidates, context)), 2)
        
        # Test pruning (top_score = 1.0, threshold = 0.20)
        candidates = [
            RetrievalResult("1", "A", 1.0),
            RetrievalResult("2", "B", 0.9),
            RetrievalResult("3", "C", 0.3),
            RetrievalResult("4", "D", 0.19), # drop
            RetrievalResult("5", "E", 0.0),  # drop
        ]
        filtered = filter_.apply(candidates, context)
        self.assertEqual(len(filtered), 3)
        self.assertNotIn("4", [c.paper_id for c in filtered])
        
        # Test non-positive top score
        candidates = [
            RetrievalResult("1", "A", 0.0),
            RetrievalResult("2", "B", -0.1),
            RetrievalResult("3", "C", -0.5),
            RetrievalResult("4", "D", -1.0),
        ]
        self.assertEqual(len(filter_.apply(candidates, context)), 4)
        
    def test_pipeline_stats_and_waterfall(self):
        pipeline = PostFilterPipeline()
        candidates = [
            RetrievalResult("1", "A", 1.0, year=2020),
            RetrievalResult("2", "B", 0.9, year=2025), # Temporal drop
            RetrievalResult("3", "C", 0.8, year=2020), # Dedup drop
            RetrievalResult("source_id", "D", 0.7, year=2020), # Self drop
            RetrievalResult("5", "E", 0.3, year=2020), # Keep (top 2)
            RetrievalResult("10", "Z", 0.25, year=2020), # Keep (top 3)
            RetrievalResult("6", "F", 0.1, year=2020), # Specificity drop (below 0.2 threshold, rank 4)
            RetrievalResult("7", "G", 0.05, year=2020), # Specificity drop (below 0.2 threshold, rank 5)
        ]
        context = FilterContext(
            source_paper_id="source_id",
            query_year=2024,
            already_cited_ids=frozenset({"3"})
        )
        
        filtered = pipeline.apply(candidates, context)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered[0].paper_id, "1")
        self.assertEqual(filtered[1].paper_id, "5")
        self.assertEqual(filtered[2].paper_id, "10")
        
        # Check stats waterfall
        self.assertEqual(pipeline.stats["TemporalFilter"].removed_count, 1)
        self.assertEqual(pipeline.stats["ReferenceDedupFilter"].removed_count, 1)
        self.assertEqual(pipeline.stats["SelfCitationFilter"].removed_count, 1)
        self.assertEqual(pipeline.stats["SpecificityFilter"].removed_count, 2)


if __name__ == '__main__':
    unittest.main()
