import unittest
from unittest.mock import patch, MagicMock

from pipeline.reference_resolver import (
    extract_doi,
    normalize_title,
    ReferenceResolver,
)

class TestReferenceResolver(unittest.TestCase):

    def test_extract_doi(self):
        self.assertEqual(extract_doi("Smith et al. 2020. 10.1234/5678. Journal."), "10.1234/5678")
        self.assertEqual(extract_doi("https://doi.org/10.1109/TSE.2018.2810892"), "10.1109/tse.2018.2810892")
        self.assertIsNone(extract_doi("No DOI here"))
        self.assertEqual(extract_doi("DOI: 10.1000/182,"), "10.1000/182")

    def test_normalize_title(self):
        self.assertEqual(normalize_title("A Study of   Stuff!"), "a study of stuff")
        self.assertEqual(normalize_title("Case-Study: The End."), "casestudy the end")
        self.assertEqual(normalize_title(""), "")

    @patch("pipeline.reference_resolver.get_session")
    def test_resolver_exact_doi(self, mock_get_session):
        # Setup mock db
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        
        mock_paper = MagicMock()
        mock_paper.paperId = "p123"
        mock_paper.title = "Found Paper"
        mock_paper.doi = "10.1234/abcd"
        mock_session.query.return_value.filter.return_value.first.return_value = mock_paper
        
        resolver = ReferenceResolver()
        result = resolver.resolve("Ref with DOI 10.1234/ABCD")
        
        self.assertTrue(result.is_resolved)
        self.assertEqual(result.method, "exact_doi")
        self.assertEqual(result.resolved_paper_id, "p123")
        self.assertEqual(resolver.stats["exact_doi"], 1)

    @patch("pipeline.reference_resolver.get_session")
    @patch("pipeline.reference_resolver.requests.get")
    def test_resolver_openalex_fallback(self, mock_requests_get, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        # Mock DB returns the paper for the OpenAlex DOI lookup
        mock_session.query.return_value.filter.return_value.first.return_value = MagicMock(paperId="local_123", title="OpenAlex Best Match")
        
        # Mock OpenAlex response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "OpenAlex Best Match",
                    "doi": "https://doi.org/10.9999/xyz",
                    "id": "https://openalex.org/W123"
                }
            ]
        }
        mock_requests_get.return_value = mock_response
        
        with patch("pipeline.reference_resolver.config") as mock_config:
            mock_config.OPEN_ALEX_EMAIL = "test@example.com"
            mock_config.OPEN_ALEX_API_KEY = None
            
            resolver = ReferenceResolver()
            result = resolver.resolve("Some raw reference without DOI")
            
            self.assertTrue(result.is_resolved)
            self.assertEqual(result.method, "openalex")
            self.assertEqual(result.resolved_paper_id, "local_123")
            self.assertEqual(result.title, "OpenAlex Best Match")
            
    @patch("pipeline.reference_resolver.get_session")
    def test_resolver_unresolved(self, mock_get_session):
        # Mock DB fails
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        # Test without OpenAlex configured
        with patch("pipeline.reference_resolver.config") as mock_config:
            mock_config.OPEN_ALEX_EMAIL = None
            
            resolver = ReferenceResolver()
            result = resolver.resolve("Just some string")
            
            self.assertFalse(result.is_resolved)
            self.assertEqual(result.method, "unresolved")

if __name__ == '__main__':
    unittest.main()
