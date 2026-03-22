import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import networkx as nx
import requests

logger = logging.getLogger(__name__)

# Constants
OPENALEX_API_URL = "https://api.openalex.org/works"
CACHE_FILE = Path(__file__).parent.parent.parent / "database" / "openalex_paper_cache.json"
MAX_REFERENCES = 40
RATE_LIMIT_DELAY = 0.2  # 5 requests per second

def _reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct an abstract from OpenAlex's inverted index format."""
    if not inverted_index:
        return ""
    
    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append((pos, word))
            
    word_index.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_index)


@dataclass
class CitationGraph:
    """
    Data class representing a citation graph built from a paper's bibliography.
    """
    graph: nx.Graph
    node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the CitationGraph into a dictionary representation."""
        return {
            "nodes": [
                {"id": n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ],
            "edges": [
                {"source": u, "target": v, "weight": d.get("weight", 1)}
                for u, v, d in self.graph.edges(data=True)
            ],
        }


class OpenAlexFetcher:
    """Handles rate-limited and cached requests to OpenAlex."""
    
    def __init__(self, cache_path: Path = CACHE_FILE):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.last_request_time = 0.0

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted. Starting fresh.")
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        cache_key = f"id:{paper_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()
        # OpenAlex expects IDs like W2740447471 or full URLs
        url = f"{OPENALEX_API_URL}/{paper_id}"
        
        try:
            resp = requests.get(url, params={"mailto": "polite-pool@example.com"})  # Important: add mailto for polite pool
            resp.raise_for_status()
            data = resp.json()
            # Normalize to match our required schema
            result = self._normalize_openalex_response(data)
            self.cache[cache_key] = result
            self._save_cache()
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch paper {paper_id}: {e}")
            return None

    def search_paper_by_string(self, query: str) -> Optional[Dict]:
        cache_key = f"search:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._rate_limit()
        
        try:
            import re
            
            # First, try standard search just in case
            resp = requests.get(OPENALEX_API_URL, params={
                "search": query,
                "per-page": 1,
                "mailto": "polite-pool@example.com"
            })
            resp.raise_for_status()
            data = resp.json()
            if data.get("results") and len(data["results"]) > 0:
                raw_result = data["results"][0]
                result = self._normalize_openalex_response(raw_result)
                self.cache[cache_key] = result
                self._save_cache()
                return result

            # Heuristic for full citation strings:
            # We split the string by typical separators and try to search the pieces as exact titles
            v = query.replace('_', ' ').replace('arXiv preprint', '')
            parts = re.split(r'[\.\?\!,]\s+', v)
            # Find parts with at least 2 words
            parts = [p.strip() for p in parts if len(p.split()) >= 2]
            # Sort by length descending, as titles are usually the longest or second longest part
            parts.sort(key=len, reverse=True)
            
            for p in parts:
                p_clean = re.sub(r'[^a-zA-Z0-9\s\-]', '', p)[:100]
                if not p_clean.strip(): continue
                
                self._rate_limit()
                resp = requests.get(OPENALEX_API_URL, params={
                    "filter": f"title.search:{p_clean}",
                    "per-page": 1,
                    "mailto": "polite-pool@example.com"
                })
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("results") and len(data["results"]) > 0:
                        raw_result = data["results"][0]
                        result = self._normalize_openalex_response(raw_result)
                        self.cache[cache_key] = result
                        self._save_cache()
                        return result
            
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return None
            
    def _normalize_openalex_response(self, raw_data: Dict) -> Dict:
        """Extract and normalize only the fields we need from OpenAlex fat payload."""
        return {
            "paperId": raw_data.get("id"),
            "title": raw_data.get("title"),
            "abstract": _reconstruct_openalex_abstract(raw_data.get("abstract_inverted_index")),
            "citationCount": raw_data.get("cited_by_count", 0),
            "references": raw_data.get("referenced_works", [])  # List of OpenAlex IDs
        }


class CitationGraphBuilder:
    """
    Class to build a citation graph (bibliographic coupling) from a list of references.
    """
    def __init__(self):
        self.fetcher = OpenAlexFetcher()

    def build_from_strings(self, references: List[str]) -> CitationGraph:
        """Build citation graph from raw reference strings."""
        resolved = []
        for ref in references:
            paper = self.fetcher.search_paper_by_string(ref)
            if paper:
                resolved.append(paper)
        return self._build_graph(resolved)

    def build_from_openalex_ids(self, oa_ids: List[str]) -> CitationGraph:
        """Build citation graph from OpenAlex IDs."""
        resolved = []
        for oa_id in oa_ids:
            paper = self.fetcher.get_paper_by_id(oa_id)
            if paper:
                resolved.append(paper)
        return self._build_graph(resolved)

    def _build_graph(self, papers: List[Dict]) -> CitationGraph:
        # Sort by citationCount descending and truncate
        papers.sort(key=lambda x: x.get("citationCount", 0) or 0, reverse=True)
        papers = papers[:MAX_REFERENCES]

        graph = nx.Graph()
        
        # Instantiate paper nodes
        for p in papers:
            s2_id = p.get("paperId")
            if not s2_id:
                continue
                
            num_citations = p.get("citationCount", 0) or 0
            
            graph.add_node(
                s2_id,
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                citation_count=num_citations
            )

        # Build Ego-networks (1-hop references)
        paper_refs = {}
        for p in papers:
            s2_id = p.get("paperId")
            if not s2_id:
                continue
                
            # OpenAlex references are directly a list of strings (IDs/URLs)
            refs = p.get("references", [])
            ref_ids = set(str(r) for r in refs)
            paper_refs[s2_id] = ref_ids

        # Construct Bibliographic Coupling Edges
        node_ids = list(graph.nodes())
        for i, id_a in enumerate(node_ids):
            for id_b in node_ids[i+1:]:
                refs_a = paper_refs.get(id_a, set())
                refs_b = paper_refs.get(id_b, set())
                
                shared = refs_a.intersection(refs_b)
                if len(shared) > 0:
                    # You could also compute Jaccard index here
                    # jaccard = len(shared) / len(refs_a.union(refs_b))
                    graph.add_edge(id_a, id_b, weight=len(shared), shared_refs=list(shared))

        return CitationGraph(
            graph=graph,
            node_ids=node_ids
        )

def build_citation_graph_from_strings(references: List[str]) -> CitationGraph:
    """Convenience function."""
    builder = CitationGraphBuilder()
    return builder.build_from_strings(references)
