from rank_bm25 import BM25Okapi

class SimpleCandidateRetriever:
    def __init__(self, corpus_papers: list[dict]):
        """
        corpus_papers: list of dicts e.g., [{'id': 'p1', 'title': '...', 'abstract': '...'}]
        """
        self.corpus = corpus_papers
        # Simple whitespace tokenization for MVP
        tokenized_corpus = [doc.get('abstract', '').lower().split() for doc in corpus_papers]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query_terms: list[str], top_k: int = 3):
        """
        query_terms: list of gap concept strings, e.g., ['transformer', 'attention mechanism']
        """
        if not query_terms:
            return []
            
        # Combine terms into a single query string, then tokenize
        query = " ".join(query_terms).lower().split()
        
        # Get BM25 scores for all docs in the corpus
        doc_scores = self.bm25.get_scores(query)
        
        # Get indices of the top K scores
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0: # Only return if there's actually a match
                results.append({
                    "id": self.corpus[idx].get('id'),
                    "title": self.corpus[idx].get('title'),
                    "score": round(doc_scores[idx], 4)
                })
        return results
