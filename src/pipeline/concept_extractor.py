from keybert import KeyBERT

class ConceptExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = KeyBERT(model_name)
    

    def extract_concepts(self, text: str, top_n: int = 10):
        if not text:
            return []

        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=top_n
        )

        return [
            {
                "label": kw[0],
                "score": float(kw[1]),
                "type": "keyword"
            } for kw in keywords
        ]